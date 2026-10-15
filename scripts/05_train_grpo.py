"""
05_train_grpo.py — Production GRPO Training for GreenRAG
"""

import os
import sys
import time
import copy
import signal
import logging
import random
import argparse
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.append(os.getcwd())

from src.data.loader import MixedStreamer
from src.env.state import GreenState, create_initial_state
from src.env.retriever import GlobalRetriever
from src.env.engine import GreenEngine
from src.agent import actions
from src.agent.prompts import format_state_for_prompt
from src.oracle.judge import SoftJudge

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

SFT_MODEL_PATH       = "models/green-rag-sft-v1"
BASE_MODEL_ID        = os.getenv("SLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
if BASE_MODEL_ID in ("qwen2.5:3b", "qwen2.5"):
    BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

NUM_GENERATIONS      = 4       # Rollouts per prompt (G)
BATCH_SIZE           = 2       # Questions per gradient step
MAX_STEPS_PER_TRAJ   = 15      
TOTAL_STEPS          = 1000
CLIP_EPS             = 0.2     # Epsilon

SAVE_EVERY           = 50
EVAL_EVERY           = 50
EVAL_SAMPLES         = 20
USE_WANDB            = True
WANDB_PROJECT        = "greenrag-grpo"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_judge = SoftJudge()
_ACTION_TOKEN_TENSOR = None

# ── Action Category Sets ──
_GENERATE_IDS   = {actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM}
_DECOMPOSE_IDS  = {actions.ACTION_DEC_LLM, actions.ACTION_DEC_RSN}
_REASON_IDS     = {actions.ACTION_RSN_SLM, actions.ACTION_RSN_LLM}
_KEYWORD_IDS    = {actions.ACTION_RET_KEY}
_DENSE_IDS      = {actions.ACTION_RET_VEC}

def get_action_token_tensor(tokenizer, device) -> torch.Tensor:
    global _ACTION_TOKEN_TENSOR
    if _ACTION_TOKEN_TENSOR is None or _ACTION_TOKEN_TENSOR.device != device:
        token_ids = [
            tokenizer.encode(str(a), add_special_tokens=False)[-1]
            for a in range(9)
        ]
        _ACTION_TOKEN_TENSOR = torch.tensor(token_ids, dtype=torch.long, device=device)
    return _ACTION_TOKEN_TENSOR


def compute_reward(final_state, ground_truth, question, format_valid, joule_lambda, **kwargs) -> Tuple[float, bool]:
    if final_state is None or not format_valid:
        return -0.10, False
    
    # Did the agent actually attempt an answer?
    final_answer = (final_state.get("answer") or "").strip()
    status = final_state.get("status")

    # If it looped out without generating an answer, penalize harder than an honest miss
    if not final_answer or status not in ("SOLVED", "FAILED"):
        return -0.25, False

    total_joules = float(final_state.get("total_joules", 0.0))
    joule_penalty = total_joules * joule_lambda
    final_answer = final_state.get("answer") or ""
    is_correct, reason = _judge.judge(final_answer, ground_truth, question)

    if is_correct:
        reward = 1.0 - joule_penalty
        return max(0.50, reward), True
    else:
        # Equal flat penalty for all failures: prevents cowardice incentive
        return -0.10, False


def sample_action(
    model, 
    tokenizer, 
    prompt: str, 
    device: torch.device, 
    temperature: float = 0.8, 
    forced_action_id: Optional[int] = None
) -> Tuple[int, int, torch.Tensor]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        action_tokens = get_action_token_tensor(tokenizer, logits.device)
        action_logits = logits[action_tokens]  # Shape: [9]

        # Log-probability strictly in the 9-action space for exact KL match
        action_log_probs = F.log_softmax(action_logits, dim=-1)

        if forced_action_id is not None:
            action_idx = forced_action_id
        else:
            scaled_logits = action_logits / max(temperature, 1e-4)
            probs = torch.softmax(scaled_logits, dim=-1)
            action_idx = torch.multinomial(probs, num_samples=1).item()
        
        old_logp = action_log_probs[action_idx].detach()

    return action_idx, action_tokens[action_idx].item(), old_logp


def rollout_one_trajectory(
    model,
    tokenizer,
    engine: GreenEngine,
    initial_state: GreenState,
    device: torch.device,
    temperature: float = 0.8,
    forced_root_action: Optional[int] = None
) -> Tuple[List[Dict[str, Any]], Optional[GreenState], bool]:
    state = copy.deepcopy(initial_state)
    trajectory_steps = []
    format_valid = False

    for step_i in range(MAX_STEPS_PER_TRAJ):
        prompt = format_state_for_prompt(state) + " Action: "
        
        # Apply forced action override only at the root decision (Step 0)
        current_force = forced_root_action if step_i == 0 else None
        action_id, token_id, old_logp = sample_action(
            model, tokenizer, prompt, device, temperature, forced_action_id=current_force
        )

        trajectory_steps.append({
            "prompt": prompt,
            "action_id": action_id,
            "target_token_id": token_id,
            "old_logprob": old_logp,
        })

        try:
            new_state = engine.step(state, action_id)
            format_valid = True
        except Exception as exc:
            logger.warning("Engine step error: %s", exc)
            break

        state = new_state
        if state.get("status") in ("SOLVED", "FAILED"):
            break

    return trajectory_steps, state, format_valid


def grpo_loss(
    model,
    tokenizer,
    trajectories: List[List[Dict[str, Any]]],
    rewards: List[float],
    device: torch.device,
    batch_scale: float = 1.0,
    kl_coef: float = 0.04,
    clip_eps: float = CLIP_EPS,
) -> Tuple[float, float]:
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    mean_r = reward_tensor.mean()
    std_r = reward_tensor.std()

    if std_r < 1e-6:
        return 0.0, 0.0
    advantages = (reward_tensor - mean_r) / (std_r + 1e-8)

    num_valid = sum(1 for t in trajectories if len(t) > 0)
    if num_valid == 0:
        return 0.0, 0.0

    total_loss = 0.0
    total_kl = 0.0

    for traj, advantage in zip(trajectories, advantages):
        if not traj:
            continue

        for step_data in traj:
            prompt = step_data["prompt"]
            old_logp = step_data["old_logprob"].to(device)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # 1. Reference Model Log-probs (Frozen SFT)
            model.set_adapter("reference")
            model.eval()
            with torch.no_grad():
                ref_out = model(**inputs).logits[0, -1, :]
                action_tokens = get_action_token_tensor(tokenizer, ref_out.device)
                ref_logits = ref_out[action_tokens]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # 2. Active Model Log-probs (Trainable Policy)
            model.set_adapter("active_rl")
            model.train()
            active_out = model(**inputs).logits[0, -1, :]
            active_logits = active_out[action_tokens]
            active_log_probs = F.log_softmax(active_logits, dim=-1)
            active_probs = F.softmax(active_logits, dim=-1)

            # 3. PPO Clipped Surrogate Loss
            active_logp = active_log_probs[step_data["action_id"]]
            ratio = torch.exp(active_logp - old_logp)
            surr1 = -advantage * ratio
            surr2 = -advantage * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            pg_loss = torch.max(surr1, surr2)

            # 4. EXACT Analytic Categorical KL: sum(P_act * (log P_act - log P_ref))
            kl = (active_probs * (active_log_probs - ref_log_probs)).sum()
            kl_loss = kl_coef * kl

            step_loss = (pg_loss + kl_loss) * (1.0 / (len(traj) * num_valid * batch_scale))
            step_loss.backward()

            total_loss += (pg_loss.item() + kl_loss.item()) / len(traj)
            total_kl += kl.item() / len(traj)

    return total_loss / num_valid, total_kl / num_valid


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for GreenRAG")
    parser.add_argument("--learning_rate", type=float, default=2.0e-6)
    parser.add_argument("--kl_coef", type=float, default=0.04)
    parser.add_argument("--danger_pay_base", type=float, default=0.10)
    parser.add_argument("--run_name", type=str, default="greenrag-grpo-baseline")
    parser.add_argument("--output_dir", type=str, default="models/green-rag-grpo-baseline")
    parser.add_argument("--gradient_accum", type=int, default=4)
    parser.add_argument("--joule_lambda", type=float, default=1.0e-5, help="Energy penalty coefficient")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    if USE_WANDB:
        # Hide SLURM variables from wandb to stop 'scontrol' shell spam
        os.environ.pop("SLURM_JOBID", None)
        os.environ.pop("SLURM_JOB_ID", None)
        
        wandb.init(project=WANDB_PROJECT, name=args.run_name)

    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading Base Model in native BF16: %s...", BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    logger.info("Setting up LoRA reference and active adapters...")
    model = PeftModel.from_pretrained(
        base_model, SFT_MODEL_PATH, adapter_name="reference", is_trainable=False
    )
    model.load_adapter(SFT_MODEL_PATH, adapter_name="active_rl", is_trainable=True)

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    model.set_adapter("active_rl")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    retriever = GlobalRetriever.get_instance(corpus_type="fullwiki", index_type="hnsw")
    engine = GreenEngine(retriever=retriever)

    streamer = MixedStreamer(
        dataset_names=["hotpot"],
        weights=[1.0],
        configs={"hotpot": {"setting": "fullwiki", "split": "train"}}
    )
    data_iter = streamer.stream()

    logger.info("Beginning GRPO training across %d steps...", TOTAL_STEPS)
    optimizer.zero_grad()

    for step in range(TOTAL_STEPS):
        batch_samples = [next(data_iter) for _ in range(BATCH_SIZE)]
        
        batch_rewards = []
        batch_scaled_loss = 0.0
        batch_mean_kl = 0.0

        batch_correct = []
        batch_joules = []
        action_counts = {
            "generate": 0, "keyword": 0, "dense": 0, "decompose": 0, "reason": 0
        }
        total_actions = 0

        for sample in batch_samples:
            question = sample["question"]
            ground_truth = sample["answer"]

            logger.info(f"\n--- Processing Q: {question[:80]}... ---")

            trajectories = []
            rewards = []

            for g in range(NUM_GENERATIONS):
                init_state = create_initial_state(question)
                
                # Structural exploration override at root step
                forced_first_action = None
                if g == 0:
                    forced_first_action = actions.ACTION_DEC_LLM  # Force decomposition


                traj, final_state, format_valid = rollout_one_trajectory(
                    model, tokenizer, engine, init_state, device, 
                    temperature=0.8, forced_root_action=forced_first_action
                )
                
                r, is_correct = compute_reward(
                    final_state, ground_truth, question, format_valid, 
                    joule_lambda=args.joule_lambda
                )

                batch_correct.append(float(is_correct))
                batch_joules.append(float(final_state.get("total_joules", 0.0)) if final_state else 0.0)
                
                for t_step in traj:
                    total_actions += 1
                    aid = t_step["action_id"]
                    if aid in _GENERATE_IDS: action_counts["generate"] += 1
                    elif aid in _KEYWORD_IDS: action_counts["keyword"] += 1
                    elif aid in _DENSE_IDS: action_counts["dense"] += 1
                    elif aid in _DECOMPOSE_IDS: action_counts["decompose"] += 1
                    elif aid in _REASON_IDS: action_counts["reason"] += 1
                
                act_sequence = " -> ".join([str(t["action_id"]) for t in traj])
                logger.info(f"  Gen {g+1}/{NUM_GENERATIONS} | Path: [{act_sequence}] | Reward: {r:.3f}")

                trajectories.append(traj)
                rewards.append(r)

            batch_rewards.extend(rewards)

            sample_scaled_loss, sample_mean_kl = grpo_loss(
                model, tokenizer, trajectories, rewards, device,
                batch_scale=BATCH_SIZE * args.gradient_accum,
                kl_coef=args.kl_coef
            )
            batch_scaled_loss += sample_scaled_loss
            batch_mean_kl += sample_mean_kl

        if (step + 1) % args.gradient_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()

        avg_loss = batch_scaled_loss / BATCH_SIZE
        avg_kl = batch_mean_kl / BATCH_SIZE
        safe_total_actions = max(1, total_actions)

        logger.info(
            "\nStep [%d/%d] | Mean Batch Reward: %.3f | Loss: %.4f | KL: %.4f",
            step + 1, TOTAL_STEPS, np.mean(batch_rewards), avg_loss, avg_kl
        )

        if USE_WANDB:
            wandb.log({
                "train/reward_mean": np.mean(batch_rewards),
                "train/loss": avg_loss,
                "train/kl": avg_kl,
                "metrics/accuracy": np.mean(batch_correct),
                "metrics/mean_joules": np.mean(batch_joules),
                "tools/generate_pct": action_counts["generate"] / safe_total_actions,
                "tools/keyword_pct": action_counts["keyword"] / safe_total_actions,
                "tools/dense_pct": action_counts["dense"] / safe_total_actions,
                "tools/decompose_pct": action_counts["decompose"] / safe_total_actions,
                "tools/reason_pct": action_counts["reason"] / safe_total_actions,
                "step": step + 1,
            })

        if (step + 1) % SAVE_EVERY == 0:
            ckpt_path = os.path.join(args.output_dir, f"step_{step+1}")
            model.save_pretrained(ckpt_path, selected_adapters=["active_rl"])
            logger.info("Saved checkpoint to %s", ckpt_path)

        if (step + 1) % EVAL_EVERY == 0:
            logger.info(f"{'='*20} Evaluation Phase (Step {step+1}) {'='*20}")
            model.eval()
            eval_accuracy = 0.0

            eval_streamer = MixedStreamer(
                dataset_names=["hotpot"],
                weights=[1.0],
                configs={"hotpot": {"setting": "fullwiki", "split": "validation"}}
            )
            eval_iter = eval_streamer.stream()
            eval_samples = []
            for _ in range(EVAL_SAMPLES):
                try:
                    eval_samples.append(next(eval_iter))
                except StopIteration:
                    break
            
            for eval_sample in eval_samples:
                logger.info(f"  [Eval] Q: {eval_sample['question'][:80]}")
                init_state = create_initial_state(eval_sample["question"])
                
                traj, final_state, format_valid = rollout_one_trajectory(
                    model, tokenizer, engine, init_state, device, temperature=0.0
                )
                
                _, is_correct = compute_reward(
                    final_state, 
                    eval_sample["answer"], 
                    eval_sample["question"], 
                    format_valid, 
                    joule_lambda=args.joule_lambda
                )

                eval_accuracy += float(is_correct)
                logger.info(f"  [Eval] Correct: {is_correct}")

            if len(eval_samples) > 0:
                mean_eval_acc = eval_accuracy / len(eval_samples)
                logger.info(f"  => Mean Evaluation Accuracy: {mean_eval_acc:.4f}")
                if USE_WANDB:
                    wandb.log({"eval/accuracy": mean_eval_acc, "step": step + 1})
            
            model.train()
            logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()