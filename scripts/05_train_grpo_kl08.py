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
OUTPUT_DIR           = "models/green-rag-grpo-v1"
BASE_MODEL_ID        = os.getenv("SLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
if BASE_MODEL_ID in ("qwen2.5:3b", "qwen2.5"):
    BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# ****OLD PARAMS
# Training config

# TOTAL_STEPS          = 2000   # Training steps (each step = one batch of questions)
# LEARNING_RATE        = 6.5e-6
# GRADIENT_ACCUM       = 1
# KL_COEF              = 0.015  # β — KL penalty coefficient (keeps policy close to ref)
# CLIP_EPS             = 0.2    # ε — PPO-style clipping (applied inside GRPO loss)

# # Reward config
# FORMAT_CONSOLATION_REWARD = 0.0   # Reward for valid format but wrong answer
# JOULE_PENALTY_SCALE       = 0.04  # Multiply total_joules by this to get the penalty
# MAX_JOULE_PENALTY         = 0   # The penalty is CAPPED at this value, set at 0 for now, to be revisited


# # Dataset config
# MAX_QUESTIONS = None   # Cap on how many examples to draw from the dataset (None = unlimited)
# OVERFIT_MODE  = False  # Set to True to train repeatedly on two specific questions
# ONLY_EASY_MODE = False  # Set to True to train only on 'level == easy' questions from HotpotQA


# # Checkpoint / logging
# SAVE_EVERY    = 25   # Save every N steps — keep low to minimise work lost on crash
# EVAL_EVERY    = 50   # Evaluate on a validation set every N steps
# EVAL_SAMPLES  = 40   # How many questions to evaluate during the eval phase   # Save every N steps — keep low to minimise work lost on crash
# LOG_FILE      = None 

# Rollout & Optimization Hyperparameters
NUM_GENERATIONS      = 4       # Rollouts per prompt (G)
BATCH_SIZE           = 2       # Questions per gradient step
MAX_STEPS_PER_TRAJ   = 15      # <-- INCREASE THIS FROM 8 TO 15
TOTAL_STEPS          = 1000
LEARNING_RATE        = 2.0e-6
GRADIENT_ACCUM       = 4
KL_COEF              = 0.04   # Beta
CLIP_EPS             = 0.2     # Epsilon

# (Half of)  Empirically Calibrated Cost Penalty: 10,000 J = -1.0 Reward
JOULE_LAMBDA         = 5.0e-5

DANGER_PAY_BASE = 0.10
DANGER_PAY_CAP = 0.05
DANGER_PAY_PER_QUERY = 0.025

SAVE_EVERY           = 50
EVAL_EVERY           = 50
EVAL_SAMPLES         = 20
USE_WANDB            = True
WANDB_PROJECT        = "greenrag-grpo-kl08"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_judge = SoftJudge()
_ACTION_TOKEN_TENSOR = None

# ── Action Category Sets (CRUCIAL FOR REWARD CALCULATION) ──
_GENERATE_IDS   = {actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM}
_DECOMPOSE_IDS  = {actions.ACTION_DEC_LLM, actions.ACTION_DEC_RSN}
_REASON_IDS     = {actions.ACTION_RSN_SLM, actions.ACTION_RSN_LLM}
_KEYWORD_IDS    = {actions.ACTION_RET_KEY}
_DENSE_IDS      = {actions.ACTION_RET_VEC}

def get_action_token_tensor(tokenizer, device) -> torch.Tensor:
    global _ACTION_TOKEN_TENSOR
    if _ACTION_TOKEN_TENSOR is None or _ACTION_TOKEN_TENSOR.device != device:
        # Cache single-digit token IDs 0 through 8 (Includes ACTION_FAIL = 8)
        token_ids = [
            tokenizer.encode(str(a), add_special_tokens=False)[-1]
            for a in range(9)
        ]
        _ACTION_TOKEN_TENSOR = torch.tensor(token_ids, dtype=torch.long, device=device)
    return _ACTION_TOKEN_TENSOR


def compute_reward(
    final_state: Optional[GreenState],
    ground_truth: str,
    question: str,
    format_valid: bool,
    joule_lambda: float = JOULE_LAMBDA,
) -> Tuple[float, bool]:
    """
    Conservative Danger Pay Reward:
    Calculates energy penalties first, floors them, and THEN applies exploration subsidies 
    so that multi-hop failures always maintain a positive advantage over single-hop failures.
    """
    if final_state is None or not format_valid:
        return 0.0, False

    history = final_state.get("history", [])
    total_joules = float(final_state.get("total_joules", 0.0))
    joule_penalty = total_joules * joule_lambda

    has_decomposition = any(step.get("action_id") in _DECOMPOSE_IDS for step in history)
    completed_subqueries = len(final_state.get("completed_subqueries", []))

    final_answer = final_state.get("answer") or ""
    is_correct, reason = _judge.judge(final_answer, ground_truth, question)

    # ── CASE 1: SOLVED ────────────────────────────────────────────────────────
    if is_correct:
        base_reward = 1.0
        # Offset slightly for effort, but keep the ceiling strictly below Cheap success
        effort_offset = 0.10 if has_decomposition else 0.0
        
        reward = base_reward + effort_offset - joule_penalty
        
        logger.info(f"  [Reward] CORRECT | Decomp={has_decomposition} | Joules={total_joules:.1f}J | Net={reward:.3f}")
        return reward, True

    # ── CASE 2: FAILED / PARTIAL (THE DANGER PAY ZONE) ───────────────────────
    # Floor the base penalty at zero so it doesn't swallow our subsidies
    base_failed_reward = max(0.0, 0.0 - joule_penalty)
    
    danger_pay = 0.0
    if has_decomposition:
        danger_pay += DANGER_PAY_BASE  # Base subsidy for the hard path
        danger_pay += min(DANGER_PAY_CAP, completed_subqueries * DANGER_PAY_PER_QUERY)  # Up to +0.15 for partial credit
        
    # The Danger Pay is strictly additive to the floored base
    reward = base_failed_reward + danger_pay
    
    logger.info(
        f"  [Reward] WRONG ({reason}) | Decomp={has_decomposition} "
        f"| DangerPay=+{danger_pay:.2f} | Joules={total_joules:.1f}J | Net={reward:.3f}"
    )
    return reward, False


def sample_action(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    temperature: float = 0.8
) -> Tuple[int, int, torch.Tensor]:
    """
    1-token constrained forward pass. Zero regex parsing, zero token slicing.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    action_tokens = get_action_token_tensor(tokenizer, device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]  # Shape: [V]
        action_logits = logits[action_tokens]       # Shape: [9]

        scaled_logits = action_logits / max(temperature, 1e-4)
        probs = torch.softmax(scaled_logits, dim=-1)
        action_idx = torch.multinomial(probs, num_samples=1).item()

        # Log-probability of the chosen action in full vocabulary space
        vocab_log_probs = F.log_softmax(logits, dim=-1)
        chosen_token_id = action_tokens[action_idx].item()
        old_logp = vocab_log_probs[chosen_token_id].detach()

    return action_idx, chosen_token_id, old_logp


def rollout_one_trajectory(
    model,
    tokenizer,
    engine: GreenEngine,
    initial_state: GreenState,
    device: torch.device,
    temperature: float = 0.8,
) -> Tuple[List[Dict[str, Any]], Optional[GreenState], bool]:
    state = copy.deepcopy(initial_state)
    trajectory_steps = []
    format_valid = False

    for step_i in range(MAX_STEPS_PER_TRAJ):
        prompt = format_state_for_prompt(state) + " Action: "
        
        # Fast 1-token action sampling
        action_id, token_id, old_logp = sample_action(model, tokenizer, prompt, device, temperature)

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
    kl_coef: float = KL_COEF,
    clip_eps: float = CLIP_EPS,
) -> Tuple[float, float]:
    reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    mean_r = reward_tensor.mean()
    std_r = reward_tensor.std()

    # Standardized Group Advantages
    if std_r < 1e-6:
        # Zero variance: rollouts were identical; skip gradient update
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
            target_token_id = step_data["target_token_id"]
            old_logp = step_data["old_logprob"].to(device)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # 1. Reference Model Log-prob (Frozen SFT)
            model.set_adapter("reference")
            model.eval()
            with torch.no_grad():
                ref_logits = model(**inputs).logits[0, -1, :]
                ref_logp = F.log_softmax(ref_logits, dim=-1)[target_token_id]

            # 2. Active Model Log-prob (Trainable RL)
            model.set_adapter("active_rl")
            model.train()
            active_logits = model(**inputs).logits[0, -1, :]
            active_logp = F.log_softmax(active_logits, dim=-1)[target_token_id]

            # 3. PPO Clipped Surrogate Loss
            ratio = torch.exp(active_logp - old_logp)
            surr1 = -advantage * ratio
            surr2 = -advantage * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            pg_loss = torch.max(surr1, surr2)

            # 4. DeepSeek Unbiased KL Divergence Estimator
            kl_log_ratio = active_logp - ref_logp
            kl = torch.exp(-kl_log_ratio) + kl_log_ratio - 1.0
            kl_loss = kl_coef * kl

            step_loss = (pg_loss + kl_loss) * (1.0 / (len(traj) * num_valid * batch_scale))
            step_loss.backward()

            total_loss += (pg_loss.item() + kl_loss.item()) / len(traj)
            total_kl += kl.item() / len(traj)

    return total_loss / num_valid, total_kl / num_valid


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name=f"grpo_{run_id}")

    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# 1. Native BF16 Loading (No 4-Bit BnB)
    logger.info("Loading Base Model in native BF16: %s...", BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # ---> ACTIVATION VRAM FIX <---
    # Slashes activation VRAM so PyTorch can coexist with Ollama's 32B model
    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 2. Dual-Adapter Setup (Frozen Reference + Trainable Active Policy)
    logger.info("Setting up LoRA reference and active adapters...")
    # ---> MAKE SURE YOU KEEP THIS WRAPPER! <---
    model = PeftModel.from_pretrained(
        base_model, SFT_MODEL_PATH, adapter_name="reference", is_trainable=False
    )
    model.load_adapter(SFT_MODEL_PATH, adapter_name="active_rl", is_trainable=True)

    # Disable dropout to maintain exact PPO ratios
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    model.set_adapter("active_rl")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    # 3. Environment & Dataset
    retriever = GlobalRetriever.get_instance(corpus_type="fullwiki", index_type="hnsw")
    engine = GreenEngine(retriever=retriever, offline_energy=True)

    streamer = MixedStreamer(
        dataset_names=["hotpot"],
        weights=[1.0],
        configs={"hotpot": {"setting": "fullwiki", "split": "train"}}
    )
    data_iter = streamer.stream()

    logger.info("Beginning GRPO training across %d steps...", TOTAL_STEPS)

    for step in range(TOTAL_STEPS):
        optimizer.zero_grad()
        batch_samples = [next(data_iter) for _ in range(BATCH_SIZE)]
        
        batch_rewards = []
        batch_scaled_loss = 0.0
        batch_mean_kl = 0.0

        # Telemetry trackers
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

            # Phase 1: Group Rollout (G independent trajectories)
            for g in range(NUM_GENERATIONS):
                init_state = create_initial_state(
                    question,
                    ground_truth,
                    dataset=sample.get("dataset", "hotpot"),
                )
                traj, final_state, format_valid = rollout_one_trajectory(
                    model, tokenizer, engine, init_state, device, temperature=0.8
                )
                r, is_correct = compute_reward(final_state, ground_truth, question, format_valid)

                # Update Telemetry
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
                
                # Print exactly what the model decided for observability
                act_sequence = " -> ".join([str(t["action_id"]) for t in traj])
                logger.info(f"  Gen {g+1}/{NUM_GENERATIONS} | Path: [{act_sequence}] | Reward: {r:.3f}")

                trajectories.append(traj)
                rewards.append(r)

            batch_rewards.extend(rewards)

            # Phase 2: GRPO Gradient Update
            sample_scaled_loss, sample_mean_kl = grpo_loss(
                model, tokenizer, trajectories, rewards, device,
                batch_scale=BATCH_SIZE * GRADIENT_ACCUM
            )
            batch_scaled_loss += sample_scaled_loss
            batch_mean_kl += sample_mean_kl

        # Safe Gradient Accumulation and Step
        if (step + 1) % GRADIENT_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

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
            ckpt_path = os.path.join(OUTPUT_DIR, f"step_{step+1}")
            model.save_pretrained(ckpt_path, selected_adapters=["active_rl"])
            logger.info("Saved checkpoint to %s", ckpt_path)

            # Rehydrate eval iterator from the MixedStreamer
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
                init_state = create_initial_state(
                    eval_sample["question"],
                    eval_sample.get("answer", ""),
                    dataset=eval_sample.get("dataset", "hotpot"),
                )
                
                # Temperature 0.0 for greedy, deterministic evaluation
                traj, final_state, format_valid = rollout_one_trajectory(
                    model, tokenizer, engine, init_state, device, temperature=0.0
                )
                _, is_correct = compute_reward(final_state, eval_sample["answer"], eval_sample["question"], format_valid)
                eval_accuracy += float(is_correct)
                logger.info(f"  [Eval] Correct: {is_correct}")

            if len(eval_samples) > 0:
                mean_eval_acc = eval_accuracy / len(eval_samples)
                logger.info(f"  => Mean Evaluation Accuracy: {mean_eval_acc:.4f}")
                if USE_WANDB:
                    wandb.log({"eval/accuracy": mean_eval_acc, "step": step + 1})
            
            model.train() # Crucial: Return to train mode!
            logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()