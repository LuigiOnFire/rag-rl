"""
05_train_grpo.py — Production GRPO Training for GreenRAG (Causal Return-to-Go)
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
from src.env.state import GreenState, create_initial_state, get_active_subquery

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
        return reward, True
    else:
        # Equal flat penalty for all failures: prevents cowardice incentive
        return -0.10, False


def compute_causal_returns(
    traj: List[Dict[str, Any]],
    root_reward: float,
    is_correct: bool,
    joule_lambda: float,
) -> List[float]:
    """
    Computes Bellman causal return-to-go G_{i,t} for each step t in a trajectory.
    For solved trajectories, G_{i,t} = 1.0 - joule_lambda * sum_{k=t}^{T-1} J_{i,k},
    stripping out sunk energy costs J_{<t} incurred prior to step t.
    """
    n_steps = len(traj)
    if n_steps == 0:
        return []
    if not is_correct:
        return [root_reward] * n_steps

    returns = [0.0] * n_steps
    remaining_joules = 0.0
    for t in range(n_steps - 1, -1, -1):
        remaining_joules += float(traj[t].get("step_joules", 0.0))
        returns[t] = 1.0 - joule_lambda * remaining_joules
    return returns


def compute_step_advantages(
    trajectories: List[List[Dict[str, Any]]],
    rewards: List[float],
    correct_flags: List[bool],
    joule_lambda: float,
    no_force_trajectory: bool,
) -> Tuple[Optional[List[List[float]]], bool]:
    """
    Computes depth-aligned Dr. GRPO advantages A_{i,t} = G_{i,t} - mean(G_{.,t}).
    At depth t, the comparison cohort includes rollouts still active at step t
    plus rollouts already absorbed into a successful terminal state (G^abs = 1.0).
    """
    step_returns = [
        compute_causal_returns(traj, r, corr, joule_lambda)
        for traj, r, corr in zip(trajectories, rewards, correct_flags)
    ]

    if no_force_trajectory:
        active_indices = [i for i, traj in enumerate(trajectories) if len(traj) > 0]
    else:
        # g=0 is forced root exploration (ACTION_DEC_LLM).
        # Include g=0 in the baseline only when it outperforms organic rollouts.
        organic_rewards = rewards[1:]
        organic_mean = float(np.mean(organic_rewards)) if organic_rewards else 0.0
        if len(trajectories[0]) > 0 and rewards[0] > organic_mean + 1e-6:
            active_indices = [i for i, traj in enumerate(trajectories) if len(traj) > 0]
        else:
            active_indices = [i for i in range(1, len(trajectories)) if len(trajectories[i]) > 0]

    if len(active_indices) < 2:
        return None, False

    root_vals = [step_returns[i][0] for i in active_indices]
    root_mean = sum(root_vals) / len(root_vals)
    max_len = max(len(trajectories[i]) for i in active_indices)

    step_advantages = [[0.0] * len(traj) for traj in trajectories]
    any_nonzero = False

    for t in range(max_len):
        cohort_vals = []
        for i in active_indices:
            if t < len(trajectories[i]):
                cohort_vals.append(step_returns[i][t])
            elif correct_flags[i]:
                # Absorbed into a solved terminal state prior to step t (0 future Joules)
                cohort_vals.append(1.0)

        if len(cohort_vals) >= 2:
            baseline_t = sum(cohort_vals) / len(cohort_vals)
            all_failed_t = max(cohort_vals) <= 0.0
        else:
            baseline_t = root_mean
            all_failed_t = max(root_vals) <= 0.0

        for i in active_indices:
            if t < len(trajectories[i]):
                adv = step_returns[i][t] - baseline_t
                if all_failed_t:
                    adv = min(0.0, adv)
                step_advantages[i][t] = adv
                if abs(adv) >= 1e-8:
                    any_nonzero = True

    if not any_nonzero:
        return None, False

    all_passed = all(correct_flags[i] for i in active_indices)
    return step_advantages, all_passed


def get_valid_action_mask(state: GreenState, device: torch.device) -> torch.Tensor:
    """
    Returns a boolean mask of shape [9] indicating valid primitive actions
    given the current subquery state machine.
    """
    mask = torch.zeros(9, dtype=torch.bool, device=device)
    history = state.get("history", [])
    subqueries = state.get("subqueries", [])
    active_sq = get_active_subquery(state)

    # 1. Root step (Step 0): allow all core actions except abort (8)
    if not history:
        mask[[0, 1, 2, 3, 4, 5, 6, 7]] = True
        return mask

    last_act = history[-1].get("action_id")

    # 2. Inside an active Decomposition Subquery Loop
    if active_sq is not None:
        if last_act in (6, 7, 0, 1):
            # Just decomposed OR just answered the previous subquery:
            # MUST retrieve documents for the new active subquery first
            mask[[2, 3]] = True
        elif last_act in (2, 3):
            # Just retrieved for the active subquery:
            # Either reason over the docs (4, 5) or answer the subquery (0, 1)
            mask[[0, 1, 4, 5]] = True
        else:
            # Just reasoned (4, 5) on the active subquery:
            # MUST answer the subquery now to advance to the next subquery
            mask[[0, 1]] = True
        return mask

    # 3. All subqueries have been ANSWERED: synthesize final answer
    if len(subqueries) > 0 and active_sq is None:
        mask[[0, 1]] = True
        return mask

    # 4. Standard (non-decomposed) trajectory after Step 0
    if last_act in (2, 3):
        # After retrieval: can generate (0, 1), reason (4, 5), retrieve again (2, 3), or decompose (6)
        mask[[0, 1, 2, 3, 4, 5, 6]] = True
    elif last_act in (4, 5):
        # After reasoning: must retrieve (2, 3) or generate (0, 1) — prevents 5->5->5 loops
        mask[[0, 1, 2, 3]] = True
    else:
        mask[[0, 1, 2, 3, 4, 5]] = True

    return mask


def sample_action(
    model, 
    tokenizer, 
    prompt: str, 
    device: torch.device, 
    valid_mask: torch.Tensor,
    temperature: float = 0.8, 
    forced_action_id: Optional[int] = None
) -> Tuple[int, int, torch.Tensor]:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        action_tokens = get_action_token_tensor(tokenizer, logits.device)
        action_logits = logits[action_tokens].float()  # Shape: [9]
        action_logits = action_logits.masked_fill(~valid_mask, -1e9)

        # Log-probability strictly in the masked 9-action space for exact KL match
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
    model.set_adapter("active_rl")
    model.eval()

    state = copy.deepcopy(initial_state)
    trajectory_steps = []
    format_valid = False

    for step_i in range(MAX_STEPS_PER_TRAJ):
        prompt = format_state_for_prompt(state) + " Action: "
        valid_mask = get_valid_action_mask(state, device)
        
        # Apply forced action override only at the root decision (Step 0)
        current_force = forced_root_action if step_i == 0 else None
        action_id, token_id, old_logp = sample_action(
            model, tokenizer, prompt, device, valid_mask,
            temperature=temperature, forced_action_id=current_force
        )

        prev_joules = float(state.get("total_joules", 0.0))
        trajectory_steps.append({
            "prompt": prompt,
            "action_id": action_id,
            "target_token_id": token_id,
            "old_logprob": old_logp,
            "valid_mask": valid_mask.detach(),
            "step_i": step_i,
            "step_joules": 0.0,
        })

        try:
            new_state = engine.step(state, action_id)
            format_valid = True
        except Exception as exc:
            logger.warning("Engine step error: %s", exc)
            break

        curr_joules = float(new_state.get("total_joules", prev_joules))
        trajectory_steps[-1]["step_joules"] = max(0.0, curr_joules - prev_joules)

        state = new_state
        if state.get("status") in ("SOLVED", "FAILED"):
            break

    return trajectory_steps, state, format_valid


def grpo_loss(
    model,
    tokenizer,
    trajectories: List[List[Dict[str, Any]]],
    rewards: List[float],
    correct_flags: List[bool],
    joule_lambda: float,
    device: torch.device,
    batch_scale: float = 1.0,
    kl_coef: float = 0.04,
    clip_eps: float = CLIP_EPS,
    no_force_trajectory: bool = False,
) -> Tuple[float, float]:
    step_advantages, all_passed = compute_step_advantages(
        trajectories=trajectories,
        rewards=rewards,
        correct_flags=correct_flags,
        joule_lambda=joule_lambda,
        no_force_trajectory=no_force_trajectory,
    )
    if step_advantages is None:
        return 0.0, 0.0

    # Conditionally decouple SFT KL tether on all-pass batches
    effective_kl_coef = 0.0 if all_passed else kl_coef

    num_valid = sum(1 for t in trajectories if len(t) > 0)
    if num_valid == 0:
        return 0.0, 0.0

    total_loss = 0.0
    total_kl = 0.0

    for traj, traj_advs, raw_r, is_corr in zip(trajectories, step_advantages, rewards, correct_flags):
        if not traj:
            continue

        is_timeout = (raw_r <= -0.20)
        in_subquery_loop = False

        for step_data, step_adv in zip(traj, traj_advs):
            step_i = step_data.get("step_i", 0)
            act_id = step_data["action_id"]

            if abs(step_adv) < 1e-8:
                if act_id in _DECOMPOSE_IDS:
                    in_subquery_loop = True
                continue

            # 1. If trajectory timed out, punish downstream loop steps (step_i > 0), not root
            if is_timeout and step_i == 0:
                continue

            # 2. If trajectory PASSED but got negative step advantage due to energy cost,
            # penalize the routing decision that triggered decomposition, NOT internal subquery solver steps
            if is_corr and step_adv < 0.0 and in_subquery_loop:
                continue
            if act_id in _DECOMPOSE_IDS:
                in_subquery_loop = True

            advantage = torch.tensor(step_adv, dtype=torch.float32, device=device)
            prompt = step_data["prompt"]
            old_logp = step_data["old_logprob"].to(device)
            valid_mask = step_data["valid_mask"].to(device)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # 1. Reference Model Log-probs (Frozen SFT)
            model.set_adapter("reference")
            model.eval()
            with torch.no_grad():
                ref_out = model(**inputs).logits[0, -1, :]
                action_tokens = get_action_token_tensor(tokenizer, ref_out.device)
                ref_logits = ref_out[action_tokens].float()
                ref_logits = ref_logits.masked_fill(~valid_mask, -1e9)
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)

            # 2. Active Model Log-probs (Trainable Policy)
            model.set_adapter("active_rl")
            model.train()
            active_out = model(**inputs).logits[0, -1, :]
            active_logits = active_out[action_tokens].float()
            active_logits = active_logits.masked_fill(~valid_mask, -1e9)
            active_log_probs = F.log_softmax(active_logits, dim=-1)
            active_probs = F.softmax(active_logits, dim=-1)

            # 3. PPO Clipped Surrogate Loss
            active_logp = active_log_probs[act_id]
            ratio = torch.exp(active_logp - old_logp)
            surr1 = -advantage * ratio
            surr2 = -advantage * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            pg_loss = torch.max(surr1, surr2)

            # 4. EXACT Analytic Categorical KL over valid actions
            kl = (active_probs * (active_log_probs - ref_log_probs)).sum()
            kl_loss = effective_kl_coef * kl

            if step_i == 0:
                traj_norm = 1.0
            else:
                traj_norm = float(max(1, len(traj) - 1))

            step_loss = (pg_loss + kl_loss) * (1.0 / (traj_norm * num_valid * batch_scale))
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
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint directory to resume from")
    parser.add_argument("--start_step", type=int, default=0, help="Step number to resume training from")
    parser.add_argument("--datasets", type=str, default="hotpot", help="Comma-separated dataset names, e.g., 'hotpot' or 'hotpot,squad'")
    parser.add_argument("--total_steps", type=int, default=2000, help="Total training steps")
    parser.add_argument(
        "--no-force-trajectory",
        "--no_force_trajectory",
        dest="no_force_trajectory",
        action="store_true",
        help="Disable the forced decomposition rollout used for structural exploration",
    )

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
    active_adapter_path = args.resume_from if args.resume_from else SFT_MODEL_PATH
    if os.path.isdir(os.path.join(active_adapter_path, "active_rl")):
        active_adapter_path = os.path.join(active_adapter_path, "active_rl")
        
    logger.info("Loading active_rl adapter from: %s", active_adapter_path)
    model.load_adapter(active_adapter_path, adapter_name="active_rl", is_trainable=True)

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0

    model.set_adapter("active_rl")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    retriever = GlobalRetriever.get_instance(corpus_type="fullwiki", index_type="hnsw")
    engine = GreenEngine(retriever=retriever)

    # 1. Initialize training data stream
    dataset_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    configs = {}
    for d in dataset_list:
        configs[d] = {"setting": "fullwiki", "split": "train"} if d == "hotpot" else {"split": "train"}

    streamer = MixedStreamer(
        dataset_names=dataset_list,
        weights=[1.0 / len(dataset_list)] * len(dataset_list),
        configs=configs
    )
    data_iter = streamer.stream()

    # Pre-load deterministic evaluation samples ONCE
    logger.info("Pre-loading %d deterministic evaluation samples...", EVAL_SAMPLES)
    eval_streamer = MixedStreamer(
        dataset_names=["hotpot"],
        weights=[1.0],
        shuffle=True,
        seed=42,  # Fully deterministic across all runs, seeds, and machines
        configs={"hotpot": {"setting": "fullwiki", "split": "validation"}}
    )
    eval_iter = eval_streamer.stream()
    fixed_eval_samples = []
    for _ in range(EVAL_SAMPLES):
        try:
            fixed_eval_samples.append(next(eval_iter))
        except StopIteration:
            break

    logger.info("Beginning GRPO training across %d steps...", args.total_steps)
    optimizer.zero_grad()

    
    for step in range(args.start_step, args.total_steps):
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
            correct_flags = []

            for g in range(NUM_GENERATIONS):
                init_state = create_initial_state(question)
                
                # Structural exploration override at root step, unless disabled by the CLI.
                forced_first_action = None
                if g == 0 and not args.no_force_trajectory:
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
                correct_flags.append(is_correct)

            batch_rewards.extend(rewards)

            sample_scaled_loss, sample_mean_kl = grpo_loss(
                model, tokenizer, trajectories, rewards, correct_flags,
                args.joule_lambda, device,
                batch_scale=BATCH_SIZE * args.gradient_accum,
                kl_coef=args.kl_coef,
                no_force_trajectory=args.no_force_trajectory,
            )
            
            batch_scaled_loss += sample_scaled_loss
            batch_mean_kl += sample_mean_kl

            # Clear PyTorch cached VRAM after every question so Ollama has headroom
            torch.cuda.empty_cache()

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
            step + 1, args.total_steps, np.mean(batch_rewards), avg_loss, avg_kl
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

            for eval_sample in fixed_eval_samples:
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

            if len(fixed_eval_samples) > 0:
                mean_eval_acc = eval_accuracy / len(fixed_eval_samples)
                logger.info(f"  => Mean Evaluation Accuracy: {mean_eval_acc:.4f}")
                if USE_WANDB:
                    wandb.log({"eval/accuracy": mean_eval_acc, "step": step + 1})
            
            model.train()
            torch.cuda.empty_cache()
            logger.info(f"{'='*60}")

if __name__ == "__main__":
    main()