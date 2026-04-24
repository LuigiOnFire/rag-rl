"""
05_train_grpo.py — GRPO Training for the GreenRAG Agent
========================================================
Architecture: Two-Phase "Rollout → GRPO Update"

Phase 1 (Rollout):
    For each question in a batch we generate NUM_GENERATIONS independent
    trajectories. Each trajectory is built by an interleaved loop:
        while not done:
            LLM generates text up to a stop token   → raw action string
            Parse action_id from the raw text
            GreenEngine.step(state, action_id)       → new_state + observation
            Append observation to the running prompt
    At the end of the loop the final state carries a reward signal.

Phase 2 (GRPO Update):
    The NUM_GENERATIONS completed trajectory strings + their scalar rewards
    are passed into a manual GRPO loss computation. No critic/value-head is
    needed; the group-relative advantage is computed purely from the reward
    spread within the generation group.

Reward shaping:
    - Correct answer (SoftJudge)  → 1.0 minus a capped Joule penalty
    - Wrong answer, valid format → FORMAT_CONSOLATION_REWARD (+0.1)
    - Wrong answer, broken format → 0.0
"""

import os
import sys
import re
import time
import copy
import signal
import logging
import traceback
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import wandb

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
from transformers import AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

sys.path.append(os.getcwd())

from src.data.loader import MixedStreamer
from src.env.state import GreenState, create_initial_state
from src.env.retriever import GlobalRetriever
from src.env.engine import GreenEngine
from src.agent import actions
from src.agent.prompts import format_state_for_prompt
from src.oracle.judge import SoftJudge

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — tweak freely
# ──────────────────────────────────────────────────────────────────────────────

SFT_MODEL_PATH       = "models/green-rag-sft-v1"
OUTPUT_DIR           = "models/green-rag-grpo-v1"

# Rollout config
NUM_GENERATIONS      = 4     # Trajectories per question (the "group" in GRPO)
BATCH_SIZE           = 2     # Questions per outer training step
MAX_STEPS_PER_TRAJ   = 8     # Max engine steps before forcing termination
MAX_NEW_TOKENS       = 4     # Tokens generated per LLM call inside the loop

# Training config
TOTAL_STEPS          = 300   # Training steps (each step = one batch of questions)
LEARNING_RATE        = 6.5e-6
GRADIENT_ACCUM       = 1
KL_COEF              = 0.015  # β — KL penalty coefficient (keeps policy close to ref)
CLIP_EPS             = 0.2    # ε — PPO-style clipping (applied inside GRPO loss)

# Reward config
FORMAT_CONSOLATION_REWARD = 0.0   # Reward for valid format but wrong answer
JOULE_PENALTY_SCALE       = 0.04  # Multiply total_joules by this to get the penalty
MAX_JOULE_PENALTY         = 0.20   # The penalty is CAPPED at this value, set at 0 for now, to be revisited

# Dataset config
MAX_QUESTIONS = None   # Cap on how many examples to draw from the dataset (None = unlimited)
OVERFIT_MODE  = False  # Set to True to train repeatedly on two specific questions
ONLY_EASY_MODE = True  # Set to True to train only on 'level == easy' questions from HotpotQA

# Checkpoint / logging
SAVE_EVERY    = 25   # Save every N steps — keep low to minimise work lost on crash
EVAL_EVERY    = 25   # Evaluate on a validation set every N steps
EVAL_SAMPLES  = 5    # How many questions to evaluate during the eval phase   # Save every N steps — keep low to minimise work lost on crash
run_id        = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE      = f"data/ppo_training/grpo_run_{run_id}.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Dashboard logging (Weights & Biases)
USE_WANDB      = True          # Set False to disable; falls back to log-only
WANDB_PROJECT  = "greenrag-grpo"
WANDB_RUN_NAME = f"grpo_{run_id}"  # forward-ref to run_id defined below


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

# Silence noisy HTTP logs from Ollama / Requests
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

trace_logger = logging.getLogger("LLM_TRACE")
trace_logger.addHandler(logging.NullHandler())

# ── Crash capture ────────────────────────────────────────────────────────────
# Route any unhandled Python exception into the log file (not just stderr).
def _log_unhandled_exception(exc_type, exc_value, exc_tb):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    logger.critical(
        "Unhandled exception — training crashed:",
        exc_info=(exc_type, exc_value, exc_tb),
    )
sys.excepthook = _log_unhandled_exception

# Log GPU memory + a clean message if we receive SIGTERM (e.g. from the scheduler).
def _sigterm_handler(signum, frame):
    msg = "Received SIGTERM — process is being terminated."
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        msg += f" GPU mem: allocated={alloc:.2f} GB  reserved={reserved:.2f} GB"
    logger.critical(msg)
    sys.exit(1)
signal.signal(signal.SIGTERM, _sigterm_handler)

# ──────────────────────────────────────────────────────────────────────────────
# REWARD CALCULATION
# ──────────────────────────────────────────────────────────────────────────────

_judge = SoftJudge()

def compute_reward(
    final_state: Optional[GreenState],
    ground_truth: str,
    question: str,
    format_valid: bool,
    joule_penalty_scale: float = JOULE_PENALTY_SCALE,
    max_joule_penalty: float = MAX_JOULE_PENALTY,
) -> Tuple[float, bool]:
    """
    Returns (scalar_reward, is_correct).

    Correct answer:    1.0  - min(max_joule_penalty, total_joules * joule_penalty_scale)
    Valid format only: FORMAT_CONSOLATION_REWARD  (+0.1)
    Broken format:     0.0
    """
    if final_state is None or not format_valid:
        return FORMAT_CONSOLATION_REWARD, False

    final_answer = final_state.get("answer") or ""

    if not final_answer:
        # No answer produced despite valid format → consolation
        return FORMAT_CONSOLATION_REWARD, False

    is_correct, reason = _judge.judge(final_answer, ground_truth, question)

    if is_correct:
        total_joules = float(final_state.get("total_joules", 0.0))
        joule_penalty = min(max_joule_penalty, total_joules * joule_penalty_scale)
        reward = 1.0 - joule_penalty
        logger.info(f"  [Reward] CORRECT | joules={total_joules:.3f} penalty={joule_penalty:.3f} reward={reward:.3f}")
        return reward, True
    else:
        logger.info(f"  [Reward] WRONG ({reason}) | answer='{final_answer[:60]}'")
        return FORMAT_CONSOLATION_REWARD, False  # right format, wrong answer


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — SINGLE TRAJECTORY ROLLOUT
# ──────────────────────────────────────────────────────────────────────────────

def _parse_action(text: str) -> Tuple[int, str]:
    """
    Parse 'Action: <id>\\nInput: <argument>' from raw LLM output.
    Returns (action_id, argument). action_id == -1 signals parse failure.

    The action ID must appear as the FIRST token in the generated text
    (possibly preceded by a space/colon), matching what the prompt ends with:
    '...\\nAction:' → model generates ' 2\\nInput: ...'
    """
    # Anchor to the start: optional whitespace/colon then a single digit
    act_match = re.match(r"[\s:]*(\d)", text)
    action_id = int(act_match.group(1)) if act_match else -1

    arg_match = re.search(r"Input:\s*(.*)", text, re.DOTALL)
    argument = arg_match.group(1).strip() if arg_match else ""
    return action_id, argument


def rollout_one_trajectory(
    model,
    tokenizer,
    engine: GreenEngine,
    initial_state: GreenState,
    generation_config: GenerationConfig,
    device: torch.device,
) -> Tuple[List[Dict[str, str]], Optional[GreenState], bool]:
    """
    Run one complete trajectory by interleaving LLM generation and GreenEngine steps.

    Returns
    -------
    trajectory_steps : List[Dict[str, str]]
        A list of state-action pairs containing {"prompt": ..., "completion": ...}
        This guarantees each decision is judged on the exact state representation
        active at that moment.
    final_state : GreenState | None
        State at termination, or None on catastrophic error.
    format_valid : bool
        True if at least one action was successfully parsed and executed.
    """
    state = copy.deepcopy(initial_state)

    # The "running prompt" starts as the formatted initial observation.
    current_prompt = format_state_for_prompt(state) + "\nAction: "

    trajectory_steps = []
    format_valid = False  # Becomes True once we successfully parse+execute an action

    for step_i in range(MAX_STEPS_PER_TRAJ):
        logger.info(f"\n--- STEP {step_i+1} ---")
        logger.info(f"[MODEL SEES]\n{current_prompt}")

        # ── LLM CALL ────────────────────────────────────────────────────────
        inputs = tokenizer(
            current_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device) # <--- Extract mask

        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True

        # 2. Generate with KV Cache explicitly enabled
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                generation_config=generation_config,
                tokenizer=tokenizer,
                stop_strings=generation_config.stop_strings,
                attention_mask=attention_mask,
                use_cache=True, # <--- Force KV Cache on
                output_logits=True,
                return_dict_in_generate=True,
            )
            output_ids = outputs.sequences
            scores = outputs.logits  # Tuple of (batch_size, vocab_size) for each generated step

            # Exact logprobs from sampling time (no dropout/drift)
            prompt_len = input_ids.shape[1]
            new_token_ids = output_ids[0, prompt_len:]
            gen_len = len(scores)
            
            old_logprobs_gen = []
            for i in range(gen_len):
                tok_logits = scores[i][0]  # (vocab_size,)
                tok_logprobs = F.log_softmax(tok_logits, dim=-1)
                # new_token_ids[i] is the token that was sampled at step i
                old_logprobs_gen.append(tok_logprobs[new_token_ids[i]].detach().cpu())
            
            # Massive Memory Leak Fix: Drop generator outputs before training
            del outputs, scores, tok_logits, tok_logprobs
            torch.cuda.empty_cache()

            # 3. Brutally force Checkpointing back ON and Cache OFF for training
        model.config.use_cache = False        
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train()

        # Decode only the *new* tokens (after the prompt)
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        logger.info(f"\n[MODEL SAID]\n{generated_text}")

        # ── PARSE ACTION ───────────────────────────────────────────────
        action_id, argument = _parse_action(generated_text)

        action_name = getattr(actions, 'get_action_name', lambda x: f"ACTION_{x}")(action_id)
        if action_id < 0 or action_id not in actions.ALL_ACTION_IDS:
            action_name = "INVALID_ACTION"

        logger.info(f"\n[ACTION TAKEN]\n{action_name} ({argument})")

        # ── APPEND TO TRAJECTORY STATES ──────────────────────────────────
        # Save a clean snapshot of exactly what prompted the model -> what it said
        # We also save the raw token IDs to guarantee Phase 2 aligns perfectly and 
        # avoids ANY tokenizer string-merging bugs (space+digit mismatches).
        trajectory_steps.append({
            "prompt": current_prompt,
            "completion": " " + generated_text + "\n",
            "prompt_ids": input_ids[0].cpu().tolist(),
            "new_token_ids": new_token_ids.cpu().tolist(),
            "old_logprobs": old_logprobs_gen
        })

        if action_id < 0 or action_id not in actions.ALL_ACTION_IDS:
            # Parsing failure — record it and end this trajectory early
            logger.info(f"\n[OBSERVATION]\nPARSE FAIL\n")
            break

        # ── ENGINE STEP ─────────────────────────────────────────────────────
        try:
            new_state = engine.step(state, action_id, argument=argument or None)
            format_valid = True
        except Exception as exc:
            logger.warning(f"\n[OBSERVATION]\nENGINE ERROR: {exc}\n")
            break

        # Extract the observation text from the last history entry
        observation = ""
        if new_state.get("history"):
            observation = new_state["history"][-1].get("observation", "")

        logger.info(f"\n[OBSERVATION]\n{observation}\n")

        # ── UPDATE PROMPT FOR NEXT STEP ──────────────────────────────────
        # We rebuild from the formatted state so that all subqueries,
        # documents, and history reflect the new state correctly.
        state = new_state
        current_prompt = format_state_for_prompt(state) + "\nAction: "

        # ── TERMINATION CHECK ────────────────────────────────────────────
        if state.get("status") in ("SOLVED", "FAILED"):
            logger.info(f"    [Rollout] Terminated with status={state['status']}")
            break

    return trajectory_steps, state, format_valid


# Action-ID groupings for dashboard metrics
_SEARCH_IDS = {actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM}
_KEYWORD_IDS    = {actions.ACTION_RET_KEY}
_DENSE_IDS      = {actions.ACTION_RET_VEC}
_DECOMPOSE_IDS  = {actions.ACTION_DEC_SLM, actions.ACTION_DEC_LLM}
_VERIFY_IDS     = {actions.ACTION_GRD_SLM, actions.ACTION_GRD_LLM}
_REWRITE_IDS     = {actions.ACTION_RWT_SLM}



def rollout_batch(
    model,
    tokenizer,
    sample: Dict[str, Any],
    generation_config: GenerationConfig,
    device: torch.device,
    num_generations: int = NUM_GENERATIONS,
    joule_penalty_scale: float = JOULE_PENALTY_SCALE,
    max_joule_penalty: float = MAX_JOULE_PENALTY,
) -> Tuple[List[List[Dict[str, str]]], List[float], List[Optional[GreenState]], Dict[str, Any]]:
    """
    Generate NUM_GENERATIONS independent trajectories for one question.

    Returns
    -------
    trajectories : List[List[Dict[str, str]]] — Full trajectory states and completions
    rewards      : List[float]        — Scalar reward per trajectory
    final_states : List[GreenState]   — Final states (for logging)
    metrics      : Dict[str, Any]     — Aggregated stats for dashboard logging
    """
    question     = sample["question"]
    ground_truth = sample["answer"]
    corpus       = sample["corpus"]

    trajectories: List[List[Dict[str, str]]] = []
    rewards:      List[float]                = []
    final_states: List[Optional[GreenState]] = []
    correct_flags: List[bool]                = []
    joules_list:   List[float]               = []
    steps_list:    List[int]                 = []

    # Action-use counters across the whole group
    action_totals: Dict[str, int] = {
        "search": 0, "keyword": 0, "dense": 0, "decompose": 0, "verify": 0, "rewrite": 0, "other": 0
    }

    for gen_idx in range(num_generations):
        logger.info(f"\n{'='*50}\nTRAJECTORY X-RAY (Gen {gen_idx+1}/{num_generations})\n{'='*50}")
        # Each trajectory gets a fresh state and a fresh engine
        retriever = GlobalRetriever.get_instance()
        engine    = GreenEngine(retriever=retriever)
        state     = create_initial_state(question, ground_truth)

        traj, final_state, format_valid = rollout_one_trajectory(
            model, tokenizer, engine, state, generation_config, device
        )

        reward, is_correct = compute_reward(final_state, ground_truth, question, format_valid, joule_penalty_scale, max_joule_penalty)

        trajectories.append(traj)
        rewards.append(reward)
        final_states.append(final_state)
        correct_flags.append(is_correct)
        joules_list.append(float(final_state.get("total_joules", 0.0)) if final_state else 0.0)

        # Tally action usage from this trajectory's history
        history = final_state.get("history", []) if final_state else []
        steps_list.append(len(history))

        traj_actions = []
        for item in history:
            aid = item.get("action_id", -1)
            traj_actions.append(getattr(actions, 'get_action_name', lambda x: f"ACTION_{x}")(aid))
            if aid in _SEARCH_IDS:
                action_totals["search"] += 1
            elif aid in _KEYWORD_IDS:
                action_totals["keyword"] += 1
            elif aid in _DENSE_IDS:
                action_totals["dense"] += 1
            elif aid in _DECOMPOSE_IDS:
                action_totals["decompose"] += 1
            elif aid in _VERIFY_IDS:
                action_totals["verify"] += 1
            elif aid in _REWRITE_IDS:
                action_totals["rewrite"] += 1
            else:
                action_totals["other"] += 1

        logger.info(f"\n{'='*50}\nSUMMARY\n{'='*50}")
        traj_path = " -> ".join(traj_actions) if traj_actions else "(None)"
        judge_status = "PASS" if is_correct else "FAIL"
        joules_final = joules_list[-1]
        
        logger.info(f"[TRAJECTORY] {traj_path}")
        logger.info(f"[JUDGE]      {judge_status}")
        logger.info(f"[COST]       {joules_final:.1f} Joules")
        logger.info(f"[REWARD]     {reward:.3f}")
        logger.info(f"{'='*50}\n")

    # Compute per-step dashboard metrics
    total_actions = sum(action_totals.values()) or 1  # avoid div-by-zero
    metrics = {
        "reward/mean":          sum(rewards) / len(rewards),
        "reward/max":           max(rewards),
        "accuracy/mean":        sum(correct_flags) / len(correct_flags),
        "cost/mean_joules":     sum(joules_list) / len(joules_list),
        "metrics/mean_steps":   sum(steps_list) / len(steps_list) if steps_list else 0,
        "tool_pct/search":      action_totals["search"] / total_actions,
        "tool_pct/keyword":     action_totals["keyword"]    / total_actions,
        "tool_pct/dense":       action_totals["dense"]      / total_actions,
        "tool_pct/decompose":   action_totals["decompose"]  / total_actions,
        "tool_pct/verify":      action_totals["verify"]     / total_actions,
        "tool_pct/rewrite":     action_totals["rewrite"]    / total_actions,
    }

    return trajectories, rewards, final_states, metrics


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — GRPO LOSS & UPDATE
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize_with_agent_mask(
    step_data: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reconstruct the exact tensor the generator saw in Phase 1 to guarantee 
    perfect token alignment. Build a loss mask covering the prompt with -100 
    so we calculate gradients ONLY on the LLM's first generated token.
    """
    prompt_ids = step_data["prompt_ids"]
    new_token_ids = step_data["new_token_ids"]
    old_logprobs = step_data.get("old_logprobs", [])
    
    # 1. Rebuild the exact sequence the generator outputted
    full_ids = prompt_ids + new_token_ids
    
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    
    # 2. Mask the prompt (standard causal-LM labeling)
    # labels[t] is the target token id at position t; the model's logits at
    # position t-1 predict labels[t] after the usual shift in grpo_loss().
    labels = input_ids.clone()
    prompt_len = len(prompt_ids)
    labels[0, :prompt_len] = -100

    # 3. Massively speed up computation by slicing off the generated arguments.
    # Keep only the first generated token (the action token) for GRPO updates.
    keep_len = prompt_len + 1
    input_ids = input_ids[:, :keep_len]
    attention_mask = attention_mask[:, :keep_len]
    labels = labels[:, :keep_len]

    # Align old_logprobs (cached at sampling time during Phase 1)
    gen_keep_len = keep_len - prompt_len
    if gen_keep_len > 0 and len(old_logprobs) >= gen_keep_len:
        old_lp_tensor = torch.stack(old_logprobs[:gen_keep_len]).unsqueeze(0).to(device)
    else:
        old_lp_tensor = torch.empty((1, 0), dtype=torch.float32, device=device)

    return input_ids, labels, attention_mask, old_lp_tensor


def grpo_loss(
    model,
    tokenizer,
    trajectories: List[List[Dict[str, str]]],
    rewards: List[float],
    device: torch.device,
    batch_scale: float = 1.0,
    kl_coef: float = KL_COEF,
    clip_eps: float = CLIP_EPS,
) -> Tuple[torch.Tensor, float]:
    """
    Compute the GRPO policy-gradient loss for one group of trajectories.

    GRPO advantage = (r_i - mean(r)) / (std(r) + ε)
    No critic is needed — the group mean acts as a baseline.

    Loss per trajectory:
        L_i = -A_i * sum( clip_ratio * log π(a|s) ) + β * KL(π || π_ref)

    where clip_ratio = clamp(π/π_ref, 1-ε, 1+ε)
    """
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    # ── Variance Catastrophe Check ────────────────────────────────────────
    mean_r = reward_tensor.mean()
    std_r  = reward_tensor.std()

    # ── Group-Relative Advantage ────────────────────────────────────────────
    # Note: NORMALLY we would normalize this with the standard deviation. In previous testing we found this didn't work with the divisor on the 
    # advantages like this. THIS IS AN EXPERIMENTAL CHANGE AND WE WILL NEED ALSO TO TEST WITHOUT.
    advantages = (reward_tensor - mean_r) 
    
    # THE CEILING CLAMP: 
    # If the batch is perfectly symmetrical and successful, the advantage vanishes to 0.0.
    # We must inject a small positive advantage to keep the Policy Gradient active 
    # so it can fight the KL penalty and maintain the SFT deviation.
    # Initialize the stat tracker for this batch
    clamp_fired = 0.0

    if advantages.abs().max() < 1e-5 and mean_r > 0.5:
        advantages = torch.ones_like(reward_tensor) * 0.1
        clamp_fired = 1.0
        logger.debug(f"Clipped gradients! (Mean Reward was {mean_r:.3f})")

    
    total_loss = 0.0
    total_kl   = 0.0

    # Pre-calculate how many trajectories are actually valid to scale gradients correctly
    num_valid = sum(1 for traj in trajectories if len(traj) > 0)
    if num_valid == 0:
        logger.warning("  [GRPO] No valid trajectories in this batch!")
        return 0.0, 0.0

    for traj, advantage in zip(trajectories, advantages):
        # We calculate the step-losses manually and sum them for the trajectory
        traj_loss = 0.0
        traj_kl = 0.0
        traj_valid = 0

        for step_data in traj:
            # 1. Use the EXACT token lists generated in Phase 1 (bypassing the tokenizer)
            #    so that no colon-space merges misalign our labels vs outputs!
            input_ids, labels, attention_mask, cached_old_logp = _tokenize_with_agent_mask(step_data, device)
            T = input_ids.shape[1]

            if T < 2:
                continue

            # Mask out any sequence where the agent made NO decisions
            if (labels != -100).sum() == 0:
                continue

            # ── [DEBUG] Backprop X-Ray ──────────────────────────────────────
            # Write to a secondary file exactly what tokens the model sees
            # as context, and exactly the target action token being reinforced.
            bp_log_file = LOG_FILE.replace(".log", "_backprop.txt")
            with open(bp_log_file, "a", encoding="utf-8") as f:
                mask = (labels[0] != -100)
                positions = mask.nonzero(as_tuple=False)
                num_targets = mask.sum().item()

                f.write(f"\n{'='*60}\n")
                f.write(f"BACKPROP X-RAY (Advantage: {advantage.item():.4f})\n")
                f.write(f"{'='*60}\n")

                # --- Structural checks ---
                f.write(f"[SUPERVISED POSITION]: {positions.tolist()}\n")

                if num_targets != 1:
                    f.write(f"[WARNING] Expected 1 target token, got {num_targets}\n")

                if num_targets == 1:
                    t = positions[0].item()

                # --- Extract tokens ---
                ctx_ids = input_ids[0, ~mask].tolist()
                tgt_ids = input_ids[0, mask].tolist()

                f.write(f"[CONTEXT TOKEN IDS]: {ctx_ids}\n")
                f.write(f"[TARGET TOKEN IDS]: {tgt_ids}\n")

                # --- Safe decode ---
                def safe_decode(ids):
                    try:
                        return tokenizer.decode(ids, skip_special_tokens=False)
                    except Exception as e:
                        return f"<DECODE ERROR: {e}>"

                f.write(f"[CONTEXT TEXT]:\n{repr(safe_decode(ctx_ids))}\n\n")
                f.write(f"[TARGET TEXT]:\n{repr(safe_decode(tgt_ids))}\n\n")

                # --- Annotated sequence (preserves ordering) ---
                annotated = []
                for i in range(input_ids.shape[1]):
                    tok = input_ids[0, i].item()
                    marker = "T" if mask[i] else "C"
                    annotated.append(f"{tok}:{marker}")
                f.write(f"[ANNOTATED SEQUENCE]: {' '.join(annotated)}\n")

                # --- Model prediction sanity check ---
                if num_targets == 1:
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits

                    pred_token = logits[0, t].argmax(dim=-1).item()
                    f.write(f"[MODEL ARGMAX @ POS]: {pred_token} ({repr(safe_decode([pred_token]))})\n")

                # --- NaN checks ---
                if torch.isnan(input_ids).any():
                    f.write("[ERROR] NaNs in input_ids\n")
                if torch.isnan(labels).any():
                    f.write("[ERROR] NaNs in labels\n")

                # --- (Optional) log old logprob if available ---
                if cached_old_logp is not None:
                    try:
                        f.write(f"[OLD LOGPROB]: {cached_old_logp.item():.4f}\n")
                    except Exception:
                        f.write(f"[OLD LOGPROB]: {cached_old_logp}\n")
            # ── Policy log-probs ───────────────────────────────────────────────
            # ── Reference Policy Log-probs ──────────────────────────────────
            # Evaluate the reference model FIRST and under no_grad() so we can free 
            # its massive activation footprint before evaluating the active policy.
            was_training = model.training
            model.set_adapter("reference")
            model.eval()
            with torch.no_grad():
                ref_outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
                ref_logits  = ref_outputs.logits
                shift_ref_logits = ref_logits[:, :-1, :]
                ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)
                
                # Gather log-probs for actual generated tokens
                shift_labels = labels[:, 1:]
                shift_ids = shift_labels.squeeze(0)
                agent_mask = (shift_ids != -100)
                
                if agent_mask.sum() == 0:
                    continue
                    
                tok_ref_logp = ref_log_probs.squeeze(0)
                chosen_ref_logp = tok_ref_logp[agent_mask].gather(1, shift_ids[agent_mask].unsqueeze(1)).squeeze(1)

            # Explicitly delete massive reference tensors mapping entire vocab dimensions
            del ref_outputs, ref_logits, shift_ref_logits, ref_log_probs, tok_ref_logp
            
            # ── Active Policy Log-probs ─────────────────────────────────────
            # MUST use train() mode! HuggingFace `gradient_checkpointing` silently
            # disables itself and hoards activations if `model.training` is False!
            model.set_adapter("active_rl")
            model.train() 
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
            logits  = outputs.logits     # (1, T, V)

            # Shift: predict token t+1 from position t
            shift_logits     = logits[:, :-1, :]       # (1, T-1, V)
            log_probs     = F.log_softmax(shift_logits, dim=-1)

            # Gather the log-prob of the actual token at each position
            tok_logp     = log_probs.squeeze(0)                              # (T-1, V)
            chosen_logp     = tok_logp[agent_mask].gather(1, shift_ids[agent_mask].unsqueeze(1)).squeeze(1)

            # ── PPO-style Ratio ────────────────────────────────────────────────
            # Ratio for PPO clip must be w.r.t. the OLD policy used during sampling.
            # We cached per-token logprobs at sampling time in Phase 1 (old_logprobs).
            if cached_old_logp is None or cached_old_logp.numel() == 0:
                logger.warning("  [GRPO] Missing cached old_logprobs for PPO ratio; skipping step.")
                continue

            # Align cached old_logprobs (generation-space) to the supervised tokens (shift-space).
            # shift index i corresponds to original token position (i+1).
            prompt_len = len(step_data["prompt_ids"])
            agent_pos = agent_mask.nonzero(as_tuple=False).squeeze(1)  # positions in 0..T-2
            gen_pos = (agent_pos + 1) - prompt_len                     # 0-based index into generation

            valid = (gen_pos >= 0) & (gen_pos < cached_old_logp.shape[1])
            if valid.sum().item() == 0:
                logger.warning("  [GRPO] No valid old_logprob alignment for this step; skipping.")
                continue

            # Filter to only positions we can align
            chosen_logp = chosen_logp[valid]
            chosen_ref_logp = chosen_ref_logp[valid]
            old_logp = cached_old_logp[0, gen_pos[valid]].to(device)

            # Sanity check: Alignment mapping validation
            with torch.no_grad():
                debug_tok_logp = log_probs.squeeze(0)
                debug_chosen = debug_tok_logp[agent_mask].gather(1, shift_ids[agent_mask].unsqueeze(1)).squeeze(1)
                debug_chosen = debug_chosen[valid]
                diff = (debug_chosen - old_logp).abs().mean().item()
                if diff > 0.1:
                    logger.warning(f"  [CRITICAL ALIGNMENT CHECK] Δ={diff:.6f} - Alignment drift detected!")

            assert chosen_logp.shape == old_logp.shape, f"chosen_logp: {chosen_logp.shape} != old_logp: {old_logp.shape}"

            token_count = chosen_logp.numel()
            if token_count == 0:
                logger.warning("  [GRPO] Zero supervised tokens after alignment; skipping step.")
                continue

            ratio = torch.exp(chosen_logp - old_logp.detach())
            adv_scalar = advantage.to(device)

            pg_loss1 = -adv_scalar * ratio
            pg_loss2 = -adv_scalar * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            # Average over supervised tokens:
            pg_loss = torch.max(pg_loss1, pg_loss2).sum() / token_count

            # ── KL Penalty (DeepSeek GRPO estimator) ───────────────────────────
            # We want to keep the policy close to the reference model.
            kl_log_ratio = chosen_logp - chosen_ref_logp.detach()
            # This uses the DeepSeek exact unbiased GRPO estimator: (pi_ref / pi_theta) - log(pi_ref / pi_theta) - 1
            kl = (torch.exp(-kl_log_ratio) + kl_log_ratio - 1.0).sum() / token_count
            kl_loss = kl_coef * kl

            # Sanity checks (Ratio distribution and Magnitudes)
            ratio_std = ratio.std(unbiased=False).item() if ratio.numel() > 1 else 0.0
            logger.debug(f"  [Sanity] ratio mean: {ratio.mean().item():.4f} std: {ratio_std:.4f}")
            logger.debug(f"  [Sanity] PG: {pg_loss.item():.4f} KL: {kl_loss.item():.4f}")

            # BACKPROP OOM FIX: Compute scaled loss for this individual token/step 
            # and immediately free the computation graph up so that back-to-back
            # steps within the trajectory do not endlessly add up in VRAM!
            step_scale = 1.0 / (len(traj) * num_valid * batch_scale)
            scaled_step_loss = (pg_loss + kl_loss) * step_scale
            scaled_step_loss.backward()

            # Clean up massive tensors IMMEDIATELY
            del outputs, logits, shift_logits, log_probs, tok_logp, chosen_logp
            torch.cuda.empty_cache()

            traj_loss += (pg_loss.item() + kl_loss.item()) / len(traj)
            traj_kl += kl.item() / len(traj)
            traj_valid += 1
            
        if traj_valid > 0:
            total_loss += traj_loss
            total_kl += traj_kl
            
    # We return the unscaled mean loss for accurate logging
    mean_loss = total_loss / num_valid
    mean_kl = total_kl / num_valid

    return mean_loss, mean_kl


# ──────────────────────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(output_dir: str) -> Tuple[Optional[str], int]:
    """
    Scans *output_dir* for sub-directories named ``step_N`` and returns
    ``(checkpoint_path, step_number)`` for the highest N found.
    Returns ``(None, 0)`` when no checkpoints exist yet.
    """
    if not os.path.isdir(output_dir):
        return None, 0

    best_step = 0
    best_path: Optional[str] = None
    for entry in os.scandir(output_dir):
        if entry.is_dir() and entry.name.startswith("step_"):
            try:
                n = int(entry.name.split("_", 1)[1])
            except ValueError:
                continue
            if n > best_step:
                best_step = n
                best_path = entry.path

    return best_path, best_step


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    logger.info("=== GreenRAG GRPO Training ===")
    logger.info(f"Run ID: {run_id}")

    # ── 0. Dashboard init ────────────────────────────────────────────────────
    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)
        # Pull the injected parameters from the sweep (with fallbacks to your defaults)
        config = wandb.config
        lr = config.get("learning_rate", LEARNING_RATE)
        kl = config.get("kl_coef", KL_COEF)
        beta1 = config.get("adam_beta1", 0.9)
        eps = config.get("clip_eps", CLIP_EPS)
        joule_scale = config.get("joule_penalty_scale", JOULE_PENALTY_SCALE)
        max_joule   = config.get("max_joule_penalty", MAX_JOULE_PENALTY)
        if wandb.run is not None:
            logger.info(f"W&B run: {wandb.run.url}")
    else:
        lr, kl, beta1, eps = LEARNING_RATE, KL_COEF, 0.9, CLIP_EPS
        joule_scale = JOULE_PENALTY_SCALE
        max_joule   = MAX_JOULE_PENALTY

    # ── 1. Detect checkpoint ─────────────────────────────────────────────────
    ckpt_path, start_step = find_latest_checkpoint(OUTPUT_DIR)
    if ckpt_path:
        logger.info(f"Resuming from checkpoint: {ckpt_path}  (step {start_step})")
        # PEFT saves each named adapter in its own subdirectory when multiple
        # adapters are present, so adapter_config.json lives at
        # <ckpt>/active_rl/adapter_config.json — not at <ckpt>/ directly.
        active_rl_source = os.path.join(ckpt_path, "active_rl")
    else:
        logger.info("No checkpoint found — starting from SFT weights.")
        active_rl_source = SFT_MODEL_PATH

    # ── 2. Load Model ────────────────────────────────────────────────────────
    BASE_MODEL_ID = os.getenv("SLM_MODEL", "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest")
    if BASE_MODEL_ID == "qwen2.5:3b":
        BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    logger.info(f"Loading model from {BASE_MODEL_ID} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # Reference adapter always stays frozen at the original SFT weights
    logger.info(f"Loading reference adapter from {SFT_MODEL_PATH} ...")
    model = PeftModel.from_pretrained(
        base_model, 
        SFT_MODEL_PATH, 
        adapter_name="reference",
        is_trainable=False
    )

    # Active RL adapter: resume from checkpoint if available, else SFT
    logger.info(f"Loading active_rl adapter from {active_rl_source} ...")
    model.load_adapter(
        active_rl_source,
        adapter_name="active_rl", 
        is_trainable=True
    )
    
    # ── Mute ALL dropout layers ───────────
    # In RLHF (unlike SFT), 5% LoRA dropout brutally destabilizes the PPO ratio 
    # resulting in a false "Alignment Drift" up to ~0.20 when comparing the 
    # clean Phase 1 outputs to the noisy Phase 2 logits. Shutting it off.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            
    # Set the active adapter as default and enable gradients
    model.set_adapter("active_rl")
    model.print_trainable_parameters()
    model.train()
    
    # ── 3. Reference model — NOT needed with LoRA ──────────────────────────
    # The frozen base weights live inside `model` already.  grpo_loss() calls
    # model.disable_adapter_layers() / enable_adapter_layers() to obtain
    # reference log-probs without allocating a second copy on GPU.

    # ── 4. Tokenizer ─────────────────────────────────────────────────────────
    # Prefer tokenizer from checkpoint (may have updated special tokens)
    tokenizer_source = ckpt_path if ckpt_path else SFT_MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 4. Generation Config (used inside the rollout loop) ──────────────────
    generation_config = GenerationConfig(
        do_sample=True, 
        temperature=0.5, # lowering this
        # top_p=0.9,
        # repetition_penalty=1.2,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Stop as soon as the agent finishes its "Input: ..." line.
        # "\n\n" catches a blank line after the argument; "Action:" catches
        # the model trying to chain a second action without our environment step.
        # This mirrors the PPO script's "\n" stop but allows one full Input: line.
        stop_strings=["\n\n", "Action:", "<|eot_id|>"],
    )

    # ── 5. Optimizer ─────────────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(float(beta1), 0.999)
    )

    # ── 6. Dataset ───────────────────────────────────────────────────────────
    active_datasets = ["hotpot", "musique", "twowiki"]
    dataset_weights = [0.8, 0.1, 0.1]

    dataset_configs = {
        "hotpot": {
            "setting": "fullwiki", 
            "split": "train"
        },
        "musique": {
            "setting": "fullwiki", 
            "split": "train"
        },
        "twowiki": {
            "setting": "fullwiki", 
            "split": "train"
        }
    }

    if OVERFIT_MODE:
        logger.info("CRITICAL WARNING: OVERFIT_MODE is ENABLED. Training on 2 questions only!")
        active_datasets = ["hotpot"]
        dataset_weights = [1.0]
        dataset_configs["hotpot"]["split"] = "train" 
        dataset_configs["hotpot"]["filter_ids"] = [
            "5ab9257b554299753720f749",  # boing/dawkins one
            "5a70fb2d5542994082a3e482"   # san jose del cabo one
        ]
        
    if ONLY_EASY_MODE and not OVERFIT_MODE:
        logger.info("ONLY_EASY_MODE is ENABLED. Training only on the first 100 'easy' difficulty HotpotQA questions.")
        active_datasets = ["hotpot"]
        dataset_weights = [1.0]
        dataset_configs["hotpot"]["level"] = "easy"
        MAX_QUESTIONS = 100
        
        # Override steps for exactly 3 epochs across the 100 questions
        global TOTAL_STEPS
        TOTAL_STEPS = (MAX_QUESTIONS // BATCH_SIZE) * 3
        logger.info(f"  => Adjusted TOTAL_STEPS to {TOTAL_STEPS} for exactly 3 epochs!")

    streamer = MixedStreamer(
        dataset_names=active_datasets, 
        weights=dataset_weights,
        limit=MAX_QUESTIONS,
        configs=dataset_configs
    )
    print(f"Streaming {streamer.n_limit} samples from: {', '.join(active_datasets)}")
    data_iter = streamer.stream()

    logger.info(f"Training will run steps {start_step + 1} → {TOTAL_STEPS}")

    # Create Eval Streamer
    eval_configs = {
        "hotpot": {
            "setting": "fullwiki",
            "split": "validation" # Use validation split for eval
        },
        "musique": {
            "setting": "fullwiki",
            "split": "validation"
        },
        "twowiki": {
            "setting": "fullwiki",
            "split": "validation"
        }
    }
    if ONLY_EASY_MODE and not OVERFIT_MODE:
        eval_configs["hotpot"]["level"] = "easy"

    eval_streamer = MixedStreamer(
        dataset_names=active_datasets,
        limit=EVAL_SAMPLES,
        configs=eval_configs
    )
    eval_iter = eval_streamer.stream()


    device = next(model.parameters()).device
    logger.info(f"Training device: {device}")

    # ── 7. Outer Training Loop ───────────────────────────────────────────────
    optimizer.zero_grad()
    accum_loss  = 0.0
    accum_kl    = 0.0
    accum_steps = 0
    from collections import deque
    acc_window = deque(maxlen=5) 

    cumulative_accuracy = 0.0
    steps_to_converge = TOTAL_STEPS
    
    global_q_step = start_step * BATCH_SIZE

    for step in range(start_step, TOTAL_STEPS):
        # ── Collect one batch of questions ──────────────────────────────────
        batch_samples = []
        for _ in range(BATCH_SIZE):
            try:
                batch_samples.append(next(data_iter))
            except StopIteration:
                logger.info("Dataset exhausted — restarting.")
                data_iter = streamer.stream()
                try:
                    batch_samples.append(next(data_iter))
                except StopIteration:
                    logger.error("Dataset produced 0 items! Stopping training to prevent loops.")
                    raise RuntimeError("Dataset is completely empty.")

        logger.info(f"\n{'='*60}")
        logger.info(f"Step {step+1}/{TOTAL_STEPS}")

        # Track metrics for the whole batch
        batch_metrics_list = []
        clamp_fired_this_step = 0.0

        for sample in batch_samples:
            q = sample["question"]
            logger.info(f"  Q: {q[:80]}")

            # ── Phase 1: Rollout ─────────────────────────────────────────────
            trajectories, rewards, final_states, single_q_metrics = rollout_batch(
                model, tokenizer, sample, generation_config, device,
                joule_penalty_scale=joule_scale, max_joule_penalty=max_joule
            )

            # Store metrics to average later
            batch_metrics_list.append(single_q_metrics)

            global_q_step += 1
            if USE_WANDB:
                # Log on a question-to-question basis without tying it to the gradient step
                wandb.log({
                    "q/accuracy": single_q_metrics["accuracy/mean"],
                    "q/reward": single_q_metrics["reward/mean"],
                    "q/joules": single_q_metrics["cost/mean_joules"],
                    "train/global_q_step": global_q_step
                }, step=global_q_step)

            mean_r = single_q_metrics["reward/mean"]
            logger.info(
                f"  Rewards: {[f'{r:.3f}' for r in rewards]}  mean={mean_r:.3f}  "
                f"accuracy={single_q_metrics['accuracy/mean']:.2f}  "
                f"joules={single_q_metrics['cost/mean_joules']:.3f}  "
                f"tool% search={single_q_metrics['tool_pct/search']:.2f} "
                f"kw={single_q_metrics['tool_pct/keyword']:.2f} "
                f"dense={single_q_metrics['tool_pct/dense']:.2f} "
                f"verify={single_q_metrics['tool_pct/verify']:.2f} "
                f"rewrite={single_q_metrics['tool_pct/rewrite']:.2f}"
            )

            # Check if clamp fired for THIS question
            _rew_tensor = torch.tensor(rewards, dtype=torch.float32)
            _mean_r = _rew_tensor.mean()
            if (_rew_tensor - _mean_r).abs().max() < 1e-5 and _mean_r > 0.5:
                clamp_fired_this_step = 1.0  # If either question clamps, flag the step
                logger.debug(f"Clipped gradients! (Mean Reward was {_mean_r:.3f})")

            # ── Phase 2: GRPO Loss ───────────────────────────────────────────
            # We pass scale to grpo_loss, which handles the localized backward passes!
            scaled_loss, mean_kl = grpo_loss(
                model, tokenizer, trajectories, rewards, device,
                batch_scale=len(batch_samples) * GRADIENT_ACCUM,
                kl_coef=kl,
                clip_eps=eps
            )
            logger.info(f"  GRPO scaled loss: {scaled_loss:.4f}  KL: {mean_kl:.4f}")

            # Track the scalar loss for logging
            accum_loss += scaled_loss
            accum_kl += mean_kl

        # ── AGGREGATE & LOG TO W&B (Outside the sample loop) ──
        # Average all keys across the questions in the batch
        avg_metrics = {}
        if batch_metrics_list:
            for key in batch_metrics_list[0].keys():
                avg_metrics[key] = sum(m[key] for m in batch_metrics_list) / len(batch_metrics_list)

        # Calculate Moving Average & Stability using the TRUE batch average
        raw_batch_acc = avg_metrics.get("accuracy/mean", 0.0)
        acc_window.append(raw_batch_acc)
        stability_score = float(np.mean(acc_window) - np.std(acc_window))

        cumulative_accuracy += raw_batch_acc
        if stability_score > 0.99 and steps_to_converge == TOTAL_STEPS:
            steps_to_converge = step + 1
            logger.info(f"  [Milestone] Perfect Stability reached at step {steps_to_converge}!")
        
        avg_metrics["accuracy/stability_score"] = stability_score
        avg_metrics["accuracy/cumulative"] = cumulative_accuracy 
        avg_metrics["stats/ceiling_clamp_fired"] = clamp_fired_this_step

        if USE_WANDB:
            wandb.log({**avg_metrics, "train/grpo_step": step + 1}, step=global_q_step)

        accum_steps += 1

        if accum_steps % GRADIENT_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                alloc    = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"  [Optim] Updated | GPU mem: alloc={alloc:.2f} GB  reserved={reserved:.2f} GB")
            else:
                logger.info(f"  [Optim] Updated")
            if USE_WANDB:
                wandb.log({
                    "train/loss": accum_loss / GRADIENT_ACCUM,
                    "train/kl":   accum_kl / GRADIENT_ACCUM
                }, step=global_q_step)
            accum_loss = 0.0
            accum_kl   = 0.0


        # ── Checkpoint ───────────────────────────────────────────────────────
        if (step + 1) % SAVE_EVERY == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{step+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir, selected_adapters=["active_rl"])
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"  [Checkpoint] Saved to {ckpt_dir}")

        # ── Evaluation ───────────────────────────────────────────────────────
        if (step + 1) % EVAL_EVERY == 0:
            logger.info(f"{'='*20} Evaluation Phase (Step {step+1}) {'='*20}")
            model.eval()
            eval_accum_loss = 0.0
            eval_accuracy = 0.0
            
            # Rehydrate eval iterator
            eval_iter = eval_streamer.stream()
            eval_samples = []
            for _ in range(EVAL_SAMPLES):
                try:
                    eval_samples.append(next(eval_iter))
                except StopIteration:
                    break
            
            for eval_sample in eval_samples:
                logger.info(f"  [Eval] Q: {eval_sample['question'][:80]}")
                _, _, _, batch_metrics = rollout_batch(
                    model, tokenizer, eval_sample, generation_config, device, num_generations=1,
                    joule_penalty_scale=joule_scale, max_joule_penalty=max_joule
                )
                eval_accuracy += batch_metrics['accuracy/mean']
                logger.info(f"  [Eval] Acc: {batch_metrics['accuracy/mean']:.2f}")

            if len(eval_samples) > 0:
                mean_eval_acc = eval_accuracy / len(eval_samples)
                logger.info(f"  => Mean Evaluation Accuracy: {mean_eval_acc:.4f}")
                if USE_WANDB:
                    wandb.log({"eval/accuracy": mean_eval_acc}, step=global_q_step)
            
            model.train() # Set back to train mode!
            logger.info(f"{'='*60}")


    # ── Final Save ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, selected_adapters=["active_rl"])
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"\nTraining complete. Model saved to {OUTPUT_DIR}")

    if USE_WANDB:
        wandb.run.summary["metrics/steps_to_converge"] = steps_to_converge # ADD THISs
        wandb.finish()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.critical("main() raised an exception:", exc_info=True)
        raise
