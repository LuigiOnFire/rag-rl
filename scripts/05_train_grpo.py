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
import logging
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

from src.data.hotpot import HotpotQAStreamer
from src.env.state import GreenState, create_initial_state
from src.env.retriever import EphemeralRetriever
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
NUM_GENERATIONS      = 8      # Trajectories per question (the "group" in GRPO)
BATCH_SIZE           = 2      # Questions per outer training step
MAX_STEPS_PER_TRAJ   = 8      # Max engine steps before forcing termination
MAX_NEW_TOKENS       = 64     # Tokens generated per LLM call inside the loop

# Training config
TOTAL_STEPS          = 2000   # Training steps (each step = one batch of questions)
LEARNING_RATE        = 1e-5
GRADIENT_ACCUM       = 4
KL_COEF              = 0.04   # β — KL penalty coefficient (keeps policy close to ref)
CLIP_EPS             = 0.2    # ε — PPO-style clipping (applied inside GRPO loss)

# Reward config
FORMAT_CONSOLATION_REWARD = 0.1   # Reward for valid format but wrong answer
JOULE_PENALTY_SCALE       = 0.05  # Multiply total_joules by this to get the penalty
MAX_JOULE_PENALTY         = 0.0   # The penalty is CAPPED at this value, set at 0 for now, to be revisited

# Checkpoint / logging
SAVE_EVERY    = 100
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

trace_logger = logging.getLogger("LLM_TRACE")
trace_logger.addHandler(logging.NullHandler())

# ──────────────────────────────────────────────────────────────────────────────
# REWARD CALCULATION
# ──────────────────────────────────────────────────────────────────────────────

_judge = SoftJudge()

def compute_reward(
    final_state: Optional[GreenState],
    ground_truth: str,
    question: str,
    format_valid: bool,
) -> Tuple[float, bool]:
    """
    Returns (scalar_reward, is_correct).

    Correct answer:    1.0  - min(MAX_JOULE_PENALTY, total_joules * JOULE_PENALTY_SCALE)
    Valid format only: FORMAT_CONSOLATION_REWARD  (+0.1)
    Broken format:     0.0
    """
    if final_state is None or not format_valid:
        return 0.0, False

    final_answer = final_state.get("answer") or ""

    if not final_answer:
        # No answer produced despite valid format → consolation
        return FORMAT_CONSOLATION_REWARD, False

    is_correct, reason = _judge.judge(final_answer, ground_truth, question)

    if is_correct:
        total_joules = float(final_state.get("total_joules", 0.0))
        joule_penalty = min(MAX_JOULE_PENALTY, total_joules * JOULE_PENALTY_SCALE)
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
) -> Tuple[str, Optional[GreenState], bool]:
    """
    Run one complete trajectory by interleaving LLM generation and GreenEngine steps.

    Returns
    -------
    full_trajectory_text : str
        The entire text string that was accumulated (prompt + all
        generated tokens + all environment observations).  This is the
        string handed to the GRPO loss later.
    final_state : GreenState | None
        State at termination, or None on catastrophic error.
    format_valid : bool
        True if at least one action was successfully parsed and executed.
    """
    state = copy.deepcopy(initial_state)

    # The "running prompt" starts as the formatted initial observation.
    # We build this as a plain string and re-tokenize on every LLM call.
    current_prompt = format_state_for_prompt(state) + "\nAction:"

    # Track the full trajectory as a single growing string.
    # We record the ENTIRE context (prompts + generated text + observations)
    # so that the GRPO loss can compute log-probs over the agent tokens later.
    full_trajectory = current_prompt

    format_valid = False  # Becomes True once we successfully parse+execute an action

    for step_i in range(MAX_STEPS_PER_TRAJ):

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

        # 2. Generate with KV Cache explicitly enabled
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                generation_config=generation_config,
                tokenizer=tokenizer,
                stop_strings=generation_config.stop_strings,
                attention_mask=attention_mask,
                use_cache=True, # <--- Force KV Cache on
            )
        
        # 3. Switch back to Train mode and turn Checkpointing back on
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train()

        # Decode only the *new* tokens (after the prompt)
        prompt_len = input_ids.shape[1]
        new_token_ids = output_ids[0, prompt_len:]
        generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        logger.info(f"    [Rollout step {step_i}] Generated: '{generated_text[:80]}'")

        # ── PARSE ACTION ───────────────────────────────────────────────
        action_id, argument = _parse_action(generated_text)

        if action_id < 0 or action_id not in actions.ALL_ACTION_IDS:
            # Parsing failure — record it and end this trajectory early
            logger.info(f"    [Rollout step {step_i}] PARSE FAIL: '{generated_text[:60]}'")
            full_trajectory += generated_text + "\n[PARSE_ERROR]\n"
            break

        # ── ENGINE STEP ─────────────────────────────────────────────────────
        try:
            new_state = engine.step(state, action_id, argument=argument or None)
            format_valid = True
        except Exception as exc:
            logger.warning(f"    [Rollout step {step_i}] Engine error: {exc}")
            full_trajectory += generated_text + f"\n[ENGINE_ERROR: {exc}]\n"
            break

        # Extract the observation text from the last history entry
        observation = ""
        if new_state.get("history"):
            observation = new_state["history"][-1].get("observation", "")

        logger.info(f"    [Rollout step {step_i}] action={action_id} arg='{argument[:40]}' obs='{observation[:60]}'")

        # ── APPEND TO TRAJECTORY TEXT ─────────────────────────────────────
        # The agent token chunk: what the LLM physically typed
        agent_chunk = generated_text + "\n"
        # The environment observation chunk (masked later for loss computation)
        env_chunk = f"Observation: {observation}\n"

        full_trajectory += agent_chunk + env_chunk

        # ── UPDATE PROMPT FOR NEXT STEP ──────────────────────────────────
        # We rebuild from the formatted state so that all subqueries,
        # documents, and history reflect the new state correctly.
        state = new_state
        next_state_prompt = format_state_for_prompt(state) + "\nAction:"
        current_prompt = next_state_prompt
        full_trajectory += current_prompt  # record new prompt boundary

        # ── TERMINATION CHECK ────────────────────────────────────────────
        if state.get("status") in ("SOLVED", "FAILED"):
            logger.info(f"    [Rollout] Terminated with status={state['status']}")
            break

    return full_trajectory, state, format_valid


# Action-ID groupings for dashboard metrics
_PARAMETRIC_IDS = {actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM}
_KEYWORD_IDS    = {actions.ACTION_RET_KEY}
_DENSE_IDS      = {actions.ACTION_RET_VEC}
_DECOMPOSE_IDS  = {actions.ACTION_DEC_SLM, actions.ACTION_DEC_LLM}


def rollout_batch(
    model,
    tokenizer,
    sample: Dict[str, Any],
    generation_config: GenerationConfig,
    device: torch.device,
) -> Tuple[List[str], List[float], List[Optional[GreenState]], Dict[str, Any]]:
    """
    Generate NUM_GENERATIONS independent trajectories for one question.

    Returns
    -------
    trajectories : List[str]          — Full trajectory strings
    rewards      : List[float]        — Scalar reward per trajectory
    final_states : List[GreenState]   — Final states (for logging)
    metrics      : Dict[str, Any]     — Aggregated stats for dashboard logging
    """
    question     = sample["question"]
    ground_truth = sample["answer"]
    corpus       = sample["corpus"]

    trajectories: List[str]                  = []
    rewards:      List[float]                = []
    final_states: List[Optional[GreenState]] = []
    correct_flags: List[bool]                = []
    joules_list:   List[float]               = []

    # Action-use counters across the whole group
    action_totals: Dict[str, int] = {
        "parametric": 0, "keyword": 0, "dense": 0, "decompose": 0, "other": 0
    }

    for gen_idx in range(NUM_GENERATIONS):
        # Each trajectory gets a fresh state and a fresh engine/retriever
        retriever = EphemeralRetriever(documents=corpus)
        engine    = GreenEngine(retriever=retriever)
        state     = create_initial_state(question, ground_truth)

        traj, final_state, format_valid = rollout_one_trajectory(
            model, tokenizer, engine, state, generation_config, device
        )

        reward, is_correct = compute_reward(final_state, ground_truth, question, format_valid)

        trajectories.append(traj)
        rewards.append(reward)
        final_states.append(final_state)
        correct_flags.append(is_correct)
        joules_list.append(float(final_state.get("total_joules", 0.0)) if final_state else 0.0)

        # Tally action usage from this trajectory's history
        history = final_state.get("history", []) if final_state else []
        for item in history:
            aid = item.get("action_id", -1)
            if aid in _PARAMETRIC_IDS:
                action_totals["parametric"] += 1
            elif aid in _KEYWORD_IDS:
                action_totals["keyword"] += 1
            elif aid in _DENSE_IDS:
                action_totals["dense"] += 1
            elif aid in _DECOMPOSE_IDS:
                action_totals["decompose"] += 1
            else:
                action_totals["other"] += 1

        logger.info(
            f"  [Gen {gen_idx+1}/{NUM_GENERATIONS}] "
            f"reward={reward:.3f}  correct={is_correct}  "
            f"steps={len(history)}  "
            f"status={final_state.get('status', 'N/A') if final_state else 'ERROR'}"
        )

    # Compute per-step dashboard metrics
    total_actions = sum(action_totals.values()) or 1  # avoid div-by-zero
    metrics = {
        "reward/mean":          sum(rewards) / len(rewards),
        "reward/max":           max(rewards),
        "accuracy/mean":        sum(correct_flags) / len(correct_flags),
        "cost/mean_joules":     sum(joules_list) / len(joules_list),
        "tool_pct/parametric":  action_totals["parametric"] / total_actions,
        "tool_pct/keyword":     action_totals["keyword"]    / total_actions,
        "tool_pct/dense":       action_totals["dense"]      / total_actions,
        "tool_pct/decompose":   action_totals["decompose"]  / total_actions,
    }

    return trajectories, rewards, final_states, metrics


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — GRPO LOSS & UPDATE
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize_with_agent_mask(
    tokenizer,
    trajectory: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tokenize a full trajectory string and build a loss mask that is 1
    only on *agent-generated* tokens and 0 on environment text.

    Strategy
    --------
    We exploit the fact that the trajectory string was built by alternating
    agent chunks and environment/prompt chunks.  Each new prompt fragment
    (from format_state_for_prompt) begins with "Goal:" — which we use as
    the delimiter.  Any text that was NOT between an "Action:" boundary and
    the following "Observation:" / "Goal:" boundary is treated as
    environment text and masked out.

    For simplicity and robustness we use a character-level approach:
    build a binary char-mask first, then map to tokens.
    """
    encoding = tokenizer(
        trajectory,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        return_offsets_mapping=True, # <--- ADD THIS
    )
    input_ids = encoding.input_ids.to(device)  # (1, T)
    attention_mask = encoding.attention_mask.to(device) # <--- Grab the mask here

    # Build agent char-mask: 1 = agent generated, 0 = env / prompt
    char_mask = [0] * len(trajectory)

    # Find all [Action: ... Observation:] spans — these are the agent spans
    # Pattern: everything from "Action:" up to the next "Observation:" or "Goal:"
    for m in re.finditer(
        r"Action:(.*?)(?=Observation:|Goal:|$)",
        trajectory,
        re.DOTALL,
    ):
        start, end = m.span(1)
        for i in range(start, end):
            char_mask[i] = 1

    # Map char-mask → token-mask using char_to_token offsets
    token_mask = torch.zeros(input_ids.shape[1], dtype=torch.bool)
    offsets = encoding.encodings[0].offsets  # list of (char_start, char_end) per token
    for tok_idx, (cs, ce) in enumerate(offsets):
        if ce > cs and any(char_mask[c] for c in range(cs, ce)):
            token_mask[tok_idx] = True

    # Labels: copy input_ids, mask out non-agent tokens with -100
    labels = input_ids.clone()
    labels[0, ~token_mask] = -100

    return input_ids, labels, attention_mask


def grpo_loss(
    model,
    tokenizer,
    trajectories: List[str],
    rewards: List[float],
    device: torch.device,
) -> torch.Tensor:
    """
    Compute the GRPO policy-gradient loss for one group of trajectories.

    GRPO advantage = (r_i - mean(r)) / (std(r) + ε)
    No critic is needed — the group mean acts as a baseline.

    Loss per trajectory:
        L_i = -A_i * sum( clip_ratio * log π(a|s) ) + β * KL(π || π_ref)

    where clip_ratio = clamp(π/π_ref, 1-ε, 1+ε)
    """
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)

    # ── Group-Relative Advantage ────────────────────────────────────────────
    mean_r = reward_tensor.mean()
    std_r  = reward_tensor.std() + 1e-8
    advantages = (reward_tensor - mean_r) / std_r  # shape (G,)

    total_loss = torch.tensor(0.0, requires_grad=True, device=device)
    num_valid  = 0

    for traj, advantage in zip(trajectories, advantages):
        input_ids, labels, attention_mask = _tokenize_with_agent_mask(tokenizer, traj, device)
        T = input_ids.shape[1]

        if T < 2:
            continue

        # Mask out any trajectory where the agent made NO decisions
        if (labels != -100).sum() == 0:
            continue

        # ── Policy log-probs ───────────────────────────────────────────────
        # Active policy forward pass (adapters ON)
        model.set_adapter("active_rl")
        outputs = model(input_ids, attention_mask=attention_mask)
        logits  = outputs.logits     # (1, T, V)

        # Reference forward pass: disable LoRA adapters to expose frozen base
        model.set_adapter("reference")
        with torch.no_grad():
            ref_outputs = model(input_ids, attention_mask=attention_mask)
            ref_logits  = ref_outputs.logits
        
        model.set_adapter("active_rl")

        # Shift: predict token t+1 from position t
        shift_logits     = logits[:, :-1, :]       # (1, T-1, V)
        shift_ref_logits = ref_logits[:, :-1, :]
        shift_labels     = labels[:, 1:]            # (1, T-1)

        log_probs     = F.log_softmax(shift_logits,     dim=-1)
        ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)

        # Gather the log-prob of the actual token at each position
        tok_logp     = log_probs.squeeze(0)                              # (T-1, V)
        tok_ref_logp = ref_log_probs.squeeze(0)
        shift_ids    = shift_labels.squeeze(0)                           # (T-1,)

        # Only compute loss on agent tokens (labels != -100 after shift)
        agent_mask = (shift_ids != -100)  # (T-1,)
        if agent_mask.sum() == 0:
            continue

        chosen_logp     = tok_logp[agent_mask].gather(1, shift_ids[agent_mask].unsqueeze(1)).squeeze(1)
        chosen_ref_logp = tok_ref_logp[agent_mask].gather(1, shift_ids[agent_mask].unsqueeze(1)).squeeze(1)

        # ── Clipped Surrogate (PPO-style) ──────────────────────────────────
        log_ratio = chosen_logp - chosen_ref_logp.detach()
        ratio     = log_ratio.exp()
        adv_scalar = advantage.to(device)

        pg_loss1 = -adv_scalar * ratio
        pg_loss2 = -adv_scalar * torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
        pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

        # ── KL Penalty (keeps policy close to reference) ───────────────────
        kl = (chosen_logp - chosen_ref_logp.detach()).mean()
        kl_loss = KL_COEF * kl

        traj_loss = pg_loss + kl_loss
        total_loss = total_loss + traj_loss
        num_valid += 1

    if num_valid == 0:
        logger.warning("  [GRPO] No valid trajectories in this batch!")
        return torch.tensor(0.0, requires_grad=True, device=device)

    return total_loss / num_valid


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=== GreenRAG GRPO Training ===")
    logger.info(f"Run ID: {run_id}")

    # ── 0. Dashboard init ────────────────────────────────────────────────────
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "num_generations":   NUM_GENERATIONS,
                "batch_size":        BATCH_SIZE,
                "max_steps_per_traj": MAX_STEPS_PER_TRAJ,
                "learning_rate":     LEARNING_RATE,
                "kl_coef":           KL_COEF,
                "clip_eps":          CLIP_EPS,
                "joule_penalty_scale": JOULE_PENALTY_SCALE,
                "max_joule_penalty": MAX_JOULE_PENALTY,
                "format_consolation": FORMAT_CONSOLATION_REWARD,
                "model_path":        SFT_MODEL_PATH,
            },
        )
        if wandb.run is not None:
            logger.info(f"W&B run: {wandb.run.url}")

    # ── 1. Load Model ────────────────────────────────────────────────────────
    BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
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

    logger.info(f"Loading SFT adapter from {SFT_MODEL_PATH} ...")

    model = PeftModel.from_pretrained(
        base_model, 
        SFT_MODEL_PATH, 
        adapter_name="reference",
        is_trainable=False
    )

# Load second copy as the active RL policy
    model.load_adapter(
        SFT_MODEL_PATH, 
        adapter_name="active_rl", 
        is_trainable=True
    )
    
    # Set the active adapter as default and enable gradients
    model.set_adapter("active_rl")
    model.print_trainable_parameters()
    model.train()
    
    # ── 2. Reference model — NOT needed with LoRA ──────────────────────────
    # The frozen base weights live inside `model` already.  grpo_loss() calls
    # model.disable_adapter_layers() / enable_adapter_layers() to obtain
    # reference log-probs without allocating a second copy on GPU.

    # ── 3. Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 4. Generation Config (used inside the rollout loop) ──────────────────
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
        repetition_penalty=1.2,
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
        lr=LEARNING_RATE,
    )

    # ── 6. Dataset ───────────────────────────────────────────────────────────
    logger.info("Streaming HotpotQA training split ...")
    streamer  = HotpotQAStreamer(split="train", limit=None)
    data_iter = streamer.stream()

    device = next(model.parameters()).device
    logger.info(f"Training device: {device}")

    # ── 7. Outer Training Loop ───────────────────────────────────────────────
    optimizer.zero_grad()
    accum_loss  = 0.0
    accum_steps = 0

    for step in range(TOTAL_STEPS):
        # ── Collect one batch of questions ──────────────────────────────────
        batch_samples = []
        for _ in range(BATCH_SIZE):
            try:
                batch_samples.append(next(data_iter))
            except StopIteration:
                logger.info("Dataset exhausted — restarting.")
                data_iter = streamer.stream()
                batch_samples.append(next(data_iter))

        logger.info(f"\n{'='*60}")
        logger.info(f"Step {step+1}/{TOTAL_STEPS}")

        for sample in batch_samples:
            q = sample["question"]
            logger.info(f"  Q: {q[:80]}")

            # ── Phase 1: Rollout ─────────────────────────────────────────────
            trajectories, rewards, final_states, batch_metrics = rollout_batch(
                model, tokenizer, sample, generation_config, device
            )

            mean_r = batch_metrics["reward/mean"]
            logger.info(
                f"  Rewards: {[f'{r:.3f}' for r in rewards]}  mean={mean_r:.3f}  "
                f"accuracy={batch_metrics['accuracy/mean']:.2f}  "
                f"joules={batch_metrics['cost/mean_joules']:.3f}  "
                f"tool% param={batch_metrics['tool_pct/parametric']:.2f} "
                f"kw={batch_metrics['tool_pct/keyword']:.2f} "
                f"dense={batch_metrics['tool_pct/dense']:.2f} "
                f"dec={batch_metrics['tool_pct/decompose']:.2f}"
            )

            # ── Dashboard logging ────────────────────────────────────────────
            if USE_WANDB:
                wandb.log({**batch_metrics, "train/grpo_step": step + 1}, step=step + 1)

            # ── Phase 2: GRPO Loss ───────────────────────────────────────────
            loss = grpo_loss(
                model, tokenizer, trajectories, rewards, device
            )
            logger.info(f"  GRPO loss: {loss.item():.4f}")
            
            # Scale the loss by batch size and gradient accumulation steps
            scaled_loss = loss / (len(batch_samples) * GRADIENT_ACCUM)

            # Backpropagate the scaled loss
            scaled_loss.backward()

            # Track the scalar loss for logging
            accum_loss += scaled_loss.item()

        accum_steps += 1

        if accum_steps % GRADIENT_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            logger.info(f"  [Optim] Updated")
            if USE_WANDB:
                wandb.log({"train/loss": accum_loss / GRADIENT_ACCUM}, step=step + 1)
            accum_loss = 0.0


        # ── Checkpoint ───────────────────────────────────────────────────────
        if (step + 1) % SAVE_EVERY == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"step_{step+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            logger.info(f"  [Checkpoint] Saved to {ckpt_dir}")

    # ── Final Save ───────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info(f"\nTraining complete. Model saved to {OUTPUT_DIR}")

    if USE_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
