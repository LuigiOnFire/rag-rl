"""
06_eval_checkpoints.py — Standardized Cross-Dataset Pareto, Stratification & Rescue Evaluation

Usage:
  python scripts/06_eval_checkpoints.py \
    --datasets hotpot squad \
    --num_samples 200 \
    --seed 42 \
    --output_json results/eval_v9_cross_dataset.json \
    --checkpoints \
      models/greenrag-grpo-v9-cond-5e5/step_350 \
      models/greenrag-grpo-v9-cond-5e5/step_800 \
      models/greenrag-grpo-v9-cond-5e5/step_500 \
      models/greenrag-grpo-v9-cond-5e6/step_350 \
      models/greenrag-grpo-v9-cond-5e6/step_650
"""

import os
import re
import sys
import json
import copy
import time
import string
import logging
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
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

SFT_MODEL_PATH     = "models/green-rag-sft-v1"
BASE_MODEL_ID      = os.getenv("SLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
if BASE_MODEL_ID in ("qwen2.5:3b", "qwen2.5"):
    BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

MAX_STEPS_PER_TRAJ = 15

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

_judge = SoftJudge()
_ACTION_TOKEN_TENSOR = None

_GENERATE_IDS  = {actions.ACTION_GEN_SLM, actions.ACTION_GEN_LLM}
_DECOMPOSE_IDS = {actions.ACTION_DEC_LLM, actions.ACTION_DEC_RSN}
_REASON_IDS    = {actions.ACTION_RSN_SLM, actions.ACTION_RSN_LLM}
_KEYWORD_IDS   = {actions.ACTION_RET_KEY}
_DENSE_IDS     = {actions.ACTION_RET_VEC}

_ABSTENTION_PHRASES = (
    "no relevant",
    "not provided",
    "not mentioned",
    "cannot determine",
    "can't determine",
    "insufficient information",
    "does not contain",
    "do not have enough",
    "unknown",
    "sorry",
)


# ──────────────────────────────────────────────────────────────────────────────
# STANDARD QA METRICS (SQuAD / HotpotQA Normalization)
# ──────────────────────────────────────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_em(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2.0 * precision * recall) / (precision + recall)


def is_abstention(prediction: str) -> bool:
    norm = prediction.strip().lower()
    if not norm:
        return True
    return any(phrase in norm for phrase in _ABSTENTION_PHRASES)


# ──────────────────────────────────────────────────────────────────────────────
# ACTION MASKING & ROLLOUTS
# ──────────────────────────────────────────────────────────────────────────────

def get_action_token_tensor(tokenizer, device) -> torch.Tensor:
    global _ACTION_TOKEN_TENSOR
    if _ACTION_TOKEN_TENSOR is None or _ACTION_TOKEN_TENSOR.device != device:
        token_ids = [
            tokenizer.encode(str(a), add_special_tokens=False)[-1]
            for a in range(9)
        ]
        _ACTION_TOKEN_TENSOR = torch.tensor(token_ids, dtype=torch.long, device=device)
    return _ACTION_TOKEN_TENSOR


def get_valid_action_mask(state: GreenState, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(9, dtype=torch.bool, device=device)
    history = state.get("history", [])
    subqueries = state.get("sub_queries", state.get("subqueries", []))
    active_sq = get_active_subquery(state)

    if not history:
        mask[[0, 1, 2, 3, 4, 5, 6, 7]] = True
        return mask

    last_act = history[-1].get("action_id")

    if active_sq is not None:
        if last_act in (6, 7, 0, 1):
            mask[[2, 3]] = True
        elif last_act in (2, 3):
            mask[[0, 1, 4, 5]] = True
        else:
            mask[[0, 1]] = True
        return mask

    if len(subqueries) > 0 and active_sq is None:
        mask[[0, 1]] = True
        return mask

    if last_act in (2, 3):
        mask[[0, 1, 2, 3, 4, 5, 6]] = True
    elif last_act in (4, 5):
        mask[[0, 1, 2, 3]] = True
    else:
        mask[[0, 1, 2, 3, 4, 5]] = True

    return mask


def greedy_action(
    model,
    tokenizer,
    prompt: str,
    valid_mask: torch.Tensor,
    device: torch.device,
) -> int:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        action_tokens = get_action_token_tensor(tokenizer, logits.device)
        action_logits = logits[action_tokens].float()
        action_logits = action_logits.masked_fill(~valid_mask, -1e9)
        action_idx = torch.argmax(action_logits).item()
    return int(action_idx)


def rollout_greedy(
    model,
    tokenizer,
    engine: GreenEngine,
    initial_state: GreenState,
    device: torch.device,
) -> Tuple[List[int], GreenState, bool]:
    model.eval()
    state = copy.deepcopy(initial_state)
    action_ids = []
    format_valid = False

    for _ in range(MAX_STEPS_PER_TRAJ):
        prompt = format_state_for_prompt(state) + " Action: "
        valid_mask = get_valid_action_mask(state, device)
        action_id = greedy_action(model, tokenizer, prompt, valid_mask, device)
        action_ids.append(action_id)

        try:
            state = engine.step(state, action_id)
            format_valid = True
        except Exception as exc:
            logger.warning("Engine step error: %s", exc)
            break

        if state.get("status") in ("SOLVED", "FAILED"):
            break

    return action_ids, state, format_valid


def rollout_fixed_policy(
    policy_type: str,
    engine: GreenEngine,
    initial_state: GreenState,
) -> Tuple[List[int], GreenState, bool]:
    """Executes deterministic static baselines without policy-model inference."""
    state = copy.deepcopy(initial_state)
    action_ids = []
    format_valid = False

    if policy_type == "fixed_2_1":
        plan = [actions.ACTION_RET_KEY, actions.ACTION_GEN_LLM]
        for aid in plan:
            action_ids.append(aid)
            try:
                state = engine.step(state, aid)
                format_valid = True
            except Exception as exc:
                logger.warning("Engine step error (%s): %s", policy_type, exc)
                break
            if state.get("status") in ("SOLVED", "FAILED"):
                break
        return action_ids, state, format_valid

    if policy_type == "fixed_3_1":
        plan = [actions.ACTION_RET_VEC, actions.ACTION_GEN_LLM]
        for aid in plan:
            action_ids.append(aid)
            try:
                state = engine.step(state, aid)
                format_valid = True
            except Exception as exc:
                logger.warning("Engine step error (%s): %s", policy_type, exc)
                break
            if state.get("status") in ("SOLVED", "FAILED"):
                break
        return action_ids, state, format_valid

    if policy_type == "fixed_decompose":
        # Step 0: Decompose
        for _ in range(MAX_STEPS_PER_TRAJ):
            history = state.get("history", [])
            active_sq = get_active_subquery(state)
            last_act = history[-1].get("action_id") if history else None

            if not history:
                aid = actions.ACTION_DEC_LLM
            elif active_sq is not None:
                if last_act in (6, 7, 0, 1):
                    aid = actions.ACTION_RET_VEC
                elif last_act in (2, 3):
                    aid = actions.ACTION_RSN_LLM
                else:
                    aid = actions.ACTION_GEN_LLM
            else:
                aid = actions.ACTION_GEN_LLM

            action_ids.append(aid)
            try:
                state = engine.step(state, aid)
                format_valid = True
            except Exception as exc:
                logger.warning("Engine step error (%s): %s", policy_type, exc)
                break
            if state.get("status") in ("SOLVED", "FAILED"):
                break
        return action_ids, state, format_valid

    raise ValueError(f"Unknown fixed policy_type: {policy_type}")


# ──────────────────────────────────────────────────────────────────────────────
# AGGREGATION, STRATIFICATION & RESCUE METRICS
# ──────────────────────────────────────────────────────────────────────────────

def summarize_records(
    dataset: str,
    label: str,
    records: List[Dict[str, Any]],
    elapsed_sec: float,
    ref_2_1_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    n = max(1, len(records))
    acc = float(np.mean([r["is_correct"] for r in records]))
    em = float(np.mean([r["em"] for r in records]))
    f1 = float(np.mean([r["f1"] for r in records]))
    mean_joules = float(np.mean([r["joules"] for r in records]))
    median_joules = float(np.median([r["joules"] for r in records]))
    mean_latency = float(np.mean([r["latency_sec"] for r in records]))
    effective_power_w = mean_joules / max(mean_latency, 1e-6)
    avg_steps = float(np.mean([r["num_steps"] for r in records]))
    timeout_pct = float(np.mean([r["is_timeout"] for r in records])) * 100.0
    multi_step_pct = float(np.mean([r["is_multi_hop"] for r in records])) * 100.0

    # Action shares
    action_counts = {"generate": 0, "keyword": 0, "dense": 0, "decompose": 0, "reason": 0}
    total_actions = 0
    path_counter = Counter()
    for r in records:
        path_counter[r["path_str"]] += 1
        for aid in r["action_ids"]:
            total_actions += 1
            if aid in _GENERATE_IDS:
                action_counts["generate"] += 1
            elif aid in _KEYWORD_IDS:
                action_counts["keyword"] += 1
            elif aid in _DENSE_IDS:
                action_counts["dense"] += 1
            elif aid in _DECOMPOSE_IDS:
                action_counts["decompose"] += 1
            elif aid in _REASON_IDS:
                action_counts["reason"] += 1

    safe_total = max(1, total_actions)
    top_paths = ", ".join(f"`[{p}]` ({c})" for p, c in path_counter.most_common(4))

    # Stratified Accuracy & Energy (Single-Hop vs Multi-Hop vs Decomposed)
    sh_recs = [r for r in records if not r["is_multi_hop"]]
    mh_recs = [r for r in records if r["is_multi_hop"]]
    dec_recs = [r for r in records if r["has_decompose"]]

    sh_acc = float(np.mean([r["is_correct"] for r in sh_recs])) if sh_recs else None
    sh_joules = float(np.mean([r["joules"] for r in sh_recs])) if sh_recs else None
    mh_acc = float(np.mean([r["is_correct"] for r in mh_recs])) if mh_recs else None
    mh_joules = float(np.mean([r["joules"] for r in mh_recs])) if mh_recs else None
    dec_acc = float(np.mean([r["is_correct"] for r in dec_recs])) if dec_recs else None
    dec_joules = float(np.mean([r["joules"] for r in dec_recs])) if dec_recs else None

    # Cross-Policy Rescue & Escalation Precision vs Fixed [2 -> 1]
    rescue_rate_pct = None
    mh_escalation_precision_pct = None
    mh_rescue_success_pct = None
    net_gain_vs_2_1 = None

    if ref_2_1_records is not None and len(ref_2_1_records) == len(records):
        rescued = 0
        broken = 0
        mh_targeted_failures = 0
        mh_rescued_failures = 0

        for r, ref in zip(records, ref_2_1_records):
            ref_ok = bool(ref["is_correct"])
            cur_ok = bool(r["is_correct"])
            if not ref_ok and cur_ok:
                rescued += 1
            elif ref_ok and not cur_ok:
                broken += 1

            if r["is_multi_hop"] and not ref_ok:
                mh_targeted_failures += 1
                if cur_ok:
                    mh_rescued_failures += 1

        rescue_rate_pct = (rescued / n) * 100.0
        net_gain_vs_2_1 = rescued - broken
        if len(mh_recs) > 0:
            mh_escalation_precision_pct = (mh_targeted_failures / len(mh_recs)) * 100.0
        if mh_targeted_failures > 0:
            mh_rescue_success_pct = (mh_rescued_failures / mh_targeted_failures) * 100.0

    return {
        "dataset": dataset,
        "label": label,
        "num_samples": len(records),
        "accuracy": acc,
        "em": em,
        "f1": f1,
        "mean_joules": mean_joules,
        "median_joules": median_joules,
        "mean_latency_sec": mean_latency,
        "effective_power_watts": effective_power_w,
        "avg_steps": avg_steps,
        "timeout_pct": timeout_pct,
        "multi_step_pct": multi_step_pct,
        "keyword_pct": (action_counts["keyword"] / safe_total) * 100.0,
        "dense_pct": (action_counts["dense"] / safe_total) * 100.0,
        "reason_pct": (action_counts["reason"] / safe_total) * 100.0,
        "decompose_pct": (action_counts["decompose"] / safe_total) * 100.0,
        # Stratification
        "single_hop_count": len(sh_recs),
        "single_hop_acc": sh_acc,
        "single_hop_mean_joules": sh_joules,
        "multi_hop_count": len(mh_recs),
        "multi_hop_acc": mh_acc,
        "multi_hop_mean_joules": mh_joules,
        "decompose_count": len(dec_recs),
        "decompose_acc": dec_acc,
        "decompose_mean_joules": dec_joules,
        # Rescue & Escalation vs Fixed [2 -> 1]
        "rescue_rate_pct": rescue_rate_pct,
        "mh_escalation_precision_pct": mh_escalation_precision_pct,
        "mh_rescue_success_pct": mh_rescue_success_pct,
        "net_gain_vs_2_1": net_gain_vs_2_1,
        "top_paths": top_paths,
        "elapsed_sec": elapsed_sec,
    }


def evaluate_policy(
    dataset: str,
    label: str,
    engine: GreenEngine,
    eval_samples: List[Dict[str, Any]],
    device: torch.device,
    model=None,
    adapter_name: Optional[str] = None,
    tokenizer=None,
    fixed_policy_type: Optional[str] = None,
    ref_2_1_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if model is not None and adapter_name is not None:
        model.set_adapter(adapter_name)
        model.eval()

    per_query_records: List[Dict[str, Any]] = []
    t0 = time.time()

    for idx, sample in enumerate(eval_samples):
        question = sample["question"]
        ground_truth = sample["answer"]
        init_state = create_initial_state(question)

        t_q0 = time.perf_counter()
        if fixed_policy_type is not None:
            action_ids, final_state, format_valid = rollout_fixed_policy(
                fixed_policy_type, engine, init_state
            )
        else:
            action_ids, final_state, format_valid = rollout_greedy(
                model, tokenizer, engine, init_state, device
            )
        q_latency = time.perf_counter() - t_q0

        final_answer = (final_state.get("answer") or "").strip() if final_state else ""
        status = final_state.get("status") if final_state else "FAILED"
        is_timeout = (not final_answer) or (status not in ("SOLVED", "FAILED"))

        if is_timeout or not format_valid:
            is_correct = False
            em_score = 0.0
            f1_score = 0.0
        else:
            is_correct, _ = _judge.judge(final_answer, ground_truth, question)
            em_score = compute_em(final_answer, ground_truth)
            f1_score = compute_f1(final_answer, ground_truth)

        joules = float(final_state.get("total_joules", 0.0)) if final_state else 0.0
        path_str = " -> ".join(str(a) for a in action_ids)
        is_multi_hop = len(action_ids) > 2
        has_decompose = any(a in _DECOMPOSE_IDS for a in action_ids)

        per_query_records.append({
            "q_idx": idx,
            "dataset": dataset,
            "label": label,
            "question": question,
            "ground_truth": ground_truth,
            "prediction": final_answer,
            "action_ids": action_ids,
            "path_str": path_str,
            "num_steps": len(action_ids),
            "is_multi_hop": is_multi_hop,
            "has_decompose": has_decompose,
            "is_timeout": bool(is_timeout),
            "is_correct": bool(is_correct),
            "em": em_score,
            "f1": f1_score,
            "joules": joules,
            "latency_sec": q_latency,
            "power_watts": joules / max(q_latency, 1e-6),
            "prev_searches": final_state.get("prev_searches", []) if final_state else [],
            "doc_titles": [
                d.get("title", "") for d in final_state.get("documents", [])
            ] if final_state else [],
            "step_args": [
                {"action_id": h.get("action_id"), "arg": h.get("argument", "")}
                for h in final_state.get("history", [])
            ] if final_state else [],
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(eval_samples):
            running_acc = np.mean([r["is_correct"] for r in per_query_records])
            running_em = np.mean([r["em"] for r in per_query_records])
            running_j = np.mean([r["joules"] for r in per_query_records])
            logger.info(
                "[%s | %s] %d/%d | Acc: %.4f | EM: %.4f | Mean J: %.1fJ",
                dataset, label, idx + 1, len(eval_samples), running_acc, running_em, running_j
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    summary = summarize_records(
        dataset=dataset,
        label=label,
        records=per_query_records,
        elapsed_sec=elapsed,
        ref_2_1_records=ref_2_1_records,
    )
    logger.info(
        "DONE [%s | %s] => Acc: %.1f%% | EM: %.1f%% | F1: %.1f%% | Joules: %.1fJ | Power: %.1fW | Multi-Hop: %.1f%%",
        dataset, label,
        summary["accuracy"] * 100, summary["em"] * 100, summary["f1"] * 100,
        summary["mean_joules"], summary["effective_power_watts"], summary["multi_step_pct"]
    )
    return summary, per_query_records


def build_abstention_cascade_baseline(
    dataset: str,
    rec_2_1: List[Dict[str, Any]],
    rec_dec: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Zero-overhead synthesis of a Heuristic Abstention Cascade:
    Runs Fixed [2 -> 1]; if it abstains/returns unknown, escalates to Fixed Decompose.
    """
    cascade_records: List[Dict[str, Any]] = []
    for r1, r2 in zip(rec_2_1, rec_dec):
        if is_abstention(r1["prediction"]):
            comb_actions = r1["action_ids"] + r2["action_ids"]
            comb_joules = r1["joules"] + r2["joules"]
            comb_latency = r1["latency_sec"] + r2["latency_sec"]
            cascade_records.append({
                "q_idx": r1["q_idx"],
                "dataset": dataset,
                "label": "Heuristic Cascade ([2->1] -> Dec)",
                "question": r1["question"],
                "ground_truth": r1["ground_truth"],
                "prediction": r2["prediction"],
                "action_ids": comb_actions,
                "path_str": f"{r1['path_str']} || {r2['path_str']}",
                "num_steps": len(comb_actions),
                "is_multi_hop": True,
                "has_decompose": True,
                "is_timeout": r2["is_timeout"],
                "is_correct": r2["is_correct"],
                "em": r2["em"],
                "f1": r2["f1"],
                "joules": comb_joules,
                "latency_sec": comb_latency,
                "power_watts": comb_joules / max(comb_latency, 1e-6),
            })
        else:
            copy_r1 = copy.deepcopy(r1)
            copy_r1["label"] = "Heuristic Cascade ([2->1] -> Dec)"
            cascade_records.append(copy_r1)

    summary = summarize_records(
        dataset=dataset,
        label="Heuristic Cascade ([2->1] -> Dec)",
        records=cascade_records,
        elapsed_sec=0.0,
        ref_2_1_records=rec_2_1,
    )
    return summary, cascade_records


def save_incremental_json(
    summary_results: List[Dict[str, Any]],
    per_query_all: Dict[str, List[Dict[str, Any]]],
    output_json: str,
) -> None:
    if os.path.dirname(output_json):
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    base, ext = os.path.splitext(output_json)
    pq_path = f"{base}_per_query{ext or '.json'}"
    with open(pq_path, "w", encoding="utf-8") as f:
        json.dump(per_query_all, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GreenRAG GRPO Checkpoints")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_sft", action="store_true", help="Skip evaluating SFT step_0 baseline")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip static [2->1], [3->1], and Decompose baselines")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="List of checkpoint paths")
    parser.add_argument("--output_json", type=str, default="results/eval_v9_full.json")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["hotpot", "squad", "nq"],
        default=None,
        help="One or more datasets to evaluate on (e.g., --datasets hotpot squad)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hotpot", "squad", "nq"],
        default="hotpot",
        help="Single dataset alias (kept for backwards compatibility)",
    )
    args = parser.parse_args()

    target_datasets = args.datasets if args.datasets else [args.dataset]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Base Model + SFT Adapter ONCE across all datasets
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading Base Model in BF16: %s...", BASE_MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base_model, SFT_MODEL_PATH, adapter_name="sft_base", is_trainable=False
    )

    all_summaries: List[Dict[str, Any]] = []
    all_per_query: Dict[str, List[Dict[str, Any]]] = {}

    for ds in target_datasets:
        logger.info("\n" + "#" * 90)
        logger.info("### STARTING EVALUATION ON DATASET: %s (N=%d, seed=%d)", ds.upper(), args.num_samples, args.seed)
        logger.info("#" * 90)

        # Clear cached retriever instances when switching corpora to save RAM/VRAM
        GlobalRetriever._instances.clear()
        corpus_type = "fullwiki" if ds == "hotpot" else "dpr_wiki"
        dataset_config = (
            {"setting": "fullwiki", "split": "validation"}
            if ds == "hotpot"
            else {"split": "validation"}
        )

        eval_streamer = MixedStreamer(
            dataset_names=[ds],
            weights=[1.0],
            shuffle=True,
            seed=args.seed,
            configs={ds: dataset_config},
        )
        eval_iter = eval_streamer.stream()
        eval_samples = []
        for _ in range(args.num_samples):
            try:
                eval_samples.append(next(eval_iter))
            except StopIteration:
                break
        logger.info("Loaded %d validation samples for %s.", len(eval_samples), ds)

        retriever = GlobalRetriever.get_instance(corpus_type=corpus_type, index_type="hnsw")
        engine = GreenEngine(retriever=retriever)

        ref_2_1_records: Optional[List[Dict[str, Any]]] = None

        # 1. Deterministic Baselines (Run Fixed [2->1] first so it serves as the Rescue Reference)
        if not args.skip_baselines:
            logger.info("=== [%s] Evaluating Fixed [2 -> 1] (Lexical -> LLM) ===", ds)
            res_2_1, rec_2_1 = evaluate_policy(
                dataset=ds,
                label="Fixed [2 -> 1] (Key->LLM)",
                engine=engine,
                eval_samples=eval_samples,
                device=device,
                fixed_policy_type="fixed_2_1",
            )
            ref_2_1_records = rec_2_1
            all_summaries.append(res_2_1)
            all_per_query[f"{ds}::Fixed [2 -> 1] (Key->LLM)"] = rec_2_1
            save_incremental_json(all_summaries, all_per_query, args.output_json)

            logger.info("=== [%s] Evaluating Fixed [3 -> 1] (Dense -> LLM) ===", ds)
            res_3_1, rec_3_1 = evaluate_policy(
                dataset=ds,
                label="Fixed [3 -> 1] (Vec->LLM)",
                engine=engine,
                eval_samples=eval_samples,
                device=device,
                fixed_policy_type="fixed_3_1",
                ref_2_1_records=ref_2_1_records,
            )
            all_summaries.append(res_3_1)
            all_per_query[f"{ds}::Fixed [3 -> 1] (Vec->LLM)"] = rec_3_1
            save_incremental_json(all_summaries, all_per_query, args.output_json)

            logger.info("=== [%s] Evaluating Fixed Decompose [6 -> (3->5->1)* -> 1] ===", ds)
            res_dec, rec_dec = evaluate_policy(
                dataset=ds,
                label="Fixed Decompose (Full Multi-Hop)",
                engine=engine,
                eval_samples=eval_samples,
                device=device,
                fixed_policy_type="fixed_decompose",
                ref_2_1_records=ref_2_1_records,
            )
            all_summaries.append(res_dec)
            all_per_query[f"{ds}::Fixed Decompose (Full Multi-Hop)"] = rec_dec

            # Zero-cost Heuristic Abstention Cascade baseline
            res_casc, rec_casc = build_abstention_cascade_baseline(ds, rec_2_1, rec_dec)
            all_summaries.append(res_casc)
            all_per_query[f"{ds}::Heuristic Cascade ([2->1] -> Dec)"] = rec_casc
            save_incremental_json(all_summaries, all_per_query, args.output_json)

        # 2. Evaluate SFT Baseline (Step 0)
        if not args.skip_sft:
            logger.info("=== [%s] Evaluating SFT Baseline (%s) ===", ds, SFT_MODEL_PATH)
            res_sft, rec_sft = evaluate_policy(
                dataset=ds,
                label="SFT (step_0)",
                engine=engine,
                eval_samples=eval_samples,
                device=device,
                model=model,
                adapter_name="sft_base",
                tokenizer=tokenizer,
                ref_2_1_records=ref_2_1_records,
            )
            all_summaries.append(res_sft)
            all_per_query[f"{ds}::SFT (step_0)"] = rec_sft
            save_incremental_json(all_summaries, all_per_query, args.output_json)

        # 3. Evaluate GRPO Checkpoints
        for i, ckpt_path in enumerate(args.checkpoints):
            load_path = ckpt_path
            if os.path.isdir(os.path.join(ckpt_path, "active_rl")):
                load_path = os.path.join(ckpt_path, "active_rl")

            if not os.path.exists(load_path):
                logger.warning("Checkpoint path not found, skipping: %s", load_path)
                continue

            adapter_name = f"ckpt_{ds}_{i}"
            label = "/".join(ckpt_path.strip("/").split("/")[-2:])
            logger.info("=== [%s] Evaluating Checkpoint: %s ===", ds, label)

            model.load_adapter(load_path, adapter_name=adapter_name, is_trainable=False)
            res, rec = evaluate_policy(
                dataset=ds,
                label=label,
                engine=engine,
                eval_samples=eval_samples,
                device=device,
                model=model,
                adapter_name=adapter_name,
                tokenizer=tokenizer,
                ref_2_1_records=ref_2_1_records,
            )
            all_summaries.append(res)
            all_per_query[f"{ds}::{label}"] = rec
            save_incremental_json(all_summaries, all_per_query, args.output_json)
            model.delete_adapter(adapter_name)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── PRINT STANDARDIZED MAIN BENCHMARK TABLE ──────────────────────────────
    print("\n" + "=" * 130)
    print(f"### Table 1: Standardized Performance & Hardware Telemetry (N = {args.num_samples}, seed = {args.seed})")
    print("| Dataset | Model / Policy | Judge Acc | EM | F1 | Mean J | Med J | Latency | Power | Multi-Hop % |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in all_summaries:
        print(
            f"| {r['dataset']} | **{r['label']}** | {r['accuracy']*100:.1f}% | {r['em']*100:.1f}% | "
            f"{r['f1']*100:.1f}% | {r['mean_joules']:.1f} J | {r['median_joules']:.1f} J | "
            f"{r['mean_latency_sec']:.2f} s | {r['effective_power_watts']:.1f} W | {r['multi_step_pct']:.1f}% |"
        )
    print("=" * 130 + "\n")

    # ── PRINT STRATIFICATION & RESCUE TABLE ──────────────────────────────────
    print("=" * 130)
    print(f"### Table 2: Query Stratification & Multi-Hop Rescue Diagnostics (vs. Fixed [2 -> 1])")
    print("| Dataset | Model / Policy | 1-Hop Acc (n) | Multi-Hop Acc (n) | Esc. Prec. | Rescue Rate | Net vs [2->1] | Top Action Paths |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
    for r in all_summaries:
        sh_str = f"{r['single_hop_acc']*100:.1f}% ({r['single_hop_count']})" if r["single_hop_acc"] is not None else "—"
        mh_str = f"{r['multi_hop_acc']*100:.1f}% ({r['multi_hop_count']})" if r["multi_hop_acc"] is not None else "—"
        ep_str = f"{r['mh_escalation_precision_pct']:.1f}%" if r["mh_escalation_precision_pct"] is not None else "—"
        rr_str = f"{r['rescue_rate_pct']:.1f}%" if r["rescue_rate_pct"] is not None else "—"
        ng_str = f"{r['net_gain_vs_2_1']:+d}" if r["net_gain_vs_2_1"] is not None else "—"
        print(
            f"| {r['dataset']} | **{r['label']}** | {sh_str} | {mh_str} | "
            f"{ep_str} | {rr_str} | {ng_str} | {r['top_paths']} |"
        )
    print("=" * 130 + "\n")


if __name__ == "__main__":
    main()