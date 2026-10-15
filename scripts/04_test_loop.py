import os
import sys
import re
import time
import logging
from typing import Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.env.state import create_initial_state, GreenState
from src.agent.prompts import format_state_for_prompt
from src.agent import actions
from src.data.hotpot import HotpotQAStreamer
from src.env.retriever import GlobalRetriever
from src.env.engine import GreenEngine

# --- LOGGING CONFIG (Silences fsspec & HTTP spam) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)
for noisy in ["fsspec", "urllib3", "filelock", "datasets", "transformers"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# --- CONFIG ---
RAW_MODEL = os.getenv("SLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct" if RAW_MODEL in ("qwen2.5:3b", "qwen2.5") else RAW_MODEL
ADAPTER_PATH = "models/green-rag-sft-v1"
MAX_STEPS = 5

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_FILE = f"data/post_sft_tests/run_{RUN_ID}.log"
os.makedirs(os.path.dirname(RUN_FILE), exist_ok=True)


# Module-level cache placeholder
_ACTION_TOKEN_TENSOR = None

import shutil
import subprocess

HAS_SCONTROL = shutil.which("scontrol") is not None

def get_node_or_job_info():
    if not HAS_SCONTROL:
        # Graceful fallback: return None or dummy defaults
        return None

    result = subprocess.check_output("scontrol show job ...", shell=True)
    return parse_output(result)


def load_director() -> Tuple[PeftModel, AutoTokenizer]:
    logger.info("Loading Base Model in native BF16 from %s...", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    logger.info("Loading SFT Adapter from %s...", ADAPTER_PATH)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def parse_action_output(output_text: str) -> int:
    """Robust parser that validates action IDs (0-8) and named action strings."""
    # Pattern 1: Look for exact single-digit action IDs (0-8)
    match = re.search(r"\b([0-8])\b", output_text)
    if match:
        return int(match.group(1))

    # Pattern 2: Fallback for action names if the model outputs identifiers
    name_map = {
        "GEN_SLM": actions.ACTION_GEN_SLM,
        "GEN_LLM": actions.ACTION_GEN_LLM,
        "RET_KEY": actions.ACTION_RET_KEY,
        "RET_VEC": actions.ACTION_RET_VEC,
        "RSN_SLM": actions.ACTION_RSN_SLM,
        "RSN_LLM": actions.ACTION_RSN_LLM,
        "DEC_LLM": actions.ACTION_DEC_LLM,
        "DEC_RSN": actions.ACTION_DEC_RSN,
        "FAIL": actions.ACTION_FAIL,
    }
    for name, a_id in name_map.items():
        if name in output_text.upper():
            return a_id

    return actions.ACTION_FAIL

def get_action_token_tensor(tokenizer, device) -> torch.Tensor:
    global _ACTION_TOKEN_TENSOR
    if _ACTION_TOKEN_TENSOR is None or _ACTION_TOKEN_TENSOR.device != device:
        token_ids = [
            tokenizer.encode(str(a), add_special_tokens=False)[-1]
            for a in range(8)
        ]
        _ACTION_TOKEN_TENSOR = torch.tensor(token_ids, dtype=torch.long, device=device)
    return _ACTION_TOKEN_TENSOR


def get_director_action(model, tokenizer, state: GreenState, temperature: float = 0.0) -> int:
    prompt = format_state_for_prompt(state) + " Action: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    action_token_tensor = get_action_token_tensor(tokenizer, model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Logits at the decision position
        next_token_logits = outputs.logits[0, -1, :]

        # Slice logits strictly for action tokens 0 through 7
        action_logits = next_token_logits[action_token_tensor]
        probs = torch.softmax(action_logits, dim=-1)

        if temperature <= 0.0:
            chosen_idx = torch.argmax(action_logits).item()
        else:
            scaled_logits = action_logits / temperature
            sampled_probs = torch.softmax(scaled_logits, dim=-1)
            chosen_idx = torch.multinomial(sampled_probs, num_samples=1).item()

    print("\n--- Action Probabilities (Constrained) ---")
    for a in range(8):
        print(f"Action {a}: {probs[a].item() * 100:5.2f}%")

    with open(RUN_FILE, "a") as f:
        f.write(f"\n--- PROMPT ---\n{prompt}\n")
        f.write("--- ACTION PROBABILITIES ---\n")
        for a in range(8):
            f.write(f"Action {a}: {probs[a].item() * 100:.2f}%\n")
        f.write(f"--- SELECTED ACTION ID: {chosen_idx} (T={temperature}) ---\n")

    return chosen_idx

def run_hotpot_episode(model, tokenizer, sample: dict):
    question = sample.get("question", "")
    ground_truth = sample.get("ground_truth") or sample.get("answer", "")

    print(f"\n{'='*70}\nHOTPOT Q: {question}\n{'='*70}")

    state = create_initial_state(question)
    # Ensure GlobalRetriever loads the HNSW index
    retriever = GlobalRetriever.get_instance(corpus_type="fullwiki", index_type="hnsw")
    engine = GreenEngine(retriever=retriever)

    with open(RUN_FILE, "a") as f:
        f.write(f"\n\n{'='*50}\nNEW EPISODE: {question}\nTarget Answer: {ground_truth}\n")

    for step in range(MAX_STEPS):
        action_id = get_director_action(model, tokenizer, state)
        action_name = getattr(actions, f"ACTION_{action_id}", str(action_id))
        print(f"Step {step+1}/{MAX_STEPS} -> Decided Action: {action_id} ({action_name})")

        new_state = engine.step(state, action_id)
        state = new_state

        if new_state.get("status") in ("SOLVED", "FAILED"):
            status = new_state.get("status")
            pred = new_state.get("answer", "")
            print(f"\n*** EPISODE ENDED [{status}] ***")
            print(f"Prediction  : {pred}")
            print(f"Ground Truth: {ground_truth}")
            break


def main():
    model, tokenizer = load_director()

    logger.info("Streaming 5 HotpotQA test samples...")
    streamer = HotpotQAStreamer(split="train", limit=5)

    for idx, sample in enumerate(streamer.stream()):
        print(f"\n>>> Running Episode {idx+1}/5 <<<")
        run_hotpot_episode(model, tokenizer, sample)

    print(f"\nTesting complete! Detailed traces written to: {RUN_FILE}")


if __name__ == "__main__":
    main()