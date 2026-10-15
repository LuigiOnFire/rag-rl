import os
import sys
import glob
import time
import random
import logging
from typing import Any, Dict, List, Union

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

sys.path.append(os.getcwd())
from src.train.dataset import load_and_clean_dataset
from src.agent import actions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- CONFIG ---
RAW_MODEL = os.getenv("SLM_MODEL", "qwen2.5:3b")
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct" if RAW_MODEL in ("qwen2.5:3b", "qwen2.5") else RAW_MODEL
RUN_LOGS_DIR = "data/trajectories"
OUTPUT_DIR = "models/green-rag-sft-v1"

COMPLEX_ACTION_IDS = {
    actions.ACTION_DEC_LLM,
    actions.ACTION_DEC_RSN,
    actions.ACTION_RSN_LLM,
}
COMPLEX_ACTION_MULTIPLIER = 2.0  # Normalized down from 4.0 to prevent overfitting
DATASET_SEED = 42


class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """
    Robust completion-only collator that searches for either 'Action:' or ' Action:'
    and masks prompt tokens with -100.
    """
    def __init__(self, tokenizer: Any, response_template: str = "Action:"):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.response_template = response_template
        # Cache token IDs for variations with and without leading space
        self.target_token_variants = [
            self.tokenizer.encode(response_template, add_special_tokens=False),
            self.tokenizer.encode(" " + response_template, add_special_tokens=False),
            self.tokenizer.encode("\n" + response_template, add_special_tokens=False),
        ]

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()

        for i in range(len(examples)):
            found_idx = -1
            
            # Scan for any valid representation of the response prefix
            for target_tokens in self.target_token_variants:
                target_len = len(target_tokens)
                target_t = torch.tensor(target_tokens, device=labels.device)
                
                for idx in range(len(labels[i]) - target_len + 1):
                    if torch.all(labels[i][idx : idx + target_len] == target_t):
                        found_idx = idx + target_len
                        break
                if found_idx != -1:
                    break

            if found_idx != -1:
                labels[i][:found_idx] = -100
            else:
                logger.warning("Response template not found in sample. Masking entire sample.")
                labels[i][:] = -100

        batch["labels"] = labels
        return batch


def main():
    logger.info("Loading Tokenizer for %s...", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 1. Discover Trajectory Files
    jsonl_files = glob.glob(f"{RUN_LOGS_DIR}/**/gold_trajectories.jsonl", recursive=True)
    if not jsonl_files:
        logger.error("No training data found in %s/**/gold_trajectories.jsonl", RUN_LOGS_DIR)
        return

    logger.info("Found %d trajectory files.", len(jsonl_files))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"data/sft_runs/run_{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    # 2. Build Dataset
    oversample_config = {
        "complex_action_ids": COMPLEX_ACTION_IDS,
        "complex_multiplier": COMPLEX_ACTION_MULTIPLIER,
    }
    rng = random.Random(DATASET_SEED)
    dataset = load_and_clean_dataset(jsonl_files, tokenizer, oversample_config, rng)
    if len(dataset) == 0:
        logger.error("Loaded dataset is empty. Halting.")
        return

    logger.info("Loaded %d formatted training samples.", len(dataset))
    dataset.select(range(min(50, len(dataset)))).to_json(
        f"{run_dir}/dataset_preview.json", orient="records", lines=True
    )

    # 3. Model Loading (Native BF16 on L40S)
    logger.info("Loading base model in native bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False,
    )

    # 4. LoRA Setup (Targeting all linear matrices)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # 5. Training Arguments
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=2048,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,          # Calibrated 3-epoch pass
        warmup_ratio=0.05,           # Reaches peak LR quickly
        lr_scheduler_type="cosine",
        optim="adamw_torch",         # Standard PyTorch AdamW for native BF16
        save_strategy="steps",
        save_steps=50,               # Never lose progress on interruption
        save_total_limit=3,
        bf16=True,
        report_to="none",
    )

    collator = CompletionOnlyCollator(tokenizer=tokenizer, response_template="Action:")

    trainer = SFTTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=collator,
    )

    logger.info("Starting SFT training pass...")
    trainer.train()

    logger.info("Saving final adapter model to %s...", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("SFT stage complete and saved successfully.")


if __name__ == "__main__":
    main()