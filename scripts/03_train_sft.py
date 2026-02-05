import torch
import logging
import copy
from typing import Any, Dict, List, Union
import glob
import json
import time
import os
import sys
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments,
    DataCollatorForLanguageModeling # We build on top of this standard class
)
from peft import LoraConfig, prepare_model_for_kbit_training
# Pylance/Linter friendly imports
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

sys.path.append(os.getcwd())
from src.train.dataset import load_and_clean_dataset
from src.agent.prompts import format_state_for_prompt

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- CONFIG ---
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
RUN_LOGS_DIR = "data/trajectories"
OUTPUT_DIR = "models/green-rag-sft-v1"
# --- 1. CLEANING & LOADING LOGIC ---
# src/train/dataset.py

import json
import glob
import logging
from datasets import Dataset

# --- MANUAL COLLATOR IMPLEMENTATION (Bypasses Import Errors) ---
class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """
    Manually implements the masking logic. 
    Everything BEFORE the 'response_template' is masked (Label = -100).
    """
    def __init__(self, response_template: str, tokenizer: Any, mlm: bool = False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        # Encode the template to find its token IDs
        self.response_token_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()

        for i in range(len(examples)):
            # Find where the response starts
            # We look for the sequence of tokens matching " Action:"
            # This is a simple linear search for the token sequence
            found_idx = -1
            for idx in range(len(labels[i]) - len(self.response_token_ids) + 1):
                if torch.all(labels[i][idx : idx + len(self.response_token_ids)] == torch.tensor(self.response_token_ids)):
                    found_idx = idx + len(self.response_token_ids)
                    break
            
            if found_idx != -1:
                # Mask everything BEFORE the response starts
                # -100 is the PyTorch "Ignore Index" for CrossEntropyLoss
                labels[i][:found_idx] = -100
            else:
                # If we didn't find the template (rare), mask everything just to be safe
                # so we don't train on garbage.
                labels[i][:] = -100
        
        batch["labels"] = labels
        return batch
# -------------------------------------------------------------

def main():
    logger.info("Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # 2. Find Data
    # Look for all gold_trajectories.jsonl recursively
    jsonl_files = glob.glob(f"{RUN_LOGS_DIR}/**/gold_trajectories.jsonl", recursive=True)
    if not jsonl_files:
        print("No training data found in data/trajectories/**/gold_trajectories.jsonl")
        return

    print(f"Found {len(jsonl_files)} trajectory files.")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"data/sft_runs/run_{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    logging.debug(f"Run Directory: {run_dir}")
    

    # 3. Create Dataset
    # Returns a Hugging Face 'datasets.Dataset'
    dataset = load_and_clean_dataset(jsonl_files, tokenizer)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return
    
    entries_to_log = min(100, len(dataset))
    print(f"DEBUG: Writing {len(dataset)} samples to 'debug_dataset.json'...")
    dataset.select(range(entries_to_log)).to_json(f"{run_dir}/dataset_preview.json", orient="records", lines=True)
    
    # --- MASKING SETUP ---
    response_template = " Action:" 
    
    # Use our custom manual class
    collator = CompletionOnlyCollator(
        response_template=response_template, 
        tokenizer=tokenizer
    )
    # ---------------------

    logger.info("Loading Base Model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False 
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"] 
    )

    logger.info("Starting SFT...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",  
        max_length=2048,    
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=10,        # 10 Epochs is safe with masking
        max_grad_norm=0.3,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        save_total_limit=5,
        # packing=False, # Optional: explicit False is safer for some versions
    )
    
    # 7. Trainer
    trainer = SFTTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        data_collator=collator,    # <--- Use custom collator
    )

    trainer.train()
    
    logger.info("Saving...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()