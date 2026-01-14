import os
import sys
import glob
import torch

# Add project root to sys.path
sys.path.append(os.getcwd())

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, prepare_model_for_kbit_training
# Pylance/Linter friendly imports
from trl.trainer.sft_trainer import SFTTrainer
from trl.trainer.sft_config import SFTConfig

from src.train.dataset import create_green_dataset

# Configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct" 
# Using 1B as proxy since 8B might OOM on smaller environments, 
# but user asked for 8B. I'll stick to 1B for safety or use param if possible.
# Start with user request: "Llama3:8b" -> usually "meta-llama/Meta-Llama-3-8B-Instruct"
# But environment likely doesn't have access. I will use the path from workers.py or standard HF text.
# For this script, let's allow an env var or default to a known accessible model.
# I will use the one requested but be ready to fallback.
TARGET_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

OUTPUT_DIR = "models/green-rag-sft-v1"
RUN_LOGS_DIR = "data/runs"

def main():
    print(f"Loading Model: {TARGET_MODEL}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Set padding side to right for generation/SFT usually, but LLama prefers left for batched inference
    # SFTTrainer defaults:
    tokenizer.padding_side = "right" 
    
    # 2. Find Data
    # Look for all gold_trajectories.jsonl recursively
    jsonl_files = glob.glob(f"{RUN_LOGS_DIR}/**/gold_trajectories.jsonl", recursive=True)
    if not jsonl_files:
        print("No training data found in data/runs/**/gold_trajectories.jsonl")
        return

    print(f"Found {len(jsonl_files)} trajectory files.")

    # 3. Create Dataset
    # Returns a Hugging Face 'datasets.Dataset'
    dataset = create_green_dataset(jsonl_files, tokenizer)
    if len(dataset) == 0:
        print("Dataset is empty. Exiting.")
        return

    # 4. Load Model (Quantized)
    # Using 4-bit quantization
    from transformers import BitsAndBytesConfig
    
    # Determine best precision
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    print(f"BF16 Support: {use_bf16}. Using compute_dtype: {compute_dtype}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=False 
    )
    
    model = prepare_model_for_kbit_training(model)
    
    # 5. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    
    # 6. Training Configuration (REPLACED TrainingArguments)
    # SFTConfig inherits from TrainingArguments but adds SFT-specific params
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",  
        max_length=2048,    
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        # packing=False, # Optional: explicit False is safer for some versions
    )
    
    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )
    
    print("Starting Training...")
    trainer.train()
    
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()
