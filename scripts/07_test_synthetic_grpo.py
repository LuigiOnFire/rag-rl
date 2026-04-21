"""
07_test_synthetic_grpo.py — Synthetic SanITY Check for GRPO
===========================================================
No RAG. No Worker Models. No Dataset.
The prompt is always exactly the same:
"You are a test agent. Your only job is to press the number 5.\nAction: "
If the model outputs '5', it gets 1.0. Otherwise, 0.0.
"""

import os
import sys
import time
import logging
import random
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import wandb

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

sys.path.append(os.getcwd())

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
SFT_MODEL_PATH       = "models/green-rag-sft-v1"
OUTPUT_DIR           = "models/synthetic-grpo-test"

NUM_GENERATIONS      = 8
BATCH_SIZE           = 2
TOTAL_STEPS          = 200
LEARNING_RATE        = 5e-5
GRADIENT_ACCUM       = 2
KL_COEF              = 0.0
CLIP_EPS             = 0.2

run_id        = time.strftime("%Y%m%d_%H%M%S")
LOG_FILE      = f"data/ppo_training/synthetic_run_{run_id}.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

USE_WANDB      = True
WANDB_PROJECT  = "greenrag-grpo"
WANDB_RUN_NAME = f"synthetic_{run_id}"

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


# ──────────────────────────────────────────────────────────────────────────────
# SYNTHETIC ROLLOUT
# ──────────────────────────────────────────────────────────────────────────────
def rollout_batch_SYNTHETIC_TEST(
    model, tokenizer, generation_config, device, num_generations=8
):
    """
    THE ULTIMATE SANITY CHECK.
    """
    dummy_prompt = "You are a test agent. Your only job is to press a number between 3 and 6.\nAction: "
    
    trajectories = []
    rewards = []
    
    for _ in range(num_generations):
        # 1. Prepare exactly the same prompt
        inputs = tokenizer(dummy_prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        prompt_len = input_ids.shape[1]
        
        # 2. Generate exactly 1 token (the action digit)
        model.eval()
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
            
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1, 
                temperature=0.5,
                do_sample=True,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            gen_ids = outputs.sequences
            new_token_id = gen_ids[0, prompt_len:]
            
            # Cache the old logprob
            logits = outputs.logits[0][0]
            logprobs = F.log_softmax(logits, dim=-1)
            old_logp = logprobs[new_token_id[0]].cpu()
            
        model.config.use_cache = False        
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        model.train()
        
        # 3. Decode and Score
        action_str = tokenizer.decode(new_token_id).strip()
        
        # THE DUMMY REWARD: +1.0 for '5', 0.0 for anything else
        if action_str == "5":
            reward = 1.0
        else:
            reward = 0.0
            
        rewards.append(reward)
        
        # 4. Package for Phase 2
        trajectories.append([{
            "prompt_ids": input_ids[0].cpu().tolist(),
            "new_token_ids": new_token_id.cpu().tolist(),
            "old_logprobs": [old_logp]
        }])
        
    mean_r = sum(rewards) / len(rewards)
    logger.info(f"[SYNTHETIC TEST] Batch Mean Reward: {mean_r:.3f} | Actions generated: {[tokenizer.decode(t[0]['new_token_ids']).strip() for t in trajectories]}")
    
    return trajectories, rewards, [], {"reward/mean": mean_r}


# ──────────────────────────────────────────────────────────────────────────────
# GRPO PHASE 2
# ──────────────────────────────────────────────────────────────────────────────
def _tokenize_with_agent_mask(step_data, device):
    prompt_ids = step_data["prompt_ids"]
    new_token_ids = step_data["new_token_ids"]
    old_logprobs = step_data.get("old_logprobs", [])
    
    full_ids = prompt_ids + new_token_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    
    labels = input_ids.clone()
    prompt_len = len(prompt_ids)
    labels[0, :prompt_len] = -100

    keep_len = prompt_len + 1
    input_ids = input_ids[:, :keep_len]
    attention_mask = attention_mask[:, :keep_len]
    labels = labels[:, :keep_len]

    gen_keep_len = keep_len - prompt_len
    if gen_keep_len > 0 and len(old_logprobs) >= gen_keep_len:
        old_lp_tensor = torch.stack(old_logprobs[:gen_keep_len]).unsqueeze(0).to(device)
    else:
        old_lp_tensor = torch.empty((1, 0), dtype=torch.float32, device=device)

    return input_ids, labels, attention_mask, old_lp_tensor

def grpo_loss(model, tokenizer, trajectories, rewards, device, batch_scale=1.0):
    reward_tensor = torch.tensor(rewards, dtype=torch.float32)
    mean_r = reward_tensor.mean()
    std_r  = reward_tensor.std()

    if std_r < 1e-5:
        if mean_r > 0.5:
            logger.info(f"  [GRPO] Uniform SUCCESS (mean={mean_r:.3f}). Skipping.")
            return 0.0, 0.0
        else:
            logger.info(f"  [GRPO] Uniform FAILURE (mean={mean_r:.3f}).")

    advantages = (reward_tensor - mean_r) 
    total_loss = 0.0
    total_kl   = 0.0

    num_valid = sum(1 for traj in trajectories if len(traj) > 0)
    if num_valid == 0:
        return 0.0, 0.0

    for traj, advantage in zip(trajectories, advantages):
        traj_loss, traj_kl, traj_valid = 0.0, 0.0, 0
        for step_data in traj:
            input_ids, labels, attention_mask, cached_old_logp = _tokenize_with_agent_mask(step_data, device)
            
            if (labels != -100).sum() == 0: continue

            # Ref Policy
            model.set_adapter("reference")
            model.eval()
            with torch.no_grad():
                ref_outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
                ref_logits  = ref_outputs.logits[:, :-1, :]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                
                shift_labels = labels[:, 1:].squeeze(0)
                agent_mask = (shift_labels != -100)
                if agent_mask.sum() == 0: continue
                chosen_ref_logp = ref_log_probs.squeeze(0)[agent_mask].gather(1, shift_labels[agent_mask].unsqueeze(1)).squeeze(1)

            del ref_outputs, ref_logits, ref_log_probs

            # Active Policy
            model.set_adapter("active_rl")
            model.train() 
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
            logits  = outputs.logits[:, :-1, :]
            log_probs = F.log_softmax(logits, dim=-1)

            chosen_logp = log_probs.squeeze(0)[agent_mask].gather(1, shift_labels[agent_mask].unsqueeze(1)).squeeze(1)

            if cached_old_logp is None or cached_old_logp.numel() == 0: continue
            
            prompt_len = len(step_data["prompt_ids"])
            gen_pos = (agent_mask.nonzero(as_tuple=False).squeeze(1) + 1) - prompt_len
            valid = (gen_pos >= 0) & (gen_pos < cached_old_logp.shape[1])
            if valid.sum().item() == 0: continue

            chosen_logp = chosen_logp[valid]
            chosen_ref_logp = chosen_ref_logp[valid]
            old_logp = cached_old_logp[0, gen_pos[valid]].to(device)

            token_count = chosen_logp.numel()
            ratio = torch.exp(chosen_logp - old_logp.detach())
            adv_scalar = advantage.to(device)

            pg_loss = torch.max(
                -adv_scalar * ratio,
                -adv_scalar * torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
            ).sum() / token_count

            kl_log_ratio = chosen_logp - chosen_ref_logp.detach()
            kl = (torch.exp(-kl_log_ratio) + kl_log_ratio - 1.0).sum() / token_count
            kl_loss = KL_COEF * kl

            step_scale = 1.0 / (len(traj) * num_valid * batch_scale)
            scaled_step_loss = (pg_loss + kl_loss) * step_scale
            scaled_step_loss.backward()

            del outputs, logits, log_probs, chosen_logp
            torch.cuda.empty_cache()

            traj_loss += (pg_loss.item() + kl_loss.item()) / len(traj)
            traj_kl += kl.item() / len(traj)
            traj_valid += 1
            
        if traj_valid > 0:
            total_loss += traj_loss
            total_kl += traj_kl
            
    return total_loss / num_valid, total_kl / num_valid

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    logger.info("=== GreenRAG SYNTHETIC GRPO Sanity Test ===")
    
    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME, config={
            "synthetic_test": True, "num_generations": NUM_GENERATIONS,
            "kl_coef": KL_COEF, "clip_eps": CLIP_EPS
        })

    BASE_MODEL_ID = os.getenv("SLM_MODEL", "hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF:latest")
    if BASE_MODEL_ID == "qwen2.5:3b": BASE_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=bnb_config, device_map="auto")
    base_model = prepare_model_for_kbit_training(base_model)

    model = PeftModel.from_pretrained(base_model, SFT_MODEL_PATH, adapter_name="reference", is_trainable=False)
    model.load_adapter(SFT_MODEL_PATH, adapter_name="active_rl", is_trainable=True)
    
    # Mute dropouts
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            
    model.set_adapter("active_rl")
    model.train()
    
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    generation_config = GenerationConfig(max_new_tokens=1)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    device = next(model.parameters()).device

    accum_loss, accum_kl, accum_steps = 0.0, 0.0, 0
    
    for step in range(TOTAL_STEPS):
        logger.info(f"\nStep {step+1}/{TOTAL_STEPS}")
        
        batch_samples = [None] * BATCH_SIZE # Dummy dataset
        
        for i in range(BATCH_SIZE):
            trajectories, rewards, _, batch_metrics = rollout_batch_SYNTHETIC_TEST(
                model, tokenizer, generation_config, device, num_generations=NUM_GENERATIONS
            )

            if USE_WANDB:
                wandb.log({**batch_metrics, "train/grpo_step": step + 1}, step=step + 1)

            scaled_loss, mean_kl = grpo_loss(
                model, tokenizer, trajectories, rewards, device,
                batch_scale=BATCH_SIZE * GRADIENT_ACCUM
            )
            logger.info(f"  GRPO scaled loss: {scaled_loss:.4f}  KL: {mean_kl:.4f}")
            accum_loss += scaled_loss
            accum_kl += mean_kl

        accum_steps += 1
        if accum_steps % GRADIENT_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            logger.info("  [Optim] Updated")
            if USE_WANDB:
                wandb.log({"train/loss": accum_loss / GRADIENT_ACCUM, "train/kl": accum_kl / GRADIENT_ACCUM}, step=step + 1)
            accum_loss, accum_kl = 0.0, 0.0

if __name__ == "__main__":
    main()
