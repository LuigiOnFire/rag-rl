import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import GenerationConfig
from trl.trainer.ppo_config import PPOConfig
from trl.models.modeling_value_head import AutoModelForCausalLMWithValueHead
from trl.models.utils import create_reference_model
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import sys
import re
import time
import logging
from typing import Dict, Any, cast
import argparse

sys.path.append(os.getcwd())

from src.data.hotpot import HotpotQAStreamer
from src.env.gym_env import GreenRAGEnv

# --- CONFIG ---
SFT_MODEL_PATH = "models/green-rag-sft-v1"
OUTPUT_DIR = "models/green-rag-rl-v1"
MAX_STEPS = 5  # Max turns per episode
EPISODE_COUNT = 4000 # Total episodes to train

run_id = time.strftime("%Y%m%d_%H%M%S")
RUN_FILE = f"data/ppo_training/run_{run_id}.log"
os.makedirs(os.path.dirname(RUN_FILE), exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,       # Set to logging.DEBUG for more verbosity
    format='%(message)s',     # Just print the message (no time, no level name)
    handlers=[
        logging.FileHandler(RUN_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)

# 2. Create a logger object
logger = logging.getLogger(__name__)

class PPOAgentTrainer:
    def __init__(self, config: PPOConfig, model, ref_model, tokenizer):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer

        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        print(f"  [Optimizer] Tracking {len(trainable_params)} tensors for optimization.")
    
        if len(trainable_params) == 0:
            raise ValueError("No trainable parameters found! Check PEFT/LoRA config.")
        
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.learning_rate)

        self.batch_counter = 0
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        
    def log_probs_from_logits(self, logits, labels):
        logp = F.log_softmax(logits, dim=-1)
        logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logpy

    def generate(self, query_tensor, **kwargs):
        # Generate in eval mode to avoid caching gradients for the rollout
        with torch.no_grad():
            response = self.model.generate(
                query_tensor, 
                tokenizer=self.tokenizer, # <--- ADD THIS ARGUMENT
                **kwargs
            )
        return response

    def step(self, query, response, score):
        device = query.device

        self.batch_counter += 1

        # 1. Setup
        q_len = query.shape[1]
        full_seq = response
        gen_tokens = response[:, q_len:]
        
        if gen_tokens.shape[1] == 0:
            return {"loss": 0.0, "reward": score}
        
        # Putting this here due to devide mismatch errors        
        attention_mask = (full_seq != self.tokenizer.pad_token_id).long().to(device)

        # 2. Forward Pass (Get Old LogProbs & Values)
        # We assume 'model' currently holds the policy we just sampled from.
        # In standard PPO, we'd detach these, but here we do single-batch updates immediately.
        with torch.no_grad():
            outputs = self.model(full_seq, attention_mask=attention_mask, output_hidden_states=True)
            ref_outputs = self.ref_model(full_seq, attention_mask=attention_mask, output_hidden_states=True)
            
            logits = outputs[0]
            values = outputs[-1]
            ref_logits = ref_outputs[0]

        # Slicing: Align logits [0...N-1] with tokens [1...N]
        # We want to predict gen_tokens. 
        # The logit that predicts gen_tokens[0] is at index q_len-1.
        gen_logits = logits[:, q_len-1 : -1, :]
        ref_gen_logits = ref_logits[:, q_len-1 : -1, :]
        gen_values = values[:, q_len-1 : -1].squeeze(-1)

        # Force reference model to gpu
        if ref_gen_logits.device != device:
            ref_gen_logits = ref_gen_logits.to(device)
        
        gen_logprobs = self.log_probs_from_logits(gen_logits, gen_tokens)
        ref_logprobs = self.log_probs_from_logits(ref_gen_logits, gen_tokens)
        
        # 3. Rewards & Advantages
        rewards = torch.zeros_like(gen_logprobs)
        # KL Penalty
        kl = gen_logprobs - ref_logprobs
        rewards = -self.config.kl_coef * kl
        # Add Task Reward to LAST token
        rewards[:, -1] += score
        
        # GAE / Returns (Simplified Monte Carlo)
        # return_t = reward_t + gamma * return_{t+1}
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(rewards.shape[1])):
            running_return = rewards[:, t] + self.config.gamma * running_return
            returns[:, t] = running_return
        
        # force value head to gpu
        if gen_values.device != device:
            gen_values = gen_values.to(device)
            
        advantages = returns - gen_values
        # Normalize
        if len(advantages.view(-1)) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        # 4. Optimization Epochs
        loss_sum = 0
        for _ in range(self.config.num_ppo_epochs):
            # Recalculate logits with Gradients enabled
            new_outputs = self.model(full_seq, attention_mask=attention_mask, output_hidden_states=True)
            new_logits = new_outputs[0]
            new_values = new_outputs[-1].squeeze(-1)
            
            new_gen_logits = new_logits[:, q_len-1 : -1, :]

            # The new values are likely on CPU. Force them to GPU
            if new_values.device != device:
                new_values = new_values.to(device)
            
            new_gen_values = new_values[:, q_len-1 : -1]
            
            new_gen_logprobs = self.log_probs_from_logits(new_gen_logits, gen_tokens)
            
            ratio = torch.exp(new_gen_logprobs - gen_logprobs)
            
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
            pg_loss = torch.max(pg_losses, pg_losses2).mean()
            
            v_loss = F.mse_loss(new_gen_values, returns)
            
            loss = pg_loss + 0.5 * v_loss
            
            self.optimizer.zero_grad()

            loss = loss / self.gradient_accumulation_steps
            
            loss.backward()
            
            if self.batch_counter % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Safety clip
                self.optimizer.step()
                self.optimizer.zero_grad() # <--- Clear the pile AFTER update

                loss_sum += loss.item()
            
        return {"loss": loss_sum / self.config.num_ppo_epochs, "reward": score}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_checkpoint", type=str, default=None, help="Path to the checkpoint folder to resume from")
    args = parser.parse_args()
    print("Initializing RL Phase...")
    
    # 1. Load Models & Config
    # We used AutoModelForCausalLMWithValueHead to wrap our SFT model
    base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        SFT_MODEL_PATH, device_map="auto", load_in_4bit=True
    )
    # Enable gradients for LoRA adapters in 4-bit mode
    base_model = prepare_model_for_kbit_training(base_model)
    peft_model = base_model.pretrained_model
    for name, param in peft_model.named_parameters():
        if "lora" in name: param.requires_grad = True
    peft_model.enable_input_require_grads()
    
    # Create Reference Model (Frozen copy for KL penalty)
    ref_model = create_reference_model(base_model)
    ref_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = PPOConfig(
        learning_rate=1.41e-5, 
        batch_size=1, 
        mini_batch_size=1, 
        gradient_accumulation_steps=8,
        kl_coef=0.05,
        temperature=1.0,
        gamma=0.99,
    )
    
    # Optimizer Check
    trainable_params = list(filter(lambda p: p.requires_grad, base_model.parameters()))
    if len(trainable_params) == 0: raise ValueError("No trainable params!")
    
    trainer = PPOAgentTrainer(config, base_model, ref_model, tokenizer)
    
    print("Starting Training Loop (Gymnasium Rollouts)...")
    
    # 2. Initialize Gymnasium Environment
    # The environment handles the dataset streaming internally
    streamer = HotpotQAStreamer(split="train", limit=None)
    env = GreenRAGEnv(streamer)
    
    GAMMA = 0.99  # Discount factor for future rewards

    # Episode Loop
    for episode in range(EPISODE_COUNT):
        # Reset Env
        obs_text, info = env.reset()
        if obs_text is None:
            print("dataset exhausted")
            break
            
        episode_buffer = [] # Stores (query, response, immediate_reward)
        
        for step_i in range(MAX_STEPS):
            # A. Prompt (Observation)
            query_txt = obs_text + " Action:"
            device = next(base_model.parameters()).device

            decay_span = int(EPISODE_COUNT * 0.25)
            force_probability = max(0.05, 1.0 - (episode / decay_span))
            
            forced_action = ""
            if step_i == 0: 
                 import random
                 if random.random() < force_probability:
                     forced_action = " 2" 
                     logger.info(f"FORCE: Pushing Agent to Search (Step 0) | Prob: {force_probability:.2f}")
                     
            # If we are forcing, we append it to the INPUT QUERY so the model completes it
            if forced_action:
                 query_txt += forced_action
                 
            query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(device)
            attention_mask = (query_tensor != tokenizer.pad_token_id).long()
            
            # B. Generate Action
            generation_config = GenerationConfig(                 
                do_sample=True,
                top_p=0.9,
                temperature=1.0, 
                repetition_penalty=1.2,
                max_new_tokens=40,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                stop_strings=["\n", "<|eot_id|>"] # we just need the digit for action
            )
            response_tensor = trainer.generate(query_tensor, generation_config=generation_config)
            
            # C. Decode Response
            # This creates 'response_txt_only' for the first time
            response_txt_only = tokenizer.decode(response_tensor[0][query_tensor.shape[1]:])
            
            # --- PART B: STITCHING (AFTER GENERATION) ---
            if forced_action:
                 # The model generated "Input: xyz". We need to attach " Action: 2" to the front.
                 
                 # Safety: Ensure there is a newline separation
                 if not response_txt_only.startswith("\n") and not response_txt_only.startswith(" "):
                     response_txt_only = "\n" + response_txt_only
                     
                 full_action = f" Action:{forced_action}{response_txt_only}"
                 response_txt_only = full_action
            # --------------------------------------------
            
            # Env.step handles: Parsing -> API Calls -> State Update -> Reward Calculation
            next_obs, reward, terminated, truncated, info = env.step(response_txt_only)
            
            # Env.step handles: Parsing -> API Calls -> State Update -> Reward Calculation
            next_obs, reward, terminated, truncated, info = env.step(response_txt_only)
            
            logger.info(f"\n--- Ep {episode} | Step {step_i} ---")
            logger.info(f"OBSERVATION (Prompt end): ...{query_txt[-200:]}")
            logger.info(f"ACTION GENERATED: {response_txt_only.strip()}")
            logger.info(f"REWARD: {reward}")
            if info:
                 logger.info(f"INFO: {info}")
            if next_obs:
                 logger.info(f"NEXT STATE (Preview): {next_obs[:100]}...")

            # Store in buffer
            episode_buffer.append({
                "query": query_tensor,
                "response": response_tensor,
                "reward": reward
            })
            
            print(f"  Ep {episode}|St {step_i} | Reward: {reward:.2f}")
            
            if terminated or truncated:
                logger.info(">>> TERMINATED")
                break
            
            obs_text = next_obs
        
        # --- ADD THIS BLOCK ---
        # "Did Not Finish" (DNF) Check
        if not terminated:
             # If the loop finished but the agent never answered (Action 0/1),
             # it means it wasted all its turns searching or looping.
             # We apply a massive penalty to the LAST action it took.
             
             DNF_PENALTY = -1.5  # Make this painful (worse than answering wrong)
             
             if len(episode_buffer) > 0:
                 episode_buffer[-1]['reward'] += DNF_PENALTY
                 print(f"  ⚠️ DNF Penalty Applied! Agent timed out.")
        
        total_episode_reward = sum(step['reward'] for step in episode_buffer)
        print(f"🏁 Episode {episode} Final Score: {total_episode_reward:.2f}")
            
        # --- 3. CREDIT ASSIGNMENT (Backpropagation of Rewards) ---
        # Calculate Discounted Return (G_t) backwards
        # G_t = r_t + gamma * G_{t+1}
        cumulative_return = 0.0
        
        # Iterate backwards to push the final success reward to the earlier steps
        for step_data in reversed(episode_buffer):
            cumulative_return = step_data["reward"] + GAMMA * cumulative_return
            step_data["discounted_return"] = cumulative_return
            
        # --- 4. OPTIMIZATION PHASE (Update Model) ---
        print(f"  >> Updating Model on {len(episode_buffer)} steps...")
        
        total_loss = 0
        for step_data in episode_buffer:
            stats = trainer.step(
                step_data["query"], 
                step_data["response"], 
                step_data["discounted_return"] # Use the FUTURE-AWARE score
            )
            total_loss += stats['loss']
            
        print(f"Episode {episode} Complete | Avg Loss: {total_loss/len(episode_buffer):.4f}")

        # Saving Logic
        if episode > 0 and episode % 50 == 0:
            if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR, exist_ok=True)
            print("Saving checkpoint...")
            base_model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)

    # Final Save
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()