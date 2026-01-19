import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead, create_reference_model, PPOConfig
from src.data.hotpot import HotpotQAStreamer
from src.env.state import create_initial_state
from src.agent.prompts import format_state_for_prompt
from src.rl.rewards import calculate_reward
from src.env.engine import execute_action
from src.env.retriever import EphemeralRetriever
import re
import math
from typing import List, Dict

# --- CONFIG ---
SFT_MODEL_PATH = "models/green-rag-sft-v1"
OUTPUT_DIR = "models/green-rag-rl-v1"

# --- CUSTOM PPO IMPLEMENTATION ---
class PPOAgentTrainer:
    """
    Custom PPO Trainer that implements the PPO algorithm manually to support
    Interaction-based environments (RAG) which the standard trl.PPOTrainer
    (dataset-based) does not easily support in v0.27+.
    """
    def __init__(self, config: PPOConfig, model, ref_model, tokenizer):
        self.config = config
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        
    def log_probs_from_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute log probs of the labels given the logits.
        """
        logp = F.log_softmax(logits, dim=-1)
        # Gather the log prob of the actual token chosen
        # labels shape: [Batch, Seq]
        # logp shape: [Batch, Seq, Vocab]
        logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
        return logpy

    def generate(self, query_tensor, **kwargs) -> torch.Tensor:
        """
        Generate response using the policy model.
        """
        # Ensure model is in eval for generation (though we act on policy)
        # But we need gradients later. For generation, no grad.
        with torch.no_grad():
            response = self.model.generate(query_tensor, **kwargs)
        return response

    def step(self, query: torch.Tensor, response: torch.Tensor, score: float):
        """
        Perform one PPO optimization step based on the collected experience.
        Args:
            query: Input IDs (Prompt) [1, Seq]
            response: Output IDs (Response) [1, Seq] (Full sequence including query in some formats, check generate)
            score: Scalar reward for this interaction
        """
        # 1. Formatting
        # AutoModel generate usually returns Cat(Query, Response)
        # We process the "Response" part mostly.
        
        # Assume response contains Query+Gen. Extract Gen.
        q_len = query.shape[1]
        gen_tokens = response[:, q_len:]
        full_seq = response
        
        # 2. Forward Pass (Old Policy / Data Collection)
        # We need gradients for the optimization, but these serve as "Old" stats for Ratio.
        # Actually, standard PPO does:
        #   Rollout -> Collect Trajectory (States, Actions, LogProbs_Old, Rewards) -> Optimize
        # Here we just did rollout. We assume 'response' is the action.
        
        with torch.no_grad():
            outputs = self.model(full_seq, output_hidden_states=True)
            ref_outputs = self.ref_model(full_seq, output_hidden_states=True)
            
            # Logic for ValueHead (Assuming (logits, loss, value) or (logits, _, value))
            # Inspect output type could be needed, but assume standard tuple unpacked by TRL wrapper
            # If standard AutoModel, it returns CausalLMOutputWithPast
            # The Wrapper usually overrides forward.
            
            # wrapper forward: return logits, _, value
            logits = outputs[0]
            values = outputs[-1] # Value head is usually last
            ref_logits = ref_outputs[0]

        # LogProbs
        # We only care about generation tokens for PPO Loss
        # Shift logits for next-token prediction
        # The logprob of token[i] is logits[i-1]
        
        # Slice for generation
        # gen_tokens = full_seq[:, q_len:]
        # corresponding logits are at full_seq indices [q_len-1 : -1]
        
        gen_logits = logits[:, q_len-1 : -1, :]
        ref_gen_logits = ref_logits[:, q_len-1 : -1, :]
        gen_values = values[:, q_len-1 : -1].squeeze(-1) # Value per token
        
        gen_logprobs = self.log_probs_from_logits(gen_logits, gen_tokens)
        ref_logprobs = self.log_probs_from_logits(ref_gen_logits, gen_tokens)
        
        # 3. Compute Rewards (Token level)
        # KL Penalty per token
        # KL approx = (log_p - log_ref)
        # We want to penalized deviation.
        
        non_score_reward = -self.config.init_kl_coef * (gen_logprobs - ref_logprobs)
        
        # Combining with valid scalar Score
        # We apply score to the LAST token usually
        rewards = non_score_reward.clone()
        rewards[:, -1] += score
        
        # 4. Advantages & Returns
        # A = R + gamma*V_next - V
        # Here simplified: Returns = Sum of future rewards
        # We use the generated values as baseline.
        # Advantage = Returns - Values
        
        # Monte Carlo Compute Returns (Backwards)
        returns = torch.zeros_like(rewards)
        running_return = 0
        gamma = self.config.gamma
        for t in reversed(range(rewards.shape[1])):
            running_return = rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            
        advantages = returns - gen_values
        
        # Normalize Advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 5. PPO Optimization Loop
        # Train on this batch multiple times (epochs)
        for _ in range(self.config.ppo_epochs):
            # Forward New Policy
            new_outputs = self.model(full_seq, output_hidden_states=True)
            new_logits = new_outputs[0]
            new_values = new_outputs[-1].squeeze(-1)
            
            new_gen_logits = new_logits[:, q_len-1 : -1, :]
            new_gen_values = new_values[:, q_len-1 : -1]
            new_gen_logprobs = self.log_probs_from_logits(new_gen_logits, gen_tokens)
            
            # Ratio
            ratio = torch.exp(new_gen_logprobs - gen_logprobs)
            
            # Policy Loss
            pg_losses = -advantages * ratio
            pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.config.cliprange, 1.0 + self.config.cliprange)
            pg_loss = torch.max(pg_losses, pg_losses2).mean()
            
            # Value Loss
            v_loss = F.mse_loss(new_gen_values, returns)
            
            # Total Loss
            loss = pg_loss + 0.5 * v_loss 
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return {"loss": loss.item(), "reward": score}

def parse_action(text):
    # Quick parser
    act_match = re.search(r"(\d+)", text)
    action_id = int(act_match.group(1)) if act_match else -1
    return action_id

def main():
    print("Initializing Models...")
    try:
        # Load Config & Models
        # PPOConfig requires no model_name arg in v0.27
        config = PPOConfig(
            learning_rate=1.41e-5,
            batch_size=1, # simplified
        )
        
        # Load Models
        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            SFT_MODEL_PATH,
            device_map="auto",
            load_in_4bit=True
        )
        
        # Reference Model (Frozen copy)
        ref_model = create_reference_model(model)
        
        tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL_PATH)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Trainer
        trainer = PPOAgentTrainer(config, model, ref_model, tokenizer)
        
        print("Starting Training Loop...")
        streamer = HotpotQAStreamer(split="train", limit=50)
        
        metrics = []
        
        for sample_idx, sample in enumerate(streamer.stream()):
            question = sample['question']
            ground_truth = sample['answer']
            corpus = sample['corpus']
            
            # Reset Env
            retriever = EphemeralRetriever(corpus)
            state = create_initial_state(question)
            
            # For this simple loop, we do ONE interaction Step per sample
            # (Ideally Multi-Step Loop)
            
            # Step 1: Prompt
            query_txt = format_state_for_prompt(state) + " Action:"
            query_tensor = tokenizer.encode(query_txt, return_tensors="pt").to(model.device)
            
            # Step 2: Generate
            generation_kwargs = {
               "min_length": -1,
               "top_k": 0.0,
               "top_p": 1.0,
               "do_sample": True,
               "pad_token_id": tokenizer.eos_token_id,
               "max_new_tokens": 40,
            }
            response_tensor = trainer.generate(query_tensor, **generation_kwargs)
            
            # Decode full response to get text
            # Step 3: Env Execute
            # Extract just the new text for Environment input
            response_txt_only = tokenizer.decode(response_tensor[0][query_tensor.shape[1]:])
            
            action_id = parse_action(response_txt_only)
            # Find argument
            arg_match = re.search(r"Input:\s*(.*)", response_txt_only, re.DOTALL)
            argument = arg_match.group(1).strip() if arg_match else ""
            
            obs, done = execute_action(state, action_id, argument, retriever)
            
            # Step 4: Reward
            # We calculate reward for this transition.
            reward_scalar, breakdown = calculate_reward(
                state, response_txt_only, ground_truth, action_id, done, obs
            )
            
            # Step 5: Optimization
            stats = trainer.step(query_tensor, response_tensor, reward_scalar)
            
            print(f"Sample {sample_idx} | Action: {action_id} | Reward: {reward_scalar:.2f} | Loss: {stats['loss']:.4f}")
            
            # Save occasionally
            if sample_idx > 0 and sample_idx % 10 == 0:
                print("Saving checkpoint...")
                model.save_pretrained(OUTPUT_DIR)
                tokenizer.save_pretrained(OUTPUT_DIR)

        # Final Save
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Training Complete.")
        
    except Exception as e:
        print(f"Critical Failure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()