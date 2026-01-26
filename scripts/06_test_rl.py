import torch
import re
import os
import sys
from typing import List, Dict, Any, cast, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add project root to sys.path
sys.path.append(os.getcwd())

# Imports
from src.env.state import create_initial_state, GreenState, Document
from src.agent.prompts import format_state_for_prompt
from src.agent import actions
from src.data.hotpot import HotpotQAStreamer
from src.env.retriever import EphemeralRetriever
from src.env.engine_old import execute_action

# --- CONFIG ---
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
# KEY CHANGE: Point to the RL-trained model
ADAPTER_PATH = "models/green-rag-rl-v1" 
MAX_STEPS = 5

def load_director():
    print(f"Loading RL Director from {ADAPTER_PATH}...")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config, 
            device_map="auto"
        )
        
        # Load the RL Adapters
        # Note: If RL training saved a Value Head, PeftModel ignores it safely.
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load RL model: {e}")
        print("Tip: Did PPO training finish and save to 'models/green-rag-rl-v1'?")
        sys.exit(1)

def get_director_action(model, tokenizer, state: GreenState):
    prompt = format_state_for_prompt(cast(Dict[str, Any], state)) + " Action:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=32, # Reduced max tokens since we want brevity
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False 
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = full_output[len(prompt):]
    
    # Parser
    action_id = -1
    argument = ""
    
    act_match = re.search(r"(\d+)", new_tokens)
    if act_match:
        action_id = int(act_match.group(1))
        
    arg_match = re.search(r"Input:\s*(.*)", new_tokens, re.DOTALL)
    if arg_match:
        argument = arg_match.group(1).strip()
        
    return action_id, argument

def run_hotpot_episode(model, tokenizer, sample: dict):
    question = sample['question']
    corpus = sample['corpus']
    
    print(f"\n{'='*60}\nHOTPOT Q: {question}\n{'='*60}")
    
    state = create_initial_state(question)
    local_retriever = EphemeralRetriever(corpus)
    
    for step in range(MAX_STEPS):
        print(f"\n--- Step {step+1} ---")
        action_id, argument = get_director_action(model, tokenizer, state)
        obs, done = execute_action(state, action_id, argument, local_retriever)
        
        print(f"  >> Obs: {obs}")
        
        state['history'].append({
            "action_id": action_id, 
            "action_name": actions.get_action_name(action_id), 
            "observation": obs,
            "argument": argument,
            "cost": 0.0 # Placeholder for test
        })
        
        if done:
            print(f"\n*** DONE ***\nPrediction: {obs}")
            print(f"Ground Truth: {sample['answer']}")
            break

def main():
    model, tokenizer = load_director()
    streamer = HotpotQAStreamer(split="train", limit=5) # Use 'validation' for real testing
    for sample in streamer.stream():
        run_hotpot_episode(model, tokenizer, sample)

if __name__ == "__main__":
    main()
