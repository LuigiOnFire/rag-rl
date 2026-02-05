import torch
import re
import random
from typing import List
import os
import sys
import time
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add project root to sys.path
sys.path.append(os.getcwd())

# Imports
from src.env.state import create_initial_state, GreenState
from src.agent.prompts import format_state_for_prompt
from src.agent import actions
from src.data.hotpot import HotpotQAStreamer
from src.env.retriever import EphemeralRetriever
from src.env.engine import GreenEngine

# --- CONFIG ---
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "models/green-rag-sft-v1"
MAX_STEPS = 5

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


run_id = time.strftime("%Y%m%d_%H%M%S")
RUN_FILE = f"data/post_sft_tests/run_{run_id}.log"
os.makedirs(os.path.dirname(RUN_FILE), exist_ok=True)
logger.debug(f"Run File: {RUN_FILE}")


def load_director():
    print(f"Loading 1B Director from {ADAPTER_PATH}...")
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
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer

def get_director_action(model, tokenizer, state: GreenState):    
    # Same logic as before
    prompt = format_state_for_prompt(state) + " Action:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False 
        )
    
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = full_output[len(prompt):]
    
    # Simple Parser
    action_id = -1
    
    act_match = re.search(r"(\d+)", new_tokens)
    if act_match:
        action_id = int(act_match.group(1))

    with open(RUN_FILE, "a") as f:
        f.write(f"\n--- PROMPT FOR MODEL ---\n{prompt}\n")
        f.write(f"\n--- MODEL OUTPUT ---\n{new_tokens}\n")
        f.write(f"\n--- PARSED ACTION ID: {action_id} ---\n")
                
    return action_id

def run_hotpot_episode(model, tokenizer, sample: dict):
    question = sample['question']
    corpus = sample['corpus']
    
    print(f"\n{'='*60}\nHOTPOT Q: {question}\n{'='*60}")
    
    # 1. Init State & Tiny Search Engine
    state = create_initial_state(question)
    local_retriever = EphemeralRetriever(corpus)

    engine = GreenEngine(retriever=local_retriever)

    with open(RUN_FILE, "a") as f:
        f.write(f"\n\n=== NEW EPISODE ===\nQuestion: {question}\n")
    
    for step in range(MAX_STEPS):
        print(f"\n--- Step {step+1} ---")
        
        # Decide
        action_id = get_director_action(model, tokenizer, state)
        
        # Act (Pass the local retriever!)
        new_state = engine.step(state, action_id)
        state = new_state
                        
        if new_state['status'] == "SOLVED":
            print(f"\n*** DONE ***\nPrediction: {new_state.get('answer')}")
            print(f"Ground Truth: {sample['answer']}")
            break

def main():
    model, tokenizer = load_director()
    
    # Stream 5 random Hotpot questions
    streamer = HotpotQAStreamer(split="train", limit=5)
    
    for sample in streamer.stream():
        run_hotpot_episode(model, tokenizer, sample)

if __name__ == "__main__":
    main()