import torch
import re
import json
import os
import logging
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add project root to sys.path
sys.path.append(os.getcwd())

# Import your existing architecture
from src.env.state import create_initial_state, GreenState
from src.agent.prompts import format_state_for_prompt
from src.agent import actions, workers
from src.env import retriever

# --- CONFIGURATION ---
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "models/green-rag-sft-v1" # Point to your best checkpoint if needed
MAX_STEPS = 5

def load_director():
    print(f"Loading Director (1B Student) from {ADAPTER_PATH}...")
    # Load in 4-bit to match training environment
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
    
    # Load the Adapter
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer

def get_director_action(model, tokenizer, state: GreenState):
    """
    1. Formats the prompt.
    2. Generates the response.
    3. Parses 'Action: X' and 'Input: Y'.
    """
    prompt = format_state_for_prompt(state)
    
    # Force the start of the generation to ensure compliance
    # We append " Action:" so the model just has to complete the ID.
    prompt_input = prompt + " Action:"
    
    inputs = tokenizer(prompt_input, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64, # Keep it short, we just need the command
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False # Greedy decoding for deterministic testing
        )
        
    # Decode ONLY the new tokens
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = full_output[len(prompt):] # Extract what the model wrote
    
    return parse_response(new_tokens)

def parse_response(text: str):
    """
    Extracts Action ID and Argument from:
    " Action: 2\nInput: capital of France"
    """
    # Defaults
    action_id = -1
    argument = "Error parsing output"
    
    # Regex for robustness
    # Looks for "Action: <digits>"
    act_match = re.search(r"Action:\s*(\d+)", text)
    if act_match:
        action_id = int(act_match.group(1))
        
    # Looks for "Input: <rest of line>"
    arg_match = re.search(r"Input:\s*(.*)", text, re.DOTALL)
    if arg_match:
        argument = arg_match.group(1).strip()
    
    return action_id, argument, text.strip()

def execute_action(state: GreenState, action_id: int, argument: str):
    """
    THE DISPATCHER: Routes commands to your actual architecture.
    """
    obs = ""
    done = False
    
    action_name = actions.get_action_name(action_id)
    print(f"  >> Executing: [{action_id}] {action_name} | Arg: {argument}")
    
    try:
        # --- ACTION 0: ANSWER (SLM) ---
        if action_id == 0:
            # In training, we mapped this to "Ready to Answer"
            # Now we actually generate the answer using the SLM Worker
            # The worker typically needs the STATE to see the gathered docs.
            obs = workers.generate_answer(state, use_llm=False)
            done = True # We assume answering ends the episode
            
        # --- ACTION 1: ANSWER (LLM) ---
        elif action_id == 1:
            obs = workers.generate_answer(state, use_llm=True)
            done = True
            
        # --- ACTION 2: KEYWORD SEARCH ---
        elif action_id == 2:
            # Argument is the search query
            results = retriever.search_bm25(argument, top_k=3)
            obs = f"Found {len(results)} docs."
            # Append docs to the current active subquery in the state
            # (Assuming you have a helper for this, simplified here)
            if state['subqueries']:
                state['subqueries'][-1]['documents'].extend(results)
        
        # --- ACTION 3: VECTOR SEARCH ---
        elif action_id == 3:
            results = retriever.search_vector(argument, top_k=3)
            obs = f"Found {len(results)} docs."
            if state['subqueries']:
                state['subqueries'][-1]['documents'].extend(results)
                
        # --- FALLBACK ---
        else:
            obs = "Unknown Action ID"
            
    except Exception as e:
        obs = f"Execution Error: {str(e)}"

    return obs, done

def run_episode(model, tokenizer, question: str):
    print(f"\n{'='*60}\nNEW GOAL: {question}\n{'='*60}")
    
    # 1. Initialize
    state = create_initial_state(question)
    
    for step in range(MAX_STEPS):
        print(f"\n--- Step {step+1} ---")
        
        # 2. DECIDE (The Director)
        action_id, argument, raw_text = get_director_action(model, tokenizer, state)
        
        # Debug print what the model actually wrote
        # print(f"  (Model Output): {raw_text}")
        
        if action_id == -1:
            print("  [!] Model failed to produce valid Action ID.")
            break
            
        # 3. ACT (The Workers)
        observation, done = execute_action(state, action_id, argument)
        
        print(f"  >> Observation: {observation}")
        
        # 4. UPDATE STATE
        # Record the history so the model sees it next time
        state['history'].append({
            "action_id": action_id,
            "action_name": actions.get_action_name(action_id),
            "observation": observation
        })
        
        if done:
            print(f"\n*** MISSION COMPLETE ***\nFinal Answer: {observation}")
            break
            
    return state

def main():
    model, tokenizer = load_director()
    
    questions = [
        "What is the capital of France?",
        "Who is the CEO of Tesla?",
        # A harder multi-step one
        "Which magazine was started first Arthur's Magazine or First for Women?"
    ]
    
    for q in questions:
        run_episode(model, tokenizer, q)

if __name__ == "__main__":
    main()