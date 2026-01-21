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
from src.agent import actions, workers
from src.data.hotpot import HotpotQAStreamer

# --- CONFIG ---
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
# KEY CHANGE: Point to the RL-trained model
ADAPTER_PATH = "models/green-rag-rl-v1" 
MAX_STEPS = 5

class EphemeralRetriever:
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        
    def search(self, query: str, top_k: int = 2):
        query_words = set(query.lower().split())
        scores = []
        for doc in self.corpus:
            doc_lower = doc.lower()
            score = sum(1 for w in query_words if w in doc_lower)
            scores.append((score, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scores[:top_k]]

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

def execute_action(state: GreenState, action_id: int, argument: str, retriever: EphemeralRetriever):
    obs = ""
    done = False
    action_name = actions.get_action_name(action_id)
    
    print(f"  >> Executing: [{action_id}] {action_name}")
    
    # 1. ANSWER (Strict Mode)
    if action_id in [0, 1]:
        print("  [>] Calling Worker for Answer Generation...")
        use_llm = (action_id == 1)
        obs = workers.generate_answer(state, use_llm=use_llm)
        done = True 

    # 2. SEARCH
    elif action_id in [2, 3]:
        clean_query = argument.strip()
        # Fallback if RL model learned to be TOO silent
        if len(clean_query) < 3: 
            clean_query = state['subqueries'][-1]['question']
            
        print(f"  [>] Searching for: '{clean_query}'")
        results = retriever.search(clean_query, top_k=2)
        obs = f"Found {len(results)} docs."
        
        # Log retrieved docs for debugging
        for i, doc in enumerate(results):
             print(f"    Doc {i}: {doc[:80]}...")

        formatted_docs: List[Document] = [
            cast(Document, {"title": "Doc", "content": r, "relevance": "UNKNOWN"}) 
            for r in results
        ]
        if state['subqueries']:
            state['subqueries'][-1]['documents'].extend(formatted_docs)

    # 3. GRADE
    elif action_id in [4, 5]:
        if state['subqueries'] and state['subqueries'][-1]['documents']:
            last_doc = state['subqueries'][-1]['documents'][-1]
            grade = workers.generate_grade(state, last_doc['content'], use_llm=(action_id==5))
            
            # Safe cast for literal
            grade_upper = grade.upper()
            if grade_upper in ["RELEVANT", "IRRELEVANT", "UNKNOWN"]:
                 last_doc['relevance'] = cast(Literal["UNKNOWN", "RELEVANT", "IRRELEVANT"], grade_upper)
            else:
                 last_doc['relevance'] = "UNKNOWN"
                 
            obs = f"Document marked {grade.upper()}"
        else:
            obs = "No documents to grade."

    # 4. REWRITE / DECOMPOSE / FAIL
    elif action_id == 6:
        if state['subqueries']:
            state['subqueries'][-1]['question'] = argument 
            obs = f"Query updated: {argument}"
        else:
            obs = "No query."
    elif action_id in [7, 8]:
        new_id = str(len(state['subqueries']) + 1)
        state['subqueries'].append({
            "id": new_id,
            "question": argument, 
            "status": "ACTIVE", 
            "documents": [], 
            "answer": None
        })
        obs = f"Plan updated: {argument}"
    elif action_id == 9:
        obs = "Failure declared."
        done = True
    else:
        obs = f"Error: Unknown Action {action_id}"

    return obs, done

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
