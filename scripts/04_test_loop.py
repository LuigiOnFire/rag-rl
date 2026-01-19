import torch
import re
import random
from typing import List
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Add project root to sys.path
sys.path.append(os.getcwd())

# Imports
from src.env.state import create_initial_state, GreenState
from src.agent.prompts import format_state_for_prompt
from src.agent import actions, workers
from src.data.hotpot import HotpotQAStreamer

# --- CONFIG ---
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_PATH = "models/green-rag-sft-v1"
MAX_STEPS = 5

class EphemeralRetriever:
    """
    A tiny search engine created instantly for just ONE question.
    Searches only the 10 documents provided by HotpotQA.
    """
    def __init__(self, corpus: List[str]):
        self.corpus = corpus
        
    def search(self, query: str, top_k: int = 2):
        # Simple Logic: Score by word overlap (Jaccard-ish)
        # In production, use RankBM25, but this is fast and dependency-free.
        query_words = set(query.lower().split())
        scores = []
        
        for doc in self.corpus:
            doc_lower = doc.lower()
            # Score = count of query words present in doc
            score = sum(1 for w in query_words if w in doc_lower)
            scores.append((score, doc))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top_k docs (just the text)
        return [item[1] for item in scores[:top_k]]

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
    argument = "Ready to Answer"
    
    act_match = re.search(r"(\d+)", new_tokens)
    if act_match:
        action_id = int(act_match.group(1))
        
    arg_match = re.search(r"Input:\s*(.*)", new_tokens, re.DOTALL)
    if arg_match:
        argument = arg_match.group(1).strip()
        
    return action_id, argument

def execute_action(state: GreenState, action_id: int, argument: str, retriever: EphemeralRetriever):
    """
    STRICT Game Engine: The Director points, the Workers shoot.
    """
    obs = ""
    done = False
    action_name = actions.get_action_name(action_id)
    
    print(f"  >> Executing: [{action_id}] {action_name}")
    print(f"  >> Query/Arg: {argument}")

    # --- 1. ANSWERING (GEN_SLM / GEN_LLM) ---
    if action_id in [0, 1]:
        print("  [>] Calling Worker for Answer Generation...")
        use_llm = (action_id == 1)
        obs = workers.generate_answer(state, use_llm=use_llm)
        done = True 

    # --- 2. RETRIEVAL (RET_KEY / RET_VEC) ---
    elif action_id in [2, 3]:
        # For Search, we DO trust the argument because the Director defines the query.
        clean_query = argument.strip()
        
        # Safety: If query is empty/noise, fall back to the main question
        if len(clean_query) < 5:
            clean_query = state['subqueries'][-1]['question']
            
        results = retriever.search(clean_query, top_k=2)
        obs = f"Found {len(results)} docs."

        # Get more meaningful logging on what the logs contain
        print("  [>] Retrieved Documents:")
        for i, doc in enumerate(results):
            print(f"    Doc {i+1}: {doc[:100]}...")  # Print first 100 chars
        
        formatted_docs = [{"title": "Doc", "content": r, "relevance": "UNKNOWN"} for r in results]
        if state['subqueries']:
            state['subqueries'][-1]['documents'].extend(formatted_docs)

    # --- 3. GRADING (GRD_SLM / GRD_LLM) ---
    elif action_id in [4, 5]:
        if state['subqueries'] and state['subqueries'][-1]['documents']:
            last_doc = state['subqueries'][-1]['documents'][-1]
            doc_content = last_doc['content']
            
            use_llm = (action_id == 5)
            grade = workers.generate_grade(state, doc_content, use_llm=use_llm)
            
            last_doc['relevance'] = grade.upper()
            obs = f"Document marked {grade.upper()}"
        else:
            obs = "No documents to grade."

    # --- 4. REWRITE (RWT_SLM) ---
    elif action_id == 6:
        if state['subqueries']:
            old_q = state['subqueries'][-1]['question']
            state['subqueries'][-1]['question'] = argument 
            obs = f"Query updated: '{old_q}' -> '{argument}'"
        else:
            obs = "No active query to rewrite."

    # --- 5. DECOMPOSITION (DEC_SLM / DEC_LLM) ---
    elif action_id in [7, 8]:
        new_subq = {
            "question": argument,
            "status": "ACTIVE",
            "documents": [],
            "answer": None
        }
        state['subqueries'].append(new_subq)
        obs = f"Plan updated. New focus: {argument}"

    # --- 6. FAIL ---
    elif action_id == 9:
        obs = "Agent declared failure."
        done = True

    else:
        obs = f"Error: Unimplemented Action ID {action_id}"

    return obs, done

def run_hotpot_episode(model, tokenizer, sample: dict):
    question = sample['question']
    corpus = sample['corpus']
    
    print(f"\n{'='*60}\nHOTPOT Q: {question}\n{'='*60}")
    
    # 1. Init State & Tiny Search Engine
    state = create_initial_state(question)
    local_retriever = EphemeralRetriever(corpus)
    
    for step in range(MAX_STEPS):
        print(f"\n--- Step {step+1} ---")
        
        # Decide
        action_id, argument = get_director_action(model, tokenizer, state)
        
        # Act (Pass the local retriever!)
        obs, done = execute_action(state, action_id, argument, local_retriever)
        
        print(f"  >> Obs: {obs}")
        
        state['history'].append({
            "action_id": action_id, 
            "action_name": actions.get_action_name(action_id),
            "observation": obs
        })
        
        if done:
            print(f"\n*** DONE ***\nPrediction: {obs}")
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