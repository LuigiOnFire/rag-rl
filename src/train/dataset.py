from typing import Any, Dict, List
import json
from datasets import Dataset 
from src.agent import actions

def create_green_dataset(jsonl_paths: List[str], tokenizer: Any) -> Dataset:
    """
    Loads trajectories from JSONL and constructs a Hugging Face Dataset ready for SFTTrainer.
    """
    samples = []
    
    # 1. Load and Flatten
    for path in jsonl_paths:
        with open(path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    steps = record.get("steps", [])
                    
                    if not steps:
                        continue
                        
                    for step in steps:
                        # Pre-process content here
                        pre_state = step["pre_state"]
                        
                        # Double Sanitize
                        if "ground_truth" in pre_state:
                            del pre_state["ground_truth"]
                            
                        # Format Text
                        prompt_text = format_prompt(pre_state)
                        completion_text = format_completion(step["action_id"], step["argument"])
                        full_text = prompt_text + completion_text + tokenizer.eos_token
                        
                        samples.append({
                            "text": full_text
                        })
                except json.JSONDecodeError:
                    continue
                    
    print(f"Loaded {len(samples)} training samples from {len(jsonl_paths)} files.")
    
    if not samples:
         # Fallback for empty runs to prevent crash
        print("Warning: No samples found. Creating dummy dataset.")
        return Dataset.from_list([{"text": "Empty"}])
    
    # Return standard HF Dataset
    return Dataset.from_list(samples)

def format_prompt(state: Dict[str, Any]) -> str:
    """
    Convert State Dict to a readable text representation.
    """
    # 1. Header
    out = f"Goal: {state.get('main_query', 'Unknown')}\n"
    out += f"Status: {state.get('status', 'SOLVING')}\n"
    out += f"Scratchpad: {state.get('scratchpad', '')}\n\n"
    
    # 2. History
    out += "History:\n"
    history = state.get('history', [])
    if not history:
        out += "(None)\n"
    else:
        for i, item in enumerate(history):
            # Handle lightweight history items
            act = item.get('action_name', 'UNKNOWN')
            obs = item.get('observation', '')
            out += f"{i+1}. {act} -> {obs}\n"
    
    # 3. Subqueries
    out += "\nSub-Tasks:\n"
    subs = state.get('subqueries', [])
    for sub in subs:
        status = sub.get('status', 'PENDING')
        q = sub.get('question', '')
        ans = sub.get('answer')
        docs = len(sub.get('documents', []))
        
        line = f"[{status}] {q}"
        if ans:
            line += f" (Ans: {ans})"
        if docs > 0:
            line += f" [Docs: {docs}]"
        out += line + "\n"
        
    out += "\nTask: Select the next best Action and Argument.\nAnswer:"
    return out

def format_completion(action_id: int, argument: str) -> str:
    action_name = actions.get_action_name(action_id)
    return f"\nAction: {action_id} ({action_name})\nInput: {argument}"
