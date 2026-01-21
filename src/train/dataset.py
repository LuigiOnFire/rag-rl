from typing import Any, Dict, List
import json
import logging
from datasets import Dataset 
from src.agent import actions
from src.agent.prompts import format_state_for_prompt

logger = logging.getLogger(__name__)

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

                        logging.warning(f"Added sample from {path}: {full_text[:300]}...")
                        
                        samples.append({
                            "text": full_text
                        })


                except json.JSONDecodeError:
                    continue
                    
    logger.info(f"Loaded {len(samples)} training samples from {len(jsonl_paths)} files.")
    
    if not samples:
         # Fallback for empty runs to prevent crash
        logger.warning("No samples found. Creating dummy dataset.")
        return Dataset.from_list([{"text": "Empty"}])

    # Logs the very first sample so you can verify the Prompt/Completion format visually.
    logger.info(f"\n{'='*40}\nSAMPLE TRAINING DATA (Item 0):\n{'='*40}\n{samples[0]['text']}\n{'='*40}\n")
    
    # Return standard HF Dataset
    return Dataset.from_list(samples)

def format_prompt(state: Dict[str, Any]) -> str:
    return format_state_for_prompt(state)

def format_completion(action_id: int, argument: str) -> str:
    """
    Constructs the target string (the Action).
    Logic:
    - If Action 0/1 (Answer): Stop immediately after the digit. No 'Input:'.
    - If Action 2+ (Tool): Include 'Input: {argument}'.
    """
    
    # --- FIX: STRICT CUTOFF FOR ANSWER ACTIONS ---
    if int(action_id) in [0, 1]:
        # The model sees: "Action: 0" -> [EOS]
        # It never sees "Input: Answer Generation" or even "Input: "
        return f" Action: {action_id}"
        
    # --- LOGIC FOR TOOLS (Search, etc.) ---
    else:
        # Sanitize garbage just in case the trajectory generator messed up
        clean_arg = argument
        if "Answer Generation" in clean_arg:
            clean_arg = "" 
            
        return f" Action: {action_id}\nInput: {clean_arg}"