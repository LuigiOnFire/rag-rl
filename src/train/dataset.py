from typing import Any, Dict, List
import json
import logging
from datasets import Dataset 
from src.agent import actions
from src.agent.prompts import format_state_for_prompt

logger = logging.getLogger(__name__)


def load_and_clean_dataset(jsonl_files: list, tokenizer) -> Dataset:
    
    samples = []
    
    for fpath in jsonl_files:
        with open(fpath, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    # Parse the Episode
                    episode = json.loads(line)
                    
                    # Validate Structure
                    if "steps" not in episode:
                        continue
                        
                    # --- THE FIX: ITERATE STEPS ---
                    for step in episode["steps"]:
                        
                        # A. Reconstruct Prompt (X)
                        # We use the 'pre_state' because that's what the agent saw 
                        # BEFORE making the decision.
                        pre_state = step["pre_state"]
                        prompt_text = format_state_for_prompt(pre_state)
                        
                        # B. Format Target (Y)
                        # The action the agent took in this step
                        action_id = step["action_id"]
                        argument = step.get("argument", "")
                        
                        # Format: " Action: 3" or " Action: 2 search query"
                        # Ensure spacing matches your tokenizer/collator expectation
                        target_text = f" Action: {action_id}"
                        if argument:
                            target_text += f" {argument}"
                            
                        # C. Combine
                        # We append an EOS token if packing=False manually, 
                        # though Trainer usually handles it. Adding it explicitly is safer.
                        full_text = prompt_text + target_text + tokenizer.eos_token
                        
                        samples.append({"text": full_text})
                        
                except json.JSONDecodeError:
                    logging.warning(f"Skipping bad JSON in {fpath} line {line_num}")
                    continue

    logging.info(f"Extracted {len(samples)} training steps from {len(jsonl_files)} files.")
    return Dataset.from_list(samples)

# def format_prompt(state: Dict[str, Any]) -> str:
    # return format_state_for_prompt(state)

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