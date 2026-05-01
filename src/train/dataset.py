from typing import Any, Dict, List, Optional, Set
import random
import json
import logging
from datasets import Dataset 
from src.agent import actions
from src.agent.prompts import format_state_for_prompt

logger = logging.getLogger(__name__)


def load_and_clean_dataset(
    jsonl_files: list,
    tokenizer,
    oversample_config: Optional[Dict[str, object]] = None,
    rng: Optional[random.Random] = None,
) -> Dataset:
    
    samples = []
    
    oversample_config = oversample_config or {}
    complex_action_ids: Set[int] = set(oversample_config.get("complex_action_ids", []))
    complex_multiplier = float(oversample_config.get("complex_multiplier", 1.0))
    rng = rng or random.Random(0)

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
                        
                        # Format: " Action: 3"
                        # Ensure spacing matches your tokenizer/collator expectation
                        target_text = f" Action: {action_id}"
                            
                        # C. Combine
                        # We append an EOS token if packing=False manually, 
                        # though Trainer usually handles it. Adding it explicitly is safer.
                        full_text = prompt_text + target_text + tokenizer.eos_token
                        
                        weight = 1.0
                        if action_id in complex_action_ids:
                            weight *= complex_multiplier

                        repeat_count = int(weight)
                        remainder = weight - repeat_count
                        if rng.random() < remainder:
                            repeat_count += 1

                        for _ in range(max(1, repeat_count)):
                            samples.append({"text": full_text})
                        
                except json.JSONDecodeError:
                    logging.warning(f"Skipping bad JSON in {fpath} line {line_num}")
                    continue

    logging.info(f"Extracted {len(samples)} training steps from {len(jsonl_files)} files.")
    return Dataset.from_list(samples)

# def format_prompt(state: Dict[str, Any]) -> str:
    # return format_state_for_prompt(state)

def format_completion(action_id: int) -> str:
    """
    Constructs the target string (the Action).
    We currently train on the action ID only (no arguments).
    """
    return f" Action: {action_id}"