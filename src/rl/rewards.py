import re
from typing import List, Tuple
from rouge_score import rouge_scorer
from src.env.state import GreenState
from src.agent import actions, workers

# --- CONFIG ---
REWARD_CORRECT = 1.0        # Huge bonus for solving it
REWARD_FORMAT_ERROR = -0.5  # Penalty for hallucinating nonsense
COST_PER_TOKEN = 0.002      # Tiny penalty to discourage "Answer Generation" artifacts
COST_TOOL_USE = 0.05        # Cost for using Search/LLM (Encourage efficiency)

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

def calculate_reward(
    state: GreenState, 
    generated_text: str, 
    ground_truth: str, 
    action_id: int, 
    done: bool,
    obs: str
) -> Tuple[float, dict]:
    """
    Calculates the reward for a single step.
    Returns: (Total Reward, Breakdown Dict)
    """
    total_reward = 0.0
    breakdown: dict[str, float] = {"correct": 0.0, "format": 0.0, "cost": 0.0}
    
    # 1. FORMAT PENALTY
    # If the parser failed (action_id is -1 or 9), punish hard.
    if action_id in [-1, 9]:
        r = REWARD_FORMAT_ERROR
        total_reward += r
        breakdown['format'] = r
        return total_reward, breakdown

    # 2. COST PENALTY (Length)
    # Penalize every character generated to cure verbosity/artifacts.
    # "Action: 2 Input: Search" (20 chars) vs "Action: 2 Input: Answer Generation... Search" (50 chars)
    token_proxy = len(generated_text) / 4.0
    cost = token_proxy * COST_PER_TOKEN
    
    # 3. TOOL COST
    # Penalize engaging the engine (Search or LLM call)
    cost += COST_TOOL_USE
    
    total_reward -= cost
    breakdown['cost'] = -cost

    # 4. OUTCOME REWARD
    # If the episode ended (Action 0/1 Answer), did we get it right?
    if done and action_id in [0, 1]:
        # Compare 'obs' (the Answer) with 'ground_truth'
        # We use ROUGE-L (Overlap) or Exact Match
        # For strict correctness:
        scores = scorer.score(ground_truth, obs)
        rouge_l = scores['rougeL'].fmeasure
        
        # Threshold for "Correct"
        if rouge_l > 0.3: # Loose threshold for HotpotQA
            r = REWARD_CORRECT * (rouge_l * 2) # Scale by confidence
            # Cap at max reward
            r = min(r, REWARD_CORRECT)
            total_reward += r
            breakdown['correct'] = r
        else:
            # Wrong answer penalty?
            # Usually better to just give 0, or small negative
            pass
            
    # 5. INTERMEDIATE REWARDS (Optional)
    # Reward finding RELEVANT docs?
    if "Found" in obs and "docs" in obs:
        # If we successfully searched, give a tiny "crumb" to encourage searching
        # But not too much, or it will just search forever (Reward Hacking)
        total_reward += 0.05

    return total_reward, breakdown