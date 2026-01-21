import re
import os
import json
import logging
from typing import List, Tuple
from rouge_score import rouge_scorer
from src.env.state import GreenState
from src.agent import actions, workers

# --- CONFIG ---
REWARD_CORRECT = 1.0        # Huge bonus for solving it
REWARD_FORMAT_ERROR = -0.5  # Penalty for hallucinating nonsense
REWARD_WRONG = -0.50        # Small penalty for wrong answer
REWARD_LAZY = -0.5          # Extra penalty for answering wrong with NO docs

# Cost Settings
COST_TOOL_USE = 0.05        # Cost for using Search/LLM (Encourage efficiency)
PENALIZE_COST = True      # Whether to penalize cost at all

# Average tokens per output (Estimated)
AVG_CALIBRATOR_TOKENS = 40 
# Scaling factor to convert Joules to Reward Penalty (to prevent scale collapse)
JOULES_TO_REWARD_SCALE = 1e-4

# Load Cost Table
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COST_TABLE_PATH = os.path.join(BASE_DIR, "data", "meta", "cost_table.json")

try:
    with open(COST_TABLE_PATH, "r") as f:
        ACT_COSTS = json.load(f)
except Exception as e:
    print(f"Warning: Could not load cost table from {COST_TABLE_PATH}: {e}")
    ACT_COSTS = {}


scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        r = REWARD_FORMAT_ERROR # Should be highly negtative
        total_reward += r
        breakdown['format'] = r
        return total_reward, breakdown

    # 2. COST PENALTY (Length)
    # Penalize every character generated to cure verbosity/artifacts.
    # "Action: 2 Input: Search" (20 chars) vs "Action: 2 Input: Answer Generation... Search" (50 chars)
    token_proxy = len(generated_text) / 4.0
    
    # Dynamic Cost Calculation
    # cost_per_token = [corresponding_action] / [average_tokens]
    # We apply a scaling factor to normalize Joules to Reward space
    action_joules = ACT_COSTS.get(str(action_id), 0.0)
    joules_per_token = action_joules / AVG_CALIBRATOR_TOKENS
    reward_cost_per_token = joules_per_token * JOULES_TO_REWARD_SCALE
    
    cost = token_proxy * reward_cost_per_token
    
    # 3. TOOL COST
    # Penalize engaging the engine (Search or LLM call)
    cost += COST_TOOL_USE
    breakdown['cost'] = -cost

    if PENALIZE_COST:
        total_reward -= cost

    # 4. OUTCOME REWARD
    # If the episode ended (Action 0/1 Answer), did we get it right?
    if done and action_id in [0, 1]:
        # Compare 'obs' (the Answer) with 'ground_truth'
        # We use ROUGE-L (Overlap) or Exact Match
        # For strict correctness:
        scores = scorer.score(ground_truth, obs)
        rouge_l = scores['rougeL'].fmeasure
        logger.info(f"ROUGE-L score: {rouge_l}")
        
        # Threshold for "Correct"
        if rouge_l > 0.3: # Loose threshold for HotpotQA
            r = REWARD_CORRECT * (rouge_l * 2) # Scale by confidence
            # Cap at max reward
            r = min(r, REWARD_CORRECT)
            total_reward += r
            breakdown['correct'] = r
        else:
            p = REWARD_WRONG
            has_docs = False
            if 'subqueries' in state:
                for sq in state['subqueries']:
                    if 'retrieved_docs' in sq and len(sq['retrieved_docs']) > 0:
                        has_docs = True
                        break
            
            if not has_docs:
                p += REWARD_LAZY # Penalize for guessing wrong without effot
            
    # 5. INTERMEDIATE REWARDS (The "Cookie")
    # Bootstrap: Tiny reward for successfully finding documents
    # if action_id in [2, 3] and "Found" in obs:
    #     if "Found 0 docs" not in obs:
    #         total_reward += 0.1
    #         breakdown['correct'] += 0.1

    return total_reward, breakdown