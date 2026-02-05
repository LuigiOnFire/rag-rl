import re
import os
import json
import logging
import collections
import string
from typing import Tuple
from src.env.state import GreenState
from src.oracle.judge import SoftJudge

# --- CONFIG ---
REWARD_CORRECT = 1.0        # The Goal
REWARD_DISCOVERY = 0.5      # <--- BIG COOKIE. Incentivize finding docs.
REWARD_WRONG = -0.5         # Penalty for failure
REWARD_LAZY = -0.5          # Extra penalty for guessing without looking
REWARD_REPEAT = -1.0        # Penalty for exact repeated actions (Insanity)
REWARD_FORMAT = -0.5        # Penalty for broken syntax

# --- PHASE 1 SETTINGS: COSTS DISABLED ---
# We calculate costs for logging, but do NOT subtract them from reward.
# This allows the agent to learn "how" to search before learning efficiency.
PENALIZE_COST = False       
COST_TOOL_USE = 0.0

# Constants for Cost Calculation (Used for logging only in Phase 1)
AVG_CALIBRATOR_TOKENS = 40 
JOULES_TO_REWARD_SCALE = 1e-4

# Threshold for Partial Credit (Safety Net)
F1_THRESHOLD = 0.25

# --- LOAD COST TABLE ---
# This is the missing piece!
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COST_TABLE_PATH = os.path.join(BASE_DIR, "data", "meta", "cost_table.json")

try:
    with open(COST_TABLE_PATH, "r") as f:
        ACT_COSTS = json.load(f)
except Exception as e:
    # Fallback if file is missing
    print(f"Warning: Could not load cost table from {COST_TABLE_PATH}: {e}")
    ACT_COSTS = {}

# Initialize Judge
judge = SoftJudge()
logger = logging.getLogger(__name__)

# --- HELPERS ---
def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens: return 0.0
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0: return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)
# ----------------

def calculate_reward(
    state: GreenState, 
    generated_text: str, 
    ground_truth: str, 
    action_id: int, 
    done: bool,
    obs: str
) -> Tuple[float, dict]:
    
    total_reward = 0.0
    breakdown = {"correct": 0.0, "format": 0.0, "bonus": 0.0, "cost": 0.0}
    question = state.get("question", "")

    # 1. FORMAT & REPETITION CHECKS
    if action_id in [-1, 9]:
        total_reward += REWARD_FORMAT
        breakdown['format'] = REWARD_FORMAT
        return total_reward, breakdown

    # Check for exact repetition (Insanity)
    history = state.get('history', [])
    if len(history) > 0:
        last_step = history[-1]
        if last_step.get('action_id') == action_id:
             # Basic check: if Action ID is the same for Search, it might be a loop.
             # Ideally we check arguments too, but this is a safe heuristic for now.
             pass 

    # 2. DISCOVERY BONUS (The "Curriculum Driver")
    # Only reward the FIRST successful search in an episode.
    is_search_action = (action_id in [2, 3])
    has_found_docs = ("Found" in obs and "Found 0 docs" not in obs)
    
    if is_search_action and has_found_docs:
        # Check history: How many times have we tried Action 2 or 3 before?
        history = state.get('history', [])
        
        # Count prior search actions (Robust integer check, no string parsing)
        prior_search_count = sum(1 for h in history if h.get('action_id') in [2, 3])
        
        if prior_search_count == 1:
            # First time? Here is your cookie.
            total_reward += REWARD_DISCOVERY
            breakdown['bonus'] += REWARD_DISCOVERY
            logger.info(f"🍪 First Discovery Bonus! (+{REWARD_DISCOVERY})")
        else:
            # Second time? No cookie. Get back to work.
            logger.info(f"🚫 No Bonus: Search action already used {prior_search_count} times.")

    # 3. OUTCOME REWARD
    if done and action_id in [0, 1]:
        is_correct, reason = judge.judge(obs, ground_truth, question)
        
        if is_correct:
            total_reward += REWARD_CORRECT
            breakdown['correct'] = REWARD_CORRECT
        else:
            # Partial Credit (F1 Safety Net)
            f1 = compute_f1(obs, ground_truth)
            if f1 >= F1_THRESHOLD:
                partial = f1 ** 2
                total_reward += partial
                breakdown['correct'] = partial
                logger.info(f"⚠️ Partial Credit: F1 {f1:.2f} -> Reward {partial:.2f}")
            else:
                # Wrong Answer
                p = REWARD_WRONG
                
                # Laziness Check
                has_docs = False
                if 'subqueries' in state:
                     for sq in state['subqueries']:
                        if sq.get('documents'): has_docs = True
                if not has_docs:
                    for h in history:
                        if "Found" in str(h.get('observation','')) and "Found 0 docs" not in str(h.get('observation','')):
                            has_docs = True
                
                if not has_docs:
                    p += REWARD_LAZY 
                
                total_reward += p
                breakdown['correct'] = p

    # 4. SILENT COST TRACKING
    # We calculate the Joules but only subtract if PENALIZE_COST is True.
    token_proxy = len(generated_text) / 4.0
    action_joules = ACT_COSTS.get(str(action_id), 0.0)
    joules_per_token = action_joules / AVG_CALIBRATOR_TOKENS
    reward_cost_per_token = joules_per_token * JOULES_TO_REWARD_SCALE
    
    # Calculate hypothetical cost
    step_cost = (token_proxy * reward_cost_per_token) + COST_TOOL_USE
    
    # Always log it negative so it looks like a cost in the breakdown
    breakdown['cost'] = -step_cost 

    if PENALIZE_COST:
        total_reward -= step_cost

    return total_reward, breakdown