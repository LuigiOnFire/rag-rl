import string
import collections
import logging
import re
from src.agent.workers import llm_worker

USE_LLM_JUDGE = False

# 1. Setup Logging (Ensures you see output immediately)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

class SoftJudge:
    def __init__(self):
        self.refusal_phrases = [
            "i don't know", "no question provided", "cannot answer", 
            "context is empty", "provide more information", "unsure", 
            "none", "null", "no answer"
        ]

    def judge(self, prediction: str, ground_truth: str) -> tuple[bool, str]:
        norm_pred = normalize_answer(prediction)
        norm_gt = normalize_answer(ground_truth)
        
        logger.info(f"JUDGE: GT='{norm_gt}' | PRED='{norm_pred}'")

        # --- Sanity Check: Reject Refusals ---
        if not norm_pred or any(phrase in norm_pred for phrase in self.refusal_phrases):
            logger.warning("JUDGE: Detected Refusal/Empty Answer -> FAIL")
            return False, "Refusal"

        # --- Tier 1: String Match (Normalized Inclusion) ---
        # Stricter Inclusion: Exact match or restricted substring
        if norm_gt:
            # 1. Exact Match (Best)
            # No substring matches here
            if norm_gt == norm_pred:
                logger.info("JUDGE: Tier 1 (Exact Match) -> PASS")
                return True, "Tier 1 (Exact Match)"
        
        # --- Tier 2: Token F1 ---
        f1 = f1_score(prediction, ground_truth)
        if f1 > 0.8: # Increased threshold
            logger.info(f"JUDGE: Tier 2 (F1={f1:.2f}) -> PASS")
            return True, f"Tier 2 (F1={f1:.2f})"
        
        if not USE_LLM_JUDGE:
            logger.info(f"JUDGE: Tier 2 (F1={f1:.2f}) -> FAIL")
            return False, f"Tier 2 (F1={f1:.2f})"
        
        # --- Tier 3: LLM Judge (The Expensive Fallback) ---
        logger.info("JUDGE: Tier 1 & 2 Failed. Escalating to LLM Judge...")
        
        prompt = f"""
        You are a strict, impartial judge avoiding false positives.
        
        Question: Are the Prediction and Ground Truth effectively the same answer?
        
        Ground Truth: "{ground_truth}"
        Prediction: "{prediction}"

        Instructions:
        1. Ignore minor differences in capitalization, punctuation, or phrasing.
        2. Identify the core entity or fact in both.
        3. If the Prediction is the opposite (e.g., "Yes" vs "No"), it is Wrong.
        4. If the Prediction contains the Ground Truth but negates it (e.g., "Did not win"), it is Wrong.
        5. If the Prediction is "I don't know", it is Wrong.

        First, explain your reasoning in one sentence.
        Then, output the final verdict as "Verdict: YES" or "Verdict: NO".
        """
        
        try:
            response = llm_worker.generate(prompt)
            # Use repr() to see hidden newlines or spaces in the log
            logger.info(f"JUDGE: LLM Raw Response: {repr(response)}") 
            
            if "verdict: yes" in response.lower():
                logger.info("JUDGE: LLM Tier 3 -> PASS")
                return True, "Tier 3 (LLM)"
            else:
                logger.info("JUDGE: LLM Tier 3 -> FAIL")
                return False, "Tier 3 (LLM Fail)"
                
        except Exception as e:
            logger.error(f"JUDGE: LLM Failed with error: {e}")
            return False, f"Error: {e}"