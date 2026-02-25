import string
import collections
import logging
import re
from difflib import SequenceMatcher

from src.agent.workers import llm_worker

USE_LLM_JUDGE = False

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

def similarity_score(a, b):
    """Returns value between 0 and 1 indicating similarity."""
    return SequenceMatcher(None, a, b).ratio()

class SoftJudge:
    def __init__(self):
        self.refusal_phrases = [
            "i don't know", "no question provided", "cannot answer", 
            "context is empty", "provide more information", "unsure", 
            "none", "null", "no answer", "not specified", "cannot determine", 
            "information is not available", "does not mention", 
            "i cannot conclude", "no information found"
        ]

    def judge(self, prediction: str, ground_truth: str, question: str) -> tuple[bool, str]:
        norm_pred = normalize_answer(prediction)
        norm_gt = normalize_answer(ground_truth)
        norm_q = normalize_answer(question)
        
        logger.info(f"JUDGE: Question='{norm_q}'")
        logger.info(f"JUDGE: GT='{norm_gt}' | PRED='{norm_pred}'")

        # --- REJECT REFUSALS ---
        if not norm_pred or any(phrase in norm_pred for phrase in self.refusal_phrases):
            logger.warning("JUDGE: Detected Refusal/Empty Answer -> FAIL")
            return False, "Refusal"
        
        # --- QUESTION-PARROT FILTER---
        if norm_q and len(norm_pred) > 5:
            threshold = 0.85
            if similarity_score(norm_pred, norm_q) >= threshold:
                logger.warning("JUDGE: Detected Question Parroting -> FAIL")
                return False, "Repeated Question, Parroting"

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
        if f1 > 0.75: # Lowering theshold now that we have the parroting guard
            logger.info(f"JUDGE: Tier 2 (F1={f1:.2f}) -> PASS")
            return True, f"Tier 2 (F1={f1:.2f})"
        
        if not USE_LLM_JUDGE:
            logger.info(f"JUDGE: Tier 2 (F1={f1:.2f}) -> FAIL")
            return False, f"Tier 2 (F1={f1:.2f})"
        
        # --- Tier 3: LLM Judge (The Expensive Fallback) ---
        logger.info("JUDGE: Tier 1 & 2 Failed. Escalating to LLM Judge...")
        
        prompt = f"""
You are a strict, impartial judge. 
Task: Validate if the Prediction is equivalent to the Ground Truth.

STRICT RULES:
1. **Core Match**: The Prediction must contain the specific entity (name, date, or number) from the Ground Truth.
2. **No Hallucinations**: If the Prediction contains *additional* specific details (dates, numbers, names) that are NOT in the Ground Truth, mark it INCORRECT.
3. **No Fluff**: Repeating the question or giving vague non-answers is INCORRECT.

First, explain your reasoning in one short sentence.
Then, output "Verdict: YES" or "Verdict: NO".

Question: "{question}"        
Ground Truth: "{ground_truth}"
Prediction: "{prediction}"
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