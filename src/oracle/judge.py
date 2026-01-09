import string
import collections
from src.agent.workers import llm_worker

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = r'\b(a|an|the)\b'
        return text # Simplification for now, or use re.sub
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
        
    return white_space_fix(remove_punc(lower(s)))

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
    def judge(self, prediction: str, ground_truth: str) -> bool:
        # Tier 1: String Match (Normalized Inclusion)
        norm_pred = normalize_answer(prediction)
        norm_gt = normalize_answer(ground_truth)
        
        if norm_gt in norm_pred: # Loose inclusion
            return True
            
        # Tier 2: Token F1
        f1 = f1_score(prediction, ground_truth)
        if f1 > 0.75:
            return True
            
        # Tier 3: LLM Judge
        prompt = f"""
Ground Truth: {ground_truth}
Prediction: {prediction}

Are these two answers semantically equivalent?
Reply with EXACTLY one word: "Yes" or "No".
"""
        response = llm_worker.generate(prompt)
        if "yes" in response.lower():
            return True
            
        return False
