import argparse
import copy
import json
import os
import sys
import urllib.request
from typing import Dict, Any, List

# Adjust import path based on repository layout
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.env.engine import GreenEngine
from src.env.retriever import EphemeralRetriever
from src.oracle.judge import SoftJudge
from src.routers.static_router import StaticHeuristicRouter
from src.routers.upfront_router import UpfrontClassifierRouter
from src.routers.thrifty_router import ThriftyEarlyExitRouter


class OllamaSLMClient:
    """Connects to the local Ollama daemon running inside the container."""
    def __init__(self, host="http://localhost:11434"):
        self.host = host
        self.model = os.getenv("SLM_MODEL", "llama3.2:3b")

    def generate(self, prompt: str) -> str:
        import requests
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0}
        }
        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            print(f"[Ollama Error] {e}")
            return ""


class MockSLMClient:
    """Mock SLM Client for local/offline dry runs."""
    def generate(self, prompt: str) -> str:
        # Simple mock response toggling based on prompt signature
        if "financial data auditor" in prompt.lower():
            return '{"status": "SUFFICIENT", "missing_info": null}'
        return '{"complexity": "SIMPLE", "reason": "Mock simplicity"}'


def load_finqa_dataset(split="test") -> List[Dict[str, Any]]:
    """Loads FinQA dataset directly from canonical raw JSON."""
    url = f"https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"
    print(f"Fetching FinQA '{split}' dataset from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))


def extract_sample_data(sample: dict) -> tuple[str, str, str]:
    """Extracts question, context, and ground truth from a FinQA record."""
    if "qa" in sample and isinstance(sample["qa"], dict):
        query = sample["qa"].get("question", "")
        ground_truth = str(sample["qa"].get("exe_ans", ""))
    else:
        query = sample.get("question", "")
        ground_truth = str(sample.get("exe_ans", ""))

    pre_text = " ".join(sample.get("pre_text", []))
    post_text = " ".join(sample.get("post_text", []))
    table_str = str(sample.get("table", []))
    context = f"Pre-text: {pre_text}\nTable: {table_str}\nPost-text: {post_text}"

    return query, context, ground_truth


def compute_trajectory_cost(history: List[Dict]) -> float:
    """Sums the total Joule cost across all actions in a state history."""
    return sum(float(step.get("cost", 0.0)) for step in history)


def main():
    parser = argparse.ArgumentParser(description="FinQA 3-Way Router Comparison")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--use_mock", action="store_true", help="Use mock client instead of live Ollama")
    args = parser.parse_args()

    print(f"=== Starting 3-Way Router Comparison ({args.num_samples} samples) ===")

    # 1. Environment Setup
    dataset = load_finqa_dataset("test")
    retriever = EphemeralRetriever()
    engine = GreenEngine(retriever=retriever)
    judge = SoftJudge()

    slm_client = MockSLMClient() if args.use_mock else OllamaSLMClient()

    # 2. Router Initialization
    r1_static = StaticHeuristicRouter(engine)
    r2_upfront = UpfrontClassifierRouter(engine, slm_client)
    r3_thrifty = ThriftyEarlyExitRouter(engine, slm_client)

    routers = {
        "Route 1 (Static Rule)": r1_static,
        "Route 2 (Upfront SLM)": r2_upfront,
        "Route 3 (Thrifty Exit)": r3_thrifty
    }

    metrics = {
        name: {"cost": 0.0, "correct": 0, "light_count": 0, "heavy_count": 0}
        for name in routers
    }

    # 3. Main Evaluation Loop
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        query, context, ground_truth = extract_sample_data(sample)

        base_state = {
            "question": query,
            "context": context,
            "history": [],
            "status": "SOLVING",
            "answer": None
        }

        print(f"\n--- Sample #{i+1} ---")
        print(f"Query: {query[:100]}...")

        for name, router in routers.items():
            # Deepcopy initial state for complete isolation
            start_state = copy.deepcopy(base_state)
            
            # Execute policy trajectory
            final_state = router.solve(start_state)
            
            # Calculate metrics
            cost = compute_trajectory_cost(final_state.get("history", []))
            answer = final_state.get("answer", "")
            
            # Judge correctness if answer was generated
            is_correct = False
            if answer:
                is_correct, _ = judge.judge(answer, ground_truth, query)

            # Record metrics
            metrics[name]["cost"] += cost
            if is_correct:
                metrics[name]["correct"] += i
            
            # Track trajectory depth/actions
            action_ids = [step["action_id"] for step in final_state.get("history", [])]
            if 7 in action_ids or 8 in action_ids:  # ACTION_GEN_LLM or heavy actions
                metrics[name]["heavy_count"] += 1
            else:
                metrics[name]["light_count"] += 1

            print(f"  [{name}] Cost: {cost:.2f} J | Correct: {is_correct} | Actions: {action_ids}")

    # 4. Comparative Summary Output
    print("\n" + "="*60)
    print("FINAL ROUTER COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Router Strategy':<25} | {'Total Cost (J)':<15} | {'Accuracy':<10} | {'Light/Heavy Traj':<15}")
    print("-" * 72)

    total_eval = min(args.num_samples, len(dataset))
    for name, m in metrics.items():
        acc = (m["correct"] / total_eval) * 100 if total_eval > 0 else 0.0
        traj_str = f"{m['light_count']}L / {m['heavy_count']}H"
        print(f"{name:<25} | {m['cost']:<15.2f} | {acc:<9.1f}% | {traj_str:<15}")

    print("="*60)


if __name__ == "__main__":
    main()