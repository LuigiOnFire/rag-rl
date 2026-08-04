import argparse
import sys
import os
import json
import urllib.request
import requests

# Adjust import path based on your repository layout
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.routers.thrifty_router import evaluate_sufficiency

def load_finqa_dataset(split="test"):
    """Loads FinQA directly from the official GitHub raw JSON file to bypass deprecated HF dataset scripts."""
    url = f"https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"
    print(f"[1/3] Fetching FinQA '{split}' set directly from GitHub ({url})...")
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
        print(f"Successfully loaded {len(data)} samples from raw JSON.")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to fetch FinQA dataset from {url}: {e}")

class MockSLMClient:
    """Mock SLM Client for initial testing without loading weights."""
    def generate(self, prompt: str) -> str:
        # Simulate SLM output for dry run sanity testing
        return '```json\n{\n  "status": "SUFFICIENT",\n  "missing_info": null\n}\n```'


class OllamaSLMClient:
    """Connects to the local Ollama daemon running inside the container."""
    def __init__(self, host="http://localhost:11434"):
        self.host = host
        # Fall back to environment variable or default SLM model
        self.model = os.getenv("SLM_MODEL", "llama3.2:3b")

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",  # Forces JSON mode in Ollama
            "options": {
                "temperature": 0.0  # Zero temp for deterministic evaluation
            }
        }
        try:
            resp = requests.post(f"{self.host}/api/generate", json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            print(f"[Ollama Error] Failed to generate: {e}")
            return ""

def format_table_to_markdown(table_data) -> str:
    """Converts a FinQA list-of-lists table into a clean Markdown table string."""
    if not table_data or not isinstance(table_data, list):
        return ""
    
    md_lines = []
    for i, row in enumerate(table_data):
        # Ensure row items are strings
        clean_row = [str(cell).strip() for cell in row]
        md_lines.append("| " + " | ".join(clean_row) + " |")
        
        # Add header separator after the first row
        if i == 0:
            md_lines.append("| " + " | ".join(["---"] * len(clean_row)) + " |")
            
    return "\n".join(md_lines)

def main():
    parser = argparse.ArgumentParser(description="FinQA Thrifty Router Sufficiency Test")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate")
    parser.add_argument("--use_mock", action="store_true", help="Use mock SLM client for fast pipeline check")
    args = parser.parse_args()

    print(f"=== Starting FinQA Sufficiency Harness ({args.num_samples} samples) ===")

    # 1. Load dataset via raw JSON helper (Bypasses HuggingFace script runner)
    dataset = load_finqa_dataset("test")

    # 2. Instantiate SLM client
    if args.use_mock:
        print("[2/3] Initializing Mock SLM Client...")
        slm_client = MockSLMClient()
    else:
        print("[2/3] Initializing Ollama SLM Client...")
        slm_client = OllamaSLMClient()

    # 3. Execution loop
    print("\n[3/3] Running evaluation loop...")
    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]

        # Extract Question (In official FinQA JSON, the question is inside the 'qa' dict)
        if "qa" in sample and isinstance(sample["qa"], dict):
            query = sample["qa"].get("question", "")
        else:
            query = sample.get("question", "")

        # Extract Context (pre-text, table, post-text)
        pre_text = " ".join(sample.get("pre_text", []))
        post_text = " ".join(sample.get("post_text", []))
        table_str = format_table_to_markdown(sample.get("table", []))
        
        context = f"Pre-text: {pre_text}\n\nTable:\n{table_str}\n\nPost-text: {post_text}"
        print(f"\n--- Sample #{i+1} ---")
        print(f"Query: {query}")
        print(f"Context Length: {len(context)} chars")

        result = evaluate_sufficiency(slm_client, query, context)
        print(f"Sufficiency Output: {result}")

    print("\n=== Harness Run Complete ===")


if __name__ == "__main__":
    main()