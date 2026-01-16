import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.agent.prompts import format_state_for_prompt

# Configuration
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct" 
ADAPTER_PATH = "models/green-rag-sft-v1"

def format_test_prompt(question):
    """
    Manually constructs a prompt in the exact format used during training.
    """
    # We simulate a "Start State"
    return f"""Goal: {question}
Status: SOLVING
Scratchpad: Goal: Answer the main query.

History:
(None)

Sub-Tasks:
- [ACTIVE] {question}

Task: Select the next best Action and Argument.
Answer:"""

def main():
    print(f"1. Loading Base Model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"2. Loading Adapter: {ADAPTER_PATH}")
    # This wraps the base model with your fine-tuned weights
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("\n--- GreenRAG 1B Model Loaded ---\n")
    
    questions = [
        "What is the capital of France?",
        "Who is the CEO of Tesla?",
        # A harder one that requires search
        "Which magazine was started first Arthur's Magazine or First for Women?" 
    ]

    for q in questions:
        print(f"\nUser: {q}")

        # Construct a 'Virtual State' instead of a raw string
        virtual_state = {
            "main_query": q,
            "status": "SOLVING",
            "scratchpad": "Goal: Answer the main query.",
            "history": [],
            "subqueries": [
                {"status": "ACTIVE", "question": q, "answer": None, "documents": []}
            ]
        }

        # Generate exact prompt
        prompt = format_state_for_prompt(virtual_state)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50, # We only need the Action/Argument line
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"GreenAgent:\n{response.strip()}")
        print("-" * 40)

if __name__ == "__main__":
    main()