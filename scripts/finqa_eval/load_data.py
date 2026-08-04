from datasets import load_dataset

# Load FinQA from Hugging Face
dataset = load_dataset("fin_qa", split="test")

print(f"Total test samples: {len(dataset)}")
sample = dataset[0]

# Inspect key keys
print("Keys available:", sample.keys())
print("\nSample Question:", sample["question"])
print("\nSample Context/Pre-text:", sample.get("pre_text", ""))