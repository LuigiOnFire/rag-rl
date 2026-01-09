import sys
import os
import json
import time

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.env.state import GreenState
from src.env.retriever import EphemeralRetriever
from src.oracle.search import OracleSearch
from src.data.hotpot import HotpotQAStreamer

def main():
    print("Starting Oracle Generation...")
    
    # 1. Initialize Streamer
    streamer = HotpotQAStreamer()
    
    # 2. Results Container
    trajectories = []
    
    # 3. Process Stream
    count = 0
    for sample in streamer.stream():
        question = sample['question']
        ground_truth = sample['answer']
        
        print(f"\n[{count+1}] Processing: {question}")
        
        # 4. Instantiate Ephemeral Retriever with specific corpus
        corpus = streamer.get_corpus(sample)
        retriever = EphemeralRetriever(documents=corpus)
        
        # 5. Initialize Oracle Search
        oracle = OracleSearch(retriever=retriever)
        
        # 6. Run Search
        start_state = GreenState(question=question, ground_truth=ground_truth)
        solution_state = oracle.solve(start_state)
        
        if solution_state:
            print(f"  -> Solution found! Cost: {sum(s['cost'] for s in solution_state.history):.4f} J")
            trajectories.append({
                "question": question,
                "ground_truth": ground_truth,
                "history": solution_state.history
            })
        else:
            print(f"  -> No solution found.")
            
        count += 1
        
    # 7. Save Results
    output_path = "data/trajectories/gold_trajectories.jsonl"
    with open(output_path, "w") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")
            
    print(f"\nGeneration complete. Saved {len(trajectories)} trajectories to {output_path}")

if __name__ == "__main__":
    main()
