import sys
import os
import json
import time

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.env.state import GreenState, create_initial_state
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
    streamer = HotpotQAStreamer(limit=100) # Start small
    for sample in streamer.stream():
        # Setup Retriever
        # ... run search ...    
        question = sample['question']
        ground_truth = sample['answer']
        
        print(f"\n[{count+1}] Processing: {question}")
        
        # # 4. Instantiate Ephemeral Retriever with specific corpus
        corpus = sample["corpus"]
        retriever = EphemeralRetriever(documents=corpus)
        
        # 5. Instantiate Oracle Search
        oracle_search = OracleSearch(retriever=retriever)
        
        # 6. Setup Oracle Search
        start_state = create_initial_state(question)
        start_state['ground_truth'] = ground_truth
        solution_state = oracle_search.solve(start_state)
        
        if solution_state:
            print(f"  -> Solution found! Cost: {solution_state['total_joules']:.4f} J")
            trajectories.append({
                "question": question,
                "ground_truth": ground_truth,
                "history": solution_state['recent_history']
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
