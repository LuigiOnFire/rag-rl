import sys
import os
import json
import time
import logging

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.env.state import GreenState, create_initial_state
from src.env.retriever import EphemeralRetriever
from src.oracle.search import OracleSearch
from src.data.hotpot import HotpotQAStreamer

def main():
    # Initialize Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(logging.INFO)
    print("Starting Oracle Generation...")

    # Create Run Directory
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"data/runs/run_{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run Directory: {run_dir}")
    
    # Gold Output File
    gold_path = f"{run_dir}/gold_trajectories.jsonl"
    
    # 1. Initialize Streamer
    streamer = HotpotQAStreamer()
    
    # 2. Results Container
    # trajectories = [] # Removed in favor of continuous file writing
    
    # 3. Process Stream
    count = 0
    streamer = HotpotQAStreamer(limit=50) # Start small
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
        
        # Run Search
        solution_state, debug_info = oracle_search.solve(start_state)
        
        # 7. Write Per-Question Debug Log
        debug_path = f"{run_dir}/q_{count}.json"
        with open(debug_path, "w") as f_debug:
            json.dump({
                "question": question,
                "ground_truth": ground_truth,
                "solution_found": solution_state is not None,
                "debug_info": debug_info
            }, f_debug, indent=2, default=str)

        # 8. Write to Gold Trajectories (Continuous Saving)
        if solution_state:
            print(f"  -> Solution found! Cost: {solution_state['total_joules']:.4f} J")
            
            # Retrieve reconstructed SFT trajectory from debug_info
            steps_out = debug_info.get("sft_trajectory", [])
            
            record = {
                "question": question,
                "ground_truth": ground_truth,
                "steps": steps_out,
                "judge_log": solution_state.get('judge_log', 'Unknown')
            }
            
            with open(gold_path, "a") as f_gold:
                f_gold.write(json.dumps(record) + "\n")
        else:
            print(f"  -> No solution found.")
            
        count += 1
        
    print(f"\nGeneration complete. Saved results to {gold_path}")

if __name__ == "__main__":
    main()
