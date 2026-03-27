import sys
import os
import json
import time
import logging
import random

# Add project root to sys.path
sys.path.append(os.getcwd())

from src.agent import workers
from src.env.state import GreenState, create_initial_state
from src.env.retriever import EphemeralRetriever
from src.oracle.search import OracleSearch, WaterfallOracle
from src.data.loader import MixedStreamer

# Frequency with which to force decomposition
FORCE_DECOMP_RATE = 1

def main():
    # Initialize Logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s\n%(message)s')
    logging.getLogger().setLevel(logging.DEBUG)
    print("Starting Oracle Generation...")

    # Create Run Directory
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = f"data/trajectories/run_{run_id}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run Directory: {run_dir}")
    
    # Gold Output File
    gold_path = f"{run_dir}/gold_trajectories.jsonl"
    
    # Initialize Streamer
    active_datasets = ["hotpot", "musique", "twowiki"]
    dataset_configs = {
        "hotpot": {
            "setting": "distractor", 
            "split": "train"
        },
        "musique": {
            "setting": "distractor", 
            "split": "train"
        },
        "twowiki": {
            "setting": "distractor", 
            "split": "train"
        }
    }
    
    # Passing limit=50 here gives an even mix; limit gives 50 samples per active dataset = 150 total.
    streamer = MixedStreamer(dataset_names=active_datasets, limit=50, configs=dataset_configs)
    
    print(f"Streaming {streamer.n_limit} of {streamer.total_available:,} available examples "
          f"from: {', '.join(active_datasets)}")

    # Process Stream
    total = streamer.n_limit
    count = 0
    for sample in streamer.stream():
        question = sample['question']
        ground_truth = sample['ground_truth'] if 'ground_truth' in sample else sample['answer']
    
        log_file = f"{run_dir}/q_{count}.log"
        workers.configure_worker_logging(log_file)

        print(f"\n[{count+1}/{total}] Processing: {question}")
        
        # Instantiate Ephemeral Retriever (Local context only)
        corpus = sample["corpus"]
        if not corpus:
            print("  -> Warning: No corpus found for Ephemeral Retriever. Ensure you are not in fullwiki mode.")
            continue
            
        retriever = EphemeralRetriever(documents=corpus)
        
        # Instantiate Oracle Search using the local retriever
        oracle_search = WaterfallOracle(retriever=retriever)
        logging.info("Initialized Oracle Search.")        
        
        # Setup Oracle Search
        start_state = create_initial_state(question)

        # Run Search
        logging.info("Starting Oracle Search...")
        solution_state, debug_info = oracle_search.solve(start_state, ground_truth)
        logging.info("Oracle Search complete.")

        # Write to Gold Trajectories (Continuous Saving)
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
