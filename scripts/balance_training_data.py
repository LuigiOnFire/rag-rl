import json
import glob
import os
from collections import defaultdict

INPUT_PATTERN = "data/runs/**/gold_trajectories.jsonl"
OUTPUT_FILE = "data/balanced_trajectories.jsonl"

def main():
    trajectories = []
    
    # 1. Load All Data
    print(f"Reading from {INPUT_PATTERN}...")
    files = glob.glob(INPUT_PATTERN, recursive=True)
    for fpath in files:
        with open(fpath, "r") as f:
            for line in f:
                if line.strip():
                    trajectories.append(json.loads(line))

    # 2. Analyze Distribution
    action_counts = defaultdict(int)
    by_action = defaultdict(list)
    
    for traj in trajectories:
        # We look at the FIRST step to determine the "Intent" of the trajectory
        if not traj.get("steps"): continue
        first_action = int(traj["steps"][0]["action_id"])
        
        # Group actions 2 and 3 as "Search"
        # Group actions 0 and 1 as "Answer"
        action_counts[first_action] += 1
        by_action[first_action].append(traj)

    print("\n--- Original Distribution ---")
    for act, count in sorted(action_counts.items()):
        print(f"Action {act}: {count} samples")

    # 3. Upsample Strategy
    # Find the max count (usually Action 1 or 0)
    max_count = max(action_counts.values())
    balanced_data = []

    print(f"\nTarget Count per Action: ~{max_count}")
    
    for act, items in by_action.items():
        # Calculate how many times to repeat to hit target
        if not items: continue
        
        # We want to boost Search (2, 3) and maybe Plan (7)
        # We leave Answer (0, 1) alone
        if act in [2, 3, 4, 5, 6, 7, 8, 9]:
            multiplier = int(max_count / len(items))
            # Cap multiplier to avoid massive overfitting on just 1-2 examples
            multiplier = min(multiplier, 10) 
            print(f"Upsampling Action {act} by {multiplier}x")
        else:
            multiplier = 1
            
        for _ in range(multiplier):
            balanced_data.extend(items)

    # 4. Save
    print(f"\nWriting {len(balanced_data)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for entry in balanced_data:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()