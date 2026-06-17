# analytics/micro_analysis.py
import argparse
import sys
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional, Iterable, List

# Programmatically inject the analytics directory into the search path
sys.path.append(str(Path(__file__).resolve().parent))

from loader import load_all_records, safe_mean, print_table

def run_micro_analysis(records: List[dict]):
    # Nested dictionary pattern: action_costs[source][action_name] = [costs]
    action_costs = defaultdict(lambda: defaultdict(list))
    
    # Flatten sequential histories into standalone operation instances
    for record in records:
        source = record.get("source", "unknown")
        history = record.get("history", [])
        
        if not history and "attempts" in record:
            # Fallback parsing strategy if embedded inside the optimal attempt structure
            opt_id = int(record.get("optimal_trajectory_id", -1))
            for attempt in record.get("attempts", []):
                if int(attempt.get("trajectory_id", -1)) == opt_id:
                    history = attempt.get("steps") or attempt.get("history") or []
                    break
                    
        for step in history:
            # Match schema variants for action identification
            action_name = step.get("action_name") or f"action_{step.get('action_id')}"
            cost = float(step.get("cost") or step.get("measured_joules") or 0.0)
            if cost > 0.0:
                action_costs[source][action_name].append(cost)

    print("==================================================================")
    print("SECTION 3.2: MICRO-ACTION HARDWARE FOOTPRINT BY DATASET (JOULES)")
    print("==================================================================")
    
    headers = ["Dataset", "Micro-Action Name", "Total Samples (N)", "Mean Cost", "Std Dev (\u03c3)"]
    rows = []
    
    # Sort by dataset first, then by action name to create clean visual groupings
    for source in sorted(action_costs.keys()):
        for action, samples in sorted(action_costs[source].items()):
            n = len(samples)
            mean_val = safe_mean(samples)
            
            # Calculate standard deviation natively
            if n > 1:
                variance = sum((x - mean_val) ** 2 for x in samples) / (n - 1)
                std_dev = math.sqrt(variance)
            else:
                std_dev = 0.0
                
            rows.append([
                source.upper(),
                action,
                str(n),
                f"{mean_val:.2f} J",
                f"\u00b1 {std_dev:.2f} J" if n > 1 else "n/a"
            ])
            
    print_table(headers, rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    records = load_all_records(args.paths)
    run_micro_analysis(records)

if __name__ == "__main__":
    main()