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

def _std_dev(samples: List[float], mean_val: float) -> float:
    """Helper to calculate standard deviation natively."""
    n = len(samples)
    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in samples) / (n - 1)
        return math.sqrt(variance)
    return 0.0

def run_micro_analysis(records: List[dict]):
    # Nested dictionary pattern: action_metrics[source][action_name][metric] = [values]
    action_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Flatten sequential histories into standalone operation instances
    for record in records:
        source = record.get("source", "unknown")
        
        # Aggregate steps from all executed attempts to build a robust statistical profile
        steps_to_process = []
        if "attempts" in record:
            for attempt in record["attempts"]:
                steps_to_process.extend(attempt.get("history", []))
        # Fallback for older JSON schema traces
        elif "history" in record:
            steps_to_process.extend(record.get("history", []))
                
        for step in steps_to_process:
            action_name = step.get("action_name") or f"action_{step.get('action_id')}"
            
            # Extract available metrics
            cost = float(step.get("cost") or step.get("measured_joules") or 0.0)
            duration = float(step.get("duration_seconds", 0.0))
            in_size = float(step.get("input_state_size", 0.0))
            out_size = float(step.get("output_state_size", 0.0))
            
            # Only append if valid values exist (handles distinct action types like RETRIEVE vs GEN)
            if cost > 0.0:
                action_metrics[source][action_name]["cost"].append(cost)
            if duration > 0.0:
                action_metrics[source][action_name]["duration"].append(duration)
            if in_size > 0.0:
                action_metrics[source][action_name]["in_size"].append(in_size)
            if out_size > 0.0:
                action_metrics[source][action_name]["out_size"].append(out_size)

    print("==========================================================================================")
    print("SECTION 3.4: MICRO-ACTION HARDWARE FOOTPRINT & LATENCY PROFILE")
    print("==========================================================================================")
    
    headers = [
        "Dataset", "Action", "Samples (N)", 
        "Energy (Joules)", "Latency (sec)", 
        "Context In (Tks)", "Gen Out (Tks)"
    ]
    rows = []
    
    # Sort by dataset first, then by action name to create clean visual groupings
    for source in sorted(action_metrics.keys()):
        for action, metrics in sorted(action_metrics[source].items()):
            n = len(metrics["cost"])
            if n == 0:
                continue
                
            # Compute Means
            mean_cost = safe_mean(metrics["cost"])
            mean_dur = safe_mean(metrics["duration"])
            mean_in = safe_mean(metrics["in_size"])
            mean_out = safe_mean(metrics["out_size"])
            
            # Compute Std Devs for primary hardware constraints
            sd_cost = _std_dev(metrics["cost"], mean_cost)
            sd_dur = _std_dev(metrics["duration"], mean_dur)
            
            # Format row
            rows.append([
                source.upper(),
                action,
                str(n),
                f"{mean_cost:.1f} \u00b1 {sd_cost:.1f} J",
                f"{mean_dur:.2f} \u00b1 {sd_dur:.2f} s",
                f"{mean_in:.1f}" if metrics["in_size"] else "n/a",
                f"{mean_out:.1f}" if metrics["out_size"] else "n/a"
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