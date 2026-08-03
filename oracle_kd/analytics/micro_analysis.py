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

def calculate_pearson_r(x: List[float], y: List[float]) -> float:
    """Computes the Pearson correlation coefficient between two variables."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den_x = sum((xi - mean_x) ** 2 for xi in x)
    den_y = sum((yi - mean_y) ** 2 for yi in y)
    
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / math.sqrt(den_x * den_y)

def run_micro_analysis(records: List[dict]):
    # action_metrics[source][action_name] = [step_data_dicts]
    action_metrics = defaultdict(lambda: defaultdict(list))
    # global_correlations[action_name] = [step_data_dicts] (combines across datasets for max sample size)
    global_correlations = defaultdict(list)
    
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
            
            step_data = {
                "cost": float(step.get("cost") or step.get("measured_joules") or 0.0),
                "duration": float(step.get("duration_seconds", 0.0)),
                "in_size": float(step.get("input_state_size", 0.0)),
                "out_size": float(step.get("output_state_size", 0.0))
            }
            
            action_metrics[source][action_name].append(step_data)
            global_correlations[action_name].append(step_data)

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
        for action, steps in sorted(action_metrics[source].items()):
            # Filter valid metrics dynamically to handle diverse action types
            costs = [s["cost"] for s in steps if s["cost"] > 0.0]
            durations = [s["duration"] for s in steps if s["duration"] > 0.0]
            in_sizes = [s["in_size"] for s in steps if s["in_size"] > 0.0]
            out_sizes = [s["out_size"] for s in steps if s["out_size"] > 0.0]
            
            n = len(costs)
            if n == 0:
                continue
                
            # Compute Means
            mean_cost = safe_mean(costs)
            mean_dur = safe_mean(durations)
            mean_in = safe_mean(in_sizes)
            mean_out = safe_mean(out_sizes)
            
            # Compute Std Devs for primary hardware constraints
            sd_cost = _std_dev(costs, mean_cost)
            sd_dur = _std_dev(durations, mean_dur)
            
            # Format row
            rows.append([
                source.upper(),
                action,
                str(n),
                f"{mean_cost:.1f} \u00b1 {sd_cost:.1f} J",
                f"{mean_dur:.2f} \u00b1 {sd_dur:.2f} s",
                f"{mean_in:.1f}" if in_sizes else "n/a",
                f"{mean_out:.1f}" if out_sizes else "n/a"
            ])
            
    print_table(headers, rows)

    print("\n==========================================================================================")
    print("SECTION 3.5: HARDWARE CORRELATION ANALYSIS (PEARSON r)")
    print("==========================================================================================")
    
    corr_headers = [
        "Action", "Global Samples (N)", 
        "In-Tks / Joules", "In-Tks / Latency", 
        "Out-Tks / Joules", "Out-Tks / Latency"
    ]
    corr_rows = []

    for action, steps in sorted(global_correlations.items()):
        # Extract paired variables to guarantee index alignment for correlation math
        in_eng_pairs = [(s["in_size"], s["cost"]) for s in steps if s["in_size"] > 0.0 and s["cost"] > 0.0]
        in_lat_pairs = [(s["in_size"], s["duration"]) for s in steps if s["in_size"] > 0.0 and s["duration"] > 0.0]
        out_eng_pairs = [(s["out_size"], s["cost"]) for s in steps if s["out_size"] > 0.0 and s["cost"] > 0.0]
        out_lat_pairs = [(s["out_size"], s["duration"]) for s in steps if s["out_size"] > 0.0 and s["duration"] > 0.0]
        
        n_base = len(in_eng_pairs)
        if n_base < 5:  # Skip actions with insufficient statistical sample size
            continue
            
        # Unzip pairs into parallel arrays
        r_in_eng = calculate_pearson_r([p[0] for p in in_eng_pairs], [p[1] for p in in_eng_pairs]) if in_eng_pairs else 0.0
        r_in_lat = calculate_pearson_r([p[0] for p in in_lat_pairs], [p[1] for p in in_lat_pairs]) if in_lat_pairs else 0.0
        r_out_eng = calculate_pearson_r([p[0] for p in out_eng_pairs], [p[1] for p in out_eng_pairs]) if out_eng_pairs else 0.0
        r_out_lat = calculate_pearson_r([p[0] for p in out_lat_pairs], [p[1] for p in out_lat_pairs]) if out_lat_pairs else 0.0
        
        corr_rows.append([
            action,
            str(n_base),
            f"{r_in_eng:.4f}",
            f"{r_in_lat:.4f}",
            f"{r_out_eng:.4f}" if out_eng_pairs else "n/a",
            f"{r_out_lat:.4f}" if out_lat_pairs else "n/a"
        ])
        
    print_table(corr_headers, corr_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    records = load_all_records(args.paths)
    run_micro_analysis(records)

if __name__ == "__main__":
    main()