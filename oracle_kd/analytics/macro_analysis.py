# analytics/macro_analysis.py
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict
import math
from typing import Optional, Iterable, List

sys.path.append(str(Path(__file__).resolve().parent))

from loader import (
    load_all_records, safe_mean, print_table,
    SIMPLE_TRAJECTORY_IDS, COMPLEX_TRAJECTORY_IDS, SINGLE_HOP_LOOKUP_IDS
)

def _get_attempt(record: dict, trajectory_id: int) -> Optional[dict]:
    for attempt in record.get("attempts", []):
        if int(attempt.get("trajectory_id", -1)) == trajectory_id:
            return attempt
    return None

def _cheapest_successful_attempt(record: dict, trajectory_ids: Iterable[int]) -> Optional[dict]:
    best_attempt, best_cost = None, None
    for trajectory_id in trajectory_ids:
        attempt = _get_attempt(record, trajectory_id)
        if not attempt or not bool(attempt.get("is_correct")):
            continue
        cost = float(attempt.get("measured_joules", 0.0))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_attempt = attempt
    return best_attempt

def run_macro_analysis(records: List[dict]):
    dataset_records = defaultdict(list)
    trajectory_names = {}
    
    # Stratify records by dataset source
    for record in records:
        dataset_records[record["source"]].append(record)
        for attempt in record.get("attempts", []):
            t_id = int(attempt.get("trajectory_id", -1))
            trajectory_names.setdefault(t_id, str(attempt.get("trajectory_name", f"traj_{t_id}")))

    print("==================================================")
    print("SECTION 3.1: PARETO MATRIX DATA (ACCURACY VS JOULES)")
    print("==================================================")
    
    datasets = [d for d in ("hotpotqa", "squad", "nq") if d in dataset_records]
    headers = ["Trajectory ID & Name"]
    for d in datasets:
        headers.extend([f"{d.upper()} Acc", f"{d.upper()} Avg Joules"])
        
    matrix_rows = []
    for t_id in sorted(trajectory_names.keys()):
        row = [f"{t_id}: {trajectory_names[t_id]}"]
        for d in datasets:
            t_correct = 0
            t_total = 0
            joules_samples = []
            
            for record in dataset_records[d]:
                attempt = _get_attempt(record, t_id)
                if attempt:
                    t_total += 1
                    if bool(attempt.get("is_correct")):
                        t_correct += 1
                    joules_samples.append(float(attempt.get("measured_joules", 0.0)))
            
            acc_str = f"{(t_correct / t_total * 100):.2f}%" if t_total else "n/a"
            joule_str = f"{safe_mean(joules_samples):.2f}J" if joules_samples else "n/a"
            row.extend([acc_str, joule_str])
        matrix_rows.append(row)
        
    print_table(headers, matrix_rows)

    print("\n==================================================")
    print("SECTION 3.2: STEP-LEVEL LATENCY & STATE SIZES")
    print("==================================================")
    
    for d in datasets:
        print(f"\nDataset: {d.upper()}")
        stats = {
            "Minimal": {"durations": [], "input_sizes": [], "steps": []},
            "Intensive": {"durations": [], "input_sizes": [], "steps": []}
        }
        
        for record in dataset_records[d]:
            for attempt in record.get("attempts", []):
                t_id = int(attempt.get("trajectory_id", -1))
                tier = "Minimal" if t_id in SIMPLE_TRAJECTORY_IDS else "Intensive"
                
                # Extract history metrics
                history = attempt.get("history", [])
                stats[tier]["steps"].append(len(history))
                
                # Aggregate across all steps in the trajectory
                total_duration = sum(step.get("duration_seconds", 0) for step in history)
                avg_input_size = safe_mean([step.get("input_state_size", 0) for step in history])
                
                if total_duration > 0:
                    stats[tier]["durations"].append(total_duration)
                if avg_input_size > 0:
                    stats[tier]["input_sizes"].append(avg_input_size)
                    
        for tier in ("Minimal", "Intensive"):
            mean_dur = safe_mean(stats[tier]["durations"])
            mean_in = safe_mean(stats[tier]["input_sizes"])
            mean_steps = safe_mean(stats[tier]["steps"])
            
            print(f"  - {tier.ljust(9)} Trajectories:")
            print(f"      Avg Steps/Query  : {mean_steps:.1f}")
            print(f"      Avg Duration     : {mean_dur:.2f} sec")
            print(f"      Avg Context Size : {mean_in:.1f} tokens")

    print("\n==================================================")
    print("SECTION 3.3: TRAJECTORY EFFICIENCY & TRADEOFFS")
    print("==================================================")
    
    # 1. Oracle Choice Distribution
    print("1. Oracle Routing Distribution (Complete Workload Breakdown):")
    for d in datasets:
        oracle_counts = Counter()
        for record in dataset_records[d]:
            # Step 1: Verify if ANY trajectory managed to solve the query
            has_successful_path = any(
                bool(attempt.get("is_correct")) 
                for attempt in record.get("attempts", [])
            )
            
            if not has_successful_path:
                oracle_counts["Unviable"] += 1
                continue
                
            # Step 2: If solvable, map the oracle's choice to the correct tier
            opt_id = int(record.get("optimal_trajectory_id", -1))
            if opt_id >= 0:
                tier = "Minimal" if opt_id in SIMPLE_TRAJECTORY_IDS else "Intensive"
                oracle_counts[tier] += 1
            else:
                # Fallback safeguard for edge cases
                oracle_counts["Unviable"] += 1

        total = sum(oracle_counts.values())
        print(f"  - {d.upper()}:")
        for tier in ("Minimal", "Intensive", "Unviable"):
            count = oracle_counts[tier]
            pct = (count / total * 100) if total else 0.0
            print(f"    * {tier.ljust(9)} Tier Selected: {count} ({pct:.1f}%)")
            
    # 2. True Cost of Misrouting
    print("\n2. Cost of Misrouting Penalty (Unnecessary Escalation):")
    for d in datasets:
        penalties_joules = []
        penalties_time = []
        
        for record in dataset_records[d]:
            simple_att = _cheapest_successful_attempt(record, SIMPLE_TRAJECTORY_IDS)
            complex_att = _cheapest_successful_attempt(record, COMPLEX_TRAJECTORY_IDS)
            
            if simple_att and complex_att:
                # Joule Penalty
                j_diff = float(complex_att.get("measured_joules", 0.0)) - float(simple_att.get("measured_joules", 0.0))
                penalties_joules.append(j_diff)
                
                # Time Penalty (using the new duration attribute)
                t_diff = float(complex_att.get("duration_seconds", 0.0)) - float(simple_att.get("duration_seconds", 0.0))
                penalties_time.append(t_diff)
                
        if penalties_joules:
            print(f"  - {d.upper()}:")
            print(f"      Mean Wasted Energy: {safe_mean(penalties_joules):.2f} J")
            print(f"      Mean Wasted Time  : {safe_mean(penalties_time):.2f} sec over {len(penalties_joules)} misrouted samples.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    records = load_all_records(args.paths)
    run_macro_analysis(records)

if __name__ == "__main__":
    main()