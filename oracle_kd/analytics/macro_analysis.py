# analytics/macro_analysis.py
import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional, Iterable, List

sys.path.append(str(Path(__file__).resolve().parent))

from loader import load_all_records, safe_mean, print_table

# -----------------------------------------------------------------------------
# Calibrated Trajectory Taxonomy (7-Route Minimal Set)
# -----------------------------------------------------------------------------
# 0: direct_slm                     (~404 J)  - Parametric
# 1: key_then_slm                   (~761 J)  - Single-Hop Lexical
# 2: direct_llm                     (~475 J)  - Parametric
# 3: vec_then_llm                   (~1104 J) - Single-Hop Dense
# 4: reason_vec_llm                 (~1471 J) - Single-Hop Guided Dense
# 5: decompose_retrieve_reason      (~8895 J) - Multi-Hop Recursive Decomposition
# 6: heavy_decompose_retrieve_reason(~9056 J) - Heavy Recursive Decomposition

PARAMETRIC_TRAJECTORY_IDS = {0, 2}
SINGLE_HOP_TRAJECTORY_IDS = {1, 3, 4}
DECOMPOSITION_TRAJECTORY_IDS = {5, 6}

# Binary Escalation Tiers
MINIMAL_TRAJECTORY_IDS = {0, 1, 2, 3, 4}      # Non-decomposed baseline tier
INTENSIVE_TRAJECTORY_IDS = {5, 6}             # Recursive sub-query decomposition


def _get_attempt(record: dict, trajectory_id: int) -> Optional[dict]:
    for attempt in record.get("attempts", []):
        if int(attempt.get("trajectory_id", -1)) == trajectory_id:
            return attempt
    return None


def _get_attempt_duration(attempt: dict) -> float:
    dur = float(attempt.get("duration_seconds", 0.0))
    if dur > 0:
        return dur
    history = attempt.get("history", [])
    return sum(float(step.get("duration_seconds", 0.0)) for step in history)


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
        source = record.get("source", "hotpotqa")
        dataset_records[source].append(record)
        for attempt in record.get("attempts", []):
            t_id = int(attempt.get("trajectory_id", -1))
            trajectory_names.setdefault(t_id, str(attempt.get("trajectory_name", f"traj_{t_id}")))

    # =========================================================================
    # SECTION 3.1: PARETO MATRIX DATA (ACCURACY VS JOULES)
    # =========================================================================
    print("==================================================")
    print("SECTION 3.1: PARETO MATRIX DATA (ACCURACY VS JOULES)")
    print("==================================================")
    
    known_datasets = ["hotpotqa", "hotpot", "squad", "nq"]
    datasets = [d for d in known_datasets if d in dataset_records] + [
        d for d in dataset_records if d not in known_datasets
    ]
    
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

    # =========================================================================
    # SECTION 3.2: STEP-LEVEL LATENCY & STATE SIZES
    # =========================================================================
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
                tier = "Minimal" if t_id in MINIMAL_TRAJECTORY_IDS else "Intensive"
                
                history = attempt.get("history", [])
                stats[tier]["steps"].append(len(history))
                
                total_duration = _get_attempt_duration(attempt)
                avg_input_size = safe_mean([
                    float(step.get("input_state_size", 0))
                    for step in history if step.get("input_state_size") is not None
                ])
                
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

    # =========================================================================
    # SECTION 3.3: TRAJECTORY EFFICIENCY & TRADEOFFS
    # =========================================================================
    print("\n==================================================")
    print("SECTION 3.3: TRAJECTORY EFFICIENCY & TRADEOFFS")
    print("==================================================")
    
    # 1. Oracle Choice Distribution
    print("1. Oracle Routing Distribution (Complete Workload Breakdown):")
    for d in datasets:
        tier_counts = Counter()
        func_counts = Counter()
        traj_counts = Counter()
        
        total_samples = len(dataset_records[d])
        
        for record in dataset_records[d]:
            has_successful_path = any(
                bool(attempt.get("is_correct")) 
                for attempt in record.get("attempts", [])
            )
            
            if not has_successful_path:
                tier_counts["Unviable"] += 1
                func_counts["Unviable"] += 1
                traj_counts["Unviable"] += 1
                continue
                
            opt_id = int(record.get("optimal_trajectory_id", -1))
            if opt_id >= 0:
                traj_counts[opt_id] += 1
                
                # Binary escalation tier
                bin_tier = "Minimal" if opt_id in MINIMAL_TRAJECTORY_IDS else "Intensive"
                tier_counts[bin_tier] += 1
                
                # Functional tier
                if opt_id in PARAMETRIC_TRAJECTORY_IDS:
                    func_counts["Parametric"] += 1
                elif opt_id in SINGLE_HOP_TRAJECTORY_IDS:
                    func_counts["Single-Hop"] += 1
                elif opt_id in DECOMPOSITION_TRAJECTORY_IDS:
                    func_counts["Decomposition"] += 1
            else:
                tier_counts["Unviable"] += 1
                func_counts["Unviable"] += 1
                traj_counts["Unviable"] += 1

        print(f"  - {d.upper()}:")
        print("    [Binary Escalation Tiers]")
        for tier in ("Minimal", "Intensive", "Unviable"):
            count = tier_counts[tier]
            pct = (count / total_samples * 100) if total_samples else 0.0
            print(f"    * {tier.ljust(9)} Tier Selected: {count} ({pct:.1f}%)")
            
        print("\n    [Functional Capability Tiers]")
        for tier in ("Parametric", "Single-Hop", "Decomposition", "Unviable"):
            count = func_counts[tier]
            pct = (count / total_samples * 100) if total_samples else 0.0
            print(f"    * {tier.ljust(13)} Tier Selected: {count} ({pct:.1f}%)")

        print("\n    [Per-Trajectory Optimal Route Distribution]")
        for t_id in sorted(trajectory_names.keys()):
            name = trajectory_names[t_id]
            count = traj_counts[t_id]
            pct = (count / total_samples * 100) if total_samples else 0.0
            print(f"    * {t_id}: {name.ljust(32)}: {count} ({pct:.1f}%)")
        unv_count = traj_counts["Unviable"]
        unv_pct = (unv_count / total_samples * 100) if total_samples else 0.0
        print(f"    * Unviable (Failed all routes)     : {unv_count} ({unv_pct:.1f}%)")

    # 2. Cost of Misrouting Penalty
    print("\n2. Cost of Misrouting Penalty (Unnecessary Escalation to Decomposition):")
    for d in datasets:
        penalties_joules = []
        penalties_time = []
        
        for record in dataset_records[d]:
            # Compare cheapest successful minimal route vs cheapest successful decomposition route
            minimal_att = _cheapest_successful_attempt(record, MINIMAL_TRAJECTORY_IDS)
            intensive_att = _cheapest_successful_attempt(record, INTENSIVE_TRAJECTORY_IDS)
            
            if minimal_att and intensive_att:
                j_diff = float(intensive_att.get("measured_joules", 0.0)) - float(minimal_att.get("measured_joules", 0.0))
                penalties_joules.append(j_diff)
                
                t_diff = _get_attempt_duration(intensive_att) - _get_attempt_duration(minimal_att)
                penalties_time.append(t_diff)
                
        if penalties_joules:
            print(f"  - {d.upper()}:")
            print(f"      Mean Wasted Energy: {safe_mean(penalties_joules):.2f} J")
            print(f"      Mean Wasted Time  : {safe_mean(penalties_time):.2f} sec over {len(penalties_joules)} misrouted samples.")
        else:
            print(f"  - {d.upper()}: No overlapping correct samples between Minimal and Intensive tiers.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate macro-level RAG trajectory metrics.")
    parser.add_argument("paths", nargs="+", type=Path, help="Paths to oracle_trajectory_history.jsonl files.")
    args = parser.parse_args()
    records = load_all_records(args.paths)
    run_macro_analysis(records)


if __name__ == "__main__":
    main()