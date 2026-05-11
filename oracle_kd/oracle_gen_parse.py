import csv
import argparse
import sys
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory usage and failure rates from the Oracle CSV.")
    parser.add_argument("file", help="Path to the generated Oracle CSV file")
    args = parser.parse_args()

    # Load the dataset
    try:
        with open(args.file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: Could not find the file '{args.file}'.")
        sys.exit(1)

    total_rows = len(rows)
    print(f"Successfully loaded: {args.file} ({total_rows} rows)\n")

    if total_rows == 0:
        print("Error: CSV is empty.")
        sys.exit(1)

    # Trackers
    trajectory_counts = Counter()
    failure_count = 0

    for row in rows:
        # Safely parse ID and boolean
        traj_id = int(float(row['optimal_trajectory_id']))
        # Handle string booleans like 'True', 'true', '1'
        is_correct = str(row['is_correct']).strip().lower() in ['true', '1', 't', 'yes']

        if not is_correct:
            failure_count += 1
        else:
            trajectory_counts[traj_id] += 1

    print("==================================================")
    print("1. OVERALL TRAJECTORY USAGE (Raw Counts & %)")
    print("==================================================")
    print(f"{'ID':<5} | {'Count':<7} | {'Percentage'}")
    print("-" * 35)
    
    # Sort by count descending
    for traj_id, count in trajectory_counts.most_common():
        pct = (count / total_rows) * 100
        print(f"{traj_id:<5} | {count:<7} | {pct:.2f}%")

    if failure_count > 0:
        fail_pct = (failure_count / total_rows) * 100
        print(f"FAIL  | {failure_count:<7} | {fail_pct:.2f}%")

    print("\n==================================================")
    print("2. FAILURE ANALYSIS: HIGHEST ID (Heaviest Trajectory)")
    print("==================================================")
    if trajectory_counts:
        highest_id = max(trajectory_counts.keys())
        total_highest = trajectory_counts[highest_id]
        print(f"Trajectory ID: {highest_id}")
        print(f"Total Times Assigned: {total_highest}")
    else:
        print("No successful trajectories recorded.")

    if failure_count > 0:
        fail_rate_overall = (failure_count / total_rows) * 100
        print(f"Total Failures: {failure_count}")
        print(f"Overall Failure Rate: {fail_rate_overall:.2f}%")
        if fail_rate_overall == 100.0:
            print("⚠️  WARNING: 100% failure rate detected. The quarantine bin is working perfectly!")

    print("\n==================================================")
    print("3. FAILURE ANALYSIS: MOST FREQUENT TRAJECTORY")
    print("==================================================")
    if trajectory_counts:
        most_frequent_id = trajectory_counts.most_common(1)[0][0]
        total_freq = trajectory_counts[most_frequent_id]

        print(f"Trajectory ID: {most_frequent_id}")
        print(f"Total Times Assigned: {total_freq}")
    else:
        print("No successful trajectories recorded.")

if __name__ == "__main__":
    main()