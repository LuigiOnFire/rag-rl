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
    trajectory_failures = Counter()

    for row in rows:
        # Safely parse ID and boolean
        traj_id = int(float(row['optimal_trajectory_id']))
        # Handle string booleans like 'True', 'true', '1'
        is_correct = str(row['is_correct']).strip().lower() in ['true', '1', 't', 'yes']

        trajectory_counts[traj_id] += 1
        if not is_correct:
            trajectory_failures[traj_id] += 1

    print("==================================================")
    print("1. OVERALL TRAJECTORY USAGE (Raw Counts & %)")
    print("==================================================")
    print(f"{'ID':<5} | {'Count':<7} | {'Percentage'}")
    print("-" * 35)
    
    # Sort by count descending
    for traj_id, count in trajectory_counts.most_common():
        pct = (count / total_rows) * 100
        print(f"{traj_id:<5} | {count:<7} | {pct:.2f}%")

    print("\n==================================================")
    print("2. FAILURE ANALYSIS: HIGHEST ID (Heaviest Trajectory)")
    print("==================================================")
    highest_id = max(trajectory_counts.keys())
    total_highest = trajectory_counts[highest_id]
    failed_highest = trajectory_failures[highest_id]

    fail_rate_highest = (failed_highest / total_highest) * 100
    print(f"Trajectory ID: {highest_id}")
    print(f"Total Times Assigned: {total_highest}")
    print(f"Total Failures: {failed_highest}")
    print(f"Failure Rate: {fail_rate_highest:.2f}%")
    
    if fail_rate_highest == 100.0:
        print("⚠️  WARNING: 100% failure rate detected. The quarantine bin is working perfectly!")

    print("\n==================================================")
    print("3. FAILURE ANALYSIS: MOST FREQUENT TRAJECTORY")
    print("==================================================")
    most_frequent_id = trajectory_counts.most_common(1)[0][0]
    total_freq = trajectory_counts[most_frequent_id]
    failed_freq = trajectory_failures[most_frequent_id]

    fail_rate_freq = (failed_freq / total_freq) * 100
    print(f"Trajectory ID: {most_frequent_id}")
    print(f"Total Times Assigned: {total_freq}")
    print(f"Total Failures: {failed_freq}")
    print(f"Failure Rate: {fail_rate_freq:.2f}%")

if __name__ == "__main__":
    main()