import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = SCRIPT_DIR / "data" / "training"


def get_source_name(file_path: Path) -> str:
    lowered = file_path.name.lower()
    if "hotpot" in lowered:
        return "hotpotqa"
    if "squad" in lowered:
        return "squad"
    raise ValueError("Unable to infer source from filename. Please ensure it contains 'hotpot' or 'squad'.")


def find_companion_file(input_path: Path) -> Path:
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        candidates = [
            input_path.with_name(input_path.name.replace("oracle_training_data_", "oracle_trajectory_history_")).with_suffix(".jsonl"),
            input_path.with_name(input_path.name.replace("training_data_", "trajectory_history_")).with_suffix(".jsonl"),
            input_path.with_name(input_path.name.replace("oracle_trajectory_history_", "oracle_training_data_")).with_suffix(".json"),
            input_path.with_name(input_path.name.replace("trajectory_history_", "training_data_")).with_suffix(".json"),
        ]
    elif suffix in {".json", ".jsonl"}:
        candidates = [
            input_path.with_name(input_path.name.replace("oracle_trajectory_history_", "oracle_training_data_")).with_suffix(".csv"),
            input_path.with_name(input_path.name.replace("trajectory_history_", "training_data_")).with_suffix(".csv"),
        ]
    else:
        raise ValueError("Input file must end in .csv, .json, or .jsonl")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find the companion file for {input_path}. Tried: {', '.join(str(p) for p in candidates)}"
    )


def create_sliced_file(file_path: Path, n: int) -> Path:
    """
    Reads the first N data entries from the file and writes them to a temporary file,
    preserving the original row/element ordering.
    """
    tmp_path = file_path.with_name(f"tmp_sliced_{n}_{file_path.name}")
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        with open(file_path, "r", encoding="utf-8") as f_in, open(tmp_path, "w", encoding="utf-8") as f_out:
            # Always copy the header first
            header = f_in.readline()
            f_out.write(header)
            for _ in range(n):
                line = f_in.readline()
                if not line:
                    break
                f_out.write(line)

    elif suffix == ".jsonl":
        with open(file_path, "r", encoding="utf-8") as f_in, open(tmp_path, "w", encoding="utf-8") as f_out:
            for _ in range(n):
                line = f_in.readline()
                if not line:
                    break
                f_out.write(line)

    elif suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Handle cases where the JSON file is a top-level list
        if isinstance(data, list):
            sliced_data = data[:n]
        else:
            raise TypeError(f"Expected top-level JSON list array in {file_path}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(sliced_data, f, indent=4)

    return tmp_path


def run_script(script_name: str, input_file: Path) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script_name), str(input_file)],
        cwd=str(REPO_ROOT),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Oracle training inputs from either a CSV or JSON/JSONL and run both merge steps."
    )
    parser.add_argument("input_file", help="Path to either the query CSV or trajectory JSON/JSONL.")
    parser.add_argument(
        "-n", "--first-n",
        type=int,
        default=None,
        help="Only process the first N sequential entries from both input and companion files."
    )
    args = parser.parse_args()

    input_path = Path(args.input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    companion_path = find_companion_file(input_path)

    # Track temporary files to ensure clean environment exit
    temp_files = []

    try:
        if args.first_n is not None:
            if args.first_n <= 0:
                raise ValueError("The value of --first-n must be a positive integer.")
            
            print(f"Slicing datasets to the first {args.first_n} items while preserving strict ordering...")
            input_path = create_sliced_file(input_path, args.first_n)
            temp_files.append(input_path)
            
            companion_path = create_sliced_file(companion_path, args.first_n)
            temp_files.append(companion_path)

        # Route to downstream merge operations
        if input_path.suffix.lower() == ".csv":
            run_script("merge_query_csv.py", input_path)
            run_script("merge_traj_json.py", companion_path)
        else:
            run_script("merge_traj_json.py", input_path)
            run_script("merge_query_csv.py", companion_path)

    finally:
        # Guarantee removal of temporary files even if execution fails midway
        if temp_files:
            print("Cleaning up temporary sliced artifacts...")
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()


if __name__ == "__main__":
    main()