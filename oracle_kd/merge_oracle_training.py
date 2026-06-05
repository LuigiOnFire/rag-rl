import argparse
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
    args = parser.parse_args()

    input_path = Path(args.input_file).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    companion_path = find_companion_file(input_path)

    if input_path.suffix.lower() == ".csv":
        run_script("merge_query_csv.py", input_path)
        run_script("merge_traj_json.py", companion_path)
    else:
        run_script("merge_traj_json.py", input_path)
        run_script("merge_query_csv.py", companion_path)


if __name__ == "__main__":
    main()