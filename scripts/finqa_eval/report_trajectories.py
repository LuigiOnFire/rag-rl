import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence

# Updated for all 9 trajectories (0 through 8)
# Minimal: Direct SLM/LLM, Key+SLM
SIMPLE_TRAJECTORY_IDS = {0, 1, 2}
# Intensive: Vec+LLM, Reason+Vec, Search-Iterate, Decompositions
COMPLEX_TRAJECTORY_IDS = {3, 4, 5, 6, 7, 8}
SINGLE_HOP_LOOKUP_IDS = {1, 2}


def _normalize_dataset_name(name: Optional[str]) -> str:
    if not name:
        return "unknown"

    normalized = str(name).strip().lower()
    if normalized in {"finqa", "fin_qa"}:
        return "finqa"
    if normalized in {"hotpot", "hotpotqa", "hotpot_qa"}:
        return "hotpotqa"
    if normalized in {"squad", "squad_v2", "squadv2"}:
        return "squad"
    if normalized in {"nq", "nq_open", "natural_questions", "naturalquestions"}:
        return "nq"
    return normalized


def _infer_dataset_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "finqa" in stem or "fin_qa" in stem:
        return "finqa"
    if "hotpot" in stem:
        return "hotpotqa"
    if "squad" in stem:
        return "squad"
    if "nq" in stem:
        return "nq"
    return "unknown"


def _safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _load_json_records(path: Path) -> List[dict]:
    with path.open(mode="r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ("records", "data", "rows", "items"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]

    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_jsonl_records(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(mode="r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_records(path: Path) -> List[dict]:
    if path.is_dir():
        records: List[dict] = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".json", ".jsonl"}:
                records.extend(_load_records(child))
        return records

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".jsonl":
        return _load_jsonl_records(path)

    if path.suffix.lower() == ".json":
        return _load_json_records(path)

    if path.suffix.lower() == ".csv":
        raise ValueError(
            f"{path} is a CSV file. This reporter expects trajectory JSON/JSONL files "
            "with per-route attempt records to compute energy, accuracy, and misrouting penalties."
        )

    raise ValueError(f"Unsupported input format: {path.suffix}")


def _normalize_record(record: dict, source_hint: str) -> dict:
    normalized = dict(record)
    normalized["source"] = _normalize_dataset_name(record.get("source") or source_hint)
    return normalized


def _load_all_records(inputs: Sequence[Path]) -> List[dict]:
    records: List[dict] = []
    for path in inputs:
        source_hint = _infer_dataset_from_path(path)
        for record in _load_records(path):
            records.append(_normalize_record(record, source_hint))
    return records


def _get_attempt(record: dict, trajectory_id: int) -> Optional[dict]:
    for attempt in record.get("attempts", []):
        if int(attempt.get("trajectory_id", -1)) == trajectory_id:
            return attempt
    return None


def _cheapest_successful_attempt(record: dict, trajectory_ids: Iterable[int]) -> Optional[dict]:
    best_attempt: Optional[dict] = None
    best_cost: Optional[float] = None

    for trajectory_id in trajectory_ids:
        attempt = _get_attempt(record, trajectory_id)
        if not attempt or not bool(attempt.get("is_correct")):
            continue

        cost = float(attempt.get("measured_joules", 0.0))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_attempt = attempt

    return best_attempt


def _record_tier(record: dict) -> str:
    if not bool(record.get("is_correct")):
        return "guardrail"

    trajectory_id = int(record.get("optimal_trajectory_id", -1))
    if trajectory_id in SIMPLE_TRAJECTORY_IDS:
        return "minimal"
    if trajectory_id in COMPLEX_TRAJECTORY_IDS:
        return "intensive"
    return "guardrail"


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))

    print(render_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render_row(row))


def _compute_stats(records: Sequence[dict]) -> Dict[str, object]:
    dataset_records: Dict[str, List[dict]] = defaultdict(list)
    trajectory_names: Dict[int, str] = {}

    for record in records:
        dataset_records[record["source"]].append(record)
        for attempt in record.get("attempts", []):
            trajectory_id = int(attempt.get("trajectory_id", -1))
            trajectory_names.setdefault(trajectory_id, str(attempt.get("trajectory_name", f"traj_{trajectory_id}")))

    tier_costs: Dict[str, List[float]] = defaultdict(list)
    misrouting_penalties: Dict[str, List[float]] = defaultdict(list)
    index_scale_samples: Dict[str, List[float]] = defaultdict(list)
    trajectory_joules: Dict[int, List[float]] = defaultdict(list)
    accuracy_counts: Dict[str, Dict[int, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for dataset_name, dataset_rows in dataset_records.items():
        for record in dataset_rows:
            tier_costs[_record_tier(record)].append(float(record.get("joules_spent", 0.0)))

            simple_attempt = _cheapest_successful_attempt(record, SIMPLE_TRAJECTORY_IDS)
            complex_attempt = _cheapest_successful_attempt(record, COMPLEX_TRAJECTORY_IDS)
            
            # Record misrouting penalty across datasets (including FinQA)
            if simple_attempt is not None and complex_attempt is not None:
                penalty = float(complex_attempt.get("measured_joules", 0.0)) - float(simple_attempt.get("measured_joules", 0.0))
                misrouting_penalties[dataset_name].append(penalty)

            for attempt in record.get("attempts", []):
                trajectory_id = int(attempt.get("trajectory_id", -1))
                if bool(attempt.get("is_correct")):
                    accuracy_counts[dataset_name][trajectory_id]["correct"] += 1
                accuracy_counts[dataset_name][trajectory_id]["total"] += 1

            if record.get("is_correct") and int(record.get("optimal_trajectory_id", -1)) in SINGLE_HOP_LOOKUP_IDS:
                index_scale_samples[dataset_name].append(float(record.get("joules_spent", 0.0)))

            try:
                chosen = int(record.get("optimal_trajectory_id", -1))
                if chosen >= 0:
                    trajectory_joules[chosen].append(float(record.get("joules_spent", 0.0)))
            except Exception:
                pass

    return {
        "tier_costs": tier_costs,
        "misrouting_penalties": misrouting_penalties,
        "index_scale_samples": index_scale_samples,
        "trajectory_joules": trajectory_joules,
        "accuracy_counts": accuracy_counts,
        "trajectory_names": trajectory_names,
    }


def _print_stats(records: Sequence[dict], source_label: str) -> None:
    stats = _compute_stats(records)
    misrouting_penalties = stats["misrouting_penalties"]
    index_scale_samples = stats["index_scale_samples"]
    accuracy_counts = stats["accuracy_counts"]
    trajectory_names = stats["trajectory_names"]

    dataset_sizes = Counter(record["source"] for record in records)
    total_rows = len(records)

    print(f"Loaded {total_rows} trajectory records from {source_label}\n")
    print("Datasets:")
    for dataset_name, count in sorted(dataset_sizes.items()):
        print(f"  - {dataset_name}: {count}")

    print("\n==================================================")
    print("1. ENERGY COST PER TRAJECTORY (Joules)")
    print("==================================================")
    traj_rows = []
    trajectory_joules = stats.get("trajectory_joules", {})
    traj_ids = sorted(set(list(trajectory_joules.keys()) + list(trajectory_names.keys())))
    for traj_id in traj_ids:
        samples = trajectory_joules.get(traj_id, [])
        name = trajectory_names.get(traj_id, f"traj_{traj_id}")
        traj_rows.append([f"{traj_id}: {name}", str(len(samples)), f"{_safe_mean(samples):.2f}" if samples else "n/a"])
    _print_table(["Trajectory", "Samples", "Avg Joules"], traj_rows)

    print("\n==================================================")
    print("2. COST OF MISROUTING (Minimal -> Complex Pipeline)")
    print("==================================================")
    for ds_name, penalties in sorted(misrouting_penalties.items()):
        print(f"[{ds_name.upper()}] Wasted Energy: {_safe_mean(penalties):.2f} J across {len(penalties)} queries")
    if not misrouting_penalties:
        print("No examples had both a successful minimal route and a successful complex route.")

    print("\n==================================================")
    print("3. FINQA & INDEX SCALE CHARACTERISTICS")
    print("==================================================")
    if "finqa" in index_scale_samples and index_scale_samples["finqa"]:
        finqa_mean = _safe_mean(index_scale_samples["finqa"])
        print(f"FinQA Ephemeral Micro-Retriever Mean (Routes 1-2): {finqa_mean:.2f} J")
    
    squad_values = index_scale_samples.get("squad", [])
    hotpot_values = index_scale_samples.get("hotpotqa", [])
    nq_values = index_scale_samples.get("nq", [])

    if squad_values:
        print(f"SQuAD DPR Mean (Routes 1-2): {_safe_mean(squad_values):.2f} J")
    if nq_values:
        print(f"NQ DPR Mean (Routes 1-2): {_safe_mean(nq_values):.2f} J")
    if squad_values and hotpot_values:
        squad_mean = _safe_mean(squad_values)
        hotpot_mean = _safe_mean(hotpot_values)
        delta = squad_mean - hotpot_mean
        pct = (delta / hotpot_mean) * 100 if hotpot_mean else float("nan")
        print(f"HotpotQA FullWiki Mean (Routes 1-2): {hotpot_mean:.2f} J")
        print(f"Index scale tax (SQuAD vs HotpotQA): {delta:.2f} J ({pct:.2f}% over HotpotQA)")

    print("\n==================================================")
    print("4. ACCURACY CEILINGS BY TRAJECTORY AND DATASET")
    print("==================================================")
    dataset_order = [dataset for dataset in ("finqa", "hotpotqa", "squad", "nq") if dataset in accuracy_counts]
    all_trajectory_ids = sorted({trajectory_id for dataset in dataset_order for trajectory_id in accuracy_counts[dataset].keys()})
    dataset_labels = {"finqa": "FinQA", "hotpotqa": "HotpotQA", "squad": "SQuAD", "nq": "NatQuestions"}
    
    header = ["Trajectory"] + [dataset_labels.get(dataset, dataset.title()) for dataset in dataset_order] + ["Overall"]
    
    rows = []
    for trajectory_id in all_trajectory_ids:
        row = [f"{trajectory_id}: {trajectory_names.get(trajectory_id, f'traj_{trajectory_id}')}"]
        overall_correct = 0
        overall_total = 0
        for dataset in dataset_order:
            counts = accuracy_counts[dataset][trajectory_id]
            correct = int(counts.get("correct", 0))
            total = int(counts.get("total", 0))
            overall_correct += correct
            overall_total += total
            row.append(_format_pct((correct / total) * 100) if total else "n/a")
        row.append(_format_pct((overall_correct / overall_total) * 100) if overall_total else "n/a")
        rows.append(row)

    _print_table(header, rows)


def _default_inputs() -> List[Path]:
    return [Path("data/oracle/")]

def main():
    parser = argparse.ArgumentParser(description="Analyze trajectory usage, joules, and accuracy ceilings.")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Trajectory JSON/JSONL file(s) or directories to analyze.",
    )
    args = parser.parse_args()

    try:
        inputs = [Path(path) for path in (args.paths or _default_inputs())]
        records = _load_all_records(inputs)
    except FileNotFoundError as error:
        print(f"Error: Could not find the file '{error.args[0]}'.")
        sys.exit(1)
    except ValueError as error:
        print(f"Error: {error}")
        sys.exit(1)

    if not records:
        print("Error: No trajectory records found in the specified path.")
        sys.exit(1)

    source_label = ", ".join(str(path) for path in inputs)
    _print_stats(records, source_label)

if __name__ == "__main__":
    main()