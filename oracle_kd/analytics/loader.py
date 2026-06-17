# analytics/loader.py
import json
from pathlib import Path
from typing import List, Optional, Sequence

SIMPLE_TRAJECTORY_IDS = {0, 1, 2}
COMPLEX_TRAJECTORY_IDS = {3, 4, 5, 6, 7}
SINGLE_HOP_LOOKUP_IDS = {1, 2}

def _normalize_dataset_name(name: Optional[str]) -> str:
    if not name:
        return "unknown"
    normalized = str(name).strip().lower()
    if normalized in {"hotpot", "hotpotqa", "hotpot_qa"}:
        return "hotpotqa"
    if normalized in {"squad", "squad_v2", "squadv2"}:
        return "squad"
    if normalized in {"nq", "nq_open", "natural_questions", "naturalquestions"}:
        return "nq"
    return normalized

def _infer_dataset_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if "hotpot" in stem:
        return "hotpotqa"
    if "squad" in stem:
        return "squad"
    if "nq" in stem:
        return "nq"
    return "unknown"

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
    raise ValueError(f"Unsupported input format: {path.suffix}")

def load_all_records(inputs: Sequence[Path]) -> List[dict]:
    records: List[dict] = []
    for path in inputs:
        source_hint = _infer_dataset_from_path(path)
        for record in _load_records(path):
            # Create fresh dict to avoid mutation side-effects
            normalized = dict(record)
            normalized["source"] = _normalize_dataset_name(record.get("source") or source_hint)
            records.append(normalized)
    return records

def safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")

def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    def render_row(row: Sequence[str]) -> str:
        return " | ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))
    print(render_row(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render_row(row))