import argparse
import os

import pandas as pd
from datasets import load_dataset

DEFAULT_SAMPLE_COUNT = 1000
DEFAULT_OUTPUT_CSV = "data/oracle/origin_training_data.csv"


def _take_questions(dataset, sample_count: int, label: int, seed: int, text_column: str = "question"):
    if len(dataset) < sample_count:
        raise ValueError(f"Requested {sample_count} rows but dataset only has {len(dataset)} available.")

    sampled = dataset.shuffle(seed=seed).select(range(sample_count))
    rows = []
    for row in sampled:
        question = row.get(text_column, "")
        if question:
            rows.append({"question": question, "dataset_label": label})
    if len(rows) != sample_count:
        raise ValueError(f"Expected {sample_count} rows after filtering, got {len(rows)}.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 2-way origin classification CSV from SQuAD and HotpotQA.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Path to the unified output CSV.")
    parser.add_argument("--num-per-dataset", type=int, default=DEFAULT_SAMPLE_COUNT, help="Rows to sample from each dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for reproducible sampling.")
    args = parser.parse_args()

    squad = load_dataset("squad", split="train")
    hotpot = load_dataset("hotpot_qa", "fullwiki", split="train", trust_remote_code=True)

    squad_rows = _take_questions(squad, args.num_per_dataset, label=0, seed=args.seed)
    hotpot_rows = _take_questions(hotpot, args.num_per_dataset, label=1, seed=args.seed)

    df = pd.DataFrame(squad_rows + hotpot_rows, columns=["question", "dataset_label"])
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(df)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
