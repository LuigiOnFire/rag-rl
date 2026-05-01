import argparse
import logging
import os
import random
from typing import Dict

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a router classifier.")
    parser.add_argument(
        "--input-csv",
        default="data/oracle/oracle_training_data.csv",
        help="Path to oracle CSV data.",
    )
    parser.add_argument(
        "--model-name",
        default="microsoft/deberta-v3-base",
        help="HF model name.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/router",
        help="Output directory.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"CSV not found: {args.input_csv}")

    set_seed(args.seed)

    dataset = load_dataset("csv", data_files=args.input_csv)["train"]
    dataset = dataset.map(lambda x: {"label": int(x["optimal_trajectory_id"])})

    label_values = sorted(set(dataset["label"]))
    num_labels = max(label_values) + 1 if label_values else 2
    id2label = {i: f"traj_{i}" for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    split_kwargs = {"test_size": args.test_size, "seed": args.seed}
    if "label" in dataset.column_names:
        split_kwargs["stratify_by_column"] = "label"
    dataset_split = dataset.train_test_split(**split_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["question"], truncation=True, max_length=args.max_length)

    tokenized = dataset_split.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in ("input_ids", "attention_mask", "label")]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logging.info("Training complete. Best checkpoint saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
