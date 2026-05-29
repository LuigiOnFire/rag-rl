import argparse
import csv
import logging
import os
import random
from typing import Dict, Tuple
import wandb
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets import load_dataset, ClassLabel
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

USE_WANDB      = True          # Set False to disable; falls back to log-only
WANDB_PROJECT  = "thrifty-rag-router"
WANDB_RUN_NAME = "training-run"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a router classifier.")
    parser.add_argument(
        "--input-csv",
        default="oracle_kd/data/training/master_queries.csv",
        help="Path to oracle CSV data.",
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        dest="model_name",
        default="microsoft/deberta-v3-base",
        help="HF model name.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/router",
        help="Output directory.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    parser.add_argument("--epochs", "--num-train-epochs", "--num_train_epochs", dest="epochs", type=int, default=50)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", dest="warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-type", "--lr_scheduler_type", dest="lr_scheduler_type", default="linear")
    parser.add_argument("--hidden-dropout-prob", "--hidden_dropout_prob", dest="hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--attention-probs-dropout-prob",
        "--attention_probs_dropout_prob",
        dest="attention_probs_dropout_prob",
        type=float,
        default=0.1,
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--routing-strategy",
        choices=["bucket", "trajectory", "origin"],
        required=True,
        help="Routing target type to train: 3-way bucket, exact trajectory ID, or 2-way dataset origin.",
    )
    parser.add_argument(
        "--pred-output",
        default=None,
        help="Optional CSV path for per-example predictions (default: <output_dir>/eval_predictions.csv)",
    )
    parser.add_argument(
        "--run-title",
        default=None,
        help="Optional run title. If omitted, a unique eval_router_* title is generated.",
    )
    return parser.parse_args()


def _build_run_title(args: argparse.Namespace) -> str:
    if args.run_title:
        return args.run_title
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"eval_router_{args.routing_strategy}_{ts}"


def compute_metrics(eval_pred) -> Dict[str, float]:
    # Support both (logits, labels) tuple and EvalPrediction objects
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": f1}


def _parse_bool(value) -> bool:
    return str(value).strip().lower() in ["true", "1", "t", "yes"]


def _finalize_dataset(
    dataset,
    num_labels: int,
    id2label: Dict[int, str],
    test_size: float,
    seed: int,
):
    dataset = dataset.cast_column(
        "label",
        ClassLabel(num_classes=num_labels, names=[id2label[i] for i in range(num_labels)]),
    )

    split_kwargs = {"test_size": test_size, "seed": seed}
    if "label" in dataset.column_names:
        split_kwargs["stratify_by_column"] = "label"
    return dataset.train_test_split(**split_kwargs)


def prep_bucket_data(raw_dataset, test_size: float, seed: int) -> Tuple[object, torch.Tensor, int, Dict[int, str], Dict[str, int]]:
    def assign_bucket_label(row):
        is_correct = _parse_bool(row["is_correct"])
        traj_id = int(row["optimal_trajectory_id"])

        if not is_correct:
            return {"label": 2}  # impossible
        if traj_id in [0, 1, 2]:
            return {"label": 0}  # simple
        if traj_id in [3, 4, 5, 6, 7]:
            return {"label": 1}  # complex
        return {"label": -1}

    dataset = raw_dataset.map(assign_bucket_label)
    dataset = dataset.filter(lambda x: x["label"] != -1)

    labels = np.asarray(dataset["label"], dtype=np.int64)
    unique_labels = np.unique(labels)
    if unique_labels.size == 0:
        raise ValueError("No valid labels found for bucket strategy.")

    # 1. Log the RAW imbalance
    unique_raw, counts_raw = np.unique(labels, return_counts=True)
    logging.info(f"RAW DATASET DISTRIBUTION: {dict(zip([str(u) for u in unique_raw], counts_raw))}")

    num_labels = 3
    id2label = {0: "simple", 1: "complex", 2: "impossible"}
    label2id = {v: k for k, v in id2label.items()}
    dataset_split = _finalize_dataset(dataset, num_labels, id2label, test_size, seed)

    train_labels = np.asarray(dataset_split["train"]["label"], dtype=np.int64)
    unique_train_labels = np.unique(train_labels)
    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_train_labels,
        y=train_labels,
    )
    class_weights = np.ones(num_labels, dtype=np.float32)
    class_weights[unique_train_labels] = balanced_weights.astype(np.float32)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    return dataset_split, class_weights_tensor, num_labels, id2label, label2id

def prep_trajectory_data(raw_dataset, test_size: float, seed: int) -> Tuple[object, torch.Tensor, int, Dict[int, str], Dict[str, int]]:
    def assign_trajectory_label(row):
        is_correct = _parse_bool(row["is_correct"])
        traj_id = int(row["optimal_trajectory_id"])
        if is_correct and 0 <= traj_id <= 7:
            return {"label": traj_id}
        return {"label": -1}

    dataset = raw_dataset.map(assign_trajectory_label)
    dataset = dataset.filter(lambda x: x["label"] != -1)

    if len(dataset) == 0:
        raise ValueError("No valid examples found for trajectory strategy.")

    num_labels = 8
    id2label = {i: f"traj_{i}" for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}
    dataset_split = _finalize_dataset(dataset, num_labels, id2label, test_size, seed)

    train_labels = np.asarray(dataset_split["train"]["label"], dtype=np.int64)
    unique_train_labels = np.unique(train_labels)
    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_train_labels,
        y=train_labels,
    )
    class_weights = np.ones(num_labels, dtype=np.float32)
    class_weights[unique_train_labels] = balanced_weights.astype(np.float32)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    return dataset_split, class_weights_tensor, num_labels, id2label, label2id


def prep_origin_data(raw_dataset, test_size: float, seed: int) -> Tuple[object, torch.Tensor, int, Dict[int, str], Dict[str, int]]:
    def assign_origin_label(row):
        dataset_label = row.get("dataset_label")
        try:
            label = int(dataset_label)
        except (TypeError, ValueError):
            return {"label": -1}

        if label in (0, 1):
            return {"label": label}
        return {"label": -1}

    dataset = raw_dataset.map(assign_origin_label)
    dataset = dataset.filter(lambda x: x["label"] != -1)

    labels = np.asarray(dataset["label"], dtype=np.int64)
    unique_labels = np.unique(labels)
    if unique_labels.size == 0:
        raise ValueError("No valid labels found for origin strategy.")

    unique_raw, counts_raw = np.unique(labels, return_counts=True)
    logging.info(f"RAW DATASET DISTRIBUTION: {dict(zip([str(u) for u in unique_raw], counts_raw))}")

    num_labels = 2
    id2label = {0: "squad", 1: "hotpot"}
    label2id = {v: k for k, v in id2label.items()}
    dataset_split = _finalize_dataset(dataset, num_labels, id2label, test_size, seed)

    train_labels = np.asarray(dataset_split["train"]["label"], dtype=np.int64)
    unique_train_labels = np.unique(train_labels)
    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_train_labels,
        y=train_labels,
    )
    class_weights = np.ones(num_labels, dtype=np.float32)
    class_weights[unique_train_labels] = balanced_weights.astype(np.float32)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    return dataset_split, class_weights_tensor, num_labels, id2label, label2id


class WeightedTrainer(Trainer):
    def __init__(self, *args, sampler_class_weights=None, **kwargs):
        # Remove kwargs that older transformers versions may not accept
        for bad in ("tokenizer", "processing_class"):
            if bad in kwargs:
                kwargs.pop(bad)
        super().__init__(*args, **kwargs)
        self.sampler_class_weights = sampler_class_weights
        self.eval_export_dir = None
        self.eval_export_prefix = "eval_predictions"
        self.eval_questions = None
        self.eval_labels = None
        self.id2label = None

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None

        if "labels" in dataset.column_names:
            train_label_values = dataset["labels"]
        elif "label" in dataset.column_names:
            train_label_values = dataset["label"]
        else:
            raise ValueError("Training dataset has no 'labels' or 'label' column")

        if self.sampler_class_weights is None:
            return super()._get_train_sampler(train_dataset)

        if torch.is_tensor(self.sampler_class_weights):
            weights_arr = self.sampler_class_weights.detach().cpu().numpy()
        else:
            weights_arr = np.asarray(self.sampler_class_weights)

        sample_weights = weights_arr[np.asarray(train_label_values, dtype=np.int64)]
        return WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self.eval_export_dir and self.eval_questions is not None and self.eval_labels is not None and self.id2label is not None:
            ds = eval_dataset if eval_dataset is not None else self.eval_dataset
            predictions = self.predict(ds)
            pred_ids = np.argmax(predictions.predictions, axis=-1)

            epoch_val = self.state.epoch
            if epoch_val is None:
                epoch_str = "na"
            else:
                epoch_str = f"{float(epoch_val):.2f}".replace(".", "p")

            filename = f"{self.eval_export_prefix}_step_{self.state.global_step}_epoch_{epoch_str}.csv"
            output_path = os.path.join(self.eval_export_dir, filename)
            os.makedirs(self.eval_export_dir, exist_ok=True)

            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "question",
                        "true_id",
                        "true_label",
                        "pred_id",
                        "pred_label",
                        "correct",
                    ],
                )
                writer.writeheader()
                for question, true_id, pred_id in zip(self.eval_questions, self.eval_labels, pred_ids):
                    writer.writerow(
                        {
                            "question": question,
                            "true_id": int(true_id),
                            "true_label": self.id2label[int(true_id)],
                            "pred_id": int(pred_id),
                            "pred_label": self.id2label[int(pred_id)],
                            "correct": int(pred_id == true_id),
                        }
                    )

            logging.info("Saved eval predictions snapshot to %s", output_path)

        return metrics


def _balance_dataset_by_label(dataset, seed: int, label_column: str = "label"):
    if label_column not in dataset.column_names:
        raise ValueError(f"Dataset has no '{label_column}' column to balance.")

    labels = np.asarray(dataset[label_column], dtype=np.int64)
    unique_labels, counts = np.unique(labels, return_counts=True)
    if unique_labels.size == 0:
        raise ValueError("Cannot balance an empty dataset.")

    target_count = int(counts.min())
    rng = np.random.default_rng(seed)
    selected_indices = []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        if label_indices.size < target_count:
            raise ValueError(f"Label {label} has fewer examples than the balancing target.")
        rng.shuffle(label_indices)
        selected_indices.extend(label_indices[:target_count].tolist())

    rng.shuffle(selected_indices)
    return dataset.select(selected_indices)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    run_title = _build_run_title(args)

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"CSV not found: {args.input_csv}")

    set_seed(args.seed)

    raw_dataset = load_dataset("csv", data_files=args.input_csv)["train"]

    if args.routing_strategy == "origin":
        dataset_split, sampler_class_weights_tensor, num_labels, id2label, label2id = prep_origin_data(
            raw_dataset,
            args.test_size,
            args.seed,
        )
    elif args.routing_strategy == "bucket":
        dataset_split, sampler_class_weights_tensor, num_labels, id2label, label2id = prep_bucket_data(
            raw_dataset,
            args.test_size,
            args.seed,
        )
    else:
        dataset_split, sampler_class_weights_tensor, num_labels, id2label, label2id = prep_trajectory_data(
            raw_dataset,
            args.test_size,
            args.seed,
        )

    logging.info("Routing strategy: %s", args.routing_strategy)
    logging.info("Computed sampler weights: %s", sampler_class_weights_tensor.tolist())

    balanced_eval_dataset = _balance_dataset_by_label(dataset_split["test"], seed=args.seed, label_column="label")

    # 1. INIT WANDB FIRST (So you can grab the sweep variables)
    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name=run_title)
        sweep_cfg = wandb.config
        epochs = sweep_cfg.get("epochs", args.epochs)
        model_name = sweep_cfg.get("model_name", args.model_name)
        learning_rate = sweep_cfg.get("learning_rate", args.learning_rate)
        weight_decay = sweep_cfg.get("weight_decay", args.weight_decay)
        warmup_ratio = sweep_cfg.get("warmup_ratio", args.warmup_ratio)
        batch_size = sweep_cfg.get("batch_size", args.batch_size)
        lr_scheduler_type = sweep_cfg.get("lr_scheduler_type", args.lr_scheduler_type)
        hidden_dropout_prob = sweep_cfg.get("hidden_dropout_prob", args.hidden_dropout_prob)
        attention_probs_dropout_prob = sweep_cfg.get("attention_probs_dropout_prob", args.attention_probs_dropout_prob)
    else:
        # Fallback to args
        epochs = args.epochs
        model_name = args.model_name
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        warmup_ratio = args.warmup_ratio
        batch_size = args.batch_size
        lr_scheduler_type = args.lr_scheduler_type
        hidden_dropout_prob = args.hidden_dropout_prob
        attention_probs_dropout_prob = args.attention_probs_dropout_prob

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["question"], truncation=True, max_length=args.max_length)

    eval_questions = balanced_eval_dataset["question"]
    eval_labels = balanced_eval_dataset["label"]

    tokenized = dataset_split.map(tokenize, batched=True)
    tokenized_eval = balanced_eval_dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in ("input_ids", "attention_mask", "label")]
    )
    tokenized_eval = tokenized_eval.remove_columns(
        [c for c in tokenized_eval.column_names if c not in ("input_ids", "attention_mask", "label")]
    )

    # Rename the label column to `labels` to match Trainer expectations
    tokenized["train"] = tokenized["train"].rename_column("label", "labels")
    tokenized["test"] = tokenized["test"].rename_column("label", "labels")
    tokenized_eval = tokenized_eval.rename_column("label", "labels")

    print(f"Before AutoModel... num_labels is {num_labels}")

    # 2. CREATE MODEL SECOND (Injecting the sweep parameters directly into the layers)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=hidden_dropout_prob,              # INJECTED HERE
        attention_probs_dropout_prob=attention_probs_dropout_prob, # INJECTED HERE
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. SET TRAINING ARGS (Using the sweep variables)
    # Use a timestamped run directory (optionally include the W&B run id)
    if USE_WANDB and wandb.run is not None:
        run_output_dir = os.path.join(args.output_dir, f"{run_title}_{wandb.run.id}")
    else:
        run_output_dir = os.path.join(args.output_dir, run_title)
    os.makedirs(run_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=run_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        report_to=["wandb"] if USE_WANDB else [],
        run_name=run_title,
        max_grad_norm=1.0, 
        warmup_ratio=warmup_ratio, 
        lr_scheduler_type=lr_scheduler_type,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        sampler_class_weights=sampler_class_weights_tensor,
    )

    trainer.eval_export_dir = run_output_dir
    trainer.eval_export_prefix = f"eval_predictions_{run_title}"
    trainer.eval_questions = eval_questions
    trainer.eval_labels = eval_labels
    trainer.id2label = id2label

    trainer.train()
    trainer.save_model(run_output_dir)
    tokenizer.save_pretrained(run_output_dir)

    # Save per-example predictions on the eval set
    pred_output = args.pred_output or os.path.join(run_output_dir, f"eval_predictions_{run_title}.csv")
    predictions = trainer.predict(tokenized_eval)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    pred_dir = os.path.dirname(pred_output)
    if pred_dir:
        os.makedirs(pred_dir, exist_ok=True)
    with open(pred_output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "true_id",
                "true_label",
                "pred_id",
                "pred_label",
                "correct",
            ],
        )
        writer.writeheader()
        for question, true_id, pred_id in zip(eval_questions, eval_labels, pred_ids):
            writer.writerow(
                {
                    "question": question,
                    "true_id": int(true_id),
                    "true_label": id2label[int(true_id)],
                    "pred_id": int(pred_id),
                    "pred_label": id2label[int(pred_id)],
                    "correct": int(pred_id == true_id),
                }
            )

    logging.info("Training complete. Best checkpoint saved to %s", args.output_dir)
    logging.info("Run title: %s", run_title)
    logging.info("Eval predictions saved to %s", pred_output)


if __name__ == "__main__":
    main()
