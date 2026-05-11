import argparse
import csv
import logging
import os
import random
from typing import Dict
import wandb

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
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pred-output",
        default=None,
        help="Optional CSV path for per-example predictions (default: <output_dir>/eval_predictions.csv)",
    )
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

    def assign_routing_label(row):
            # 1. Parse the boolean and integer safely
            is_correct = str(row["is_correct"]).strip().lower() in ["true", "1", "t", "yes"]
            traj_id = int(row["optimal_trajectory_id"])
            
            # 2. Apply our routing philosophy (3-way buckets)
            if not is_correct:
                return {"label": 2}  # Bucket 2: Impossible / Abort
            if traj_id in [0, 1, 2]:
                return {"label": 0}  # Bucket 0: Simple Route
            if traj_id in [3, 4, 5, 6, 7]:
                return {"label": 1}  # Bucket 1: Complex Route
            return {"label": -1}

    # First, create the 'label' column using our logic
    dataset = dataset.map(assign_routing_label)
    
    # Then, filter out anything we marked as invalid (-1)
    dataset = dataset.filter(lambda x: x["label"] != -1)
    label_values = sorted(set(dataset["label"]))
    num_labels = max(label_values) + 1 if label_values else 2
    if num_labels == 3:
        id2label = {0: "simple", 1: "complex", 2: "impossible"}
    else:
        id2label = {i: f"traj_{i}" for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    dataset = dataset.cast_column("label", ClassLabel(num_classes=num_labels, names=list(id2label.values())))

    split_kwargs = {"test_size": args.test_size, "seed": args.seed}
    if "label" in dataset.column_names:
        split_kwargs["stratify_by_column"] = "label"
    dataset_split = dataset.train_test_split(**split_kwargs)
    
    train_labels = dataset_split["train"]["label"]
    unique_labels = np.unique(train_labels)

    # Calculate balanced weights (race classes get high weights)
    class_weights = compute_class_weight("balanced", classes=unique_labels, y=train_labels)

    # Move this to the GPU in the trainer
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    logging.info(f"Computed Class Weights: {class_weights_tensor}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["question"], truncation=True, max_length=args.max_length)

    eval_questions = dataset_split["test"]["question"]
    eval_labels = dataset_split["test"]["label"]

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
        eval_strategy="epoch",
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
        report_to=["wandb"],
        run_name=f"thrifty-router-{args.model_name.split('/')[-1]}",
        bf16=False,                # Use Brain Float 16 to prevent overflow
        max_grad_norm=1.0,        # Clip exploding gradients
        warmup_ratio=0.1,         # Gently warm up the learning rate over the first 10% of training
    )

    wandb.init(project=WANDB_PROJECT, name=WANDB_RUN_NAME)

    class WeightedTrainer(Trainer):
        def get_train_dataloader(self):
            # Use weighted sampling to balance classes per batch
            train_dataset = self.train_dataset
            train_label_values = train_dataset["label"]
            sample_weights = np.asarray(class_weights)[np.asarray(train_label_values, dtype=np.int64)]
            sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )

            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
            )

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # Extract labels
            labels = inputs.pop("labels")
            # Run the forward passs
            outputs = model(**inputs)
            logits = outputs.logits

            # Move weights to the same device/dtype as logits to avoid Half/Float mismatch
            weights = class_weights_tensor.to(device=logits.device, dtype=logits.dtype)

            # Apply weighted Cross Entropy Loss
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

            return (loss, outputs) if return_outputs else loss


    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save per-example predictions on the eval set
    pred_output = args.pred_output or os.path.join(args.output_dir, "eval_predictions.csv")
    predictions = trainer.predict(tokenized["test"])
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    os.makedirs(os.path.dirname(pred_output), exist_ok=True)
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
    logging.info("Eval predictions saved to %s", pred_output)


if __name__ == "__main__":
    main()
