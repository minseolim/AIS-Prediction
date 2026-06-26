# -*- coding: utf-8 -*-
"""
Train hierarchical D1 and D2 multi-label AIS classifiers.

Input JSONL schema:
    {
      "case_id": "CASE_000001",
      "prompt": "CT radiology report text ...",
      "completion": ["1402023", "4502022"]
    }

Notes:
- D1 model predicts AIS first digit labels from the report text.
- D2 model predicts AIS second digit suffix labels conditioned on gold D1:
      [D1=x] + report text
- Raw clinical data and preprocessing logic are intentionally excluded.
"""

import argparse
import ast
import gc
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int = 77) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_codes(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            pass

    return [v.strip() for v in text.split(",") if v.strip()]


def clean_code7(code: Any) -> str:
    digits = re.sub(r"[^0-9]", "", str(code).strip())
    return digits.zfill(7)


def unique_sorted(values: Iterable[str]) -> List[str]:
    return sorted(set(values))


def get_text_and_code(example: Dict[str, Any]) -> Tuple[str, Any]:
    text = example.get("prompt", example.get("text", ""))
    codes = example.get("completion", example.get("code", ""))
    return str(text), codes


def make_onehot(labels: Sequence[str], label2id: Dict[str, int]) -> List[float]:
    vec = np.zeros(len(label2id), dtype=np.float32)
    for label in labels:
        if label in label2id:
            vec[label2id[label]] = 1.0
    return vec.tolist()


# -----------------------------------------------------------------------------
# Label spaces and training records
# -----------------------------------------------------------------------------
def build_label_spaces(train_raw: Dataset, val_raw: Dataset):
    d1_vocab = set()
    d2_vocab = set()
    d12_vocab = set()

    for ds in [train_raw, val_raw]:
        for ex in ds:
            _, codes = get_text_and_code(ex)
            for code in [clean_code7(c) for c in parse_codes(codes)]:
                d1_vocab.add(code[0])
                d2_vocab.add(code[1])
                d12_vocab.add(code[:2])

    d1_label2id = {label: i for i, label in enumerate(sorted(d1_vocab))}
    d2_label2id = {label: i for i, label in enumerate(sorted(d2_vocab))}
    d12_label2id = {label: i for i, label in enumerate(sorted(d12_vocab))}

    return d1_label2id, d2_label2id, d12_label2id


def build_d1_records(raw_ds: Dataset) -> List[Dict[str, Any]]:
    records = []
    for i, ex in enumerate(raw_ds):
        text, codes = get_text_and_code(ex)
        code_list = [clean_code7(c) for c in parse_codes(codes)]
        records.append(
            {
                "sample_id": str(i),
                "text": text,
                "d1_labels": unique_sorted(code[0] for code in code_list),
            }
        )
    return records


def build_d2_records(raw_ds: Dataset) -> List[Dict[str, Any]]:
    records = []
    for i, ex in enumerate(raw_ds):
        text, codes = get_text_and_code(ex)
        code_list = [clean_code7(c) for c in parse_codes(codes)]

        d1_to_d2 = {}
        for code in code_list:
            d1_to_d2.setdefault(code[0], set()).add(code[1])

        for gold_d1, d2_set in sorted(d1_to_d2.items()):
            records.append(
                {
                    "sample_id": str(i),
                    "text": f"[D1={gold_d1}] {text}",
                    "gold_d1": gold_d1,
                    "d2_labels": sorted(d2_set),
                }
            )
    return records


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weight: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        loss = criterion(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


def compute_pos_weight(dataset: Dataset, max_weight: float = 10.0) -> torch.Tensor:
    labels = np.stack(dataset["labels"]).astype(np.float32)
    pos_cnt = labels.sum(axis=0)
    neg_cnt = len(labels) - pos_cnt
    pos_cnt_safe = np.maximum(pos_cnt, 1.0)
    return torch.tensor(neg_cnt / pos_cnt_safe, dtype=torch.float32).clamp(max=max_weight)


def build_model(model_name: str, label2id: Dict[str, int], tokenizer) -> AutoModelForSequenceClassification:
    id2label = {i: label for label, i in label2id.items()}
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(label2id),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.reference_compile = False
    model.gradient_checkpointing_enable()
    return model


def compute_metrics_fn(threshold: float):
    def compute_metrics(pred):
        logits, labels = pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= threshold).astype(int)
        return {
            "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
            "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
            "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
            "subset_accuracy": accuracy_score(labels, preds),
        }
    return compute_metrics


def build_training_args(args, output_dir: Path) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        ddp_find_unused_parameters=False,
        disable_tqdm=False,
    )


def train_one_task(
    task_name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    label2id: Dict[str, int],
    tokenizer,
    args,
    output_dir: Path,
) -> None:
    task_output_dir = output_dir / f"{task_name}_model"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    pos_weight = compute_pos_weight(train_ds, max_weight=args.max_pos_weight)
    model = build_model(args.model_name, label2id, tokenizer)

    trainer = WeightedBCETrainer(
        pos_weight=pos_weight,
        model=model,
        args=build_training_args(args, task_output_dir),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_fn(args.threshold),
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    print(f"\n========== TRAIN {task_name.upper()} ==========")
    trainer.train()
    trainer.save_model(str(task_output_dir))
    tokenizer.save_pretrained(str(task_output_dir))

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train D1 and D2 AIS classifiers.")
    parser.add_argument("--train-file", required=True, help="Path to train JSONL file.")
    parser.add_argument("--val-file", required=True, help="Path to validation JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save models and label maps.")
    parser.add_argument("--model-name", default="thomas-sounack/BioClinical-ModernBERT-base")
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--max-pos-weight", type=float, default=10.0)
    parser.add_argument("--skip-d1", action="store_true")
    parser.add_argument("--skip-d2", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load_dataset("json", data_files=args.train_file, split="train")
    val_raw = load_dataset("json", data_files=args.val_file, split="train")

    print("train rows:", len(train_raw))
    print("val rows:", len(val_raw))
    print("train columns:", train_raw.column_names)
    print("val columns:", val_raw.column_names)

    d1_label2id, d2_label2id, d12_label2id = build_label_spaces(train_raw, val_raw)
    for name, label2id in [
        ("d1_label2id.json", d1_label2id),
        ("d2_label2id.json", d2_label2id),
        ("d12_label2id.json", d12_label2id),
    ]:
        with open(output_dir / name, "w", encoding="utf-8") as f:
            json.dump(label2id, f, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_d1(ex):
        encoded = tokenizer(ex["text"], truncation=True, max_length=args.max_len)
        encoded["labels"] = make_onehot(ex["d1_labels"], d1_label2id)
        return encoded

    def tokenize_d2(ex):
        encoded = tokenizer(ex["text"], truncation=True, max_length=args.max_len)
        encoded["labels"] = make_onehot(ex["d2_labels"], d2_label2id)
        return encoded

    if not args.skip_d1:
        train_d1_raw = Dataset.from_pandas(pd.DataFrame(build_d1_records(train_raw)), preserve_index=False)
        val_d1_raw = Dataset.from_pandas(pd.DataFrame(build_d1_records(val_raw)), preserve_index=False)
        train_d1 = train_d1_raw.map(tokenize_d1, remove_columns=train_d1_raw.column_names, num_proc=args.num_workers)
        val_d1 = val_d1_raw.map(tokenize_d1, remove_columns=val_d1_raw.column_names, num_proc=args.num_workers)
        train_one_task("d1", train_d1, val_d1, d1_label2id, tokenizer, args, output_dir)

    if not args.skip_d2:
        train_d2_raw = Dataset.from_pandas(pd.DataFrame(build_d2_records(train_raw)), preserve_index=False)
        val_d2_raw = Dataset.from_pandas(pd.DataFrame(build_d2_records(val_raw)), preserve_index=False)
        train_d2 = train_d2_raw.map(tokenize_d2, remove_columns=train_d2_raw.column_names, num_proc=args.num_workers)
        val_d2 = val_d2_raw.map(tokenize_d2, remove_columns=val_d2_raw.column_names, num_proc=args.num_workers)
        train_one_task("d2", train_d2, val_d2, d2_label2id, tokenizer, args, output_dir)

    print("\nSaved outputs to:", output_dir)


if __name__ == "__main__":
    main()
