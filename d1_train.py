# -*- coding: utf-8 -*-

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import argparse
import ast
import gc
import json
import random
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import Dataset, load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ais_text_rules import (
    contains_uncertain_expression,
    get_uncertainty_policy_text,
    normalize_uncertainty,
    summarize_uncertainty_texts,
)


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "thomas-sounack/BioClinical-ModernBERT-base"

SEED = 77
NUM_EPOCHS = 30
BATCH_SIZE = 8
GRAD_ACCUM = 1
LEARNING_RATE = 2e-5
MAX_LEN = 4096

# Used only for metrics displayed during training.
# Final threshold calibration is performed during inference
# using the internal validation set.
TRAIN_METRIC_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the D1 AIS multi-label classifier."
    )

    parser.add_argument(
        "--train-file",
        required=True,
        help="Path to the training JSONL file.",
    )
    parser.add_argument(
        "--val-file",
        required=True,
        help="Path to the validation JSONL file.",
    )
    parser.add_argument(
        "--lvl-map-12",
        required=True,
        help="Path to the D1/D2 AIS hierarchy mapping file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory used to save the trained model and metadata.",
    )

    return parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Basic utilities
# ============================================================

def parse_codes(x):
    if x is None:
        return []

    if isinstance(x, list):
        return x

    if isinstance(x, np.ndarray):
        return x.tolist()

    x = str(x).strip()

    if not x:
        return []

    if x.startswith("[") and x.endswith("]"):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return [v.strip() for v in x.split(",") if v.strip()]


def clean_code7(code7):
    code7 = re.sub(r"[^0-9]", "", str(code7).strip())
    return code7.zfill(7)


def get_text_and_code(example):
    text = example.get(
        "prompt",
        example.get("text", ""),
    )

    codes = example.get(
        "completion",
        example.get("code", ""),
    )

    return text, codes


def get_case_id(example, fallback):
    for col in ["case_id", "caseid", "CaseID", "CASE_ID"]:
        if col in example and example[col] is not None:
            return str(example[col])

    return str(fallback)


def load_lvl_map_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    if not txt.startswith("{"):
        txt = "{" + txt + "}"

    return ast.literal_eval(txt)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# Dataset QC
# ============================================================

def check_case_uniqueness(raw_ds, split_name):
    case_ids = [
        get_case_id(example, i)
        for i, example in enumerate(raw_ds)
    ]

    n_rows = len(case_ids)
    n_unique = len(set(case_ids))
    n_duplicates = n_rows - n_unique

    print(
        f"{split_name:8s} | rows={n_rows:,} | "
        f"unique_case={n_unique:,} | "
        f"duplicate_case_rows={n_duplicates:,}"
    )

    return {
        "split": split_name,
        "rows": n_rows,
        "unique_cases": n_unique,
        "duplicate_case_rows": n_duplicates,
    }


def dataset_uncertainty_qc(raw_ds, split_name):
    texts = []

    for example in raw_ds:
        text, _ = get_text_and_code(example)
        texts.append("" if text is None else str(text))

    summary = summarize_uncertainty_texts(texts)

    print("\n" + "-" * 80)
    print(f"UNCERTAINTY QC: {split_name}")
    print("-" * 80)

    print(
        "texts with uncertainty:",
        summary["texts_with_uncertainty"],
        "/",
        summary["total_texts"],
        f"({summary['texts_with_uncertainty_ratio']:.2%})",
    )
    print(
        "texts with negative:",
        summary["texts_with_negative"],
    )
    print(
        "texts with both:",
        summary["texts_with_both_uncertain_and_negative"],
    )

    return summary


# ============================================================
# Label space
# ============================================================

def build_d1_label_space_from_codebook(lvl_map_12):
    d1_map = lvl_map_12.get((1, ()), {})

    if not d1_map:
        raise ValueError("No D1 entries found in the hierarchy mapping.")

    vocab = sorted(str(k) for k in d1_map.keys())

    label2id = {
        label: idx
        for idx, label in enumerate(vocab)
    }
    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    return label2id, id2label


def build_d1_label_space_from_train_val(train_raw, val_raw):
    """
    Final classifier label space used in the experiment:
    D1 labels observed in the training and validation datasets.
    """

    vocab = set()

    for raw_ds in [train_raw, val_raw]:
        for example in raw_ds:
            _, codes = get_text_and_code(example)

            for code in parse_codes(codes):
                code7 = clean_code7(code)
                vocab.add(code7[0])

    vocab = sorted(vocab)

    if not vocab:
        raise ValueError(
            "No observed D1 labels were found in training + validation."
        )

    label2id = {
        label: idx
        for idx, label in enumerate(vocab)
    }
    id2label = {
        idx: label
        for label, idx in label2id.items()
    }

    return label2id, id2label


# ============================================================
# D1 input construction
# ============================================================

def get_d1_codebook_text(lvl_map_12):
    d1_map = lvl_map_12.get((1, ()), {})

    return "\n".join(
        f"{d1} : {description}"
        for d1, description in sorted(d1_map.items())
    )


def make_d1_input_text(raw_text, lvl_map_12):
    normalized_text = normalize_uncertainty(raw_text)

    return (
        f"{get_uncertainty_policy_text()}\n\n"
        "[AVAILABLE D1 CODES]\n"
        f"{get_d1_codebook_text(lvl_map_12)}\n\n"
        "[REPORT]\n"
        f"{normalized_text}"
    ).strip()


# ============================================================
# Ground-truth / hierarchy QC
# ============================================================

def validate_d1_labels_against_codebook(
    raw_ds,
    split_name,
    d1_label2id,
    d1_codebook,
    qc_dir,
):
    errors = []
    observed = set()

    counts = {
        label: 0
        for label in d1_label2id.keys()
    }

    for i, example in enumerate(raw_ds):
        _, codes = get_text_and_code(example)

        code_list = [
            clean_code7(code)
            for code in parse_codes(codes)
        ]

        for code7 in code_list:
            d1 = code7[0]
            observed.add(d1)

            if d1 not in d1_label2id:
                errors.append({
                    "split": split_name,
                    "row_idx": i,
                    "case_id": get_case_id(example, i),
                    "code7": code7,
                    "d1": d1,
                })
            else:
                counts[d1] += 1

    print("\n" + "-" * 80)
    print(f"D1 GT/HIERARCHY CONSISTENCY: {split_name}")
    print("-" * 80)
    print("Observed D1:", sorted(observed))
    print("Missing from classifier label space:", len(errors))

    if errors:
        error_df = pd.DataFrame(errors)
        error_df.to_csv(
            os.path.join(
                qc_dir,
                f"d1_{split_name}_hierarchy_errors.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

        raise ValueError(
            f"{split_name}: some ground-truth D1 labels are not "
            "included in the classifier label space."
        )

    return pd.DataFrame({
        "d1": list(d1_label2id.keys()),
        "description": [
            str(d1_codebook.get(label, ""))
            for label in d1_label2id.keys()
        ],
        f"{split_name}_positive_code_count": [
            counts[label]
            for label in d1_label2id.keys()
        ],
    })


# ============================================================
# Build D1 records
# ============================================================

def build_d1_records(raw_ds, d1_label2id, lvl_map_12):
    records = []

    for i, example in enumerate(raw_ds):
        text, codes = get_text_and_code(example)

        code_list = [
            clean_code7(code)
            for code in parse_codes(codes)
        ]

        d1_labels = sorted({
            code7[0]
            for code7 in code_list
        })

        unknown = [
            d1
            for d1 in d1_labels
            if d1 not in d1_label2id
        ]

        if unknown:
            raise ValueError(
                f"Unknown D1 label(s) in row {i}: {unknown}"
            )

        records.append({
            "sample_id": str(i),
            "case_id": get_case_id(example, i),
            "raw_text": "" if text is None else str(text),
            "text": make_d1_input_text(text, lvl_map_12),
            "has_uncertainty": int(
                contains_uncertain_expression(text)
            ),
            "codes_raw": ",".join(code_list),
            "d1_labels": d1_labels,
        })

    return records


# ============================================================
# Target encoding
# ============================================================

def make_onehot_from_labels(labels, label2id):
    vector = np.zeros(
        len(label2id),
        dtype=np.float32,
    )

    for label in labels:
        idx = label2id.get(label)

        if idx is None:
            raise ValueError(f"Unknown target label: {label}")

        vector[idx] = 1.0

    return vector.tolist()


# ============================================================
# Weighted BCE Trainer
# ============================================================

class WeightedBCETrainer(Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        **kwargs,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device)
        )

        loss = criterion(
            logits.view(-1, model.config.num_labels),
            labels.float().view(-1, model.config.num_labels),
        )

        return (loss, outputs) if return_outputs else loss


# ============================================================
# Class weighting
# ============================================================

def compute_pos_weight(dataset_with_labels):
    label_arr = np.stack(
        dataset_with_labels["labels"]
    ).astype(np.float32)

    pos_count = label_arr.sum(axis=0)
    neg_count = len(label_arr) - pos_count

    pos_count_safe = np.maximum(pos_count, 1.0)

    pos_weight = neg_count / pos_count_safe

    pos_weight = torch.tensor(
        pos_weight,
        dtype=torch.float32,
    ).clamp(max=10.0)

    return pos_weight, pos_count


# ============================================================
# Training metrics
# ============================================================

def compute_metrics_multilabel(threshold=TRAIN_METRIC_THRESHOLD):
    def compute_metrics(pred):
        logits, labels = pred

        probs = 1.0 / (1.0 + np.exp(-logits))
        predictions = (probs >= threshold).astype(int)

        return {
            "f1_macro": f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "precision_macro": precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "f1_micro": f1_score(
                labels,
                predictions,
                average="micro",
                zero_division=0,
            ),
            "precision_micro": precision_score(
                labels,
                predictions,
                average="micro",
                zero_division=0,
            ),
            "recall_micro": recall_score(
                labels,
                predictions,
                average="micro",
                zero_division=0,
            ),
            "accuracy": accuracy_score(
                labels,
                predictions,
            ),
        }

    return compute_metrics


# ============================================================
# Model / training configuration
# ============================================================

def build_d1_model(tokenizer, d1_label2id, d1_id2label):
    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        num_labels=len(d1_label2id),
        problem_type="multi_label_classification",
        id2label=d1_id2label,
        label2id=d1_label2id,
        trust_remote_code=True,
    )

    if hasattr(config, "reference_compile"):
        config.reference_compile = False

    config.use_cache = False

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config=config,
        trust_remote_code=True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if hasattr(model.config, "reference_compile"):
        model.config.reference_compile = False

    model.gradient_checkpointing_enable()

    return model


def build_train_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,
        logging_strategy="epoch",
        logging_dir=os.path.join(output_dir, "tb_logs"),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
        tf32=torch.cuda.is_available(),
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        disable_tqdm=True,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    set_seed()

    train_file = os.path.abspath(args.train_file)
    val_file = os.path.abspath(args.val_file)
    lvl_map_file = os.path.abspath(args.lvl_map_12)
    output_root = os.path.abspath(args.output_dir)

    model_output_dir = os.path.join(output_root, "d1_model")
    label_map_dir = os.path.join(output_root, "label_maps")
    qc_dir = os.path.join(output_root, "train_qc")

    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(label_map_dir, exist_ok=True)
    os.makedirs(qc_dir, exist_ok=True)

    for required_file in [
        train_file,
        val_file,
        lvl_map_file,
    ]:
        if not os.path.isfile(required_file):
            raise FileNotFoundError(
                f"Required file does not exist: {required_file}"
            )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    train_raw = load_dataset(
        "json",
        data_files=train_file,
        split="train",
    )

    val_raw = load_dataset(
        "json",
        data_files=val_file,
        split="train",
    )

    print("\n" + "=" * 80)
    print("DATASET")
    print("=" * 80)
    print("Train rows:", len(train_raw))
    print("Val rows  :", len(val_raw))
    print("Train columns:", train_raw.column_names)
    print("Val columns  :", val_raw.column_names)

    # --------------------------------------------------------
    # Case-level QC
    # --------------------------------------------------------

    merge_qc = pd.DataFrame([
        check_case_uniqueness(train_raw, "train"),
        check_case_uniqueness(val_raw, "val"),
    ])

    merge_qc.to_csv(
        os.path.join(qc_dir, "d1_case_uniqueness_qc.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Uncertainty QC
    # --------------------------------------------------------

    uncertainty_summary = {
        "train": dataset_uncertainty_qc(
            train_raw,
            "train",
        ),
        "val": dataset_uncertainty_qc(
            val_raw,
            "val",
        ),
    }

    save_json(
        uncertainty_summary,
        os.path.join(
            qc_dir,
            "d1_uncertainty_summary.json",
        ),
    )

    # --------------------------------------------------------
    # Load hierarchy
    # --------------------------------------------------------

    lvl_map_12 = load_lvl_map_txt(lvl_map_file)

    if (1, ()) not in lvl_map_12:
        raise ValueError(
            "The hierarchy mapping does not contain "
            "the D1 key (1, ())."
        )

    d1_codebook = lvl_map_12[(1, ())]

    print("\n" + "=" * 80)
    print("D1 CODEBOOK")
    print("=" * 80)

    for d1, description in sorted(d1_codebook.items()):
        print(f"{d1} : {description}")

    # Keep the complete codebook vocabulary for QC.
    codebook_label2id, _ = build_d1_label_space_from_codebook(
        lvl_map_12
    )

    # Final classifier label space used in the experiment:
    # labels observed in training + validation.
    d1_label2id, d1_id2label = (
        build_d1_label_space_from_train_val(
            train_raw,
            val_raw,
        )
    )

    print("\n" + "=" * 80)
    print("D1 CLASSIFIER LABEL SPACE")
    print("=" * 80)
    print("Observed train + validation labels:", list(d1_label2id))
    print("Complete codebook labels:", list(codebook_label2id))

    # Ensure every observed classifier label exists in the hierarchy.
    missing_from_codebook = sorted(
        set(d1_label2id) - set(codebook_label2id)
    )

    if missing_from_codebook:
        raise ValueError(
            "Observed D1 labels missing from the hierarchy mapping: "
            f"{missing_from_codebook}"
        )

    save_json(
        d1_label2id,
        os.path.join(
            label_map_dir,
            "d1_label2id.json",
        ),
    )

    save_json(
        {
            str(idx): label
            for idx, label in d1_id2label.items()
        },
        os.path.join(
            label_map_dir,
            "d1_id2label.json",
        ),
    )

    # --------------------------------------------------------
    # Label QC
    # --------------------------------------------------------

    train_label_qc = validate_d1_labels_against_codebook(
        train_raw,
        "train",
        d1_label2id,
        d1_codebook,
        qc_dir,
    )

    val_label_qc = validate_d1_labels_against_codebook(
        val_raw,
        "val",
        d1_label2id,
        d1_codebook,
        qc_dir,
    )

    label_qc = train_label_qc.merge(
        val_label_qc[
            [
                "d1",
                "val_positive_code_count",
            ]
        ],
        on="d1",
        how="left",
    )

    label_qc["zero_positive_in_train"] = (
        label_qc["train_positive_code_count"] == 0
    )

    label_qc.to_csv(
        os.path.join(
            qc_dir,
            "d1_label_counts.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Build records
    # --------------------------------------------------------

    train_records = build_d1_records(
        train_raw,
        d1_label2id,
        lvl_map_12,
    )

    val_records = build_d1_records(
        val_raw,
        d1_label2id,
        lvl_map_12,
    )

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    # Do not export raw radiology text to QC files.
    qc_columns = [
        "sample_id",
        "case_id",
        "has_uncertainty",
        "codes_raw",
        "d1_labels",
    ]

    train_df[qc_columns].to_csv(
        os.path.join(
            qc_dir,
            "d1_train_records_summary.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    val_df[qc_columns].to_csv(
        os.path.join(
            qc_dir,
            "d1_val_records_summary.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    train_ds_raw = Dataset.from_pandas(
        train_df,
        preserve_index=False,
    )

    val_ds_raw = Dataset.from_pandas(
        val_df,
        preserve_index=False,
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            max_length=MAX_LEN,
        )

        tokenized["labels"] = make_onehot_from_labels(
            example["d1_labels"],
            d1_label2id,
        )

        return tokenized

    print("\nTokenizing training dataset...")

    train_ds = train_ds_raw.map(
        tokenize,
        remove_columns=train_ds_raw.column_names,
        num_proc=4,
    )

    print("Tokenizing validation dataset...")

    val_ds = val_ds_raw.map(
        tokenize,
        remove_columns=val_ds_raw.column_names,
        num_proc=4,
    )

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    pos_weight, pos_count = compute_pos_weight(train_ds)

    training_label_stats = pd.DataFrame({
        "d1": [
            d1_id2label[i]
            for i in range(len(d1_id2label))
        ],
        "label_id": list(range(len(d1_id2label))),
        "description": [
            str(
                d1_codebook.get(
                    d1_id2label[i],
                    "",
                )
            )
            for i in range(len(d1_id2label))
        ],
        "positive_count": pos_count.astype(int),
        "pos_weight": pos_weight.cpu().numpy(),
    })

    training_label_stats["zero_positive"] = (
        training_label_stats["positive_count"] == 0
    )

    training_label_stats.to_csv(
        os.path.join(
            qc_dir,
            "d1_training_label_counts_and_weights.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 80)
    print("D1 TRAINING")
    print("=" * 80)
    print("Number of labels:", len(d1_label2id))
    print(
        "Zero-positive labels in train:",
        int((pos_count == 0).sum()),
    )
    print(
        "pos_weight min/max:",
        float(pos_weight.min()),
        float(pos_weight.max()),
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = build_d1_model(
        tokenizer,
        d1_label2id,
        d1_id2label,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = WeightedBCETrainer(
        pos_weight=pos_weight,
        model=model,
        args=build_train_args(model_output_dir),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics_multilabel(
            TRAIN_METRIC_THRESHOLD
        ),
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=5
            )
        ],
    )

    train_result = trainer.train()

    print("\nTraining finished.")
    print(train_result)

    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    print("\nSaved D1 model:", model_output_dir)
    print("Saved label maps:", label_map_dir)
    print("Saved QC outputs:", qc_dir)

    del trainer
    del model

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
