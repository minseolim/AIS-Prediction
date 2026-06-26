# -*- coding: utf-8 -*-
"""
Run case-level inference and evaluation for hierarchical D1/D2 AIS classifiers.

Input JSONL schema:
    {
      "case_id": "CASE_000001",
      "prompt": "CT radiology report text ...",
      "completion": ["1402023", "4502022"]
    }

This script saves only compact summary metrics. It does not export case-level
prediction details or raw clinical text.
"""

import argparse
import ast
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def get_text_col(df: pd.DataFrame) -> str:
    for col in ["prompt", "text"]:
        if col in df.columns:
            return col
    raise ValueError("Input JSONL must contain either 'prompt' or 'text'.")


def get_code_col(df: pd.DataFrame) -> str:
    for col in ["completion", "code"]:
        if col in df.columns:
            return col
    raise ValueError("Input JSONL must contain either 'completion' or 'code'.")


def get_caseid_col(df: pd.DataFrame) -> str:
    for col in ["case_id", "caseid", "CaseID", "CASE_ID"]:
        if col in df.columns:
            return col
    raise ValueError("Input JSONL must contain a case_id column.")


def build_true_d1_set(codes: Any) -> Set[str]:
    return {clean_code7(code)[0] for code in parse_codes(codes)}


def build_true_d12_set(codes: Any) -> Set[str]:
    return {clean_code7(code)[:2] for code in parse_codes(codes)}


def build_true_d2_by_d1(codes: Any) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for code in [clean_code7(c) for c in parse_codes(codes)]:
        out.setdefault(code[0], set()).add(code[1])
    return out


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
@torch.no_grad()
def predict_multilabel_sets(
    model,
    tokenizer,
    texts: List[str],
    id2label: Dict[int, str],
    max_len: int,
    batch_size: int,
    threshold: float,
    device: str,
) -> List[Set[str]]:
    model.eval()
    pred_sets: List[Set[str]] = []
    use_bf16 = device == "cuda"

    for start in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch_texts = texts[start:start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**encoded)
        else:
            outputs = model(**encoded)

        probs = torch.sigmoid(outputs.logits).detach().float().cpu().numpy()
        preds = (probs >= threshold).astype(np.int32)

        for row in preds:
            idxs = np.where(row == 1)[0]
            pred_sets.append({id2label[int(i)] for i in idxs})

    return pred_sets


# -----------------------------------------------------------------------------
# Case-level aggregation and metrics
# -----------------------------------------------------------------------------
def union_sets(values: Iterable[Set[str]]) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        if isinstance(value, set):
            out |= value
    return out


def union_dict_of_sets(values: Iterable[Dict[str, Set[str]]]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for d in values:
        if isinstance(d, dict):
            for key, value in d.items():
                out.setdefault(key, set()).update(value)
    return out


def compute_case_summary(true_sets: List[Set[str]], pred_sets: List[Set[str]]) -> Dict[str, Any]:
    per_precision, per_recall, per_f1, exact, over, missed = [], [], [], [], [], []
    total_tp, total_fp, total_fn = 0, 0, 0

    for true_set, pred_set in zip(true_sets, pred_sets):
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        per_precision.append(precision)
        per_recall.append(recall)
        per_f1.append(f1)
        exact.append(float(true_set == pred_set))
        over.append(fp)
        missed.append(fn)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_precision = safe_div(total_tp, total_tp + total_fp)
    micro_recall = safe_div(total_tp, total_tp + total_fn)
    micro_f1 = safe_div(2 * micro_precision * micro_recall, micro_precision + micro_recall)

    return {
        "n_cases": len(true_sets),
        "precision": float(np.mean(per_precision)) if per_precision else 0.0,
        "recall": float(np.mean(per_recall)) if per_recall else 0.0,
        "f1": float(np.mean(per_f1)) if per_f1 else 0.0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "exact_match": float(np.mean(exact)) if exact else 0.0,
        "over_prediction": float(np.mean(over)) if over else 0.0,
        "missed_injury": float(np.mean(missed)) if missed else 0.0,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def build_predictions(df_raw, split_name, models, tokenizer, label_maps, args, device):
    d1_model, d2_model = models
    d1_id2label, d2_id2label = label_maps

    text_col = get_text_col(df_raw)
    code_col = get_code_col(df_raw)
    caseid_col = get_caseid_col(df_raw)

    df = df_raw.copy()
    df["case_id"] = df[caseid_col].astype(str)
    df["true_d1"] = df[code_col].apply(build_true_d1_set)
    df["true_d12"] = df[code_col].apply(build_true_d12_set)
    df["true_d2_by_d1"] = df[code_col].apply(build_true_d2_by_d1)

    texts = df[text_col].fillna("").astype(str).tolist()
    df["pred_d1"] = predict_multilabel_sets(
        d1_model, tokenizer, texts, d1_id2label,
        args.max_len, args.batch_size, args.threshold, device,
    )

    # D2 evaluation is gold-D1 conditioned, matching the training formulation.
    d2_meta = []
    d2_texts = []
    for row_idx, row in df.iterrows():
        raw_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
        for gold_d1 in sorted(row["true_d2_by_d1"].keys()):
            d2_meta.append((row_idx, gold_d1))
            d2_texts.append(f"[D1={gold_d1}] {raw_text}")

    pred_d2_sets = predict_multilabel_sets(
        d2_model, tokenizer, d2_texts, d2_id2label,
        args.max_len, args.batch_size, args.threshold, device,
    ) if d2_texts else []

    row_to_pred_d2 = {idx: {} for idx in df.index}
    for (row_idx, gold_d1), pred_suffix_set in zip(d2_meta, pred_d2_sets):
        row_to_pred_d2[row_idx][gold_d1] = pred_suffix_set
    df["pred_d2_by_gold_d1"] = df.index.map(lambda idx: row_to_pred_d2[idx])

    pred_d12_eval = []
    true_d12_eval = []
    inter_d1_list = []
    for _, row in df.iterrows():
        inter_d1 = row["true_d1"] & row["pred_d1"]
        inter_d1_list.append(inter_d1)
        true_d12_eval.append({d12 for d12 in row["true_d12"] if d12[0] in inter_d1})

        pred_set = set()
        for d1 in inter_d1:
            for d2 in row["pred_d2_by_gold_d1"].get(d1, set()):
                pred_set.add(d1 + d2)
        pred_d12_eval.append(pred_set)

    df["inter_d1"] = inter_d1_list
    df["true_d12_eval"] = true_d12_eval
    df["pred_d12_eval"] = pred_d12_eval
    return df


def build_case_dataframe(df_row: pd.DataFrame) -> pd.DataFrame:
    df_case = df_row.groupby("case_id", as_index=False).agg(
        true_d1=("true_d1", union_sets),
        pred_d1=("pred_d1", union_sets),
        true_d12=("true_d12", union_sets),
        true_d12_eval=("true_d12_eval", union_sets),
        pred_d12_eval=("pred_d12_eval", union_sets),
        true_d2_by_d1=("true_d2_by_d1", union_dict_of_sets),
        pred_d2_by_gold_d1=("pred_d2_by_gold_d1", union_dict_of_sets),
    )

    true_d2_suffix = []
    pred_d2_suffix = []
    for _, row in df_case.iterrows():
        true_all, pred_all = set(), set()
        for suffixes in row["true_d2_by_d1"].values():
            true_all |= suffixes
        for suffixes in row["pred_d2_by_gold_d1"].values():
            pred_all |= suffixes
        true_d2_suffix.append(true_all)
        pred_d2_suffix.append(pred_all)

    df_case["true_d2_suffix"] = true_d2_suffix
    df_case["pred_d2_suffix"] = pred_d2_suffix
    return df_case


def evaluate_split(jsonl_path: str, split_name: str, models, tokenizer, label_maps, args, device) -> pd.DataFrame:
    df_raw = load_jsonl(jsonl_path)
    df_row = build_predictions(df_raw, split_name, models, tokenizer, label_maps, args, device)
    df_case = build_case_dataframe(df_row)

    task_defs = [
        ("d1", "true_d1", "pred_d1"),
        ("d2_gold_conditioned", "true_d2_suffix", "pred_d2_suffix"),
        ("d1d2_intersection", "true_d12_eval", "pred_d12_eval"),
    ]

    rows = []
    for task, true_col, pred_col in task_defs:
        summary = compute_case_summary(df_case[true_col].tolist(), df_case[pred_col].tolist())
        rows.append({"split": split_name, "task": task, **summary})

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate D1/D2 AIS classifiers.")
    parser.add_argument("--model-dir", required=True, help="Directory containing d1_model, d2_model, and label maps.")
    parser.add_argument("--input-file", action="append", required=True, help="Input JSONL file. Can be used multiple times.")
    parser.add_argument("--split-name", action="append", required=True, help="Split name. Must match --input-file order.")
    parser.add_argument("--output-dir", required=True, help="Directory to save summary metrics.")
    parser.add_argument("--max-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=77)
    args = parser.parse_args()

    if len(args.input_file) != len(args.split_name):
        raise ValueError("The number of --input-file values must match the number of --split-name values.")

    set_seed(args.seed)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "d1_label2id.json", "r", encoding="utf-8") as f:
        d1_label2id = json.load(f)
    with open(model_dir / "d2_label2id.json", "r", encoding="utf-8") as f:
        d2_label2id = json.load(f)

    d1_id2label = {int(v): k for k, v in d1_label2id.items()}
    d2_id2label = {int(v): k for k, v in d2_label2id.items()}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir / "d1_model", use_fast=True)
    d1_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir / "d1_model",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    d2_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir / "d2_model",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)

    summaries = []
    for input_file, split_name in zip(args.input_file, args.split_name):
        print(f"\nEvaluating {split_name}: {input_file}")
        summaries.append(
            evaluate_split(
                input_file,
                split_name,
                models=(d1_model, d2_model),
                tokenizer=tokenizer,
                label_maps=(d1_id2label, d2_id2label),
                args=args,
                device=device,
            )
        )

    final_summary = pd.concat(summaries, ignore_index=True)
    final_summary.to_csv(output_dir / "d1_d2_summary.csv", index=False, encoding="utf-8-sig")
    print("\n===== SUMMARY =====")
    print(final_summary.round(4))
    print("\nSaved summary to:", output_dir / "d1_d2_summary.csv")


if __name__ == "__main__":
    main()
