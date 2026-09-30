# -*- coding: utf-8 -*-

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import ast
import json
import re

import numpy as np
import pandas as pd
import torch

from transformers import AutoConfig

from ais_text_rules import normalize_uncertainty, get_uncertainty_policy_text


# ============================================================
# Runtime configuration
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL_NAME = "thomas-sounack/BioClinical-ModernBERT-base"


# ============================================================
# Basic IO
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path):
    return pd.read_json(path, lines=True)


def save_excel(sheet_dict, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheet_dict.items():
            safe_name = str(sheet_name)[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)


def load_lvl_map_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    if not txt.startswith("{"):
        txt = "{" + txt + "}"

    return ast.literal_eval(txt)


# ============================================================
# Dataset helpers
# ============================================================

def get_text_col(df):
    for col in ["prompt", "text", "report"]:
        if col in df.columns:
            return col

    raise KeyError(f"No text column found. columns={list(df.columns)}")


def get_code_col(df):
    for col in ["completion", "code", "codes"]:
        if col in df.columns:
            return col

    raise KeyError(f"No code column found. columns={list(df.columns)}")


def get_caseid_col(df):
    for col in ["case_id", "caseid", "CaseID", "CASE_ID"]:
        if col in df.columns:
            return col

    raise KeyError(f"No case_id column found. columns={list(df.columns)}")


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


# ============================================================
# AIS policy / input construction
# ============================================================

def get_policy_body():
    policy = get_uncertainty_policy_text()
    prefix = "[AIS CODING POLICY]\n"

    if policy.startswith(prefix):
        return policy[len(prefix):]

    return policy


# ============================================================
# D1 input
# ============================================================

def get_d1_codebook_text(lvl12):
    d1_map = lvl12.get((1, ()), {})
    lines = []

    for d1, desc in sorted(d1_map.items()):
        lines.append(f"{str(d1)} : {desc}")

    return "\n".join(lines)


def make_d1_input(raw, lvl12):
    return (
        "[AIS CODING POLICY]\n"
        f"{get_policy_body()}\n\n"
        "[AVAILABLE D1 CODES]\n"
        f"{get_d1_codebook_text(lvl12)}\n\n"
        "[REPORT]\n"
        f"{normalize_uncertainty(raw)}"
    ).strip()


# ============================================================
# D2 input
# ============================================================

def get_d2_codebook_text(d1, lvl12):
    child_map = lvl12.get((2, (str(d1),)), {})
    lines = []

    for d2, desc in sorted(child_map.items()):
        lines.append(f"{str(d1)}{str(d2)} : {desc}")

    return "\n".join(lines)


def make_d2_input(d1, raw, lvl12):
    return (
        "[AIS CODING POLICY]\n"
        f"{get_policy_body()}\n\n"
        f"[D1={d1}]\n\n"
        "[AVAILABLE D2 CODES]\n"
        f"{get_d2_codebook_text(d1, lvl12)}\n\n"
        "[REPORT]\n"
        f"{normalize_uncertainty(raw)}"
    ).strip()


# ============================================================
# D3-D4 input
# ============================================================

def get_d34_codebook_text(d1, d2, lvl34):
    child_map = lvl34.get((3, (str(d1), str(d2))), {})
    lines = []

    for d34, desc in sorted(child_map.items()):
        d34 = str(d34).zfill(2)
        lines.append(f"{str(d1)}{str(d2)}{d34} : {desc}")

    return "\n".join(lines)


def make_d34_input(d1, d2, raw, lvl34):
    return (
        "[AIS CODING POLICY]\n"
        f"{get_policy_body()}\n\n"
        f"[D1={d1}]\n"
        f"[D2={d2}]\n\n"
        "[AVAILABLE D3D4 CODES]\n"
        f"{get_d34_codebook_text(d1, d2, lvl34)}\n\n"
        "[REPORT]\n"
        f"{normalize_uncertainty(raw)}"
    ).strip()


# ============================================================
# D5-D6 input
# ============================================================

def get_d56_codebook_text(d1, d2, d34, lvl56):
    child_map = lvl56.get(
        (4, (str(d1), str(d2), str(d34).zfill(2))),
        {},
    )

    lines = []
    seen_d56 = set()

    for d567, desc in sorted(child_map.items()):
        d567 = str(d567).zfill(3)
        d56 = d567[:2]

        if d56 in seen_d56:
            continue

        seen_d56.add(d56)

        full6 = (
            str(d1)
            + str(d2)
            + str(d34).zfill(2)
            + d56
        )

        lines.append(f"{full6} : {desc}")

    return "\n".join(lines)


def make_d56_input(d1, d2, d34, raw, lvl56):
    return (
        "[AIS CODING POLICY]\n"
        f"{get_policy_body()}\n\n"
        f"[D1={d1}]\n"
        f"[D2={d2}]\n"
        f"[D3D4={str(d34).zfill(2)}]\n\n"
        "[AVAILABLE D5D6 CODES]\n"
        f"{get_d56_codebook_text(d1, d2, d34, lvl56)}\n\n"
        "[REPORT]\n"
        f"{normalize_uncertainty(raw)}"
    ).strip()


# ============================================================
# Safe ModernBERT model loading
# ============================================================

def load_modernbert_classifier_model(model_dir):
    """
    Load a ModernBERT sequence classifier with reference
    compilation disabled.
    """

    config = AutoConfig.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    if hasattr(config, "reference_compile"):
        config.reference_compile = False

    if hasattr(config, "use_cache"):
        config.use_cache = False

    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        config=config,
        trust_remote_code=True,
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available()
            else torch.float32
        ),
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# Model inference
# ============================================================

@torch.no_grad()
def predict_logits_probs(
    model,
    tokenizer,
    texts,
    batch_size=4,
    max_len=4096,
    desc="infer",
):
    all_logits = []

    print(
        f"{desc}: n={len(texts):,}, "
        f"batch_size={batch_size}, device={DEVICE}"
    )

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        out = model(**enc)

        all_logits.append(
            out.logits.float().cpu().numpy()
        )

        batch_number = start // batch_size + 1

        if start == 0 or batch_number % 500 == 0:
            processed = min(start + batch_size, len(texts))
            print(f"  processed {processed:,}/{len(texts):,}")

    if not all_logits:
        empty = np.empty((0, 0), dtype=np.float32)
        return empty, empty

    logits = np.concatenate(all_logits, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))

    return logits, probs


# ============================================================
# Global micro-F1 threshold calibration
# ============================================================

def micro_counts(y_true, y_pred, valid_mask=None):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    if valid_mask is None:
        valid_mask = np.ones_like(y_true, dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]

    tp = int(np.logical_and(yt == 1, yp == 1).sum())
    fp = int(np.logical_and(yt == 0, yp == 1).sum())
    fn = int(np.logical_and(yt == 1, yp == 0).sum())
    tn = int(np.logical_and(yt == 0, yp == 0).sum())

    return tp, fp, fn, tn


def scores_from_counts(tp, fp, fn, tn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    fpr = fp / (fp + tn) if fp + tn > 0 else 0.0

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "youden_j": recall + specificity - 1.0,
    }


def calibrate_global_micro_f1_threshold(
    y_true,
    probs,
    valid_mask=None,
    thresholds=None,
):
    if thresholds is None:
        thresholds = np.round(
            np.arange(0.01, 1.00, 0.01),
            2,
        )

    rows = []

    for threshold in thresholds:
        pred = (probs >= threshold).astype(np.int32)

        tp, fp, fn, tn = micro_counts(
            y_true,
            pred,
            valid_mask,
        )

        scores = scores_from_counts(
            tp,
            fp,
            fn,
            tn,
        )

        rows.append({
            "threshold": float(threshold),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            **scores,
        })

    sweep = pd.DataFrame(rows)

    best_f1 = sweep["micro_f1"].max()

    candidates = sweep[
        np.isclose(sweep["micro_f1"], best_f1)
    ].copy()

    # Tie-break:
    # 1) larger recall
    # 2) threshold closer to 0.5
    # 3) smaller threshold
    candidates["distance_to_0_5"] = (
        candidates["threshold"] - 0.5
    ).abs()

    candidates = candidates.sort_values(
        [
            "micro_recall",
            "distance_to_0_5",
            "threshold",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )

    best_row = candidates.iloc[0]
    best_threshold = float(best_row["threshold"])

    return (
        best_threshold,
        sweep,
        best_row.to_dict(),
    )


def apply_global_threshold(
    probs,
    threshold,
    valid_mask=None,
):
    pred = probs >= float(threshold)

    if valid_mask is not None:
        pred = np.logical_and(
            pred,
            valid_mask,
        )

    return pred


# ============================================================
# Case-level metrics
# ============================================================

def set_metrics(true_set, pred_set):
    true_set = set(true_set)
    pred_set = set(pred_set)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    union = len(true_set | pred_set)
    jaccard = tp / union if union > 0 else 1.0

    exact_match = int(true_set == pred_set)
    inclusive_accuracy = int(true_set.issubset(pred_set))

    # A case is counted as sample-level correct when at least
    # one predicted label overlaps with the ground truth.
    sample_accuracy = int(len(true_set & pred_set) > 0)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "sample_level_accuracy": sample_accuracy,
        "exact_match": exact_match,
        "inclusive_accuracy": inclusive_accuracy,
        "over_prediction": int(fp > 0),
        "missed_injury": int(fn > 0),
    }


def summarize_case_df(
    case_df,
    true_col,
    pred_col,
    universe_size,
    split,
    stage,
):
    detail_rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for _, row in case_df.iterrows():
        true_set = set(row[true_col])
        pred_set = set(row[pred_col])

        metrics = set_metrics(
            true_set,
            pred_set,
        )

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        out = row.to_dict()
        out.update(metrics)

        detail_rows.append(out)

    detail = pd.DataFrame(detail_rows)

    n_case = len(detail)

    total_tn = (
        n_case * universe_size
        - total_tp
        - total_fp
        - total_fn
    )

    global_scores = scores_from_counts(
        total_tp,
        total_fp,
        total_fn,
        total_tn,
    )

    precision = global_scores["micro_precision"]
    recall = global_scores["micro_recall"]

    def fbeta(beta):
        denominator = beta * beta * precision + recall

        if denominator <= 0:
            return 0.0

        return (
            (1 + beta * beta)
            * precision
            * recall
            / denominator
        )

    base_summary = {
        "split": split,
        "stage": stage,
        "n_cases": n_case,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": global_scores["micro_f1"],
        "micro_f1_5": fbeta(1.5),
        "micro_f2": fbeta(2.0),
        "specificity": global_scores["specificity"],
        "fpr": global_scores["fpr"],
        "youden_j": global_scores["youden_j"],
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }

    if n_case == 0:
        summary = pd.DataFrame([
            {
                **base_summary,
                "precision_case_mean": 0.0,
                "recall_case_mean": 0.0,
                "f1_case_mean": 0.0,
                "jaccard_case_mean": 0.0,
                "sample_level_accuracy": 0.0,
                "exact_match": 0.0,
                "inclusive_accuracy": 0.0,
                "case_accuracy": 0.0,
                "over_prediction_rate": 0.0,
                "missed_injury_rate": 0.0,
            }
        ])

        return summary, detail

    summary = pd.DataFrame([
        {
            **base_summary,
            "precision_case_mean": detail["precision"].mean(),
            "recall_case_mean": detail["recall"].mean(),
            "f1_case_mean": detail["f1"].mean(),
            "jaccard_case_mean": detail["jaccard"].mean(),
            "sample_level_accuracy": detail[
                "sample_level_accuracy"
            ].mean(),
            "exact_match": detail["exact_match"].mean(),
            "inclusive_accuracy": detail[
                "inclusive_accuracy"
            ].mean(),
            "case_accuracy": detail["exact_match"].mean(),
            "over_prediction_rate": detail[
                "over_prediction"
            ].mean(),
            "missed_injury_rate": detail[
                "missed_injury"
            ].mean(),
        }
    ])

    return summary, detail


# ============================================================
# Export helpers
# ============================================================

def stringify_set_columns(df, columns):
    out = df.copy()

    for col in columns:
        out[col] = out[col].apply(
            lambda x: ",".join(sorted(set(x)))
            if isinstance(x, (set, list, tuple))
            else x
        )

    return out
