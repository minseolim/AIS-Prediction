# -*- coding: utf-8 -*-

import argparse
import os

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from filteredv3_infer_common_observed import (
    BASE_MODEL_NAME,
    DEVICE,
    apply_global_threshold,
    calibrate_global_micro_f1_threshold,
    clean_code7,
    get_caseid_col,
    get_code_col,
    get_text_col,
    load_json,
    load_jsonl,
    load_lvl_map_txt,
    load_modernbert_classifier_model,
    make_d1_input,
    parse_codes,
    predict_logits_probs,
    save_excel,
    save_json,
    stringify_set_columns,
    summarize_case_df,
)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run D1 AIS inference and evaluate the trained "
            "BioClinical ModernBERT classifier."
        )
    )

    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the trained D1 model directory.",
    )

    parser.add_argument(
        "--label-map-dir",
        required=True,
        help="Directory containing d1_label2id.json.",
    )

    parser.add_argument(
        "--lvl-map-12",
        required=True,
        help="Path to the corrected D1/D2 hierarchy mapping file.",
    )

    parser.add_argument(
        "--internal-val",
        required=True,
        help="Path to the internal validation JSONL file.",
    )

    parser.add_argument(
        "--internal-test",
        required=True,
        help="Path to the internal test JSONL file.",
    )

    parser.add_argument(
        "--external-val",
        required=True,
        help="Path to the external validation JSONL file.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory used to save inference results.",
    )

    return parser.parse_args()


# ============================================================
# Input construction
# ============================================================

def build(df, lvl12):
    """
    Build D1 inference inputs.

    Each row corresponds to one case-level merged CT report.

    Returns
    -------
    rows : list of dict
        Case metadata and ground-truth D1 labels.

    texts : list of str
        Model inputs containing the AIS coding policy,
        available D1 codebook, and normalized CT report.
    """

    text_col = get_text_col(df)
    code_col = get_code_col(df)
    case_col = get_caseid_col(df)

    rows = []
    texts = []

    for _, row in df.iterrows():
        case_id = str(row[case_col])

        raw_text = (
            str(row[text_col])
            if pd.notna(row[text_col])
            else ""
        )

        codes = [
            clean_code7(code)
            for code in parse_codes(row[code_col])
        ]

        true_d1 = {
            code[0]
            for code in codes
        }

        rows.append(
            {
                "case_id": case_id,
                "raw_text": raw_text,
                "true": true_d1,
            }
        )

        texts.append(
            make_d1_input(
                raw_text,
                lvl12,
            )
        )

    return rows, texts


# ============================================================
# Ground-truth matrix
# ============================================================

def make_y(rows, labels, label2id):
    """
    Convert case-level D1 ground truth into a multi-hot matrix.
    """

    y = np.zeros(
        (
            len(rows),
            len(labels),
        ),
        dtype=np.int32,
    )

    for i, row in enumerate(rows):
        for label in row["true"]:
            if label in label2id:
                y[i, label2id[label]] = 1

    return y


# ============================================================
# Case-level prediction table
# ============================================================

def make_case_df(
    rows,
    probs,
    threshold,
    labels,
):
    """
    Apply the global threshold and construct case-level
    ground-truth and prediction sets.
    """

    pred = apply_global_threshold(
        probs,
        threshold,
    )

    output_rows = []

    for row, prediction in zip(rows, pred):
        pred_set = {
            labels[j]
            for j in range(len(labels))
            if prediction[j]
        }

        output_rows.append(
            {
                "case_id": row["case_id"],
                "raw_text": row["raw_text"],
                "true_d1": set(row["true"]),
                "pred_d1": pred_set,
            }
        )

    return pd.DataFrame(output_rows)


# ============================================================
# Split inference
# ============================================================

def run_split(
    split,
    path,
    save_dir,
    model,
    tokenizer,
    lvl12,
    labels,
    label2id,
    threshold=None,
    calibrate=False,
):
    """
    Run D1 inference for one dataset split.

    If calibrate=True, a global threshold maximizing validation
    micro-F1 is selected from this split.
    """

    split_dir = os.path.join(
        save_dir,
        split,
    )

    os.makedirs(
        split_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_jsonl(path)

    rows, texts = build(
        df,
        lvl12,
    )

    print("\n" + "=" * 80)
    print(f"D1 INFERENCE: {split}")
    print("=" * 80)
    print("Cases:", len(rows))

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    logits, probs = predict_logits_probs(
        model,
        tokenizer,
        texts,
        desc=f"D1 {split}",
    )

    # --------------------------------------------------------
    # Save label-level logits and probabilities
    # --------------------------------------------------------

    long_rows = []

    for i, row in enumerate(rows):
        for j, label in enumerate(labels):
            long_rows.append(
                {
                    "split": split,
                    "case_id": row["case_id"],
                    "d1_label": label,
                    "is_true": int(
                        label in row["true"]
                    ),
                    "logit": float(
                        logits[i, j]
                    ),
                    "probability": float(
                        probs[i, j]
                    ),
                }
            )

    long_df = pd.DataFrame(long_rows)

    long_df.to_csv(
        os.path.join(
            split_dir,
            "d1_case_label_logits_probs_long.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Threshold calibration
    # --------------------------------------------------------

    if calibrate:
        y_true = make_y(
            rows,
            labels,
            label2id,
        )

        threshold, sweep, best = (
            calibrate_global_micro_f1_threshold(
                y_true,
                probs,
            )
        )

        save_json(
            {
                "stage": "d1",
                "method": "global_micro_f1",
                "threshold": threshold,
                "best_validation_row": best,
            },
            os.path.join(
                save_dir,
                "selected_threshold_d1_global_microf1.json",
            ),
        )

        sweep.to_csv(
            os.path.join(
                save_dir,
                "threshold_sweep_d1_global_microf1.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

        print(
            "[D1] selected global threshold = "
            f"{threshold:.4f}"
        )

    if threshold is None:
        raise ValueError(
            "A threshold must be supplied or calibrated."
        )

    # --------------------------------------------------------
    # Case-level predictions
    # --------------------------------------------------------

    case_df = make_case_df(
        rows,
        probs,
        threshold,
        labels,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    summary, detail = summarize_case_df(
        case_df,
        "true_d1",
        "pred_d1",
        len(labels),
        split,
        "d1",
    )

    summary["global_threshold"] = threshold

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    summary.to_csv(
        os.path.join(
            split_dir,
            "d1_summary.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    detail.to_csv(
        os.path.join(
            split_dir,
            "d1_detail.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Prediction export
    # --------------------------------------------------------

    export = stringify_set_columns(
        case_df,
        [
            "true_d1",
            "pred_d1",
        ],
    )

    export.to_csv(
        os.path.join(
            split_dir,
            "d1_case_prediction_export.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    return threshold, summary, export


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    model_dir = os.path.abspath(
        args.model_dir
    )

    label_map_dir = os.path.abspath(
        args.label_map_dir
    )

    lvl_map_file = os.path.abspath(
        args.lvl_map_12
    )

    save_dir = os.path.abspath(
        args.output_dir
    )

    files = {
        "internal_val": os.path.abspath(
            args.internal_val
        ),
        "internal_test": os.path.abspath(
            args.internal_test
        ),
        "external_val": os.path.abspath(
            args.external_val
        ),
    }

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # File sanity check
    # --------------------------------------------------------

    required_files = [
        lvl_map_file,
        os.path.join(
            label_map_dir,
            "d1_label2id.json",
        ),
        files["internal_val"],
        files["internal_test"],
        files["external_val"],
    ]

    for path in required_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required file does not exist: {path}"
            )

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"Model directory does not exist: {model_dir}"
        )

    # --------------------------------------------------------
    # Load AIS hierarchy
    # --------------------------------------------------------

    lvl12 = load_lvl_map_txt(
        lvl_map_file
    )

    # --------------------------------------------------------
    # Load classifier label space
    # --------------------------------------------------------

    label2id = {
        str(key): int(value)
        for key, value in load_json(
            os.path.join(
                label_map_dir,
                "d1_label2id.json",
            )
        ).items()
    }

    id2label = {
        value: key
        for key, value in label2id.items()
    }

    labels = [
        id2label[i]
        for i in range(len(id2label))
    ]

    print("\n" + "=" * 80)
    print("D1 INFERENCE CONFIGURATION")
    print("=" * 80)
    print("Backbone:", BASE_MODEL_NAME)
    print("Device:", DEVICE)
    print("Number of labels:", len(labels))
    print("Labels:", labels)

    # --------------------------------------------------------
    # Tokenizer / model
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_modernbert_classifier_model(
        model_dir
    )

    # --------------------------------------------------------
    # Internal validation
    #
    # The global threshold is selected only on the internal
    # validation dataset.
    # --------------------------------------------------------

    threshold, val_summary, val_export = run_split(
        split="internal_val",
        path=files["internal_val"],
        save_dir=save_dir,
        model=model,
        tokenizer=tokenizer,
        lvl12=lvl12,
        labels=labels,
        label2id=label2id,
        calibrate=True,
    )

    # --------------------------------------------------------
    # Internal test
    #
    # Reuse the threshold selected on internal validation.
    # --------------------------------------------------------

    _, test_summary, test_export = run_split(
        split="internal_test",
        path=files["internal_test"],
        save_dir=save_dir,
        model=model,
        tokenizer=tokenizer,
        lvl12=lvl12,
        labels=labels,
        label2id=label2id,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # External validation
    #
    # Reuse exactly the same validation-selected threshold.
    # --------------------------------------------------------

    _, ext_summary, ext_export = run_split(
        split="external_val",
        path=files["external_val"],
        save_dir=save_dir,
        model=model,
        tokenizer=tokenizer,
        lvl12=lvl12,
        labels=labels,
        label2id=label2id,
        threshold=threshold,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    final_summary = pd.concat(
        [
            val_summary,
            test_summary,
            ext_summary,
        ],
        ignore_index=True,
    )

    final_summary.to_csv(
        os.path.join(
            save_dir,
            "final_summary_d1_global_microf1.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Excel export
    # --------------------------------------------------------

    save_excel(
        {
            "summary": final_summary,
            "val": val_export,
            "test": test_export,
            "ext": ext_export,
        },
        os.path.join(
            save_dir,
            "d1_case_predictions_global_microf1.xlsx",
        ),
    )

    print("\n" + "=" * 80)
    print("FINAL D1 RESULTS")
    print("=" * 80)

    print(
        final_summary.round(4)
    )

    print("\nSelected threshold:", threshold)
    print("Results saved to:", save_dir)


if __name__ == "__main__":
    main()
