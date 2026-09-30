# -*- coding: utf-8 -*-

import argparse
import os
from collections import defaultdict

os.environ["TORCH_COMPILE_DISABLE"] = "1"

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from infer_common import (
    DEVICE,
    apply_global_threshold,
    clean_code7,
    get_caseid_col,
    get_code_col,
    get_text_col,
    load_json,
    load_jsonl,
    load_lvl_map_txt,
    load_modernbert_classifier_model,
    make_d1_input,
    make_d2_input,
    make_d34_input,
    make_d56_input,
    parse_codes,
    predict_logits_probs,
    save_excel,
    stringify_set_columns,
    summarize_case_df,
)


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run hierarchical AIS inference: "
            "D1 -> D2 -> D3-D4 -> D5-D6 -> D7."
        )
    )

    # Evaluation datasets
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

    # Stage models
    parser.add_argument(
        "--d1-model-dir",
        required=True,
        help="Path to the trained D1 model.",
    )
    parser.add_argument(
        "--d2-model-dir",
        required=True,
        help="Path to the trained D2 model.",
    )
    parser.add_argument(
        "--d34-model-dir",
        required=True,
        help="Path to the trained D3-D4 model.",
    )
    parser.add_argument(
        "--d56-model-dir",
        required=True,
        help="Path to the trained D5-D6 model.",
    )

    # Label maps
    parser.add_argument(
        "--label-map-dir",
        required=True,
        help="Directory containing stage label2id JSON files.",
    )

    # AIS hierarchy
    parser.add_argument(
        "--lvl-map-12",
        required=True,
        help="Path to the D1/D2 hierarchy mapping.",
    )
    parser.add_argument(
        "--lvl-map-34",
        required=True,
        help="Path to the D3-D4 hierarchy mapping.",
    )
    parser.add_argument(
        "--lvl-map-56",
        required=True,
        help="Path to the D5-D6/D7 hierarchy mapping.",
    )

    # Thresholds
    parser.add_argument(
        "--d1-threshold",
        required=True,
        type=float,
        help="Validation-selected global threshold for D1.",
    )
    parser.add_argument(
        "--d2-threshold",
        required=True,
        type=float,
        help="Validation-selected global threshold for D2.",
    )
    parser.add_argument(
        "--d34-threshold",
        required=True,
        type=float,
        help="Validation-selected global threshold for D3-D4.",
    )
    parser.add_argument(
        "--d56-threshold",
        required=True,
        type=float,
        help="Validation-selected global threshold for D5-D6.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory used to save hierarchical evaluation results.",
    )

    return parser.parse_args()


# ============================================================
# Label-map utilities
# ============================================================

def load_stage_labels(label_map_dir, stage):
    path = os.path.join(
        label_map_dir,
        f"{stage}_label2id.json",
    )

    label2id = {
        str(key): int(value)
        for key, value in load_json(path).items()
    }

    id2label = {
        value: key
        for key, value in label2id.items()
    }

    labels = [
        id2label[i]
        for i in range(len(id2label))
    ]

    return label2id, id2label, labels


# ============================================================
# Dataset construction
# ============================================================

def build_cases(df):
    text_col = get_text_col(df)
    code_col = get_code_col(df)
    case_col = get_caseid_col(df)

    cases = []

    for _, row in df.iterrows():
        case_id = str(row[case_col])

        raw_text = (
            str(row[text_col])
            if pd.notna(row[text_col])
            else ""
        )

        codes7 = {
            clean_code7(code)
            for code in parse_codes(row[code_col])
        }

        true6 = {
            code[:6]
            for code in codes7
        }

        cases.append(
            {
                "case_id": case_id,
                "raw_text": raw_text,
                "true7": codes7,
                "true6": true6,
            }
        )

    return cases


# ============================================================
# D7 reconstruction
# ============================================================

def build_d7_mapping(lvl56):
    """
    Construct a six-digit AIS -> D7 mapping from the final
    hierarchy level.

    A six-digit code is considered deterministic only when
    exactly one D7 value is associated with that code.
    """

    mapping = defaultdict(set)

    for key, child_map in lvl56.items():
        level, parents = key

        if level != 4:
            continue

        if len(parents) != 3:
            continue

        d1 = str(parents[0])
        d2 = str(parents[1])
        d34 = str(parents[2]).zfill(2)

        for d567 in child_map.keys():
            d567 = str(d567).zfill(3)

            d56 = d567[:2]
            d7 = d567[2]

            full6 = (
                d1
                + d2
                + d34
                + d56
            )

            mapping[full6].add(d7)

    return mapping


def reconstruct_d7(pred6_set, d7_mapping):
    pred7 = set()

    missing = set()
    ambiguous = set()

    for code6 in pred6_set:
        d7_values = d7_mapping.get(
            code6,
            set(),
        )

        if len(d7_values) == 1:
            d7 = next(iter(d7_values))
            pred7.add(code6 + d7)

        elif len(d7_values) == 0:
            missing.add(code6)

        else:
            ambiguous.add(code6)

    return pred7, missing, ambiguous


# ============================================================
# Stage prediction helpers
# ============================================================

def predict_d1(
    cases,
    model,
    tokenizer,
    labels,
    threshold,
    lvl12,
):
    texts = [
        make_d1_input(
            case["raw_text"],
            lvl12,
        )
        for case in cases
    ]

    _, probs = predict_logits_probs(
        model,
        tokenizer,
        texts,
        desc="Hierarchical D1",
    )

    pred = apply_global_threshold(
        probs,
        threshold,
    )

    outputs = []

    for row in pred:
        outputs.append({
            labels[j]
            for j in range(len(labels))
            if row[j]
        })

    return outputs


def predict_d2(
    cases,
    pred_d1,
    model,
    tokenizer,
    labels,
    label2id,
    threshold,
    lvl12,
):
    """
    Predict D2 labels conditioned on predicted D1 labels.

    Returns complete D1-D2 prefixes.
    """

    outputs = []

    for case, d1_set in zip(cases, pred_d1):
        case_predictions = set()

        for d1 in sorted(d1_set):
            child_map = lvl12.get(
                (2, (str(d1),)),
                {},
            )

            valid_d2 = {
                str(value)
                for value in child_map.keys()
            }

            if not valid_d2:
                continue

            text = make_d2_input(
                d1,
                case["raw_text"],
                lvl12,
            )

            _, probs = predict_logits_probs(
                model,
                tokenizer,
                [text],
                desc=f"D2 case={case['case_id']} D1={d1}",
            )

            probs = probs[0]

            for d2 in valid_d2:
                if d2 not in label2id:
                    continue

                idx = label2id[d2]

                if probs[idx] >= threshold:
                    case_predictions.add(
                        str(d1) + str(d2)
                    )

        outputs.append(case_predictions)

    return outputs


def predict_d34(
    cases,
    pred_d12,
    model,
    tokenizer,
    labels,
    label2id,
    threshold,
    lvl34,
):
    """
    Predict D3-D4 labels conditioned on predicted D1-D2
    prefixes.

    Returns complete D1-D4 prefixes.
    """

    outputs = []

    for case, d12_set in zip(cases, pred_d12):
        case_predictions = set()

        for d12 in sorted(d12_set):
            if len(d12) != 2:
                continue

            d1 = d12[0]
            d2 = d12[1]

            child_map = lvl34.get(
                (
                    3,
                    (
                        str(d1),
                        str(d2),
                    ),
                ),
                {},
            )

            valid_d34 = {
                str(value).zfill(2)
                for value in child_map.keys()
            }

            if not valid_d34:
                continue

            text = make_d34_input(
                d1,
                d2,
                case["raw_text"],
                lvl34,
            )

            _, probs = predict_logits_probs(
                model,
                tokenizer,
                [text],
                desc=(
                    f"D34 case={case['case_id']} "
                    f"D12={d12}"
                ),
            )

            probs = probs[0]

            for d34 in valid_d34:
                if d34 not in label2id:
                    continue

                idx = label2id[d34]

                if probs[idx] >= threshold:
                    case_predictions.add(
                        d12 + d34
                    )

        outputs.append(case_predictions)

    return outputs


def predict_d56(
    cases,
    pred_d1234,
    model,
    tokenizer,
    labels,
    label2id,
    threshold,
    lvl56,
):
    """
    Predict D5-D6 labels conditioned on predicted D1-D4
    prefixes.

    Returns complete six-digit AIS codes.
    """

    outputs = []

    for case, d1234_set in zip(cases, pred_d1234):
        case_predictions = set()

        for d1234 in sorted(d1234_set):
            if len(d1234) != 4:
                continue

            d1 = d1234[0]
            d2 = d1234[1]
            d34 = d1234[2:4]

            child_map = lvl56.get(
                (
                    4,
                    (
                        str(d1),
                        str(d2),
                        str(d34).zfill(2),
                    ),
                ),
                {},
            )

            valid_d56 = {
                str(value).zfill(3)[:2]
                for value in child_map.keys()
            }

            if not valid_d56:
                continue

            text = make_d56_input(
                d1,
                d2,
                d34,
                case["raw_text"],
                lvl56,
            )

            _, probs = predict_logits_probs(
                model,
                tokenizer,
                [text],
                desc=(
                    f"D56 case={case['case_id']} "
                    f"D1234={d1234}"
                ),
            )

            probs = probs[0]

            for d56 in valid_d56:
                if d56 not in label2id:
                    continue

                idx = label2id[d56]

                if probs[idx] >= threshold:
                    case_predictions.add(
                        d1234 + d56
                    )

        outputs.append(case_predictions)

    return outputs


# ============================================================
# Evaluation helpers
# ============================================================

def make_stage_case_df(
    cases,
    pred_sets,
    digits,
    true_col,
    pred_col,
):
    rows = []

    for case, pred_set in zip(cases, pred_sets):
        if digits == 1:
            true_set = {
                code[0]
                for code in case["true7"]
            }

        elif digits == 2:
            true_set = {
                code[:2]
                for code in case["true7"]
            }

        elif digits == 4:
            true_set = {
                code[:4]
                for code in case["true7"]
            }

        elif digits == 6:
            true_set = set(
                case["true6"]
            )

        elif digits == 7:
            true_set = set(
                case["true7"]
            )

        else:
            raise ValueError(
                f"Unsupported AIS digit length: {digits}"
            )

        rows.append(
            {
                "case_id": case["case_id"],
                true_col: true_set,
                pred_col: set(pred_set),
            }
        )

    return pd.DataFrame(rows)


def count_stage_universe(
    lvl12,
    lvl34,
    lvl56,
):
    d1_universe = set()
    d12_universe = set()
    d1234_universe = set()
    d6_universe = set()
    d7_universe = set()

    # D1 / D12
    d1_map = lvl12.get(
        (1, ()),
        {},
    )

    for d1 in d1_map.keys():
        d1 = str(d1)
        d1_universe.add(d1)

        child_map = lvl12.get(
            (2, (d1,)),
            {},
        )

        for d2 in child_map.keys():
            d12_universe.add(
                d1 + str(d2)
            )

    # D1234
    for key, child_map in lvl34.items():
        level, parents = key

        if level != 3 or len(parents) != 2:
            continue

        d1 = str(parents[0])
        d2 = str(parents[1])

        for d34 in child_map.keys():
            d1234_universe.add(
                d1
                + d2
                + str(d34).zfill(2)
            )

    # D6 / D7
    for key, child_map in lvl56.items():
        level, parents = key

        if level != 4 or len(parents) != 3:
            continue

        d1 = str(parents[0])
        d2 = str(parents[1])
        d34 = str(parents[2]).zfill(2)

        for d567 in child_map.keys():
            d567 = str(d567).zfill(3)

            code6 = (
                d1
                + d2
                + d34
                + d567[:2]
            )

            code7 = code6 + d567[2]

            d6_universe.add(code6)
            d7_universe.add(code7)

    return {
        "d1": len(d1_universe),
        "d12": len(d12_universe),
        "d1234": len(d1234_universe),
        "d6": len(d6_universe),
        "d7": len(d7_universe),
    }


# ============================================================
# Split-level hierarchical inference
# ============================================================

def run_split(
    split,
    path,
    output_dir,
    models,
    tokenizers,
    stage_labels,
    stage_label2id,
    thresholds,
    lvl12,
    lvl34,
    lvl56,
    d7_mapping,
    universe_sizes,
):
    print("\n" + "=" * 80)
    print(f"HIERARCHICAL INFERENCE: {split}")
    print("=" * 80)

    df = load_jsonl(path)
    cases = build_cases(df)

    print("Cases:", len(cases))

    # --------------------------------------------------------
    # D1
    # --------------------------------------------------------

    pred_d1 = predict_d1(
        cases=cases,
        model=models["d1"],
        tokenizer=tokenizers["d1"],
        labels=stage_labels["d1"],
        threshold=thresholds["d1"],
        lvl12=lvl12,
    )

    # --------------------------------------------------------
    # D2
    # --------------------------------------------------------

    pred_d12 = predict_d2(
        cases=cases,
        pred_d1=pred_d1,
        model=models["d2"],
        tokenizer=tokenizers["d2"],
        labels=stage_labels["d2"],
        label2id=stage_label2id["d2"],
        threshold=thresholds["d2"],
        lvl12=lvl12,
    )

    # --------------------------------------------------------
    # D3-D4
    # --------------------------------------------------------

    pred_d1234 = predict_d34(
        cases=cases,
        pred_d12=pred_d12,
        model=models["d34"],
        tokenizer=tokenizers["d34"],
        labels=stage_labels["d34"],
        label2id=stage_label2id["d34"],
        threshold=thresholds["d34"],
        lvl34=lvl34,
    )

    # --------------------------------------------------------
    # D5-D6
    # --------------------------------------------------------

    pred_d6 = predict_d56(
        cases=cases,
        pred_d1234=pred_d1234,
        model=models["d56"],
        tokenizer=tokenizers["d56"],
        labels=stage_labels["d56"],
        label2id=stage_label2id["d56"],
        threshold=thresholds["d56"],
        lvl56=lvl56,
    )

    # --------------------------------------------------------
    # D7 reconstruction
    # --------------------------------------------------------

    pred_d7 = []
    missing_d7 = []
    ambiguous_d7 = []

    for pred6_set in pred_d6:
        reconstructed, missing, ambiguous = reconstruct_d7(
            pred6_set,
            d7_mapping,
        )

        pred_d7.append(reconstructed)
        missing_d7.append(missing)
        ambiguous_d7.append(ambiguous)

    # --------------------------------------------------------
    # Stage-level case dataframes
    # --------------------------------------------------------

    d1_df = make_stage_case_df(
        cases,
        pred_d1,
        digits=1,
        true_col="true_d1",
        pred_col="pred_d1",
    )

    d12_df = make_stage_case_df(
        cases,
        pred_d12,
        digits=2,
        true_col="true_d12",
        pred_col="pred_d12",
    )

    d1234_df = make_stage_case_df(
        cases,
        pred_d1234,
        digits=4,
        true_col="true_d1234",
        pred_col="pred_d1234",
    )

    d6_df = make_stage_case_df(
        cases,
        pred_d6,
        digits=6,
        true_col="true_d6",
        pred_col="pred_d6",
    )

    d7_df = make_stage_case_df(
        cases,
        pred_d7,
        digits=7,
        true_col="true_d7",
        pred_col="pred_d7",
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    summaries = []
    details = {}

    stage_configs = [
        (
            "d1",
            d1_df,
            "true_d1",
            "pred_d1",
            universe_sizes["d1"],
        ),
        (
            "d12",
            d12_df,
            "true_d12",
            "pred_d12",
            universe_sizes["d12"],
        ),
        (
            "d1234",
            d1234_df,
            "true_d1234",
            "pred_d1234",
            universe_sizes["d1234"],
        ),
        (
            "d6",
            d6_df,
            "true_d6",
            "pred_d6",
            universe_sizes["d6"],
        ),
        (
            "d7",
            d7_df,
            "true_d7",
            "pred_d7",
            universe_sizes["d7"],
        ),
    ]

    for stage, stage_df, true_col, pred_col, universe_size in stage_configs:
        summary, detail = summarize_case_df(
            stage_df,
            true_col,
            pred_col,
            universe_size,
            split,
            stage,
        )

        summaries.append(summary)
        details[stage] = detail

    summary_df = pd.concat(
        summaries,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Final case-level export
    # --------------------------------------------------------

    final_rows = []

    for i, case in enumerate(cases):
        final_rows.append(
            {
                "case_id": case["case_id"],
                "true_d1": {
                    code[0]
                    for code in case["true7"]
                },
                "pred_d1": pred_d1[i],
                "true_d12": {
                    code[:2]
                    for code in case["true7"]
                },
                "pred_d12": pred_d12[i],
                "true_d1234": {
                    code[:4]
                    for code in case["true7"]
                },
                "pred_d1234": pred_d1234[i],
                "true_d6": case["true6"],
                "pred_d6": pred_d6[i],
                "true_d7": case["true7"],
                "pred_d7": pred_d7[i],
                "d7_missing_mapping": missing_d7[i],
                "d7_ambiguous_mapping": ambiguous_d7[i],
            }
        )

    final_case_df = pd.DataFrame(
        final_rows
    )

    # --------------------------------------------------------
    # D7 mapping QC
    # --------------------------------------------------------

    n_missing_cases = sum(
        bool(value)
        for value in missing_d7
    )

    n_ambiguous_cases = sum(
        bool(value)
        for value in ambiguous_d7
    )

    n_missing_codes = sum(
        len(value)
        for value in missing_d7
    )

    n_ambiguous_codes = sum(
        len(value)
        for value in ambiguous_d7
    )

    d7_qc = pd.DataFrame([
        {
            "split": split,
            "n_cases": len(cases),
            "cases_with_missing_d7_mapping": n_missing_cases,
            "cases_with_ambiguous_d7_mapping": n_ambiguous_cases,
            "predicted_d6_codes_without_d7_mapping": n_missing_codes,
            "predicted_d6_codes_with_ambiguous_d7_mapping": n_ambiguous_codes,
        }
    ])

    # --------------------------------------------------------
    # Save split outputs
    # --------------------------------------------------------

    split_dir = os.path.join(
        output_dir,
        split,
    )

    os.makedirs(
        split_dir,
        exist_ok=True,
    )

    summary_df.to_csv(
        os.path.join(
            split_dir,
            "hierarchical_summary.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    d7_qc.to_csv(
        os.path.join(
            split_dir,
            "d7_mapping_qc.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    export_case_df = stringify_set_columns(
        final_case_df,
        [
            "true_d1",
            "pred_d1",
            "true_d12",
            "pred_d12",
            "true_d1234",
            "pred_d1234",
            "true_d6",
            "pred_d6",
            "true_d7",
            "pred_d7",
            "d7_missing_mapping",
            "d7_ambiguous_mapping",
        ],
    )

    export_case_df.to_csv(
        os.path.join(
            split_dir,
            "hierarchical_case_predictions.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Excel export
    # --------------------------------------------------------

    excel_sheets = {
        "summary": summary_df,
        "d7_qc": d7_qc,
        "case_predictions": export_case_df,
    }

    for stage, detail in details.items():
        true_col = {
            "d1": "true_d1",
            "d12": "true_d12",
            "d1234": "true_d1234",
            "d6": "true_d6",
            "d7": "true_d7",
        }[stage]

        pred_col = {
            "d1": "pred_d1",
            "d12": "pred_d12",
            "d1234": "pred_d1234",
            "d6": "pred_d6",
            "d7": "pred_d7",
        }[stage]

        excel_sheets[stage] = stringify_set_columns(
            detail,
            [
                true_col,
                pred_col,
            ],
        )

    save_excel(
        excel_sheets,
        os.path.join(
            split_dir,
            "hierarchical_evaluation.xlsx",
        ),
    )

    print("\nResults:")
    print(summary_df.round(4))

    print(
        "\nD7 missing mapping cases:",
        n_missing_cases,
    )

    print(
        "D7 ambiguous mapping cases:",
        n_ambiguous_cases,
    )

    return (
        summary_df,
        d7_qc,
        export_case_df,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    output_dir = os.path.abspath(
        args.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
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

    model_dirs = {
        "d1": os.path.abspath(
            args.d1_model_dir
        ),
        "d2": os.path.abspath(
            args.d2_model_dir
        ),
        "d34": os.path.abspath(
            args.d34_model_dir
        ),
        "d56": os.path.abspath(
            args.d56_model_dir
        ),
    }

    label_map_dir = os.path.abspath(
        args.label_map_dir
    )

    lvl_map_files = {
        "12": os.path.abspath(
            args.lvl_map_12
        ),
        "34": os.path.abspath(
            args.lvl_map_34
        ),
        "56": os.path.abspath(
            args.lvl_map_56
        ),
    }

    thresholds = {
        "d1": args.d1_threshold,
        "d2": args.d2_threshold,
        "d34": args.d34_threshold,
        "d56": args.d56_threshold,
    }

    # --------------------------------------------------------
    # Validate required inputs
    # --------------------------------------------------------

    for path in files.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Dataset does not exist: {path}"
            )

    for path in lvl_map_files.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Hierarchy mapping does not exist: {path}"
            )

    for stage, path in model_dirs.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"{stage} model directory does not exist: {path}"
            )

    for stage in ["d1", "d2", "d34", "d56"]:
        path = os.path.join(
            label_map_dir,
            f"{stage}_label2id.json",
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Label map does not exist: {path}"
            )

    # --------------------------------------------------------
    # Load AIS hierarchy
    # --------------------------------------------------------

    lvl12 = load_lvl_map_txt(
        lvl_map_files["12"]
    )

    lvl34 = load_lvl_map_txt(
        lvl_map_files["34"]
    )

    lvl56 = load_lvl_map_txt(
        lvl_map_files["56"]
    )

    # --------------------------------------------------------
    # Load stage label spaces
    # --------------------------------------------------------

    stage_label2id = {}
    stage_labels = {}

    for stage in [
        "d1",
        "d2",
        "d34",
        "d56",
    ]:
        label2id, _, labels = (
            load_stage_labels(
                label_map_dir,
                stage,
            )
        )

        stage_label2id[stage] = label2id
        stage_labels[stage] = labels

    # --------------------------------------------------------
    # Load tokenizers and models
    # --------------------------------------------------------

    tokenizers = {}
    models = {}

    print("\n" + "=" * 80)
    print("LOADING HIERARCHICAL MODELS")
    print("=" * 80)
    print("Device:", DEVICE)

    for stage in [
        "d1",
        "d2",
        "d34",
        "d56",
    ]:
        print(f"Loading {stage}...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_dirs[stage],
            use_fast=True,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizers[stage] = tokenizer

        models[stage] = (
            load_modernbert_classifier_model(
                model_dirs[stage]
            )
        )

    # --------------------------------------------------------
    # D7 reconstruction map
    # --------------------------------------------------------

    d7_mapping = build_d7_mapping(
        lvl56
    )

    n_unique_d7 = sum(
        len(values) == 1
        for values in d7_mapping.values()
    )

    n_ambiguous_d7 = sum(
        len(values) > 1
        for values in d7_mapping.values()
    )

    print("\nD7 mapping")
    print("Six-digit codes:", len(d7_mapping))
    print("Unique mappings:", n_unique_d7)
    print("Ambiguous mappings:", n_ambiguous_d7)

    # --------------------------------------------------------
    # Universe sizes
    # --------------------------------------------------------

    universe_sizes = count_stage_universe(
        lvl12,
        lvl34,
        lvl56,
    )

    print("\nAIS hierarchy universe sizes")

    for stage, size in universe_sizes.items():
        print(f"{stage}: {size}")

    # --------------------------------------------------------
    # Run all evaluation splits
    # --------------------------------------------------------

    all_summaries = []
    all_d7_qc = []

    for split in [
        "internal_val",
        "internal_test",
        "external_val",
    ]:
        summary, d7_qc, _ = run_split(
            split=split,
            path=files[split],
            output_dir=output_dir,
            models=models,
            tokenizers=tokenizers,
            stage_labels=stage_labels,
            stage_label2id=stage_label2id,
            thresholds=thresholds,
            lvl12=lvl12,
            lvl34=lvl34,
            lvl56=lvl56,
            d7_mapping=d7_mapping,
            universe_sizes=universe_sizes,
        )

        all_summaries.append(
            summary
        )

        all_d7_qc.append(
            d7_qc
        )

    # --------------------------------------------------------
    # Combined summary
    # --------------------------------------------------------

    final_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    final_d7_qc = pd.concat(
        all_d7_qc,
        ignore_index=True,
    )

    final_summary.to_csv(
        os.path.join(
            output_dir,
            "hierarchical_final_summary.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    final_d7_qc.to_csv(
        os.path.join(
            output_dir,
            "hierarchical_d7_mapping_qc.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    save_excel(
        {
            "summary": final_summary,
            "d7_qc": final_d7_qc,
        },
        os.path.join(
            output_dir,
            "hierarchical_final_results.xlsx",
        ),
    )

    print("\n" + "=" * 80)
    print("FINAL HIERARCHICAL RESULTS")
    print("=" * 80)

    print(
        final_summary.round(4)
    )

    print(
        "\nResults saved to:",
        output_dir,
    )


if __name__ == "__main__":
    main()
