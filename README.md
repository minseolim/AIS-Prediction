# AIS Prediction from Trauma CT Reports

## Overview

This project presents a hierarchical Clinical NLP framework for predicting **Abbreviated Injury Scale (AIS)** codes from trauma CT radiology reports.

The framework predicts AIS codes using CT radiology reports without incorporating additional patient-level clinical metadata such as age, sex, or other demographic variables.

The backbone model is **BioClinical ModernBERT**, an encoder-only Transformer pretrained on biomedical and clinical text. The model is used for multi-label classification at successive levels of the AIS hierarchy.

Because AIS codes have a hierarchical structure, prediction is decomposed into multiple stages rather than directly predicting the complete AIS code in a single classification step.

The overall prediction pipeline is:

```text
Trauma CT Reports
        │
        ▼
Text Preprocessing
        │
        ▼
Uncertainty-aware Input Construction
        │
        ▼
BioClinical ModernBERT
        │
        ▼
D1 Prediction
        │
        ▼
D2 Prediction
        │
        ▼
D3-D4 Prediction
        │
        ▼
D5-D6 Prediction
        │
        ▼
D7 Reconstruction
        │
        ▼
Final AIS Code
```

---

# Dataset

## 1. Development and Internal Validation Dataset

The development and internal validation datasets consist of trauma CT radiology reports collected from three medical institutions.

Multiple CT reports belonging to the same patient visit were grouped into a single **Case ID**, generated from the patient identifier and visit date.

A total of **24,005 cases** were used for model development and internal validation.

| Dataset | Number of Cases |
| --- | ---: |
| Development & Internal Validation | 24,005 |

Patient-level data and institution-specific source files are **not included in this repository**.

---

## 2. External Validation Dataset

External validation was performed using trauma CT radiology reports collected from an independent medical institution.

A total of **1,460 cases** were used to evaluate the generalization performance of the framework.

| Dataset | Number of Cases |
| --- | ---: |
| External Validation | 1,460 |

The external validation data are also not publicly distributed.

---

# Method

## 1. Dataset Construction

AIS codes from the trauma registry were matched with CT radiology reports using their corresponding Case IDs.

To improve consistency between radiologic findings and AIS labels, AIS codes were filtered according to the anatomical region represented by each CT examination.

Since the first digit (**D1**) of an AIS code represents the body region, CT examination types were mapped to the corresponding AIS body regions. AIS codes inconsistent with the anatomical region of the examination were excluded during dataset construction.

Conceptually:

```text
CT Examination
      │
      ▼
Anatomical Region Mapping
      │
      ▼
Compatible AIS D1 Region
      │
      ▼
Region-consistent AIS Labels
```

This filtering step was performed before model development to reduce mismatches between the radiology report content and the associated AIS labels.

---

## 2. Case-level Input Construction

Multiple CT radiology reports belonging to the same Case ID were merged into a single case-level input document.

Minimal text preprocessing was applied, including normalization of whitespace and line breaks.

The resulting case-level report is used as the textual input to the hierarchical AIS classifiers.

The public input format is described in:

```text
input_schema.json
```

Each sample contains the following conceptual fields:

```text
case_id
prompt
completion
```

where:

- `case_id` represents a unique patient visit.
- `prompt` contains the merged trauma CT radiology report.
- `completion` contains the corresponding ground-truth AIS code list.

The schema is provided only to document the expected data structure. **No real patient examples are included.**

---

## 3. Uncertainty-aware Text Processing

Radiology reports frequently contain uncertain diagnostic expressions such as:

```text
suspicious for
suspected
possible
possibly
probable
R/O
cannot exclude
suggestive of
may represent
```

For AIS prediction, uncertain traumatic findings may still contain clinically relevant evidence.

The preprocessing module therefore detects predefined uncertainty expressions and marks uncertain findings as candidate-positive evidence while distinguishing them from explicitly negated findings.

For example:

```text
Suspicious fracture of the left rib.
```

is transformed conceptually into:

```text
[UNCERTAIN_POSITIVE] Suspicious fracture of the left rib.
```

Explicitly negative findings are not marked as positive evidence.

The implementation is provided in:

```text
ais_text_rules.py
```

The uncertainty policy is also incorporated into the model input together with the AIS codebook information.

---

# Hierarchical AIS Prediction

AIS codes contain increasingly specific information across their digit positions.

The prediction task is therefore decomposed into the following stages:

| Stage | Target |
| --- | --- |
| D1 | Body region |
| D2 | Anatomical structure |
| D3-D4 | Injury type / anatomical specification |
| D5-D6 | Injury specification / severity-related code component |
| D7 | AIS severity score |

The hierarchical prediction flow is:

```text
D1
 │
 ▼
D1-D2
 │
 ▼
D1-D2-D3D4
 │
 ▼
D1-D2-D3D4-D5D6
 │
 ▼
6-digit AIS code
 │
 ▼
D7 reconstruction
 │
 ▼
7-digit AIS code
```

Each classifier predicts only labels compatible with the upstream hierarchical prediction and the corresponding AIS codebook.

This constrains the prediction space to hierarchically valid AIS candidates.

---

## 4. Stage-wise Classification

Each stage uses **BioClinical ModernBERT** as the text encoder.

For a given hierarchical stage, the input consists conceptually of:

```text
[AIS CODING POLICY]

[UPSTREAM AIS CONTEXT]

[AVAILABLE AIS CODES]

[REPORT]
<CT radiology report>
```

The encoder output is passed to a multi-label classification head.

Sigmoid probabilities are generated independently for candidate labels, allowing multiple injuries to be predicted for a single case.

The general structure is:

```text
Case-level CT Report
        +
AIS Hierarchical Context
        +
Candidate Codebook
        │
        ▼
BioClinical ModernBERT
        │
        ▼
Multi-label Classification Head
        │
        ▼
Sigmoid Probabilities
        │
        ▼
Thresholding
        │
        ▼
Predicted AIS Labels
```

---

## 5. Representative D1 Implementation

The repository provides the complete D1 training and inference implementation as a representative example of the stage-wise hierarchical classifiers.

The D1 training implementation is provided in:

```text
d1_train.py
```

and includes:

- case-level input construction,
- uncertainty-aware preprocessing,
- AIS codebook context,
- observed training/validation label-space construction,
- multi-label target encoding,
- class-weighted binary cross-entropy loss,
- BioClinical ModernBERT fine-tuning,
- early stopping,
- and training quality-control outputs.

The corresponding D1 inference implementation is provided in:

```text
d1_infer.py
```

It performs:

```text
Internal Validation
        │
        ▼
Global Micro-F1 Threshold Calibration
        │
        ▼
Selected Threshold
       / \
      /   \
     ▼     ▼
Internal   External
 Test      Validation
```

The classification threshold is selected using the **internal validation dataset only**.

The same validation-selected threshold is subsequently applied to the internal test and external validation datasets without additional threshold optimization.

---

## 6. Remaining Hierarchical Stages

The D2, D3-D4, and D5-D6 classifiers follow the same general stage-wise training framework while using different target label spaces and hierarchical conditioning.

Their individual training scripts are not included in this public repository.

Instead, the complete hierarchical inference procedure is provided in:

```text
hierarchical_infer.py
```

This script demonstrates how independently trained stage classifiers are connected during hierarchical prediction:

```text
D1 model
   │
   ▼
Predicted D1
   │
   ▼
D2 model
   │
   ▼
Predicted D1-D2
   │
   ▼
D3-D4 model
   │
   ▼
Predicted D1-D4
   │
   ▼
D5-D6 model
   │
   ▼
Predicted 6-digit AIS code
```

Predicted upstream digits are used to restrict the candidate label space at subsequent stages.

Therefore, the hierarchical inference procedure reflects the full sequential prediction setting rather than independently evaluating each stage using ground-truth upstream labels.

---

## 7. D7 Reconstruction

After prediction of the six-digit AIS code, the final D7 severity digit is reconstructed using the AIS hierarchy mapping.

Conceptually:

```text
Predicted D1-D6
      │
      ▼
AIS Hierarchy Mapping
      │
      ▼
D7
      │
      ▼
Final D1-D7 Code
```

A D7 value is assigned only when the corresponding six-digit code has a deterministic mapping in the provided hierarchy.

Missing or ambiguous mappings are tracked separately during evaluation.

---

# Evaluation

Performance is evaluated at the **case level** under a multi-label classification setting.

The implementation calculates and exports a broad set of evaluation metrics for detailed analysis. The primary metrics used to evaluate model performance are:

- **Micro Precision**
- **Micro Recall**
- **Micro F1-score**
- **Exact Match**
- **Sample-level Accuracy**
- **Inclusive Accuracy**

Additional metrics, including F1.5, F2, specificity, false positive rate, Youden's J, case-level precision/recall/F1, Jaccard similarity, over-prediction rate, and missed-injury rate, are also calculated by the evaluation utilities and are available for supplementary analysis.

### Micro Precision, Recall, and F1-score

Micro-averaged metrics aggregate true positives, false positives, and false negatives across all cases and labels before calculating the corresponding metric.

These metrics are used to evaluate overall multi-label classification performance across the AIS label space.

### Exact Match

A case is counted as an exact match when the complete predicted label set is identical to the ground-truth label set:

```text
Predicted AIS set = Ground-truth AIS set
```

### Sample-level Accuracy

A case is counted as sample-level correct when at least one predicted label overlaps with the ground-truth label set:

```text
Predicted AIS set ∩ Ground-truth AIS set ≠ ∅
```

### Inclusive Accuracy

A case is counted as inclusive when every ground-truth label is contained in the prediction set:

```text
Ground-truth AIS set ⊆ Predicted AIS set
```

This metric allows additional predicted labels while requiring all ground-truth injuries to be recovered.

---

# Repository Structure

```text
AIS-Prediction/
│
├── README.md
├── .gitignore
├── requirements.txt
├── input_schema.json
│
├── ais_text_rules.py
├── infer_common.py
│
├── d1_train.py
├── d1_infer.py
└── hierarchical_infer.py
```

### File Description

| File | Description |
| --- | --- |
| `README.md` | Project overview and methodology |
| `.gitignore` | Prevents local data, model artifacts, and temporary files from being committed |
| `requirements.txt` | Python package dependencies |
| `input_schema.json` | Expected case-level input structure |
| `ais_text_rules.py` | Uncertainty and negation handling for radiology reports |
| `infer_common.py` | Shared input construction, model loading, thresholding, metrics, and export utilities |
| `d1_train.py` | Representative D1 training implementation |
| `d1_infer.py` | D1 inference, threshold calibration, and evaluation |
| `hierarchical_infer.py` | Full D1 → D2 → D3-D4 → D5-D6 → D7 inference pipeline |

---

# Usage

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Train the Representative D1 Model

The training script expects local paths to the training and validation datasets and the AIS hierarchy mapping.

Example:

```bash
python d1_train.py \
    --train-file /path/to/train.jsonl \
    --val-file /path/to/val.jsonl \
    --lvl-map-12 /path/to/lvl_map_12.txt \
    --output-dir /path/to/output
```

The actual datasets and AIS mapping files are not included in this repository.

---

## 3. D1 Inference and Evaluation

Example:

```bash
python d1_infer.py \
    --model-dir /path/to/output/d1_model \
    --label-map-dir /path/to/output/label_maps \
    --lvl-map-12 /path/to/lvl_map_12.txt \
    --internal-val /path/to/internal_val.jsonl \
    --internal-test /path/to/internal_test.jsonl \
    --external-val /path/to/external_val.jsonl \
    --output-dir /path/to/evaluation/d1
```

The global classification threshold is calibrated on the internal validation dataset and reused for internal test and external validation.

---

## 4. Full Hierarchical Inference

The hierarchical inference script requires trained D1, D2, D3-D4, and D5-D6 models together with their label mappings, hierarchy mappings, and validation-selected thresholds.

Conceptually:

```bash
python hierarchical_infer.py \
    --internal-val /path/to/internal_val.jsonl \
    --internal-test /path/to/internal_test.jsonl \
    --external-val /path/to/external_val.jsonl \
    --d1-model-dir /path/to/d1_model \
    --d2-model-dir /path/to/d2_model \
    --d34-model-dir /path/to/d34_model \
    --d56-model-dir /path/to/d56_model \
    --label-map-dir /path/to/label_maps \
    --lvl-map-12 /path/to/lvl_map_12.txt \
    --lvl-map-34 /path/to/lvl_map_34.txt \
    --lvl-map-56 /path/to/lvl_map_56.txt \
    --d1-threshold <threshold> \
    --d2-threshold <threshold> \
    --d34-threshold <threshold> \
    --d56-threshold <threshold> \
    --output-dir /path/to/evaluation/hierarchical
```

The example paths are placeholders only.

---

# Data Availability and Privacy

This repository **does not contain patient data**.

The following materials are not publicly distributed:

- original CT radiology reports,
- patient or visit identifiers,
- AIS annotations linked to individual cases,
- institution-specific source datasets,
- internal train/validation/test files,
- external validation data,
- trained model checkpoints,
- experiment outputs containing case-level clinical information.

These materials are excluded because of institutional data-security, privacy, and research-governance requirements.

The repository contains only source code and documentation required to describe the modeling framework.

Users who wish to run the code must prepare their own appropriately authorized dataset following the structure documented in `input_schema.json`.

---

# Reproducibility

Because the clinical datasets and AIS hierarchy resources used in the study cannot be publicly distributed, the repository is intended to provide **methodological and implementation transparency rather than a fully self-contained reproduction dataset**.

The D1 classifier is provided as the representative stage-wise implementation, while `hierarchical_infer.py` demonstrates the complete sequential inference architecture.

No patient information is required or included in the repository itself.
