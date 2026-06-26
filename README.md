# AIS Prediction from Trauma CT Reports

## Overview

This project presents a hierarchical Clinical NLP model for predicting Abbreviated Injury Scale (AIS) codes from trauma CT radiology reports.

The proposed framework predicts AIS codes using only CT radiology reports, without incorporating additional clinical information such as age, sex, examination name, or other patient metadata.

The backbone model is BioClinical ModernBERT, an encoder-only Transformer pre-trained on biomedical and clinical corpora. Compared with the original BERT architecture, BioClinical ModernBERT supports longer context lengths, making it more suitable for processing lengthy clinical documents.

---

# Dataset

## 1. Development & Validation Dataset

The development and internal validation datasets consist of trauma CT radiology reports collected from three medical institutions.

Multiple CT reports from the same patient visit were grouped into a single Case ID, generated using the patient identifier and visit date.

A total of 24,005 cases were used for model development and internal validation.

| Dataset | Number of Cases |
|---------|----------------:|
| Development & Validation | 24,005 |

---

## 2. External Validation Dataset

External validation was performed using trauma CT radiology reports collected from an independent medical institution.

A total of 1,460 cases were used to evaluate the generalization performance of the proposed model.

| Dataset | Number of Cases |
|---------|----------------:|
| External Validation | 1,460 |

---

# Method

## 1. Preprocessing

AIS codes from the Korean Trauma Data Bank (KTDB) were first matched with CT radiology reports using the corresponding Case IDs.

To ensure that only CT-identifiable injuries were included in the training labels, an AIS code filtering process was performed.

Since the first digit (D1) of an AIS code represents the injured body region, examination names were mapped to the corresponding AIS body regions. AIS codes that were inconsistent with the examination region were removed.

For example,

- Brain CT → AIS D1 = 1 (Head)
- Chest CT → AIS D1 = 4 (Thorax)
- Abdomen CT → AIS D1 = 5 (Abdomen)

This filtering process improved the consistency between CT reports and AIS labels.

---

## 2. Model Development

CT radiology reports underwent minimal text preprocessing, including whitespace and newline normalization. All reports belonging to the same Case ID were merged into a single input document.

The merged report was encoded using BioClinical ModernBERT, followed by a fully connected classification layer with sigmoid activation for multi-label classification.

Since AIS codes have a hierarchical structure in which the semantic meaning becomes increasingly specific from D1 to D7, the prediction task was decomposed into multiple hierarchical stages:

- D1: Body Region
- D2: Anatomical Structure
- D3-D4: Injury Type
- D5-D6: Injury Severity
- D7: Severity Score

The prediction pipeline follows the hierarchical order:

D1 → D2 → D3-D4 + D5-D6 → D7

Each stage is trained independently, and a corresponding AIS codebook is provided to restrict the prediction space to valid labels for that hierarchical level.

---

## 3. Evaluation

The filtered AIS labels were regarded as the ground truth for model evaluation.

Performance was evaluated using case-level multi-label classification with the following metrics:

- Micro Precision
- Micro Recall
- Micro F1-score
- Exact Match
- Over Prediction
- Missed Injury

---

# File Structure


---

# Data Availability

This repository does not include any patient data or institution-specific clinical information.

Original CT radiology reports, AIS annotations, and other clinical data are not publicly available due to institutional data security and privacy regulations. This repository provides only the source code and project documentation.
