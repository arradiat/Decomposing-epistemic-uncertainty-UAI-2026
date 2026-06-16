# Supplemental Code: Selective Prediction for Diabetic Retinopathy Grading

This folder contains the code and saved experimental artifacts accompanying the UAI paper submission. It is designed to allow reproduction of all analysis results, figures, and metrics **without requiring access to the original datasets** — using pre-saved Monte Carlo samples instead.

---

## Overview

The code implements and evaluates **selective prediction** (also called selective classification or learn-to-abstain) for 4-class diabetic retinopathy (DR) grading using two Bayesian deep learning approaches:

1. **Bayesian Neural Network (BNN)** — EfficientNet-B4 with Bayesian convolutional layers (full variational inference).
2. **MC Dropout** — Deterministic EfficientNet-B4 with dropout enabled at test time (30 Monte Carlo samples).

The key contribution is the **CBEC** (Cross-Boundary Epistemic Confusion) metric and the related **C\_critical\_max** / **C\_critical\_sum** metrics, which identify samples that should be deferred to a clinician by detecting epistemic uncertainty specifically at the safe/critical class boundary (Grades 0–1 vs. Grades 2–3).

---

## Folder Structure

```
Selective_prediction_DR/
│
├── README.md                                       ← this file
│
├── selective_prediction_BNN.ipynb                  ← Main notebook: BNN selective prediction
│                                                      (CBEC, C_critical_*, baselines, bootstrap)
│
├── selective_prediction_bootstrap MC_dropout.ipynb ← MC Dropout notebook: deferral policies,
│                                                      bootstrap CIs, diagnostic figures
│
├── ensemble_supplement.ipynb                       ← Deep Ensemble supplementary validation
│
├── modules/
│   ├── data_pipeline.py          ← Dataset loading (EyePACS + APTOS + Messidor),
│   │                                patient-level splitting, tf.data pipeline,
│   │                                Ben Graham preprocessing
│   ├── efficientnet_builder.py   ← EfficientNet-B0 model builder (deterministic)
│   └── uncertaintybis.py         ← Uncertainty metrics (CBEC, C_critical_max/sum,
│                                    entropy, MI, MaxProb, AUSC, bootstrap)
│
├── results_bayesian/
│   ├── mc_predictions.npy        ← Pre-saved BNN MC samples  [S × N × 4]
│   ├── y_true.npy                ← Ground truth class labels for the test set [N]
│   ├── bayesian_final_weightsbaseline1.h5  ← BNN weights [NOT INCLUDED — see note below]
│   └── bayesian_full_model/      ← Full BNN SavedModel   [NOT INCLUDED — see note below]
│
├── results_stable/
│   ├── mcdo_det_preds_30samples.npy  ← Pre-saved MC Dropout samples [30 × N × 4]
│   ├── mcdo_det_y_true.npy           ← Ground truth labels for MC Dropout test set [N]
│   ├── best_phase1.keras             ← Deterministic model Phase 1 [NOT INCLUDED]
│   └── best_phase2.keras             ← Deterministic model Phase 2 [NOT INCLUDED]
│
└── results_ensemble/
    ├── ensemble_preds.npy        ← Pre-saved ensemble predictions [5 × N × 4]
    ├── ensemble_y_true.npy       ← Ground truth labels for ensemble test set [N]
    └── member_00–04.keras        ← Ensemble member models         [NOT INCLUDED]
```

> **Note on missing files:** Model weights and trained `.keras` / `.h5` / SavedModel files are
> **not included in this archive** due to file-size constraints (combined ~1.4 GB). All analysis
> cells use the pre-saved `.npy` prediction arrays and run without the weight files. The weight
> files can be made available on request.

---

## Running the Notebooks Without Re-downloading Data

**All three notebooks are configured to load pre-saved `.npy` prediction arrays**, so all
analysis, metrics, figures, and bootstrap confidence intervals can be reproduced immediately
without the datasets or model weights.

Cells that require model weights or raw image data (data loading, model rebuild, live MC
inference, saving new `.npy` files) have been set to **Raw** cell type in Jupyter — they are
visible for inspection but will not execute when running the notebook. These cells are included
to document the full pipeline; they are inert due to the file-size constraints of this archive.

To reproduce the full analysis, open each notebook and run all cells. Each notebook will start
from the pre-saved prediction arrays:

| Notebook | Start from | Loads |
|----------|-----------|-------|
| `selective_prediction_BNN.ipynb` | Cell 10 | `results_bayesian/mc_predictions.npy`, `results_bayesian/y_true.npy` |
| `selective_prediction_bootstrap MC_dropout.ipynb` | Cell 7 | `results_stable/mcdo_det_preds_30samples.npy`, `results_stable/mcdo_det_y_true.npy` |
| `ensemble_supplement.ipynb` | Cell 6 | `results_ensemble/ensemble_preds.npy`, `results_ensemble/ensemble_y_true.npy` |

### Dependencies

```
tensorflow >= 2.12
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
```

Install with:

```bash
pip install tensorflow numpy pandas scikit-learn scipy matplotlib seaborn
```

---

## Datasets (Not Included)

The raw image datasets are **not included** in this supplemental material due to their size (combined ~30 GB) and licensing restrictions. They must be downloaded separately from the sources below.

The expected directory layout once downloaded is:

```
Selective_prediction_DR/
└── datasets/
    ├── eyepacs_resized_v7/
    │   ├── resized_train_cropped/
    │   │   └── resized_train_cropped/
    │   │       └── *.jpeg
    │   └── trainLabels_cropped.csv
    ├── aptos_224x224_v4/
    │   └── colored_images/
    │       ├── No_DR/
    │       ├── Mild/
    │       ├── Moderate/
    │       ├── Severe/
    │       └── Proliferate_DR/
    ├── preprocessed/          ← Messidor images converted to .npy
    └── messidor_labels.csv
```

### 1. EyePACS — Kaggle Diabetic Retinopathy Detection (2015)

- **URL:** https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data
- Requires a free Kaggle account.
- Download the `resized_train_cropped.zip` archive and `trainLabels_cropped.csv`.
- Place under `datasets/eyepacs_resized_v7/`.
- Images are JPEG; original 5-class labels are clipped to 4 classes (Grade 3 and 4 merged into Grade 3).

### 2. APTOS 2019 — Kaggle Blindness Detection

- **URL:** https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
- Requires a free Kaggle account.
- Download the training images and labels.
- Organize images into per-class subdirectories matching the grade names above (`No_DR`, `Mild`, `Moderate`, `Severe`, `Proliferate_DR`) and place under `datasets/aptos_224x224_v4/colored_images/`.
- Images are resized to 224×224 for this project.

### 3. Messidor — ADCIS / TECHNO-VISION

- **URL:** http://www.adcis.net/en/third-party/messidor/
- Messidor is freely available for research upon request from the ADCIS platform.
- After downloading, images must be preprocessed (center-cropped, resized) and saved as `.npy` arrays.
- The label file `messidor_labels.csv` should have columns `image_id` and `grade` (0–3, with original grade 3 mapped to class 3 and grade 4 also mapped to class 3 if present).
- Place preprocessed arrays under `datasets/preprocessed/` and the CSV at `datasets/messidor_labels.csv`.

---

## Re-running Live Inference (Optional)

To re-run MC sampling from scratch (e.g., to use a different number of samples):

1. Obtain the model weight files (available on request) and place them in `results_bayesian/`, `results_stable/`, and `results_ensemble/` as shown in the folder structure above.
2. Place the datasets as described in the section below.
3. In each notebook, locate the **Raw** cells (data loading and inference blocks). Change their cell type back to **Code** in Jupyter (`Cell → Cell Type → Code`).
4. Comment out or remove the `np.load(...)` lines that load pre-saved predictions.
5. Run the notebook; new `.npy` files will be saved to `results_bayesian/` or `results_stable/`.

---

## Grade Mapping

All three datasets are unified to a 4-class scheme:

| Class | DR Grade        | Clinical Significance |
|-------|-----------------|----------------------|
| 0     | No DR           | Safe                 |
| 1     | Mild NPDR       | Safe                 |
| 2     | Moderate NPDR   | **Critical**         |
| 3     | Severe NPDR + PDR (grades 3 & 4 merged) | **Critical** |

The safe/critical boundary (between grades 1 and 2) is the focus of the deferral policies.

---
