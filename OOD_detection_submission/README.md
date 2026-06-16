# OOD Detection — Supplementary Code

Code for the out-of-distribution (OOD) detection experiments reported in the paper.
Two settings are covered: a clinical tabular dataset (MIMIC-III) and an image dataset (FashionMNIST → KMNIST).

---

## Folder structure

```
OOD_detection_submission/
│
├── mimic_deep_ensemble_ood.ipynb              # Deep Ensemble (5 members) on MIMIC-III
├── mimic_lowrank_gauss_ood_multiseed.ipynb    # Low-Rank Gaussian BNN on MIMIC-III (5 seeds)
├── fmnist_kmnist_lowrank_gauss_ood_multiseed.ipynb  # Low-Rank Gaussian BNN on FMNIST→KMNIST (5 seeds)
│
├── results_csv/          # Pre-computed aggregated results (see below)
├── mimic_deep_ensemble/  # Trained ensemble member weights (5 × .h5)
├── mimic_multiseed_checkpoints/   # MIMIC low-rank BNN checkpoints (5 seeds)
├── fmnist_multiseed_checkpoints/  # FMNIST low-rank BNN checkpoints (5 seeds)
│
├── kmnist-test-imgs.npz     # KMNIST test images (OOD set for the FMNIST experiment)
├── kmnist-test-labels.npz   # KMNIST test labels
│
└── modules/              # Shared model and training utilities (see below)
```

---

## Running the notebooks

### What runs out of the box

Each notebook contains **active code cells** that load pre-computed results from `results_csv/`
and reproduce all figures reported in the paper. No data or GPU is required for these cells.

Simply open a notebook and run all cells — the raw cells (see below) will be skipped automatically.

### Raw cells (training and inference)

The cells that perform data loading, model training, and MC inference are set to **raw** cell type.
They are visible for inspection but are not executed when running the notebook.

If you have access to the datasets (see *Data* below), you can convert these cells back to code
(`Cell → Cell Type → Code` in JupyterLab, or change `"cell_type": "raw"` to `"cell_type": "code"`
in the `.ipynb` JSON) and run the full pipeline end-to-end. The trained weights are included,
so re-training is optional — set `FORCE_RETRAIN = False` in the config cell to load from checkpoints.

---

## Data

Data files are **not included** in this archive.

### MIMIC-III (clinical tabular)

The MIMIC-III dataset is available from PhysioNet under a data use agreement:

> https://physionet.org/content/mimiciii/

Access requires completion of a CITI training course and approval by PhysioNet.
Once downloaded, preprocessing follows the script in `modules/MIMIC_3_data_preprocessing.py`
(not included here; available in the full codebase).

The experiments use:
- **ID set**: adult ICU admissions (test split)
- **OOD set**: neonatal (newborn) admissions

### FashionMNIST (images)

FashionMNIST loads automatically via TensorFlow/Keras:

```python
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

No manual download is needed.

The **KMNIST** test set (OOD) is included as `kmnist-test-imgs.npz` and `kmnist-test-labels.npz`.

---

## Pre-computed results (`results_csv/`)

| File | Contents |
|---|---|
| `mimic_deep_ensemble_ood.csv` | AUROC, mean ID/OOD scores, OOD/ID ratio per metric (ensemble) |
| `mimic_lowrank_gauss_multiseed_agg.csv` | AUROC mean ± std across 5 seeds (MIMIC low-rank BNN) |
| `mimic_lowrank_gauss_multiseed_per_seed.csv` | Per-seed AUROC and scores (MIMIC) |
| `fmnist_kmnist_lowrank_gauss_multiseed_agg.csv` | AUROC mean ± std across 5 seeds (FMNIST) |
| `fmnist_kmnist_lowrank_gauss_multiseed_per_seed.csv` | Per-seed AUROC and scores (FMNIST) |
| `fmnist_perclass_ck_id_vs_ood.csv` | Per-class epistemic uncertainty C_k, ID vs OOD |
| `fmnist_perclass_rho_id_vs_ood.csv` | Per-class skewness diagnostic rho_k, ID vs OOD |

---

## Modules (`modules/`)

| File | Description |
|---|---|
| `bayesian_layers.py` | Custom Keras layers: `LowRankDenseVariational` (low-rank Gaussian weight posterior) and `DenseVariational` (mean-field). Implements the reparameterisation trick and KL divergence. |
| `model_builders.py` | Functions to build and compile models: `build_lowrank_gauss` for the BNN, `build_dense_model` for ensemble members. Also provides `set_kl_scale` and `compile_binary`. |
| `data_utils.py` | MIMIC-III data loading helpers: `load_mimic_data` (ICU train/test split), `load_ood_data` (newborn admissions), `compute_class_weights`. Reads from the preprocessed CSV files. |
| `config.py` | Centralized path configuration for MIMIC-III data files. Update `DATA_DIR` to point to your local preprocessed data before running. |

---

## Dependencies

```
tensorflow >= 2.12
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
```
