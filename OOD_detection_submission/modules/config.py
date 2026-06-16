"""
Configuration Constants for MIMIC-III Bayesian Deep Learning Experiments

This module centralizes all hyperparameters, paths, and settings.

IMPORTANT: Import this module BEFORE importing tensorflow in other modules
to ensure GPU configuration is set properly.
"""

import os
import random

# Set GPU environment variables BEFORE importing TensorFlow
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Now import TensorFlow
import tensorflow as tf
import numpy as np

# Random Seed Configuration

SEED = 42


def set_global_seed(seed=SEED):
    """
    Set global random seed for reproducibility across all libraries.

    Parameters:
    -----------
    seed : int, random seed (default: 42)
    """
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


# Data Paths


DATA_DIR = "data2/processed"

# Training data paths
X_TRAIN_PATH = f"{DATA_DIR}/X_train_processed.csv"
X_TEST_PATH = f"{DATA_DIR}/X_test_processed.csv"
Y_TRAIN_PATH = f"{DATA_DIR}/y_train.csv"
Y_TEST_PATH = f"{DATA_DIR}/y_test.csv"

# OOD data paths
X_NEWBORN_PATH = f"{DATA_DIR}/X_newborns.csv"
Y_NEWBORN_PATH = f"{DATA_DIR}/y_newborns.csv"


# Model Architecture Configuration

# Network architecture (shared across all models)
HIDDEN_UNITS = [128, 128]  # Two hidden layers with 128 units each

# Low-rank factorization ranks
DEFAULT_RANK = 15  # For Low-Rank Gaussian and Laplace models

# Weight initialization ranges
WEIGHT_INIT_RANGE = (-0.2, 0.2)
RHO_INIT_RANGE = (-5.0, -4.0)


# Training Hyperparameters

# Training settings
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# Epochs for different models
EPOCHS_BAYESIAN = 256  # Full-Rank BBB, Low-Rank Gaussian/Laplace
EPOCHS_RANK1 = 32  # Rank-1 multiplicative (faster convergence)
EPOCHS_ENSEMBLE = 256  # Deep ensemble members

# Deep ensemble configuration
N_ENSEMBLE_MEMBERS = 5


# Bayesian Inference Configuration
# 

# Monte Carlo sampling
N_MC_SAMPLES = 200  # Number of weight samples for predictions

# KL divergence scaling
# Note: kl_factor is computed dynamically based on dataset size
# kl_factor = 0.5 / num_batches, where num_batches = ceil(N_train / BATCH_SIZE)


# 
# Evaluation Configuration
# 

# Calibration error (ECE) settings
ECE_N_BINS_DEFAULT = 15
ECE_BINNING_METHOD = "equal_mass"  # Options: "equal_width", "equal_mass"

# ECE optimization configurations to try
ECE_BIN_CONFIGS = [
    ('equal_width', 10),
    ('equal_width', 15),
    ('equal_width', 20),
    ('equal_mass', 10),
    ('equal_mass', 15),
    ('equal_mass', 20)
]


# 
# GPU and Mixed Precision Configuration
# 

# Configure GPUs at module import time (before they are initialized)
try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"GPU configuration warning: {e}")

# Set mixed precision policy
tf.keras.mixed_precision.set_global_policy("float32")


def configure_gpu():
    """
    Configure GPU settings for optimal performance and memory usage.

    Note: Most GPU configuration is now done at module import time.
    This function is kept for backward compatibility and to ensure
    mixed precision is set correctly.
    """
    # Use float32 (disable mixed precision)
    tf.keras.mixed_precision.set_global_policy("float32")


# 
# Prior Configuration
# 

# Scale-mixture Gaussian prior (Blundell/Ruhe style)
PRIOR_PI = 0.5  # Mixture weight
PRIOR_SIGMA1 = 1.0  # Scale of first Gaussian component
PRIOR_SIGMA2 = tf.exp(-6.0).numpy()  # Scale of second Gaussian ≈ 0.0025


# 
# Feature Dimension
# 

FEATURE_DIM = 44  # Number of input features for MIMIC-III dataset


# 
# Initialization Function
# 

def initialize_environment(seed=SEED):
    """
    Initialize the complete environment for reproducible experiments.

    Parameters:
    -----------
    seed : int, random seed (default: 42)
    """
    set_global_seed(seed)
    configure_gpu()
    print(f"Environment initialized with seed={seed}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
