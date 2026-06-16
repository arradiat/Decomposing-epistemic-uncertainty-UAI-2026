"""
Data Loading and Preprocessing Utilities for MIMIC-III Experiments

This module handles:
- Loading training, test, and OOD data
- Computing class weights for imbalanced datasets
- Data preprocessing and imputation
- Dataset summary statistics
"""

import pandas as pd
import numpy as np
from modules.config import (
    X_TRAIN_PATH,
    X_TEST_PATH,
    Y_TRAIN_PATH,
    Y_TEST_PATH,
    X_NEWBORN_PATH,
    Y_NEWBORN_PATH
)

import pandas as pd

def save_model_stats(models, save_path="results_csv/model_statistics.csv"):
    """Save parameter counts and memory to CSV."""
    stats = []
    for name, model in models.items():
        if isinstance(model, list):
            params = model[0].count_params() * len(model)
        else:
            params = model.count_params()
        
        stats.append({
            'Model': name,
            'Total_Params': params,
            'Memory_MB': params * 4 / (1024**2)
        })
    
    df = pd.DataFrame(stats)
    df.to_csv(save_path, index=False)
    print(df)
    return df
def load_mimic_data(verbose=True):
    """
    Load MIMIC-III processed training and test data.

    Parameters:
    -----------
    verbose : bool, whether to print dataset summaries (default: True)

    Returns:
    --------
    X_train : pandas DataFrame, training features (N_train, 44)
    X_test : pandas DataFrame, test features (N_test, 44)
    y_train : pandas Series, training labels (N_train,)
    y_test : pandas Series, test labels (N_test,)
    feature_dim : int, number of features (should be 44)
    """
    # Load features and labels
    X_train = pd.read_csv(X_TRAIN_PATH).astype('float32')
    X_test = pd.read_csv(X_TEST_PATH).astype('float32')
    y_train = pd.read_csv(Y_TRAIN_PATH, header=0).squeeze().astype('float32')
    y_test = pd.read_csv(Y_TEST_PATH, header=0).squeeze().astype('float32')

    # Determine input dimension
    feature_dim = X_train.shape[1]

    if verbose:
        print("="*80)
        print("MIMIC-III DATA LOADED")
        print("="*80)
        show_dataset_summary('Training Set', X_train, y_train)
        show_dataset_summary('Test Set', X_test, y_test)
        print(f"Feature dimension: {feature_dim}\n")

    return X_train, X_test, y_train, y_test, feature_dim


def load_ood_data(X_train, verbose=True):
    """
    Load out-of-distribution (newborn) data and preprocess.

    Parameters:
    -----------
    X_train : pandas DataFrame, training features (for computing imputation values)
    verbose : bool, whether to print dataset summary (default: True)

    Returns:
    --------
    X_newborn : pandas DataFrame, OOD features (N_ood, 44)
    y_newborn : pandas Series, OOD labels (N_ood,)
    """
    # Load newborn (out-of-domain) data
    X_newborn = pd.read_csv(X_NEWBORN_PATH).astype("float32")
    y_newborn = pd.read_csv(Y_NEWBORN_PATH, header=0).squeeze().astype(int)

    # Preprocessing: Fill missing age and weight with training set means
    mean_age = X_train['age'].mean()
    X_newborn['age'] = mean_age

    if 'weight' in X_newborn.columns:
        mean_weight = X_train['weight'].mean()
        X_newborn['weight'] = mean_weight

    if verbose:
        print("="*80)
        print("OOD DATA LOADED (Newborns)")
        print("="*80)
        show_dataset_summary('Newborn (OOD)', X_newborn, y_newborn)
        print()

    return X_newborn, y_newborn


def compute_class_weights(y_train, verbose=True):
    """
    Compute class weights for imbalanced binary classification.

    Uses inverse frequency weighting:
    - Majority class (y=0): weight = 1.0
    - Minority class (y=1): weight = 1 / positive_fraction

    Parameters:
    -----------
    y_train : pandas Series or numpy array, training labels {0, 1}
    verbose : bool, whether to print weight information (default: True)

    Returns:
    --------
    class_weight : dict, {0: weight_negative, 1: weight_positive}
    """
    # Compute positive class fraction
    pos_fraction = y_train.mean()

    # Compute positive class weight (inverse frequency)
    pos_weight = 1.0 / pos_fraction

    # Create class weight dictionary
    class_weight = {0: 1.0, 1: pos_weight}

    if verbose:
        print("="*80)
        print("CLASS WEIGHTS COMPUTED")
        print("="*80)
        print(f"Positive class fraction: {pos_fraction:.4f} ({pos_fraction*100:.2f}%)")
        print(f"Class weights: {{0: {class_weight[0]:.2f}, 1: {class_weight[1]:.2f}}}")
        print()

    return class_weight


def show_dataset_summary(name, X, y):
    """
    Print summary statistics for a dataset.

    Parameters:
    -----------
    name : str, dataset name
    X : pandas DataFrame, features
    y : pandas Series or numpy array, labels
    """
    n_samples = len(X)
    n_survived = (y == 0).sum()
    n_deceased = (y == 1).sum()

    print(f"{name}:")
    print(f"  Total samples: {n_samples}")
    print(f"  Survived (y=0): {n_survived} ({n_survived/n_samples:.1%})")
    print(f"  Deceased (y=1): {n_deceased} ({n_deceased/n_samples:.1%})")
    print(f"  Features: {X.shape[1]}")
    print()


def prepare_data_for_training(X_train, y_train, X_test, y_test):
    """
    Convert pandas DataFrames to numpy arrays for training.

    Parameters:
    -----------
    X_train : pandas DataFrame
    y_train : pandas Series
    X_test : pandas DataFrame
    y_test : pandas Series

    Returns:
    --------
    x_tr : numpy array, training features
    y_tr : numpy array, training labels
    x_te : numpy array, test features
    y_te : numpy array, test labels
    """
    x_tr = X_train.values
    y_tr = y_train.values
    x_te = X_test.values
    y_te = y_test.values

    return x_tr, y_tr, x_te, y_te


def load_all_data(verbose=True):
    """
    Convenience function to load all data (training, test, OOD) and compute class weights.

    Parameters:
    -----------
    verbose : bool, whether to print summaries (default: True)

    Returns:
    --------
    X_train : pandas DataFrame, training features
    X_test : pandas DataFrame, test features
    y_train : pandas Series, training labels
    y_test : pandas Series, test labels
    X_ood : pandas DataFrame, OOD features
    y_ood : pandas Series, OOD labels
    feature_dim : int, number of features
    class_weight : dict, class weights for imbalanced data
    """
    # Load training and test data
    X_train, X_test, y_train, y_test, feature_dim = load_mimic_data(verbose=verbose)

    # Compute class weights
    class_weight = compute_class_weights(y_train, verbose=verbose)

    # Load OOD data
    X_ood, y_ood = load_ood_data(X_train, verbose=verbose)

    return X_train, X_test, y_train, y_test, X_ood, y_ood, feature_dim, class_weight
