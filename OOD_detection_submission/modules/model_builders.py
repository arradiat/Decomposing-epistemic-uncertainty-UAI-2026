"""
Model Architecture Builders for Bayesian Neural Networks

This module provides builders for different Bayesian NN architectures:
- Full-rank Bayes by Backprop (BBB)
- Low-rank Gaussian factorization
- Low-rank Laplace factorization
- Rank-1 multiplicative reparameterization
- Deterministic models for ensemble baselines
"""

import tensorflow as tf
from modules.bayesian_layers import (
    DenseVariational,
    LowRankDenseVariational,
    LowRankDenseVariationalLap,
    Rank1DenseVariational
)


def set_kl_scale(model, value):
    """
    Set kl_scale on all variational layers.

    Parameters:
    -----------
    model : keras Model
    value : float, new KL scale value
    """
    for layer in model.layers:
        if hasattr(layer, 'kl_scale'):
            layer.kl_scale = float(value)


def compile_binary(model):
    """
    Compile model with binary cross-entropy loss and Adam optimizer.

    Metrics:
    - AUC (AUROC)
    - AUPRC (Area Under Precision-Recall Curve)

    Parameters:
    -----------
    model : keras Model

    Returns:
    --------
    model : compiled keras Model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.AUC(curve="PR", name="auprc")]
    )
    return model


def build_fullrank_bbb(input_dim: int) -> tf.keras.Model:
    """
    Build full-rank Bayesian Neural Network with diagonal Gaussian posterior.
    Implements Bayes by Backprop (Blundell et al., 2015).

    Architecture:
    - Input layer (input_dim features)
    - DenseVariational(128, relu)
    - DenseVariational(128, relu)
    - DenseVariational(1, sigmoid)

    Parameters:
    -----------
    input_dim : int, number of input features

    Returns:
    --------
    model : compiled keras Model
    """
    x = tf.keras.Input(shape=(input_dim,), name="input")
    h = DenseVariational(128, kl_scale=0.0, activation="relu", name="bbb_dense1")(x)
    h = DenseVariational(128, kl_scale=0.0, activation="relu", name="bbb_dense2")(h)
    y = DenseVariational(1, kl_scale=0.0, activation="sigmoid")(h)
    model = tf.keras.Model(x, y, name="FullRank_BBB")
    set_kl_scale(model, 0)
    compile_binary(model)
    return model


def build_lowrank_gauss(input_dim: int, rank1: int = 15,rank2: int = 15,init_from_deterministic: tf.keras.Model | None = None,) -> tf.keras.Model:
    """
    Build low-rank Gaussian factorization model.
    Weight matrix W ≈ AB^T where A ∈ R^(d_in × rank), B ∈ R^(d_out × rank).

    Architecture:
    - Input layer (input_dim features)
    - LowRankDenseVariational(128, rank, relu)
    - LowRankDenseVariational(128, rank, relu)
    - LowRankDenseVariational(1, rank=1, sigmoid)

    Parameters:
    -----------
    input_dim : int, number of input features
    rank : int, rank of factorization (default: 15)

    Returns:
    --------
    model : compiled keras Model
    """
    x = tf.keras.Input(shape=(input_dim,), name="input")
    h = LowRankDenseVariational(128, rank=rank1, kl_scale=0.0, activation="relu", name="layer0")(x)
    h = LowRankDenseVariational(128, rank=rank2, kl_scale=0.0, activation="relu", name="layer1")(h)
    y = LowRankDenseVariational(1, rank=1, kl_scale=0.0, activation="sigmoid",name="output")(h)
    model = tf.keras.Model(x, y, name=f"LowRank_Gaussian_r{rank1}")
    if init_from_deterministic is not None:
        # Build Bayesian model
        _ = model(tf.zeros((1, input_dim), dtype=tf.float32), training=False)
        for layer in model.layers:
            if isinstance(layer, LowRankDenseVariational) and hasattr(layer, "init_from_full_matrix"):
                try:
                    det_layer = init_from_deterministic.get_layer(layer.name)
                except ValueError:
                    continue
                det_weights = det_layer.get_weights()
                if det_weights:
                    layer.init_from_full_matrix(det_weights[0])
    set_kl_scale(model, 0)
    compile_binary(model)
    return model


def build_lowrank_laplace(input_dim: int, rank: int = 15) -> tf.keras.Model:
    """
    Build low-rank Laplace factorization model.
    Uses Laplace distribution instead of Gaussian for potentially sparser solutions.

    Architecture:
    - Input layer (input_dim features)
    - LowRankDenseVariationalLap(128, rank, relu)
    - LowRankDenseVariationalLap(128, rank, relu)
    - LowRankDenseVariationalLap(1, rank=1, sigmoid)

    Parameters:
    -----------
    input_dim : int, number of input features
    rank : int, rank of factorization (default: 15)

    Returns:
    --------
    model : compiled keras Model
    """
    x = tf.keras.Input(shape=(input_dim,), name="input")
    h = LowRankDenseVariationalLap(128, rank=rank, kl_scale=0.0, activation="relu", name=f"lrL_dense1_r{rank}")(x)
    h = LowRankDenseVariationalLap(128, rank=rank, kl_scale=0.0, activation="relu", name=f"lrL_dense2_r{rank}")(h)
    y = LowRankDenseVariationalLap(1, rank=1, kl_scale=0.0, activation="sigmoid")(h)
    model = tf.keras.Model(x, y, name=f"LowRank_Laplace_r{rank}")
    set_kl_scale(model, 0)
    compile_binary(model)
    return model


def build_rank1(input_dim: int) -> tf.keras.Model:
    """
    Build rank-1 multiplicative Bayesian model (Dusenberry et al., 2020).
    Most parameter-efficient approach: W_eff = W0 ⊙ (1 + s)(1 + r)^T

    Architecture:
    - Input layer (input_dim features)
    - Rank1DenseVariational(128, relu)
    - Rank1DenseVariational(128, relu)
    - Dense(1, sigmoid) [deterministic output layer]

    Parameters:
    -----------
    input_dim : int, number of input features

    Returns:
    --------
    model : compiled keras Model
    """
    x = tf.keras.Input(shape=(input_dim,), name="input")
    h = Rank1DenseVariational(128, kl_scale=0.0, activation="relu", name="rank1_dense1")(x)
    h = Rank1DenseVariational(128, kl_scale=0.0, activation="relu", name="rank1_dense2")(h)
    y = tf.keras.layers.Dense(1, activation="sigmoid", name="out")(h)
    model = tf.keras.Model(x, y, name="Rank1_Gaussian")
    set_kl_scale(model, 1)
    compile_binary(model)
    return model


def build_dense_model(input_dim: int) -> tf.keras.Model:
    """
    Build a simple deterministic feed-forward neural network for ensemble members.
    No Bayesian layers - standard deep learning model.

    Architecture:
    - Input layer (input_dim features)
    - Dense(128, relu)
    - Dense(128, relu)
    - Dense(1, sigmoid)

    Parameters:
    -----------
    input_dim : int, number of input features

    Returns:
    --------
    model : compiled keras Model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(128, activation='relu',name="layer0"),
        tf.keras.layers.Dense(128, activation='relu',name="layer1"),
        tf.keras.layers.Dense(1, activation='sigmoid',name="output")
    ])
    compile_binary(model)
    return model


import os
import json
import tensorflow as tf

# =============================================================================
# Save trained models (preserves exact dict keys)
# =============================================================================

def load_trained_models(configs, feature_dim, save_dir="checkpoints", 
                       ensemble_builder=None, n_ensemble_members=5):
    """
    Load model weights.
    
    Args:
        configs: configs dict for Bayesian models
        feature_dim: input feature dimension
        save_dir: where weights are saved
        ensemble_builder: function to build single ensemble member (e.g., build_dense_model)
        n_ensemble_members: number of ensemble members (used if ensemble found)
    """
    print(f"\n{'='*80}")
    print(f"LOADING MODEL WEIGHTS")
    print(f"{'='*80}")
    
    # Load key mapping
    with open(os.path.join(save_dir, "key_mapping.json"), 'r') as f:
        key_mapping = json.load(f)
    
    models = {}
    
    for original_key, info in key_mapping.items():
        item_path = os.path.join(save_dir, info["path"])
        
        if info["type"] == "ensemble":
            # Rebuild and load ensemble
            member_files = sorted([f for f in os.listdir(item_path) if f.endswith('.h5')])
            ensemble = []
            for f in member_files:
                # Rebuild using ensemble_builder
                model = ensemble_builder(feature_dim)
                # Load weights
                model.load_weights(os.path.join(item_path, f))
                ensemble.append(model)
            models[original_key] = ensemble
            print(f"✓ Loaded ensemble: {original_key} ({len(ensemble)} members)")
        else:
            # Rebuild single Bayesian model from configs
            model = configs[original_key]["builder"](feature_dim)
            # Load weights
            model.load_weights(item_path)
            models[original_key] = model
            print(f"✓ Loaded: {original_key}")
    
    print(f"{'='*80}\n")
    return models

import os
import json
def save_trained_models(models, save_dir="checkpoints"):
    """
    Save model weights only (works with custom layers, no custom_objects needed).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"SAVING MODEL WEIGHTS")
    print(f"{'='*80}")
    
    key_mapping = {}
    
    for idx, (name, model) in enumerate(models.items()):
        safe_name = f"model_{idx}"
        
        if isinstance(model, list):
            # Ensemble
            model_dir = os.path.join(save_dir, safe_name)
            os.makedirs(model_dir, exist_ok=True)
            for i, member in enumerate(model):
                member.save_weights(os.path.join(model_dir, f"member_{i}.h5"))
            key_mapping[name] = {"path": safe_name, "type": "ensemble", "n_members": len(model)}
            print(f"✓ Saved ensemble: {name} ({len(model)} members)")
        else:
            # Single model
            weight_path = os.path.join(save_dir, f"{safe_name}.h5")
            model.save_weights(weight_path)
            key_mapping[name] = {"path": f"{safe_name}.h5", "type": "single"}
            print(f"✓ Saved: {name}")
    
    # Save key mapping
    with open(os.path.join(save_dir, "key_mapping.json"), 'w') as f:
        json.dump(key_mapping, f, indent=2)
    
    print(f"{'='*80}\n")
