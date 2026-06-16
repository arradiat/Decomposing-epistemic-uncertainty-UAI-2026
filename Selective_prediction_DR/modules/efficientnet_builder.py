"""
Model Builder for Diabetic Retinopathy Grading

EfficientNet-B4 backbone + explicit trainable head.
Head layers (Conv2D, Dense) are designed for later Bayesian conversion
to LowRankConv2DVariational / LowRankDenseVariational.

Two-phase training:
    Phase 1: Freeze backbone, train head only (5 epochs warmup)
    Phase 2: Unfreeze all layers, fine-tune end-to-end with cosine LR

Architecture:
    EfficientNet-B4 → Conv2D(256) → GAP → Dense(128) → Dropout → Dense(K, softmax)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB4


def build_efficientnet_dr(input_shape=(256, 256, 3), num_classes=4, dropout_rate=0.4):
    """
    Build EfficientNet-B4 + explicit trainable head for DR grading.

    The head layers are explicit so they can later be replaced with
    LowRankConv2DVariational / LowRankDenseVariational.

    Args:
        input_shape: Input image shape. Use (256, 256, 3) for EfficientNet-B4.
        num_classes: Number of output classes (4 for DR grading).
        dropout_rate: Dropout rate for regularization.

    Returns:
        model: Keras functional model (backbone starts frozen).
    """
    inputs = keras.Input(shape=input_shape, name='input')

    # === BACKBONE (starts frozen, unfrozen in Phase 2) ===
    backbone = EfficientNetB4(
        weights='imagenet',
        include_top=False,
        input_tensor=inputs,
    )
    backbone.trainable = False

    x = backbone.output  # (batch, 8, 8, 1792) for 256x256 input

    # === TRAINABLE HEAD ===
    # Conv2D to reduce channels: 1792 → 256
    # Later: LowRankConv2DVariational(256, (3,3), rank=8, activation='relu')
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same',
        kernel_initializer='he_normal',
        name='head_conv1'
    )(x)

    # GlobalAveragePooling: (8, 8, 256) → (256,)
    x = layers.GlobalAveragePooling2D(name='head_gap')(x)

    # Dense: 256 → 128
    # Later: LowRankDenseVariational(128, rank=8, activation='relu')
    x = layers.Dense(
        128, activation='relu',
        kernel_initializer='he_normal',
        name='head_dense1'
    )(x)

    x = layers.Dropout(rate=dropout_rate, name='head_dropout')(x)

    # Output: softmax for classification (required for per-class C_k metric)
    # Later: LowRankDenseVariational(num_classes, rank=8, activation='softmax')
    x = layers.Dense(
        num_classes, activation='softmax',
        kernel_initializer='glorot_normal',
        name='head_output'
    )(x)

    model = keras.Model(inputs=inputs, outputs=x, name='EfficientNetB4-DR')
    return model


def unfreeze_backbone(model, unfreeze_from=None):
    """
    Unfreeze backbone layers for fine-tuning.

    Args:
        model: Keras model returned by build_efficientnet_dr()
        unfreeze_from: Name of the first backbone layer to unfreeze.
            - None: unfreeze entire backbone (recommended for Phase 2)

    Returns:
        Number of trainable parameters after unfreezing.
    """
    for layer in model.layers:
        layer.trainable = True

    # Freeze BatchNorm layers — critical for fine-tuning pretrained models.
    # BN layers have running mean/var from ImageNet; updating them with
    # small batches (32) produces noisy statistics that destabilize training.
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable = sum(
        tf.keras.backend.count_params(w) for w in model.trainable_weights
    )
    frozen = sum(
        tf.keras.backend.count_params(w) for w in model.non_trainable_weights
    )
    print(f"After unfreezing backbone (BN layers frozen):")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Frozen params:    {frozen:,}")
    return trainable
