"""
Data Pipeline for DR Grading

Reusable data loading, patient-level splitting, and tf.data pipeline
for the combined EyePACS + APTOS + Messidor dataset.

Includes Ben Graham preprocessing (local average color subtraction)
used by winning Kaggle DR solutions.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def load_and_combine_datasets(eyepacs_dir, aptos_dir, messidor_img, messidor_csv):
    """
    Load and combine EyePACS, APTOS, and Messidor datasets.

    All datasets are mapped to 4-class DR grading:
        Grade 0: No DR
        Grade 1: Mild NPDR
        Grade 2: Moderate NPDR
        Grade 3: Severe NPDR + PDR (grades 3+4 merged)
    """
    eyepacs_dir = Path(eyepacs_dir)
    aptos_dir = Path(aptos_dir)
    messidor_img = Path(messidor_img)

    # --- EyePACS ---
    eyepacs_img = eyepacs_dir / 'resized_train_cropped' / 'resized_train_cropped'
    eyepacs_csv = eyepacs_dir / 'trainLabels_cropped.csv'

    df_eye = pd.read_csv(eyepacs_csv)[['image', 'level']].copy()
    df_eye.columns = ['image_id', 'grade']
    df_eye['grade'] = df_eye['grade'].clip(upper=3)
    df_eye['path'] = df_eye['image_id'].apply(
        lambda x: str(eyepacs_img / f'{x}.jpeg')
    )
    df_eye['source'] = 'eyepacs'
    df_eye['patient'] = df_eye['image_id'].str.replace(
        r'_(left|right)$', '', regex=True
    )
    df_eye['format'] = 'jpeg'

    # --- APTOS ---
    aptos_img = aptos_dir / 'colored_images'
    grade_map = {
        'No_DR': 0, 'Mild': 1, 'Moderate': 2,
        'Severe': 3, 'Proliferate_DR': 3,
    }
    aptos_rows = []
    for folder, grade in grade_map.items():
        folder_path = aptos_img / folder
        if folder_path.exists():
            for f in folder_path.iterdir():
                if f.suffix in ('.jpeg', '.jpg', '.png'):
                    aptos_rows.append({
                        'image_id': f.stem,
                        'grade': grade,
                        'path': str(f),
                        'source': 'aptos',
                        'patient': f'aptos_{f.stem}',
                        'format': 'jpeg',
                    })
    df_aptos = pd.DataFrame(aptos_rows)

    # --- Messidor ---
    df_mess = pd.read_csv(messidor_csv)
    df_mess['path'] = df_mess['image_id'].apply(
        lambda x: str(messidor_img / f'{x}.npy')
    )
    df_mess['source'] = 'messidor'
    df_mess['patient'] = 'messidor_' + df_mess['image_id'].astype(str)
    df_mess['format'] = 'npy'

    # --- Combine ---
    df = pd.concat([df_eye, df_aptos, df_mess], ignore_index=True)
    df['exists'] = df['path'].apply(os.path.exists)
    missing = (~df['exists']).sum()
    if missing > 0:
        print(f"Warning: {missing} files not found, removing.")
    df = df[df['exists']].reset_index(drop=True)
    df.drop(columns='exists', inplace=True)

    print(f"Combined dataset: {len(df)} images")
    print(f"  EyePACS: {len(df_eye)}, APTOS: {len(df_aptos)}, "
          f"Messidor: {len(df_mess)}")
    for g in range(4):
        c = (df['grade'] == g).sum()
        print(f"  Grade {g}: {c} ({c / len(df) * 100:.1f}%)")

    return df


def patient_level_split(df, test_size=0.3, val_ratio=1 / 3, seed=42):
    """
    Patient-level stratified split (70/10/20).
    EyePACS left/right eyes of the same patient stay in the same split.
    """
    patient_grades = df.groupby('patient')['grade'].max().reset_index()

    train_patients, temp_patients = train_test_split(
        patient_grades, test_size=test_size,
        stratify=patient_grades['grade'], random_state=seed,
    )
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=(1 - val_ratio),
        stratify=temp_patients['grade'], random_state=seed,
    )

    train_df = df[df['patient'].isin(train_patients['patient'])].reset_index(drop=True)
    val_df = df[df['patient'].isin(val_patients['patient'])].reset_index(drop=True)
    test_df = df[df['patient'].isin(test_patients['patient'])].reset_index(drop=True)

    assert len(set(train_df['patient']) & set(test_df['patient'])) == 0
    assert len(set(train_df['patient']) & set(val_df['patient'])) == 0
    print(f"Patient leakage check: PASSED")
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    return train_df, val_df, test_df


def compute_dr_class_weights(train_df, num_classes=4):
    """Compute balanced class weights from training set."""
    classes = np.arange(num_classes)
    weights = compute_class_weight(
        'balanced', classes=classes, y=train_df['grade'].values
    )
    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    print("Class weights:")
    for c, w in class_weights.items():
        n = (train_df['grade'] == c).sum()
        print(f"  Grade {c}: weight={w:.3f} (n={n})")

    return class_weights


def ben_graham_preprocess(img, sigmaX=10):
    """
    Ben Graham preprocessing (2015 Kaggle DR winner).

    Subtracts local average color to enhance fine retinal structures
    (microaneurysms, hemorrhages) that are otherwise hidden by
    illumination variation across the fundus image.

    Formula: result = addWeighted(img, 4, GaussianBlur(img), -4, 128)

    Args:
        img: uint8 image [H, W, 3]
        sigmaX: Gaussian blur sigma (10 = standard)

    Returns:
        Preprocessed uint8 image [H, W, 3]
    """
    import cv2
    img_np = img.numpy() if hasattr(img, 'numpy') else img
    img_np = img_np.astype(np.uint8)
    blur = cv2.GaussianBlur(img_np, (0, 0), sigmaX)
    result = cv2.addWeighted(img_np, 4, blur, -4, 128)
    return result.astype(np.float32)


def make_dataset(split_df, img_size, batch_size, preprocess_fn=None,
                 shuffle=True, augment=False, num_classes=4, seed=42,
                 use_ben_graham=False):
    """
    Create a tf.data.Dataset from a split DataFrame.

    Pipeline: load raw → Ben Graham (optional) → augment → normalize.

    Args:
        split_df: DataFrame with 'path', 'grade', 'format' columns
        img_size: Target image size (height = width)
        batch_size: Batch size
        preprocess_fn: Model-specific preprocessing (e.g. resnet50.preprocess_input)
                       or None to keep raw [0, 255] (for models with internal normalization).
        shuffle: Whether to shuffle
        augment: Whether to apply data augmentation (training only)
        num_classes: Number of classes for one-hot encoding
        seed: Random seed for shuffling
        use_ben_graham: Apply Ben Graham preprocessing (recommended for DR)
    """

    def _load_and_preprocess_jpeg(p, l):
        """Load JPEG, apply Ben Graham, return float32 [0, 255]."""
        img_raw = tf.io.read_file(p)
        img = tf.image.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32)

        if use_ben_graham:
            img = tf.py_function(
                lambda x: ben_graham_preprocess(x),
                [tf.cast(img, tf.uint8)],
                tf.float32
            )
            img.set_shape([img_size, img_size, 3])

        lbl = tf.one_hot(l, num_classes)
        return img, lbl

    def _load_raw_npy_fn(p, l):
        """Load NPY, apply Ben Graham, return float32 [0, 255]."""
        img = np.load(p.numpy().decode('utf-8')).astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0
        img = tf.image.resize(img, [img_size, img_size]).numpy()

        if use_ben_graham:
            img = ben_graham_preprocess(img)

        lbl = tf.one_hot(int(l.numpy()), num_classes)
        return img.astype(np.float32), lbl

    def load_raw_npy(path, label):
        img, lbl = tf.py_function(
            _load_raw_npy_fn, [path, label], [tf.float32, tf.float32]
        )
        img.set_shape([img_size, img_size, 3])
        lbl.set_shape([num_classes])
        return img, lbl

    # --- Augmentation (on [0, 255] RGB images) ---

    def train_augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        # Random 90-degree rotations (retinal images are rotationally invariant)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
        # Color augmentation
        img = tf.image.random_brightness(img, max_delta=30.0)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, 0.0, 255.0)
        return img, label

    # --- Normalization ---
    # preprocess_fn=None  → raw [0, 255] (for models with internal normalization, e.g. EfficientNet)
    # preprocess_fn=func  → apply func (e.g. resnet50.preprocess_input for caffe-style)

    def normalize(img, label):
        if preprocess_fn is not None:
            img = preprocess_fn(img)
        # If preprocess_fn is None, keep raw [0, 255] — model normalizes internally
        return img, label

    # --- Build pipeline ---

    jpeg_df = split_df[split_df['format'] == 'jpeg']
    npy_df = split_df[split_df['format'] == 'npy']

    datasets = []

    if len(jpeg_df) > 0:
        ds_jpeg = tf.data.Dataset.from_tensor_slices(
            (jpeg_df['path'].values, jpeg_df['grade'].values)
        )
        ds_jpeg = ds_jpeg.map(
            _load_and_preprocess_jpeg, num_parallel_calls=tf.data.AUTOTUNE
        )
        datasets.append(ds_jpeg)

    if len(npy_df) > 0:
        ds_npy = tf.data.Dataset.from_tensor_slices(
            (npy_df['path'].values, npy_df['grade'].values)
        )
        ds_npy = ds_npy.map(
            load_raw_npy, num_parallel_calls=tf.data.AUTOTUNE
        )
        datasets.append(ds_npy)

    ds = datasets[0]
    for d in datasets[1:]:
        ds = ds.concatenate(d)

    # Order: augment → normalize
    if augment:
        ds = ds.map(train_augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(split_df), 10000), seed=seed
        )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
