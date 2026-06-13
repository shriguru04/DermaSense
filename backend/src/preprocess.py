# backend/src/preprocess.py
"""
Preprocessing Pipeline for DermaSense
--------------------------------------
Handles the UCI Dermatology dataset (366 samples, 34 features, 6 classes)
Steps: clean → impute → scale → SMOTE → PCA (optional)

The 34 manual features:
  F1–F10:  Clinical (observable during examination)
  F11–F33: Histopathological (from biopsy/microscopy)
  F34:     Age (continuous, 8 missing values)
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE

# ── COLUMN DEFINITIONS ───────────────────────────────────────────
# CRITICAL: This order MUST match the training column order
# Any mismatch causes silent wrong predictions
MANUAL_FEATURE_COLS = [
    # --- Clinical features (F1-F10) ---
    'erythema', 'scaling', 'definite_borders', 'itching',
    'koebner_phenomenon', 'polygonal_papules', 'follicular_papules',
    'oral_mucosal_involvement', 'knee_elbow_involvement', 'scalp_involvement',
    # --- Family history (binary) ---
    'family_history',
    # --- Histopathological features (F12-F33) ---
    'melanin_incontinence', 'eosinophils_infiltrate', 'pnl_infiltrate',
    'fibrosis_papillary_dermis', 'exocytosis', 'acanthosis', 'hyperkeratosis',
    'parakeratosis', 'clubbing_rete_ridges', 'elongation_rete_ridges',
    'thinning_suprapapillary', 'spongiform_pustule', 'munro_microabscess',
    'focal_hypergranulosis', 'disappearance_granular_layer',
    'vacuolisation_basal_layer', 'spongiosis', 'saw_tooth_retes',
    'follicular_horn_plug', 'perifollicular_parakeratosis',
    'inflammatory_infiltrate', 'band_like_infiltrate',
    # --- Age (continuous) ---
    'age'
]

CLASS_NAMES = [
    'Psoriasis', 'Seborrheic_Dermatitis', 'Lichen_Planus',
    'Pityriasis_Rosea', 'Chronic_Dermatitis', 'PRP'
]

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_uci_dataset(csv_path=None):
    """
    Load UCI Dermatology dataset.
    If no path provided, generates synthetic demo data for testing.
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path, header=None, na_values='?')
        df.columns = MANUAL_FEATURE_COLS + ['target']
    else:
        print("No dataset CSV found — generating synthetic demo data for testing.")
        print("For real training: place dermatology.csv in backend/data/raw/")
        df = _generate_synthetic_data()
    return df


def _generate_synthetic_data(n=366, random_state=42):
    """
    Generate synthetic dataset that mimics UCI Dermatology distributions.
    Used when real dataset is not available (demo/testing purposes).
    """
    rng = np.random.RandomState(random_state)

    # Class distribution matching real UCI dataset
    class_counts = [112, 61, 72, 49, 52, 20]
    records = []

    # Disease-specific feature patterns (simplified from clinical literature)
    patterns = {
        0: {'munro_microabscess':(2,3), 'elongation_rete_ridges':(2,3), 'parakeratosis':(2,3),
            'scaling':(2,3), 'erythema':(2,3), 'knee_elbow_involvement':(2,3)},
        1: {'scalp_involvement':(2,3), 'follicular_papules':(1,3), 'scaling':(1,2)},
        2: {'band_like_infiltrate':(2,3), 'saw_tooth_retes':(2,3), 'polygonal_papules':(2,3),
            'oral_mucosal_involvement':(1,3), 'vacuolisation_basal_layer':(2,3)},
        3: {'focal_hypergranulosis':(2,3), 'spongiosis':(1,2), 'erythema':(1,2)},
        4: {'spongiosis':(2,3), 'acanthosis':(2,3), 'itching':(2,3)},
        5: {'follicular_horn_plug':(2,3), 'perifollicular_parakeratosis':(2,3)}
    }

    for cls_idx, count in enumerate(class_counts):
        for _ in range(count):
            row = {col: rng.randint(0, 2) for col in MANUAL_FEATURE_COLS[:-1]}  # 0 or 1 base
            row['age'] = rng.randint(20, 75)
            row['family_history'] = rng.randint(0, 2)
            # Apply disease-specific pattern
            for feat, (lo, hi) in patterns.get(cls_idx, {}).items():
                row[feat] = rng.randint(lo, hi + 1)
            row['target'] = cls_idx + 1  # UCI uses 1-indexed
            records.append(row)

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def clean_and_prepare(df):
    """
    Full cleaning pipeline:
    1. Handle missing age values (8 NaN in real dataset)
    2. Clip ordinal features to valid range [0,3]
    3. Remap class labels 1-6 → 0-5
    4. Return X (feature array), y (label array)
    """
    df = df.copy()

    # ── Missing age: median imputation ───────────────────────────
    age_median = df['age'].median()
    df['age'] = df['age'].fillna(age_median)
    df['age_missing'] = df['age'].isna().astype(int)  # indicator (not used in features)

    # ── Clip ordinal features to [0, 3] ──────────────────────────
    ordinal_cols = [c for c in MANUAL_FEATURE_COLS if c != 'age']
    for col in ordinal_cols:
        df[col] = df[col].clip(0, 3)

    # ── Remap classes 1-6 → 0-5 ──────────────────────────────────
    df['target'] = df['target'] - 1

    X = df[MANUAL_FEATURE_COLS].values.astype(np.float32)
    y = df['target'].values.astype(int)
    return X, y


def build_preprocessors(X_train, y_train, apply_smote=True, n_components_pca=0.95):
    """
    Fit scaler and PCA on training data only.
    Returns transformed X_train + fitted objects.
    """
    # ── Scale ────────────────────────────────────────────────────
    # RobustScaler: robust to outliers in the 0-3 ordinal scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_train)

    # ── SMOTE: handle class imbalance (112 Psoriasis vs 20 PRP) ──
    if apply_smote:
        # k_neighbors=5: PRP has 20 samples → k must be < 20
        smote = SMOTE(k_neighbors=5, random_state=42)
        X_scaled, y_train = smote.fit_resample(X_scaled, y_train)
        print(f"After SMOTE: {X_scaled.shape[0]} samples (was {sum(y_train==y_train)}, balanced to {np.bincount(y_train)})")

    # ── PCA: dimensionality reduction ────────────────────────────
    pca = PCA(n_components=n_components_pca, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA: {X_train.shape[1]} → {X_pca.shape[1]} components ({n_components_pca*100:.0f}% variance)")

    return X_pca, y_train, scaler, pca


def transform_input(X_raw, scaler, pca, include_cnn=False, cnn_features=None):
    """
    Transform a new input (1 or more samples) using fitted scaler + PCA.
    If CNN features provided, they are appended AFTER PCA transform of manual features.

    Args:
        X_raw:        np.ndarray (n, 34) — raw manual features
        scaler:       fitted RobustScaler
        pca:          fitted PCA
        include_cnn:  bool — whether CNN features are provided
        cnn_features: np.ndarray (n, 512) or None

    Returns:
        np.ndarray ready for XGBoost prediction
    """
    X_scaled = scaler.transform(X_raw)
    X_pca    = pca.transform(X_scaled)

    if include_cnn and cnn_features is not None:
        # Normalize CNN features before merging (they have different scale)
        cnn_norm = cnn_features / (np.linalg.norm(cnn_features, axis=-1, keepdims=True) + 1e-8)
        return np.hstack([X_pca, cnn_norm])

    return X_pca


def save_preprocessors(scaler, pca, label_encoder, age_median, feature_cols, models_dir=None):
    """Save all fitted preprocessors as PKL files."""
    d = models_dir or MODELS_DIR
    os.makedirs(d, exist_ok=True)
    joblib.dump(scaler,        os.path.join(d, 'scaler.pkl'))
    joblib.dump(pca,           os.path.join(d, 'pca.pkl'))
    joblib.dump(label_encoder, os.path.join(d, 'label_encoder.pkl'))
    joblib.dump({'age_median': age_median, 'feature_cols': feature_cols},
                os.path.join(d, 'config.pkl'))
    print(f"Preprocessors saved to {d}/")


def load_preprocessors(models_dir=None):
    """Load all fitted preprocessors from PKL files."""
    d = models_dir or MODELS_DIR
    scaler   = joblib.load(os.path.join(d, 'scaler.pkl'))
    pca      = joblib.load(os.path.join(d, 'pca.pkl'))
    le       = joblib.load(os.path.join(d, 'label_encoder.pkl'))
    config   = joblib.load(os.path.join(d, 'config.pkl'))
    return scaler, pca, le, config
