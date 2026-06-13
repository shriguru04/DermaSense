# backend/src/train.py
"""
DermaSense Training Script
--------------------------
Run this once to train XGBoost and save all model artifacts.

Command:  python src/train.py

What it does:
1. Loads UCI Dermatology dataset (or synthetic demo data)
2. Cleans + preprocesses (imputation, scaling, SMOTE, PCA)
3. Trains XGBoost with 10-fold stratified CV
4. Calibrates probabilities (Platt scaling)
5. Computes class centroids + Isolation Forest for anomaly detection
6. Saves all PKL files to backend/models/
7. Prints final performance metrics

Total time: ~60 seconds on CPU
"""

import os, sys, json, time, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.model_selection import StratifiedKFold, cross_validate, StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (f1_score, matthews_corrcoef, classification_report,
                              roc_auc_score, accuracy_score, confusion_matrix)
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from src.preprocess import (load_uci_dataset, clean_and_prepare, build_preprocessors,
                             save_preprocessors, MANUAL_FEATURE_COLS, CLASS_NAMES, MODELS_DIR)

# ── XGBOOST HYPERPARAMETERS ──────────────────────────────────────
# These are tuned for UCI Dermatology (366 samples, 6 classes, imbalanced)
XGBOOST_PARAMS = {
    'n_estimators':       500,
    'max_depth':          5,
    'learning_rate':      0.05,
    'subsample':          0.8,
    'colsample_bytree':   0.8,
    'min_child_weight':   3,
    'gamma':              0.1,
    'reg_alpha':          0.1,     # L1 regularization
    'reg_lambda':         1.0,     # L2 regularization
    'objective':          'multi:softprob',
    'num_class':          6,
    'eval_metric':        'mlogloss',
    'use_label_encoder':  False,
    'random_state':       42,
    'n_jobs':             -1,
    'tree_method':        'hist',  # fast histogram method
}


def train():
    print("=" * 60)
    print("  DermaSense AI — Model Training")
    print("=" * 60)
    t0 = time.time()

    # ── 1. Load dataset ──────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'dermatology.csv')
    df = load_uci_dataset(csv_path)
    print(f"\n[1/7] Dataset loaded: {df.shape[0]} samples, {len(MANUAL_FEATURE_COLS)} features")
    print(f"  Class distribution: {dict(zip(CLASS_NAMES, np.bincount(df['target'].values - 1)))}")

    # ── 2. Clean & prepare ───────────────────────────────────────
    X_raw, y = clean_and_prepare(df)
    print(f"\n[2/7] Data cleaned: X={X_raw.shape}, y={y.shape}")
    print(f"  Age median imputation for {df['age'].isna().sum()} missing values")

    # ── 3. Train/test split (stratified) ────────────────────────
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(sss.split(X_raw, y))
    X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
    y_train, y_test         = y[train_idx], y[test_idx]
    print(f"\n[3/7] Split: train={len(y_train)}, test={len(y_test)} (80/20 stratified)")

    # ── 4. Preprocess (scaler + SMOTE + PCA) ─────────────────────
    print(f"\n[4/7] Preprocessing (RobustScaler + SMOTE + PCA)...")
    X_train, y_train_bal, scaler, pca = build_preprocessors(
        X_train_raw, y_train, apply_smote=True, n_components_pca=0.95
    )
    X_test = scaler.transform(X_test_raw)
    X_test = pca.transform(X_test)
    print(f"  Train after SMOTE+PCA: {X_train.shape}")

    # ── 5. LabelEncoder ─────────────────────────────────────────
    le = LabelEncoder()
    le.fit(np.arange(6))
    le.classes_ = np.array(CLASS_NAMES)

    # ── 6. 10-Fold Cross Validation ─────────────────────────────
    print(f"\n[5/7] 10-Fold Stratified Cross-Validation...")
    xgb_base = XGBClassifier(**XGBOOST_PARAMS)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    cv_results = cross_validate(
        xgb_base, X_train, y_train_bal, cv=skf,
        scoring=['f1_macro', 'balanced_accuracy', 'roc_auc_ovr_weighted'],
        return_train_score=False, n_jobs=-1
    )
    cv_f1  = cv_results['test_f1_macro'].mean()
    cv_std = cv_results['test_f1_macro'].std()
    cv_mcc = cv_results['test_balanced_accuracy'].mean()
    cv_auc = cv_results['test_roc_auc_ovr_weighted'].mean()
    print(f"  CV F1-Macro:         {cv_f1:.4f} ± {cv_std:.4f}")
    print(f"  CV Balanced Acc:     {cv_mcc:.4f}")
    print(f"  CV AUC (OvR):        {cv_auc:.4f}")

    # ── 7. Train final model + calibration ───────────────────────
    print(f"\n[6/7] Training final XGBoost + Platt Calibration...")
    xgb_final = XGBClassifier(**XGBOOST_PARAMS)
    # CalibratedClassifierCV with cv=5 applies Platt scaling
    # Makes predicted probabilities accurate (not just ranking)
    calibrated_model = CalibratedClassifierCV(
        xgb_final, cv=5, method='sigmoid'
    )
    calibrated_model.fit(X_train, y_train_bal)

    # ── 8. Test set evaluation ───────────────────────────────────
    y_pred  = calibrated_model.predict(X_test)
    y_proba = calibrated_model.predict_proba(X_test)

    test_f1  = f1_score(y_test, y_pred, average='macro')
    test_mcc = matthews_corrcoef(y_test, y_pred)
    test_acc = accuracy_score(y_test, y_pred)
    try:
        test_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    except Exception:
        test_auc = 0.0

    print(f"\n  --- TEST SET RESULTS ---")
    print(f"  Accuracy:    {test_acc:.4f}")
    print(f"  F1-Macro:    {test_f1:.4f}")
    print(f"  MCC:         {test_mcc:.4f}")
    print(f"  AUC (OvR):   {test_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=CLASS_NAMES)}")

    # ── 9. Class centroids (for anomaly detection) ───────────────
    print(f"\n[7/7] Computing centroids + Isolation Forest...")
    centroids = {}
    for cls in range(6):
        mask = y_train_bal == cls
        if mask.sum() > 0:
            centroids[cls] = X_train[mask].mean(axis=0)
    centroids_arr = np.stack([centroids[i] for i in range(6)])

    # Compute per-class distance thresholds (mean + 2*std of training distances)
    all_dists = []
    for i, x in enumerate(X_train):
        cls = y_train_bal[i]
        dist = np.linalg.norm(x - centroids[cls])
        all_dists.append(dist)
    all_dists = np.array(all_dists)
    anomaly_threshold = float(all_dists.mean() + 2.0 * all_dists.std())

    # Isolation Forest for second-layer anomaly detection
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_train)

    # ── 10. Save confusion matrix + per-class metrics ─────────────
    cm = confusion_matrix(y_test, y_pred).tolist()
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask = y_test == i
        if mask.sum() > 0:
            per_class[name] = {
                'precision': float(f1_score(y_test == i, y_pred == i, average='binary', zero_division=0)),
                'recall':    float((y_pred[mask] == i).mean()),
                'support':   int(mask.sum())
            }

    # ── 11. Save all artifacts ────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(calibrated_model, os.path.join(MODELS_DIR, 'xgboost_calibrated.pkl'))
    joblib.dump(iso,              os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
    np.save(os.path.join(MODELS_DIR, 'centroids.npy'), centroids_arr)
    save_preprocessors(scaler, pca, le, df['age'].median(), MANUAL_FEATURE_COLS, MODELS_DIR)

    # Save performance stats as JSON (for frontend model stats page)
    stats = {
        'cv_f1_macro':       round(cv_f1, 4),
        'cv_f1_std':         round(cv_std, 4),
        'cv_balanced_acc':   round(cv_mcc, 4),
        'cv_auc':            round(cv_auc, 4),
        'test_accuracy':     round(test_acc, 4),
        'test_f1_macro':     round(test_f1, 4),
        'test_mcc':          round(test_mcc, 4),
        'test_auc':          round(test_auc, 4),
        'confusion_matrix':  cm,
        'per_class_metrics': per_class,
        'anomaly_threshold': round(anomaly_threshold, 4),
        'n_pca_components':  int(pca.n_components_),
        'class_names':       CLASS_NAMES,
        'model':             'XGBoost (CalibratedClassifierCV, Platt)',
        'train_samples':     int(len(y_train_bal)),
        'test_samples':      int(len(y_test)),
    }
    with open(os.path.join(MODELS_DIR, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Models saved to: {os.path.abspath(MODELS_DIR)}")
    print(f"  Start Flask: python app.py")
    print(f"{'='*60}\n")

    return stats


if __name__ == '__main__':
    train()
