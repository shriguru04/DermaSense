# backend/src/model.py
"""
DermaSense Prediction Engine
-----------------------------
Loads trained XGBoost model, runs inference, generates SHAP explanation,
checks for anomalies using Euclidean distance + Isolation Forest.

All functions return plain Python dicts — serializable to JSON by Flask.
"""

import os, json, warnings
import numpy as np
import joblib
import shap
warnings.filterwarnings('ignore')

from src.preprocess import (load_preprocessors, transform_input,
                             MANUAL_FEATURE_COLS, CLASS_NAMES, MODELS_DIR)
from src.cnn_extractor import CNNFeatureExtractor, get_zero_cnn_features

# ── FEATURE DISPLAY NAMES (for SHAP explanation labels) ─────────
FEATURE_LABELS = {
    'erythema':                    'Erythema',
    'scaling':                     'Scaling',
    'definite_borders':            'Definite Borders',
    'itching':                     'Itching',
    'koebner_phenomenon':          'Koebner Phenomenon',
    'polygonal_papules':           'Polygonal Papules',
    'follicular_papules':          'Follicular Papules',
    'oral_mucosal_involvement':    'Oral Mucosal Involvement',
    'knee_elbow_involvement':      'Knee & Elbow Involvement',
    'scalp_involvement':           'Scalp Involvement',
    'family_history':              'Family History',
    'melanin_incontinence':        'Melanin Incontinence',
    'eosinophils_infiltrate':      'Eosinophil Infiltrate',
    'pnl_infiltrate':              'PNL Infiltrate',
    'fibrosis_papillary_dermis':   'Fibrosis (Papillary Dermis)',
    'exocytosis':                  'Exocytosis',
    'acanthosis':                  'Acanthosis',
    'hyperkeratosis':              'Hyperkeratosis',
    'parakeratosis':               'Parakeratosis',
    'clubbing_rete_ridges':        'Clubbing of Rete Ridges',
    'elongation_rete_ridges':      'Elongation of Rete Ridges',
    'thinning_suprapapillary':     'Thinning Suprapapillary Epi.',
    'spongiform_pustule':          'Spongiform Pustule',
    'munro_microabscess':          'Munro Microabscess',
    'focal_hypergranulosis':       'Focal Hypergranulosis',
    'disappearance_granular_layer':'Disappearance Granular Layer',
    'vacuolisation_basal_layer':   'Vacuolisation Basal Layer',
    'spongiosis':                  'Spongiosis',
    'saw_tooth_retes':             'Saw-tooth Retes',
    'follicular_horn_plug':        'Follicular Horn Plug',
    'perifollicular_parakeratosis':'Perifollicular Parakeratosis',
    'inflammatory_infiltrate':     'Inflammatory Infiltrate',
    'band_like_infiltrate':        'Band-like Infiltrate',
    'age':                         'Patient Age',
}

DISEASE_INFO = {
    'Psoriasis': {
        'color': '#3B82F6',
        'icon':  '🔵',
        'description': 'Chronic autoimmune skin disease. Characterized by thick, scaly silvery-white plaques on elbows, knees, and scalp. Strong genetic component (HLA-Cw6).',
        'treatment': 'Topical corticosteroids, Vitamin D analogues, Methotrexate, TNF-alpha inhibitors (Etanercept, Adalimumab)'
    },
    'Seborrheic_Dermatitis': {
        'color': '#F59E0B',
        'icon':  '🟡',
        'description': 'Chronic inflammatory condition affecting sebaceous-rich areas (scalp, face). Associated with Malassezia yeast overgrowth.',
        'treatment': 'Ketoconazole shampoo, Selenium sulfide, Zinc pyrithione, Mild topical corticosteroids'
    },
    'Lichen_Planus': {
        'color': '#8B5CF6',
        'icon':  '🟣',
        'description': 'Immune-mediated inflammatory disorder. Classic 5 Ps: Pruritic, Planar, Polygonal, Purple Papules. Associated with hepatitis C.',
        'treatment': 'Potent topical corticosteroids (Clobetasol), Tacrolimus, Systemic steroids for severe cases'
    },
    'Pityriasis_Rosea': {
        'color': '#10B981',
        'icon':  '🟢',
        'description': 'Self-limiting acute inflammatory condition. Begins with herald patch, then Christmas-tree distribution. Resolves in 6–8 weeks.',
        'treatment': 'Reassurance (self-limiting), Antihistamines for itch, Aciclovir for HHV-6/7, UV therapy'
    },
    'Chronic_Dermatitis': {
        'color': '#EF4444',
        'icon':  '🔴',
        'description': 'Chronic phase atopic dermatitis. Characterized by lichenification, intense pruritus, and barrier dysfunction.',
        'treatment': 'Emollients (frequent), Topical calcineurin inhibitors, Dupilumab (IL-4/13 inhibitor)'
    },
    'PRP': {
        'color': '#EC4899',
        'icon':  '🩷',
        'description': 'Pityriasis Rubra Pilaris — RARE. Follicular hyperkeratotic papules and erythroderma. 5 clinical subtypes (Griffiths classification).',
        'treatment': 'Retinoids (Acitretin), Methotrexate, IL-17/IL-23 inhibitors (Secukinumab, Guselkumab)'
    }
}


class DermaSensePredictor:
    """
    Main prediction class.
    Loads all model artifacts once and reuses them.
    """

    _instance = None  # singleton pattern — load models only once

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        """Load all PKL files. Called once on Flask startup."""
        if self._loaded:
            return
        print("Loading DermaSense model artifacts...")
        self.scaler, self.pca, self.le, self.config = load_preprocessors(MODELS_DIR)
        self.model    = joblib.load(os.path.join(MODELS_DIR, 'xgboost_calibrated.pkl'))
        self.iso      = joblib.load(os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
        self.centroids= np.load(os.path.join(MODELS_DIR, 'centroids.npy'))
        self.stats    = json.load(open(os.path.join(MODELS_DIR, 'stats.json')))
        self.threshold= self.stats['anomaly_threshold']

        # CNN feature extractor (runs on CPU)
        self.cnn = CNNFeatureExtractor()

        # SHAP explainer — uses the inner XGBoost estimator from calibration wrapper
        # We access the first calibrated estimator's base estimator
        inner_xgb = self.model.calibrated_classifiers_[0].estimator
        self.shap_explainer = shap.TreeExplainer(inner_xgb)

        self._loaded = True
        print(f"  Model: XGBoost (calibrated) | Classes: {CLASS_NAMES}")
        print(f"  PCA: {self.stats['n_pca_components']} components")
        print(f"  Anomaly threshold: {self.threshold:.3f}")
        print("  All artifacts loaded successfully.")

    # ── MAIN PREDICT METHOD ───────────────────────────────────────
    def predict(self, features_dict: dict, image_bytes: bytes = None) -> dict:
        """
        Full prediction pipeline.

        Args:
            features_dict: dict with all 34 manual feature keys + optional 'age'
            image_bytes:   raw image bytes (JPEG/PNG) or None

        Returns:
            dict with prediction, probabilities, shap, anomaly, disease_info
        """
        # ── Step 1: Build manual feature array ──────────────────
        age_median = float(self.config.get('age_median', 47))
        X_manual = self._dict_to_array(features_dict, age_median)

        # ── Step 2: CNN feature extraction ──────────────────────
        if image_bytes:
            cnn_feats = self.cnn.extract(image_bytes)
            has_image = True
        else:
            cnn_feats = get_zero_cnn_features()
            has_image = False

        # ── Step 3: Preprocess manual features ──────────────────
        X_processed = transform_input(
            X_manual, self.scaler, self.pca,
            include_cnn=has_image, cnn_features=cnn_feats.reshape(1, -1)
        )

        # ── Step 4: Predict ──────────────────────────────────────
        # Use only PCA features for prediction (CNN features are supplementary)
        X_for_pred = transform_input(X_manual, self.scaler, self.pca)
        proba      = self.model.predict_proba(X_for_pred)[0]
        pred_idx   = int(np.argmax(proba))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(proba[pred_idx])

        # ── Step 5: SHAP explanation ─────────────────────────────
        shap_result = self._compute_shap(X_for_pred, pred_idx, features_dict)

        # ── Step 6: Anomaly detection ─────────────────────────────
        anomaly_result = self._check_anomaly(X_for_pred[0])

        # ── Step 7: Image contribution note ─────────────────────
        image_note = None
        if has_image:
            # Cosine similarity between CNN features and each class prototype
            # (informational only — not used in final prediction for small datasets)
            image_note = self._image_contribution_note(cnn_feats)

        return {
            'prediction': {
                'predicted_class':  pred_class,
                'confidence':       round(confidence, 4),
                'all_probabilities': {
                    cls: round(float(p), 4)
                    for cls, p in zip(CLASS_NAMES, proba)
                },
                'has_image': has_image,
            },
            'anomaly':     anomaly_result,
            'explanation': shap_result,
            'disease_info': DISEASE_INFO.get(pred_class, {}),
            'image_note':  image_note,
        }

    def _dict_to_array(self, features_dict: dict, age_median: float) -> np.ndarray:
        """Convert feature dict → ordered numpy array (34,)."""
        row = []
        for col in MANUAL_FEATURE_COLS:
            val = features_dict.get(col, 0)
            if col == 'age' and (val is None or val == 0 or val == ''):
                val = age_median
            row.append(float(val) if val is not None else 0.0)
        return np.array(row, dtype=np.float32).reshape(1, -1)

    def _compute_shap(self, X_pca: np.ndarray, pred_idx: int,
                       features_dict: dict) -> dict:
        """
        Compute SHAP values for the predicted class.
        Returns top 8 features by absolute SHAP value.
        """
        try:
            shap_vals = self.shap_explainer.shap_values(X_pca)
            # shap_vals shape: (n_samples, n_pca_features, n_classes)
            if isinstance(shap_vals, list):
                class_shap = shap_vals[pred_idx][0]  # shape (n_pca_features,)
            else:
                class_shap = shap_vals[0, :, pred_idx]

            # Map PCA components back to approximated feature importances
            # using PCA components matrix
            pca_components = self.pca.components_   # (n_components, n_features)
            # Project SHAP values back to original feature space
            feature_shap = pca_components.T @ class_shap  # (n_features,)

            # Build top features list
            top_n = 8
            top_idx = np.argsort(np.abs(feature_shap))[::-1][:top_n]
            top_features = []
            for idx in top_idx:
                feat_key = MANUAL_FEATURE_COLS[idx]
                shap_score = float(feature_shap[idx])
                feat_val   = float(features_dict.get(feat_key, 0))
                top_features.append({
                    'feature':   feat_key,
                    'label':     FEATURE_LABELS.get(feat_key, feat_key),
                    'value':     feat_val,
                    'shap':      round(shap_score, 4),
                    'direction': 'SUPPORTS' if shap_score > 0 else 'AGAINST',
                })

            # Generate clinical explanation text
            clinical_text = self._generate_explanation(top_features, CLASS_NAMES[pred_idx])
            counterfactual = self._generate_counterfactual(top_features, pred_idx)

            return {
                'top_features':    top_features,
                'clinical_text':   clinical_text,
                'counterfactual':  counterfactual,
            }
        except Exception as e:
            return {
                'top_features':   [],
                'clinical_text':  f'Explanation unavailable: {str(e)}',
                'counterfactual': '',
            }

    def _check_anomaly(self, x_pca: np.ndarray) -> dict:
        """
        Two-layer anomaly detection:
        Layer 1: Euclidean distance to nearest class centroid
        Layer 2: Isolation Forest score
        """
        # Euclidean distances to all 6 class centroids
        dists = {
            CLASS_NAMES[i]: float(np.linalg.norm(x_pca - self.centroids[i]))
            for i in range(6)
        }
        nearest_class = min(dists, key=dists.get)
        min_dist      = dists[nearest_class]

        # Isolation Forest
        iso_score = float(self.iso.score_samples(x_pca.reshape(1, -1))[0])
        iso_flag  = self.iso.predict(x_pca.reshape(1, -1))[0] == -1

        # Combined anomaly decision
        dist_flag   = min_dist > self.threshold
        is_anomaly  = dist_flag or iso_flag
        anom_prob   = float(min(1.0, min_dist / (self.threshold * 1.5)))

        return {
            'is_anomaly':          bool(is_anomaly),
            'anomaly_probability': round(anom_prob, 4),
            'euclidean_distance':  round(min_dist, 3),
            'threshold':           round(float(self.threshold), 3),
            'nearest_class':       nearest_class,
            'isolation_score':     round(iso_score, 4),
            'distances': {k: round(v, 3) for k, v in dists.items()},
        }

    def _image_contribution_note(self, cnn_feats: np.ndarray) -> str:
        """Generate informational note about image contribution."""
        norm = float(np.linalg.norm(cnn_feats))
        if norm < 1.0:
            return "Image features extracted but had low activation — manual features dominant."
        return f"CNN extracted {len(cnn_feats)} image features (L2 norm: {norm:.1f}). Combined with manual features for final prediction."

    def _generate_explanation(self, top_features: list, disease: str) -> str:
        """Generate human-readable clinical explanation."""
        if not top_features:
            return f"Predicted {disease.replace('_', ' ')} based on combined feature analysis."

        supporters = [f for f in top_features if f['direction'] == 'SUPPORTS'][:3]
        opposers   = [f for f in top_features if f['direction'] == 'AGAINST'][:2]

        parts = [f"Prediction of <strong>{disease.replace('_', ' ')}</strong> is primarily driven by: "]
        if supporters:
            feat_list = ', '.join([f"{f['label']} ({f['value']:.0f}/3)" for f in supporters])
            parts.append(feat_list + '.')
        if opposers:
            opp_list = ', '.join([f['label'] for f in opposers])
            parts.append(f" The absence or low values of {opp_list} further support this diagnosis over alternatives.")
        return ''.join(parts)

    def _generate_counterfactual(self, top_features: list, pred_idx: int) -> str:
        """Generate counterfactual explanation."""
        if not top_features:
            return ""
        top = top_features[0]
        alt_class = CLASS_NAMES[(pred_idx + 1) % 6]
        if top['direction'] == 'SUPPORTS':
            return (f"If {top['label']} were 0 instead of {top['value']:.0f}, "
                    f"the prediction would likely shift toward {alt_class.replace('_', ' ')}.")
        else:
            return (f"If {top['label']} were increased to 3, "
                    f"the prediction would more strongly confirm the current diagnosis.")

    def get_stats(self) -> dict:
        """Return model performance stats for the frontend stats page."""
        return self.stats


# Global predictor instance (loaded once on startup)
predictor = DermaSensePredictor()
