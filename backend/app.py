# backend/app.py
"""
DermaSense Flask API Server
----------------------------
Run:  python app.py

Endpoints:
  GET  /api/health            — server + model status
  POST /api/predict           — manual features only
  POST /api/predict-with-image — manual features + image (multipart)
  GET  /api/stats             — model performance metrics
"""

import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from src.model import predictor, DISEASE_INFO, FEATURE_LABELS
from src.preprocess import MANUAL_FEATURE_COLS, CLASS_NAMES

# ── APP SETUP ────────────────────────────────────────────────────
app = Flask(__name__)

# CORS: allow React dev server + production Netlify URL
ORIGINS = os.getenv(
    'ALLOWED_ORIGINS',
    'http://localhost:3000,http://localhost:3001'
).split(',')

CORS(app, resources={
    r'/api/*': {
        'origins': [o.strip() for o in ORIGINS],
        'methods': ['GET', 'POST', 'OPTIONS'],
        'allow_headers': ['Content-Type', 'Authorization'],
    }
})

# ── LOAD MODELS ON STARTUP ───────────────────────────────────────
# This runs before the first request — models are loaded once
try:
    predictor.load()
    MODEL_READY = True
    print("\n✅ DermaSense ready at http://localhost:5000\n")
except FileNotFoundError:
    MODEL_READY = False
    print("\n⚠️  Models not found — run: python src/train.py first\n")


# ── HELPER: parse feature dict from request ──────────────────────
def _parse_features(data: dict) -> dict:
    """
    Extract and validate all feature values from request JSON.
    Defaults missing features to 0.
    Returns clean dict ready for predictor.
    """
    features = {}
    for col in MANUAL_FEATURE_COLS:
        raw = data.get(col, data.get('features', {}).get(col, 0))
        try:
            val = float(raw) if raw not in (None, '', 'null') else 0.0
        except (TypeError, ValueError):
            val = 0.0
        # Clip ordinal features to valid range
        if col != 'age':
            val = max(0.0, min(3.0, val))
        features[col] = val
    return features


# ════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health():
    """Health check — used by React to show API status indicator."""
    return jsonify({
        'status':       'ok' if MODEL_READY else 'model_not_trained',
        'model_ready':  MODEL_READY,
        'model':        'XGBoost (CalibratedClassifierCV)',
        'classes':      CLASS_NAMES,
        'n_features':   len(MANUAL_FEATURE_COLS),
        'cnn_enabled':  True,
        'version':      '2.0',
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict from manual features only (no image).

    Request body (JSON):
    {
        "erythema": 2,
        "scaling": 3,
        "munro_microabscess": 3,
        "age": 45,
        ... (all 34 features, missing ones default to 0)
    }

    OR nested:
    { "features": { "erythema": 2, ... } }
    """
    if not MODEL_READY:
        return jsonify({'error': 'Model not trained. Run: python src/train.py'}), 503

    try:
        data = request.get_json(force=True, silent=True) or {}
        # Support both flat and nested { features: {...} } format
        if 'features' in data and isinstance(data['features'], dict):
            features = _parse_features(data['features'])
        else:
            features = _parse_features(data)

        result = predictor.predict(features, image_bytes=None)
        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict-with-image', methods=['POST'])
def predict_with_image():
    """
    Predict from manual features + dermoscopy image.

    Request: multipart/form-data
      - features: JSON string of feature values
      - image:    image file (JPEG, PNG, WEBP)

    OR: JSON body with base64 image
      { "features": {...}, "image_base64": "data:image/jpeg;base64,..." }
    """
    if not MODEL_READY:
        return jsonify({'error': 'Model not trained. Run: python src/train.py'}), 503

    try:
        image_bytes = None

        # ── Parse image ──────────────────────────────────────────
        if request.files.get('image'):
            image_bytes = request.files['image'].read()
            features_raw = request.form.get('features', '{}')
            features = _parse_features(json.loads(features_raw))

        elif request.is_json:
            data = request.get_json()
            features_raw = data.get('features', data)
            features = _parse_features(features_raw if isinstance(features_raw, dict) else data)

            # Handle base64 image
            img_b64 = data.get('image_base64', '')
            if img_b64:
                import base64
                # Strip data URL prefix if present
                if ',' in img_b64:
                    img_b64 = img_b64.split(',', 1)[1]
                image_bytes = base64.b64decode(img_b64)
        else:
            return jsonify({'error': 'Send multipart form-data with features + image, or JSON with image_base64'}), 400

        result = predictor.predict(features, image_bytes=image_bytes)
        return jsonify(result), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Return model performance metrics for the stats dashboard."""
    if not MODEL_READY:
        return jsonify({'error': 'Model not trained'}), 503
    return jsonify(predictor.get_stats()), 200


@app.route('/api/features', methods=['GET'])
def feature_info():
    """Return feature metadata for dynamic form generation."""
    return jsonify({
        'features':     MANUAL_FEATURE_COLS,
        'labels':       FEATURE_LABELS,
        'disease_info': DISEASE_INFO,
        'class_names':  CLASS_NAMES,
    }), 200


# ── START SERVER ──────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)
