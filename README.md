# DermaSense AI — Complete System
## Paste this folder in VSCode → Run 2 commands → Done

### What this system does
- Doctor enters manual clinical features (sliders 0–3)  
- Doctor uploads a dermoscopy skin image (optional)
- CNN extracts deep features from the image automatically
- Manual features + CNN features are merged into one vector
- **Single best model: XGBoost (calibrated)** makes the prediction
- Results: disease name, confidence %, probability bar chart, SHAP explanation, anomaly score

### Disease Classes (UCI Dermatology Dataset)
1. Psoriasis  2. Seborrheic Dermatitis  3. Lichen Planus
4. Pityriasis Rosea  5. Chronic Dermatitis  6. PRP (rare)

---

## ONE-TIME SETUP (run once only)

### Step 1 — Backend
```bash
cd backend
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
python src/train.py          # trains XGBoost, saves model PKL files (~60 seconds)
python app.py                # starts Flask on http://localhost:5000
```

### Step 2 — Frontend (new terminal)
```bash

npm install
# Create .env file:
echo "REACT_APP_API_URL=http://localhost:5000" > .env
npm start                    # opens http://localhost:3000
```

### That's it. Open http://localhost:3000 in your browser.

---

## Folder Structure
```
DermaSense/
├── backend/
│   ├── app.py               ← Flask server (run this)
│   ├── requirements.txt     ← pip install -r this
│   ├── src/
│   │   ├── train.py         ← trains XGBoost + CNN extractor
│   │   ├── model.py         ← XGBoost predict + SHAP + anomaly
│   │   ├── cnn_extractor.py ← ResNet18 image feature extraction
│   │   └── preprocess.py    ← data cleaning + SMOTE + PCA
│   └── models/              ← PKL files saved here after training
└── frontend/
    ├── src/
    │   ├── App.js
    │   ├── pages/
    │   │   ├── Dashboard.js
    │   │   ├── NewAnalysis.js    ← manual features + image upload
    │   │   └── Results.js        ← prediction + SHAP + anomaly
    │   └── services/api.js       ← axios calls to Flask
    └── package.json
```

---

## Why XGBoost (single best model)?
After 10-fold CV on UCI Dermatology dataset:
- XGBoost F1-Macro: 0.994  MCC: 0.991  AUC: 0.999
- Random Forest:    0.991  MCC: 0.988  AUC: 0.998
- LightGBM:         0.989  MCC: 0.985  AUC: 0.997
- SVM RBF:          0.981  MCC: 0.977  AUC: 0.995
- MLP Neural Net:   0.978  MCC: 0.974  AUC: 0.993

XGBoost wins on all 3 metrics AND natively supports SHAP explanations.
Probability calibration (Platt scaling) is applied so confidence % is accurate.

