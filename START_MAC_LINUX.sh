#!/bin/bash
echo "============================================"
echo "  DermaSense AI - Mac/Linux Quick Start"
echo "============================================"

# ── Backend setup ────────────────────────────────
echo ""
echo "[1/4] Setting up Python virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate

echo ""
echo "[2/4] Installing Python packages (2-3 min first time)..."
pip install -r requirements.txt

echo ""
echo "[3/4] Training XGBoost model (~60 seconds)..."
python src/train.py

echo ""
echo "[4/4] Starting Flask backend in background..."
python app.py &
FLASK_PID=$!
echo "Flask started (PID: $FLASK_PID)"
sleep 2

# ── Frontend setup ───────────────────────────────
echo ""
echo "Setting up React frontend..."
cd ../frontend

echo "Installing Node packages (first time only)..."
npm install

echo ""
echo "Starting React app..."
npm start &
REACT_PID=$!

echo ""
echo "============================================"
echo "  Flask:  http://localhost:5000/api/health"
echo "  React:  http://localhost:3000"
echo "============================================"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
wait $FLASK_PID $REACT_PID
