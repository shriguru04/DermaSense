@echo off
echo ============================================
echo   DermaSense AI - Windows Quick Start
echo ============================================

echo.
echo [1/4] Setting up Python virtual environment...
cd backend
python -m venv venv
call venv\Scripts\activate

echo.
echo [2/4] Installing Python packages (takes 2-3 minutes)...
pip install -r requirements.txt

echo.
echo [3/4] Training XGBoost model (takes ~60 seconds)...
python src/train.py

echo.
echo [4/4] Starting Flask backend...
start "DermaSense Flask" cmd /k "cd backend && venv\Scripts\activate && python app.py"

echo.
echo Waiting for Flask to start...
timeout /t 3 /nobreak > nul

echo.
echo Setting up React frontend...
cd ..\frontend

echo Installing Node packages (first time only)...
call npm install

echo.
echo Starting React app...
start "DermaSense React" cmd /k "cd frontend && npm start"

echo.
echo ============================================
echo   DONE! Open http://localhost:3000
echo ============================================
pause
