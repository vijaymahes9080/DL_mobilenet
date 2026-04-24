@echo off
SETLOCAL EnableDelayedExpansion
TITLE ORIEN Neural Synergy - Emotion HUD

echo ============================================================
echo      NEURAL SYNERGY RESEARCH PIPELINE: REAL-TIME HUD
echo ============================================================
echo [STATE] INITIALIZING ENGINE...
echo.

:: Detect Python
set PYTHON_CMD=python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    set PYTHON_CMD=python3
    python3 --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Python was not found in your PATH.
        echo Please ensure Python is installed and added to your environment variables.
        pause
        exit /b 1
    )
)

:: Check for model
if not exist "models\optimized\champion_model.tflite" (
    echo [ERROR] Optimized TFLite model not found.
    echo Expected path: models\optimized\champion_model.tflite
    echo Please run the training pipeline first.
    echo.
    pause
    exit /b 1
)

echo [STATE] HUD ACTIVE - SYNCHRONIZING WITH CAMERA...
echo [INFO]  Using: !PYTHON_CMD!
echo [INFO]  Press 'Q' inside the window to terminate mission.
echo.

!PYTHON_CMD! inference_hud.py

if %errorlevel% neq 0 (
    echo.
    echo [CRITICAL ERROR] HUD Engine encountered a neural desync (Exit Code: %errorlevel%).
    echo.
    echo Potential issues:
    echo 1. Camera is already in use by another app.
    echo 2. Missing dependencies (Run: pip install opencv-python tensorflow numpy).
    echo 3. Model file is corrupted.
    echo.
    pause
)

echo.
echo [STATE] MISSION COMPLETE. TERMINATING...
timeout /t 3
