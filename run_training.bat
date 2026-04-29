@echo off
SETLOCAL EnableDelayedExpansion
TITLE Neural Synergy Master Orchestrator

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

:MENU
cls
echo ============================================================
echo      NEURAL SYNERGY RESEARCH PIPELINE: MASTER ORCHESTRATOR
echo ============================================================
echo.
echo [1] RUN FULL TRAINING PIPELINE (High-Fidelity)
echo [2] RUN ABLATION STUDY (Performance Drop Analysis)
echo [3] GENERATE FINAL CONSOLIDATED REPORT
echo [4] LAUNCH REAL-TIME EMOTION HUD
echo [5] EXIT
echo.
set "choice="
set /p choice="Enter your selection (1-5): "

if "%choice%"=="1" goto TRAINING
if "%choice%"=="2" goto ABLATION
if "%choice%"=="3" goto REPORT
if "%choice%"=="4" goto HUD
if "%choice%"=="5" goto EXIT
goto MENU

:TRAINING
echo.
if not exist "train_local.py" (
    echo [ERROR] train_local.py not found in current directory.
    pause
    goto MENU
)
echo [!] STARTING HIGH-FIDELITY TRAINING (Memory Optimized)...
SET TF_ENABLE_ONEDNN_OPTS=0
!PYTHON_CMD! train_local.py
pause
goto MENU

:ABLATION
echo.
if not exist "ablation_study.py" (
    echo [ERROR] ablation_study.py not found in current directory.
    pause
    goto MENU
)
echo [!] STARTING ABLATION STUDY...
!PYTHON_CMD! ablation_study.py
pause
goto MENU

:REPORT
echo.
if not exist "generate_report.py" (
    echo [ERROR] generate_report.py not found in current directory.
    pause
    goto MENU
)
echo [!] GENERATING FINAL RESEARCH REPORT...
!PYTHON_CMD! generate_report.py
pause
goto MENU

:HUD
echo.
if not exist "run_hud.bat" (
    echo [ERROR] run_hud.bat not found in current directory.
    pause
    goto MENU
)
echo [!] LAUNCHING REAL-TIME HUD...
call run_hud.bat
goto MENU

:EXIT
echo.
echo Terminating Orchestrator. Good luck with your research!
timeout /t 2 >nul
exit /b 0
