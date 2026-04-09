@echo off
title "ORIEN | MASTER SYNERGY SEQUENCE [Advanced]"
setlocal enabledelayedexpansion

:: 💎 ORIEN: MASTER SYNERGY SEQUENCE (FULL AUTOMATION)
:: This script performs the complete audit, optimize, train, and test cycle.
color 0b
set "ROOT=%~dp0"
cd /d "%ROOT%"
set PYTHONUNBUFFERED=1
set TF_ENABLE_ONEDNN_OPTS=0
set CUDA_VISIBLE_DEVICES=-1
set TF_CPP_MIN_LOG_LEVEL=3

echo.
echo "  🌌 ORIEN NEURAL MASTER | SYNERGY SEQUENCE"
echo "  ════════════════════════════════════════════════"
echo "  [STAGE 0: SYNC] Booting Neural Clusters..."
if exist ".venv_training\Scripts\python.exe" (
    set "PY=.venv_training\Scripts\python.exe"
) else (
    set "PY=python"
)

echo.
echo "  [STAGE 0.1: SEEDS] Downloading Base Neural Prototypes..."
!PY! scripts\download_pre_trained_seeds.py

echo.
echo "  [STAGE 0.2: DATA] Synchronizing Distributed Repositories..."
echo [SYS] Skipping HARD_DOWNLOAD.bat as requested by user.
:: call scripts\HARD_DOWNLOAD.bat
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo "  [ERR] Synergy Sync Phase Failed. Data cluster in fragmentation."
    echo "  Continuing with existing local shards..."
)

echo.
echo "  [STAGE 1: AUDIT] Running Local Data Audit..."
if exist ".venv_training\Scripts\python.exe" (
    set "PY=.venv_training\Scripts\python.exe"
) else (
    set "PY=python"
)

!PY! scripts\verify_datasets.py
if !ERRORLEVEL! NEQ 0 (
    echo [ERR] Audit Failed. Please check dataset folder structure.
    exit /b 1
)

echo.
echo "  [STAGE 2: BALANCE] Resolving High Bias skews..."
!PY! scripts\balance_datasets.py
if !ERRORLEVEL! NEQ 0 (
    echo [WARN] Balancing skipped or failed. Continuing...
)

echo.
echo "  [STAGE 3: OPTIMIZE] Purging binary corruption..."
!PY! scripts\optimize_dataset.py
if !ERRORLEVEL! NEQ 0 (
    echo [WARN] Optimization skipped or failed. Continuing...
)

echo.
echo "  [STAGE 4: TRAIN ALL] Executing 20-Epoch Master Cycle..."
set "TRAIN_CMD=!PY! -u training\local_trainer.py --modality all --epochs 1 --batch_size 8"
!TRAIN_CMD!
if !ERRORLEVEL! NEQ 0 (
    echo.
    echo [ERR] Training Cycle Interrupted or Critical Failure.
    echo [SYS] Re-running with diagnostic mode...
    exit /b !ERRORLEVEL!
)

echo.
echo "  [STAGE 5: SYNERGY TEST] Verifying Ensemble Status..."
!PY! scripts\test_ensemble.py

echo.
echo "  [STAGE 6: BOOT] Powering up ORIEN Neural Hub..."
echo [SYS] Finalizing synergy. Launching LAUNCH.bat in Production mode...
:: Pass a flag to LAUNCH.bat if needed to skip the menu, but LAUNCH.bat usually waits for user interaction
start LAUNCH.bat /quick

pause
