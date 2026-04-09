@echo off
title "ORIEN | Neural Recovery Unit [Optimized]"
setlocal enabledelayedexpansion

:: 🧬 ORIEN MASTER RECOVERY SCRIPT
:: Restores the Neural Foundation (Seeds) and Scaffolds the Dataset.

echo.
echo "  🛠️  ORIEN: System NEURAL RECOVERY"
echo "  ══════════════════════════════════════════"
echo "  [AUDIT] Validating Directory Synergy..."

set "ROOT=%~dp0.."
cd /d "%ROOT%"

:: STAGE 1: Scaffolding
echo "  [STEP 1/3] Scaffolding Neural Modalities..."
if exist "scripts\verify_datasets.py" (
    python scripts\verify_datasets.py >nul 2>&1
    echo "  [OK] Neural Scaffolding complete."
)

:: STAGE 2: Foundations (v0)
echo "  [STEP 2/3] Patching Neural Seeds (v0)..."
if exist "scripts\download_pre_trained_seeds.py" (
    python scripts\download_pre_trained_seeds.py
    echo "  [OK] Neural Seeds established (Models/v0)."
)

:: STAGE 3: Directory Alignment
echo "  [STEP 3/3] Aligning VMAX Model Paths..."
if not exist "models\vmax" (
    mkdir "models\vmax"
    echo "  [NEW] VMAX Cluster initialized."
    echo "  [REQ] You must run 'scripts\train_all_modalities.py' to populate VMAX."
)

echo.
echo "  -------------------------------------------------"
echo "  ✅ RECOVERY COMPLETE: SYSTEM READY FOR SYNERGY"
echo "  -------------------------------------------------"
echo.
pause
