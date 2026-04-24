@echo off
SETLOCAL EnableDelayedExpansion
TITLE Neural Synergy Master Orchestrator

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
set /p choice="Enter your selection (1-5): "

if "%choice%"=="1" goto TRAINING
if "%choice%"=="2" goto ABLATION
if "%choice%"=="3" goto REPORT
if "%choice%"=="4" goto HUD
if "%choice%"=="5" goto EXIT
goto MENU

:TRAINING
echo.
echo [!] STARTING HIGH-FIDELITY TRAINING...
python train_local.py
pause
goto MENU

:ABLATION
echo.
echo [!] STARTING ABLATION STUDY...
python ablation_study.py
pause
goto MENU

:REPORT
echo.
echo [!] GENERATING FINAL RESEARCH REPORT...
python generate_report.py
pause
goto MENU

:HUD
echo.
echo [!] LAUNCHING REAL-TIME HUD...
call run_hud.bat
goto MENU

:EXIT
echo.
echo Terminating Orchestrator. Good luck with your research!
timeout /t 2 >nul
exit /b 0
