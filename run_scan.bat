@echo off
title Ampolin AI Trading Agent - One-Click Scan
echo ======================================================================
echo Executing Autonomous Swing Scan & Trade Evaluation
echo ======================================================================
cd /d "%~dp0"

call .\venv\Scripts\activate.bat
python main.py --dry-run
pause
