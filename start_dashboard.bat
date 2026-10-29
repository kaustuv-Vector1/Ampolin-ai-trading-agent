@echo off
title Ampolin AI Trading Agent - Web Dashboard
echo ======================================================================
echo Launching Ampolin AI Trading Agent Web Terminal
echo ======================================================================
cd /d "%~dp0"

echo Activating Virtual Environment...
call .\venv\Scripts\activate.bat

echo Opening browser at http://127.0.0.1:8000 ...
start http://127.0.0.1:8000

echo Starting Web UI Server...
python ui.py
pause
