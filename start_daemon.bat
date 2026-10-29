@echo off
title Ampolin AI Trading Agent - 24/7 Market Daemon
echo ======================================================================
echo Launching Ampolin AI Trading Agent Autonomous Market Daemon (IST)
echo ======================================================================
cd /d "%~dp0"

echo Activating Virtual Environment...
call .\venv\Scripts\activate.bat

echo Starting Market Hours Scheduler (09:15 to 15:45 IST)...
python cloud_scheduler.py
pause
