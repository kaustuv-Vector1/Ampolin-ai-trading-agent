@echo off
title Stop Ampolin Trading Agent
echo Stopping any running Ampolin Python / Uvicorn tasks...
taskkill /F /IM python.exe /T 2>nul
echo All Ampolin processes stopped.
pause
