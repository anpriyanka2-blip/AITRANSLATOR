@echo off
cd /d "%~dp0"
set OPEN_BROWSER=1
".venv\Scripts\python.exe" app.py
pause
