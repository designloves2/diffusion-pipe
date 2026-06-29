@echo off
cd /d "%~dp0"
call "%~dp0run_wsl.bat" python tools/dp_run.py
pause
