@echo off
cd /d "%~dp0"
call "%~dp0run_wsl.bat" python tools/dp_tensorboard.py
pause
