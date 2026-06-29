@echo off
cd /d "%~dp0"
call "%~dp0run_wsl.bat" python tools/krea2_wizard.py
pause
