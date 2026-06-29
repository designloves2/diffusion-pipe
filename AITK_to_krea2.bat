@echo off
cd /d "%~dp0"
call "%~dp0run_wsl.bat" python tools/aitk_to_krea2.py
pause
