@echo off
cd /d "%~dp0"
echo Starting Krea2 wizard...
echo Loading WSL/micromamba environment. This can take 10-30 seconds.
echo.
call "%~dp0run_wsl.bat" python tools/krea2_wizard.py
pause
