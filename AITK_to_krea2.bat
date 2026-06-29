@echo off
cd /d "%~dp0"
echo Starting AI-Toolkit to Krea2 converter...
echo Loading WSL/micromamba environment. This can take 10-30 seconds.
echo.
call "%~dp0run_wsl.bat" python tools/aitk_to_krea2.py
pause
