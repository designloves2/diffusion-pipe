@echo off
cd /d "%~dp0"
echo Starting diffusion-pipe local UI...
echo Loading WSL/micromamba environment. This can take 30-60 seconds.
echo.
call "%~dp0run_wsl.bat" python tools/dp_ui.py
pause
