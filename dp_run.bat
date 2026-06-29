@echo off
cd /d "%~dp0"
echo Starting diffusion-pipe config runner...
echo Loading WSL/micromamba environment. This can take 10-30 seconds.
echo.
call "%~dp0run_wsl.bat" python tools/dp_run.py
pause
