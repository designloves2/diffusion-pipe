@echo off
cd /d "%~dp0"
call "%~dp0run_wsl.bat" python tools/aitk_to_diffusion_pipe.py
pause
