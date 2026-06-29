@echo off
setlocal
set "REPO_DIR=%~dp0"
wsl.exe --cd "%REPO_DIR%" bash ./run_wsl.sh %*
