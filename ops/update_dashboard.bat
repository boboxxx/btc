@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT_DIR=%%~fI"
cd /d "%ROOT_DIR%"

where bash >nul 2>nul
if errorlevel 1 (
  echo bash not found in PATH.
  echo Install Git for Windows and run this script from Git Bash, or add bash to PATH.
  exit /b 1
)

bash "%ROOT_DIR%\ops\update_dashboard.sh"
exit /b %errorlevel%
