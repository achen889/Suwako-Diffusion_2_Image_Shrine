@echo off
setlocal
cd /d "%~dp0"
set "SUWAKO_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%SUWAKO_PYTHON%" (
    echo Suwako's project virtual environment is missing: "%SUWAKO_PYTHON%"
    echo Create it with: py -3.12 -m venv .venv
    echo Then install CUDA-enabled PyTorch and requirements-desktop.txt into .venv.
    pause
    exit /b 1
)

"%SUWAKO_PYTHON%" -m suwako.cli --check-cuda
if errorlevel 1 (
    echo CUDA preflight failed. Repair the .venv shown above before launching Suwako.
    pause
    exit /b 1
)

"%SUWAKO_PYTHON%" -m streamlit run app.py --server.address 127.0.0.1
set "SUWAKO_EXIT=%ERRORLEVEL%"
pause
exit /b %SUWAKO_EXIT%
pause
