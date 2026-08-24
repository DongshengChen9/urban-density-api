@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_CMD=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_CMD=.venv\Scripts\python.exe"

%PYTHON_CMD% --version >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.11 or 3.12, then follow README.md.
    pause
    exit /b 1
)

%PYTHON_CMD% -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Streamlit is not installed. Run: python -m pip install -e .
    pause
    exit /b 1
)

%PYTHON_CMD% -m streamlit run 03_code/app.py

