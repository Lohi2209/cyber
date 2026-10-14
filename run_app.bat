@echo off
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"
set "PYTHON_EXE=C:\ProgramData\anaconda3\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Could not find the configured Python interpreter:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)
start "Cyberbullying Detection Studio" "%PYTHON_EXE%" app.py
ping 127.0.0.1 -n 3 > nul
start "" http://127.0.0.1:5000/
