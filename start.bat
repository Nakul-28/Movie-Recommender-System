@echo off
setlocal

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

set "BACKEND_DIR=%ROOT%"
set "FRONTEND_DIR=%ROOT%\frontend"
set "VENV_PY=%ROOT%\..\venv\Scripts\python.exe"

if not exist "%VENV_PY%" set "VENV_PY=%ROOT%\venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo [ERROR] Python executable not found in venv.
    echo Expected one of:
    echo   %ROOT%\..\venv\Scripts\python.exe
    echo   %ROOT%\venv\Scripts\python.exe
    pause
    exit /b 1
)

if not exist "%FRONTEND_DIR%\package.json" (
    echo [ERROR] Frontend package.json not found at:
    echo   %FRONTEND_DIR%
    pause
    exit /b 1
)

echo Starting backend and frontend...
echo Backend URL:  http://127.0.0.1:8000
echo Frontend URL: http://127.0.0.1:3000
echo.

start "Backend - FastAPI" cmd /k "cd /d ""%BACKEND_DIR%"" && ""%VENV_PY%"" -m uvicorn main:app --reload --host 127.0.0.1 --port 8000"
start "Frontend - Vite" cmd /k "cd /d ""%FRONTEND_DIR%"" && npm.cmd run dev -- --host 127.0.0.1 --port 3000"

echo Both services launched in separate terminal windows.
echo You can close this launcher window.
