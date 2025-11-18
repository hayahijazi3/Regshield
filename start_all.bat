@echo off
setlocal enableextensions

REM ---- paths ----
set "BACKEND_DIR=/Users/haya/Desktop/Regshield/Backend"
set "PY=%BACKEND_DIR%\.venv\Scripts\python.exe"

REM ---- optional: force proxy to talk to this FastAPI URL ----
set "SEARCH_URL=http://127.0.0.1:8000"

REM ---- 1) Start FastAPI (port 8000) ----
start "FastAPI" cmd /c "cd /d %BACKEND_DIR% && "%PY%" server_app.py"

REM ---- wait for FastAPI /health ----
echo Waiting for FastAPI on http://127.0.0.1:8000/health ...
:wait_fastapi
curl -fs http://127.0.0.1:8000/health >nul 2>&1
if errorlevel 1 (
  timeout /t 1 >nul
  goto wait_fastapi
)

REM ---- 2) Start Flask proxy/auth (port 5001) ----
start "Flask" cmd /c "cd /d %BACKEND_DIR% && "%PY%" app.py"

REM ---- wait for Flask /health (which proxies FastAPI /health) ----
echo Waiting for Flask proxy on http://127.0.0.1:5001/health ...
:wait_flask
curl -fs http://127.0.0.1:5001/health >nul 2>&1
if errorlevel 1 (
  timeout /t 1 >nul
  goto wait_flask
)

REM ---- 3) Open the UI served by Flask (same-origin, no CORS) ----
start "" "http://127.0.0.1:5001/login.html"

echo All set. You can close this window.
endlocal
