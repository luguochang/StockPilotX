@echo off
setlocal

REM StockPilotX backend startup script (Windows)
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing .venv\Scripts\python.exe
  echo Please create venv and install dependencies first.
  pause
  exit /b 1
)

set "HOST=127.0.0.1"
set "PORT=8000"
set "LLM_EXTERNAL_ENABLED=true"
set "LLM_CONFIG_PATH=%ROOT_DIR%backend\config\llm_providers.local.json"
set "LLM_FALLBACK_TO_LOCAL=true"
set "LLM_RETRY_COUNT=1"

echo [INFO] Starting backend: http://%HOST%:%PORT%
start "StockPilotX Backend" cmd /k "cd /d ""%ROOT_DIR%"" && set LLM_EXTERNAL_ENABLED=%LLM_EXTERNAL_ENABLED% && set LLM_CONFIG_PATH=%LLM_CONFIG_PATH% && set LLM_FALLBACK_TO_LOCAL=%LLM_FALLBACK_TO_LOCAL% && set LLM_RETRY_COUNT=%LLM_RETRY_COUNT% && .venv\Scripts\python.exe -m uvicorn backend.app.http_api:create_app --factory --host %HOST% --port %PORT%"

echo [OK] Backend start command sent.
endlocal
