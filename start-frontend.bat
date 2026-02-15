@echo off
setlocal

REM StockPilotX frontend startup script (Windows)
set "ROOT_DIR=%~dp0"
set "FRONTEND_DIR=%ROOT_DIR%frontend"

cd /d "%FRONTEND_DIR%"

if not exist "package.json" (
  echo [ERROR] Missing frontend\package.json
  echo Make sure this script is in the StockPilotX root directory.
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo [WARN] node_modules not found. Run npm install if needed.
)

echo [INFO] Starting frontend: http://127.0.0.1:3000
start "StockPilotX Frontend" cmd /k "cd /d ""%FRONTEND_DIR%"" && npm run dev"

echo [OK] Frontend start command sent.
endlocal
