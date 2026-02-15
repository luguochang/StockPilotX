@echo off
setlocal

REM StockPilotX one-click startup script
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

call "%ROOT_DIR%start-backend.bat"
timeout /t 2 /nobreak >nul
call "%ROOT_DIR%start-frontend.bat"

echo [OK] Backend and frontend start commands sent.
echo Backend: http://127.0.0.1:8000/docs
echo Frontend: http://127.0.0.1:3000
endlocal
