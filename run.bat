@echo off
title PosChair - Full Local Stack (Web App + Heavy Pose Engine)
color 0A

echo ======================================================================
echo    ____   ____   _____  _____  _    _          _____  _____  
echo   ^|  _ \ / __ \ / ____^|/ ____^|^| ^|  ^| ^|   /\   ^|_   _^|^|  __ \ 
echo   ^| ^|_) ^| ^|  ^| ^| (___ ^| ^|     ^| ^|__^| ^|  /  \    ^| ^|  ^| ^|__) ^|
echo   ^|  _ ^<^| ^|  ^| ^|\___ \^| ^|     ^|  __  ^| / /\ \   ^| ^|  ^|  _  / 
echo   ^| ^|_) ^| ^|__^| ^|____) ^| ^|____ ^| ^|  ^| ^|/ ____ \ _^| ^|_ ^| ^| \ \ 
echo   ^|____/ \____/^|_____/ \_____^|^|_^|  ^|_/_/    \_\_____^|^|_^|  \_\
echo.
echo           v3.0 DUAL-LAYER // FULL LOCAL STACK LAUNCHER
echo ======================================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: 2. Check Node
where npm >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Node/NPM not found! Please install Node.js and add it to PATH.
    pause
    exit /b 1
)

echo [1/3] Launching Heavy Pose Engine (YOLOv8m-Pose on Port 8000)...
start "PosChair Heavy Pose Engine (Port 8000)" cmd /k "color 0E && python server/heavy_pose_server.py"

echo [2/3] Waiting 2 seconds for heavy engine initialization...
timeout /t 2 /nobreak >nul

echo [3/3] Launching Next.js Web App on Port 3000...
start "PosChair Web App (Port 3000)" cmd /k "color 0A && npm run dev"

echo.
echo Waiting for Next.js to start...
timeout /t 3 /nobreak >nul

echo Opening browser at http://localhost:3000...
start http://localhost:3000

echo.
echo ======================================================================
echo  [SUCCESS] Full PosChair stack is running locally!
echo   * Web App:           http://localhost:3000
echo   * Heavy Pose Engine: http://localhost:8000
echo.
echo  Keep the opened command windows running while using PosChair.
echo ======================================================================
echo.
pause
