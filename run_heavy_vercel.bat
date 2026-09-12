@echo off
title PosChair - Heavy Model Host for Vercel Deployment
color 0B

echo ======================================================================
echo    ____   ____   _____  _____  _    _          _____  _____  
echo   ^|  _ \ / __ \ / ____^|/ ____^|^| ^|  ^| ^|   /\   ^|_   _^|^|  __ \ 
echo   ^| ^|_) ^| ^|  ^| ^| (___ ^| ^|     ^| ^|__^| ^|  /  \    ^| ^|  ^| ^|__) ^|
echo   ^|  _ ^<^| ^|  ^| ^|\___ \^| ^|     ^|  __  ^| / /\ \   ^| ^|  ^|  _  / 
echo   ^| ^|_) ^| ^|__^| ^|____) ^| ^|____ ^| ^|  ^| ^|/ ____ \ _^| ^|_ ^| ^| \ \ 
echo   ^|____/ \____/^|_____/ \_____^|^|_^|  ^|_/_/    \_\_____^|^|_^|  \_\
echo.
echo        HEAVY MODEL HOST (CLOUDFLARE TUNNEL -^> VERCEL)
echo ======================================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found! Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: 2. Check Cloudflared
where cloudflared >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] cloudflared CLI not found in PATH!
    echo Please install it using: scoop install cloudflared
    echo Or download from: https://github.com/cloudflare/cloudflared/releases
    pause
    exit /b 1
)

echo [1/2] Starting Heavy Pose Engine (YOLOv8m-Pose on Port 8000)...
start "PosChair Heavy Pose Engine (Port 8000)" cmd /k "color 0E && python server/heavy_pose_server.py"

echo [2/2] Waiting 2 seconds for model initialization...
timeout /t 2 /nobreak >nul

echo.
echo ======================================================================
echo  HOW TO CONNECT THIS TO YOUR VERCEL DEPLOYMENT:
echo ======================================================================
echo  1. Cloudflare will print a public HTTPS URL below like:
echo     https://[random-words].trycloudflare.com
echo.
echo  2. Copy that URL.
echo.
echo  3. In your Vercel Project Dashboard -^> Settings -^> Environment Variables:
echo     Add:
echo     Key:   HEAVY_MODEL_URL
echo     Value: https://[random-words].trycloudflare.com
echo.
echo  4. Redeploy or trigger a new deployment on Vercel.
echo     Your deployed app will now use your laptop GPU for deep verification!
echo ======================================================================
echo.
echo Starting Cloudflare Tunnel now...
echo.

cloudflared tunnel --url http://localhost:8000
