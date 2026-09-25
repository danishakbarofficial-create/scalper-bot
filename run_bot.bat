@echo off
title MEXC Scalper PRO 25% Bot
color 0A
echo ========================================================
echo        MEXC SCALPER PRO 25% - LAUNCHER
echo        Target: 20-25% Monthly Return (~1% Daily)
echo ========================================================
echo.

cd /d "%~dp0"

echo Checking Python environment...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10+ from python.org
    pause
    exit /b
)

echo Starting MEXC Scalping Bot & Web Dashboard...
echo Dashboard will automatically open at http://localhost:8000
echo Press Ctrl+C in this terminal window to stop the bot.
echo.

python main.py

pause
