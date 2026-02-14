@echo off
REM ============================================
REM TranscribeAI - Setup Script (Windows)
REM Engine: faster-whisper (100% Local)
REM ============================================

echo.
echo ============================================
echo   TranscribeAI - Setup
echo   Engine: faster-whisper (Local)
echo ============================================
echo.

REM Check Python
echo [1/4] Checking Python...
python --version 2>nul || (
    echo ERROR: Python 3 is required. Install from https://python.org
    pause
    exit /b 1
)

REM Check ffmpeg
echo [2/4] Checking ffmpeg...
ffmpeg -version >nul 2>&1 || (
    echo.
    echo WARNING: ffmpeg not found!
    echo.
    echo Install ffmpeg:
    echo   1. Download from https://ffmpeg.org/download.html
    echo   2. Or use: winget install ffmpeg
    echo   3. Or use: choco install ffmpeg
    echo.
    echo ffmpeg is required for audio processing.
    echo.
    pause
)

REM Create virtual environment
echo [3/4] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo   Virtual environment created.
) else (
    echo   Virtual environment already exists.
)

REM Install dependencies
echo [4/4] Installing dependencies...
call venv\Scripts\activate.bat
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo.
echo ============================================
echo   Setup Complete!
echo ============================================
echo.
echo   Tanpa API key! 100%% lokal dan gratis.
echo.
echo   Jalankan:
echo     run.bat
echo.
echo   Lalu buka http://localhost:5000
echo.
echo   Model akan di-download otomatis saat
echo   pertama kali digunakan (~244MB untuk 'small').
echo.
echo ============================================
echo.
pause
