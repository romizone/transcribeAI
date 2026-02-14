@echo off
REM ============================================
REM TranscribeAI - Run Script (Windows)
REM Engine: faster-whisper (100% Local)
REM ============================================

if not exist "venv" (
    echo Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs

echo.
echo Starting TranscribeAI...
echo Open http://localhost:5000 in your browser.
echo.
python app.py
