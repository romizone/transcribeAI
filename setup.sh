#!/bin/bash
# ============================================
# TranscribeAI - Setup Script
# Engine: faster-whisper (100% Local)
# ============================================

set -e

echo ""
echo "============================================"
echo "  TranscribeAI - Setup"
echo "  Engine: faster-whisper (Local)"
echo "============================================"
echo ""

# Check Python version
echo "[1/4] Checking Python..."
python3 --version 2>/dev/null || { echo "ERROR: Python 3 is required. Install from https://python.org"; exit 1; }

# Check ffmpeg
echo "[2/4] Checking ffmpeg..."
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "WARNING: ffmpeg not found!"
    echo ""
    echo "Install ffmpeg:"
    echo "  macOS:   brew install ffmpeg"
    echo "  Ubuntu:  sudo apt install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org/download.html"
    echo ""
    echo "ffmpeg is required for audio processing."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "  ffmpeg found: $(ffmpeg -version 2>&1 | head -1)"
fi

# Create virtual environment
echo "[3/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created."
else
    echo "  Virtual environment already exists."
fi

# Activate and install dependencies
echo "[4/4] Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "  Tanpa API key! 100% lokal & gratis."
echo ""
echo "  Jalankan:"
echo "    ./run.sh"
echo ""
echo "  Lalu buka http://localhost:5000"
echo ""
echo "  Model akan di-download otomatis saat"
echo "  pertama kali digunakan (~244MB untuk 'small')."
echo ""
echo "============================================"
echo ""
