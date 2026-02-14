#!/bin/bash
# ============================================
# TranscribeAI - Run Script
# Engine: faster-whisper (100% Local)
# ============================================

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Create directories
mkdir -p uploads outputs

# Run the application
echo ""
echo "Starting TranscribeAI..."
echo "Open http://localhost:5000 in your browser."
echo ""
python3 app.py
