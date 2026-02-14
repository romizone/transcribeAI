<p align="center">
  <img src="https://img.icons8.com/fluency/96/microphone.png" width="80" />
</p>

<h1 align="center">🎙️ TranscribeAI</h1>

<p align="center">
  <strong>100% Local AI Transcription with Speaker Diarization</strong><br>
  <em>No API key. No cloud. No cost. Runs completely offline on your machine.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/🐍_Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/🍎_Apple_Silicon-Optimized-000000?style=for-the-badge&logo=apple&logoColor=white" />
  <img src="https://img.shields.io/badge/🔒_Privacy-100%25_Offline-2ea44f?style=for-the-badge" />
  <img src="https://img.shields.io/badge/📄_License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/macOS-supported-999?style=flat-square&logo=apple" />
  <img src="https://img.shields.io/badge/Windows-supported-999?style=flat-square&logo=windows" />
  <img src="https://img.shields.io/badge/Linux-supported-999?style=flat-square&logo=linux" />
</p>

---

## ✨ Features

| | Feature | Description |
|---|---------|-------------|
| ⚡ | **Dual Engine** | faster-whisper (CPU) + mlx-whisper (Apple Silicon GPU, 2-5x faster) |
| 🗣️ | **Speaker Diarization** | Auto-identifies Speaker 1, 2, 3... using MFCC + Agglomerative Clustering |
| 🌍 | **99+ Languages** | Indonesian, English, and 99+ languages with auto-detection |
| 📁 | **Multi-Format** | Input: MP3, MP4, WAV, M4A, OGG, FLAC, WEBM → Output: SRT, TXT, DOCX |
| 🧠 | **5 AI Models** | tiny (39M) → large-v3 (1.5B) — choose speed vs accuracy |
| 📊 | **Smart Progress** | 5-stage: Upload → Model → Transcription → Speaker ID → Export |
| 💾 | **Auto Cache** | Downloads model once, loads instantly from cache afterward |
| 🌙 | **Dark Theme UI** | Professional web UI with audio player, search & drag-drop |
| 🔒 | **100% Offline** | Zero data leaves your machine. Your audio stays yours. |

---

## 🚀 Quick Start

### 🍎 macOS — Apple Silicon (M1/M2/M3/M4)

```bash
git clone https://github.com/romizone/transcribeAI.git
cd transcribeAI

# ⚙️ Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install mlx-whisper          # 🔥 GPU acceleration

# ▶️ Run
python3 app.py
```

> 🌐 Open **http://localhost:8080** in your browser

### 🍎 macOS (Intel) / 🐧 Linux

```bash
git clone https://github.com/romizone/transcribeAI.git
cd transcribeAI

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 app.py
```

### 🪟 Windows

```cmd
git clone https://github.com/romizone/transcribeAI.git
cd transcribeAI

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python app.py
```

> 💡 Or use setup scripts: `./setup.sh` (macOS/Linux) or `setup.bat` (Windows)

---

## 📥 Pre-download Models

Download models ahead of time so transcription starts instantly:

```bash
source venv/bin/activate

python3 download_models.py small    # 📦 Download recommended model
python3 download_models.py all      # 📦 Download all models
python3 download_models.py          # 📋 Check download status
```

> 🍎 On Apple Silicon, MLX models are auto-downloaded for GPU acceleration

---

## 🧠 Models

| Model | Params | Size | Speed | Best For |
|:------|:------:|:----:|:-----:|:---------|
| `tiny` | 39M | ~75 MB | ⚡⚡⚡⚡⚡ | Quick drafts, short clips |
| `base` | 74M | ~145 MB | ⚡⚡⚡⚡ | Casual transcription |
| **`small`** ⭐ | **244M** | **~465 MB** | **⚡⚡⚡** | **Recommended — best balance** |
| `medium` | 769M | ~1.5 GB | ⚡⚡ | Higher accuracy needed |
| `large-v3` | 1550M | ~2.9 GB | ⚡ | Maximum accuracy |

---

## 🔧 Engines

| Engine | Device | Speed | Install |
|:-------|:-------|:-----:|:--------|
| 🔥 **mlx-whisper** | Apple Silicon GPU | **2-5x faster** | `pip install mlx-whisper` |
| 🖥️ faster-whisper | CPU (all platforms) | Baseline | Included in `requirements.txt` |

> 🤖 The app auto-detects Apple Silicon and defaults to mlx-whisper when available

---

## 💻 CLI Usage

Transcribe directly from terminal — no browser needed:

```bash
# 🎵 Simple transcription
python3 transcribe_cli.py audio.mp3

# 🇮🇩 Indonesian, medium model, 3 speakers
python3 transcribe_cli.py video.mp4 --language id --model medium --speakers 3

# 📂 Custom output folder + multiple formats
python3 transcribe_cli.py audio.wav --output ./results --format srt txt docx
```

---

## 📂 Project Structure

```
transcribeAI/
├── 🐍 app.py               # Flask backend (dual engine, diarization, API)
├── 📄 templates/
│   └── index.html           # Web UI (dark theme, progress, audio player)
├── 🖥️ transcribe_cli.py     # CLI version
├── 📥 download_models.py    # Pre-download models offline
├── 📋 requirements.txt      # Python dependencies
├── ⚙️ setup.sh / setup.bat  # Setup scripts
├── ▶️ run.sh / run.bat      # Run scripts
└── 🔧 .env.example          # Configuration template
```

---

## ⚙️ How It Works

```
🎤 Audio Input
    │
    ▼
🧠 Whisper Transcription
    │  faster-whisper (CTranslate2 INT8)
    │  mlx-whisper (Apple MLX GPU)
    │
    ▼
🔇 VAD Filter
    │  Silero VAD removes silence
    │
    ▼
🗣️ Speaker Diarization
    │  MFCC (20 coeff) + Delta + Spectral + Pitch
    │  → StandardScaler → Agglomerative Clustering
    │
    ▼
📄 Export
    ├── 🎬 SRT (subtitles)
    ├── 📝 TXT (readable transcript)
    └── 📑 DOCX (formatted document)
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| 🚫 Port 5000 in use (macOS) | AirPlay uses port 5000. TranscribeAI uses **port 8080** by default |
| ❌ `ModuleNotFoundError` | Activate venv first: `source venv/bin/activate` |
| ⚠️ `python3` aliased wrong | Use venv directly: `./venv/bin/python3 app.py` |
| ⏳ Stuck at "Memuat model..." | First run downloads ~465MB model (one-time). Pre-download: `python3 download_models.py small` |

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| 🖥️ Backend | Flask, faster-whisper, mlx-whisper |
| 🎵 Audio | librosa, numpy, pydub |
| 🗣️ Speaker ID | scikit-learn (Agglomerative Clustering) |
| 📄 Export | python-docx |
| 🎨 Frontend | Vanilla HTML/CSS/JS (zero framework dependencies) |

---

## 📜 License

MIT License — free for personal and commercial use.

---

<p align="center">
  <strong>🇮🇩 Made in Indonesia</strong><br>
  Built with ❤️ and AI by <a href="https://github.com/romizone">@romizone</a>
</p>

<p align="center">
  <a href="https://github.com/romizone/transcribeAI/stargazers">⭐ Star this repo</a> if you find it useful!
</p>
