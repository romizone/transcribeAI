"""
TranscribeAI - Model Downloader
================================
Download model sebelum digunakan agar tidak menunggu saat transkripsi.
Jalankan sekali saja, setelah itu model tersimpan lokal secara permanen.

Usage:
  python3 download_models.py              # Download model 'small' (default)
  python3 download_models.py tiny         # Download model 'tiny' saja
  python3 download_models.py small medium # Download beberapa model
  python3 download_models.py all          # Download semua model
  python3 download_models.py --mlx small  # Download MLX version (Apple Silicon)
"""

import sys
import time
import os
import platform
from pathlib import Path


MODELS = {
    'tiny':     {'size': '~75 MB',   'params': '39M',   'speed': 'Tercepat (1-2 menit/10min audio)'},
    'base':     {'size': '~145 MB',  'params': '74M',   'speed': 'Sangat Cepat (2-4 menit/10min audio)'},
    'small':    {'size': '~465 MB',  'params': '244M',  'speed': 'Cepat (4-8 menit/10min audio)'},
    'medium':   {'size': '~1.5 GB',  'params': '769M',  'speed': 'Sedang (10-20 menit/10min audio)'},
    'large-v3': {'size': '~2.9 GB',  'params': '1550M', 'speed': 'Lambat tapi paling akurat'},
}

MLX_MODEL_REPOS = {
    'tiny':     'mlx-community/whisper-tiny-mlx',
    'base':     'mlx-community/whisper-base-mlx',
    'small':    'mlx-community/whisper-small-mlx',
    'medium':   'mlx-community/whisper-medium-mlx',
    'large-v3': 'mlx-community/whisper-large-v3-mlx',
}


def is_cached(model_size, engine='faster-whisper'):
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'hub'
    if engine == 'mlx':
        repo = MLX_MODEL_REPOS.get(model_size, f"mlx-community/whisper-{model_size}-mlx")
        repo_name = f"models--{repo.replace('/', '--')}"
    else:
        repo_name = f"models--Systran--faster-whisper-{model_size}"
    model_dir = cache_dir / repo_name / 'snapshots'
    return model_dir.exists() and any(model_dir.iterdir()) if model_dir.exists() else False


def download_model_fw(model_size):
    """Download faster-whisper model"""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  [SKIP] faster-whisper not installed. Run: pip install faster-whisper")
        return False

    info = MODELS.get(model_size, {})

    if is_cached(model_size, 'faster-whisper'):
        print(f"  [OK] faster-whisper '{model_size}' sudah ada di cache.")
        return True

    print(f"\n  Downloading faster-whisper '{model_size}' ({info.get('size', '?')})...")
    print(f"  Mohon tunggu...")

    t0 = time.time()
    try:
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=os.cpu_count() or 4,
        )
        elapsed = time.time() - t0
        print(f"  [OK] faster-whisper '{model_size}' berhasil! ({elapsed:.0f}s)")
        del model
        return True
    except Exception as e:
        print(f"  [ERROR] Gagal: {e}")
        return False


def download_model_mlx(model_size):
    """Download mlx-whisper model"""
    try:
        import mlx_whisper
    except ImportError:
        print("  [SKIP] mlx-whisper not installed. Run: pip install mlx-whisper")
        return False

    if is_cached(model_size, 'mlx'):
        print(f"  [OK] mlx-whisper '{model_size}' sudah ada di cache.")
        return True

    repo = MLX_MODEL_REPOS.get(model_size, f"mlx-community/whisper-{model_size}-mlx")
    print(f"\n  Downloading mlx-whisper '{model_size}' from {repo}...")
    print(f"  Mohon tunggu...")

    t0 = time.time()
    try:
        # Create a tiny silent audio to trigger model download
        import numpy as np
        tmp_wav = Path('/tmp/transcribeai_test.wav')
        import wave
        with wave.open(str(tmp_wav), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            silence = np.zeros(16000, dtype=np.int16)  # 1 second silence
            wf.writeframes(silence.tobytes())

        # This triggers the model download
        mlx_whisper.transcribe(
            str(tmp_wav),
            path_or_hf_repo=repo,
        )
        tmp_wav.unlink(missing_ok=True)

        elapsed = time.time() - t0
        print(f"  [OK] mlx-whisper '{model_size}' berhasil! ({elapsed:.0f}s)")
        return True
    except Exception as e:
        print(f"  [ERROR] Gagal: {e}")
        return False


def show_status():
    is_apple = platform.system() == 'Darwin' and platform.machine() == 'arm64'

    print()
    print("  STATUS MODEL:")
    print("  " + "-" * 65)
    for name, info in MODELS.items():
        fw_cached = is_cached(name, 'faster-whisper')
        fw_status = "\033[92mREADY\033[0m" if fw_cached else "\033[93mBELUM\033[0m"
        line = f"  [{fw_status}] FW  {name:10s}  {info['size']:>10s}  {info['params']:>6s} param"

        if is_apple:
            mlx_cached = is_cached(name, 'mlx')
            mlx_status = "\033[92mREADY\033[0m" if mlx_cached else "\033[93mBELUM\033[0m"
            line += f"  |  [{mlx_status}] MLX"

        print(line)
    print("  " + "-" * 65)
    print()


def main():
    is_apple = platform.system() == 'Darwin' and platform.machine() == 'arm64'

    print()
    print("=" * 58)
    print("  TranscribeAI - Model Downloader")
    print("  Download model offline untuk transkripsi cepat")
    print("=" * 58)

    show_status()

    # Parse args
    args = sys.argv[1:]
    use_mlx = '--mlx' in args
    args = [a for a in args if a != '--mlx']

    # Auto-detect: on Apple Silicon, download MLX by default
    if is_apple and not use_mlx:
        use_mlx = True
        print("  Apple Silicon terdeteksi! Download MLX model (lebih cepat).")

    if not args:
        to_download = ['small']
        print("  Default: download model 'small' (direkomendasikan)")
        print("  Tip: 'python3 download_models.py all' untuk semua model")
    elif 'all' in args:
        to_download = list(MODELS.keys())
        print("  Downloading SEMUA model...")
    else:
        to_download = [m for m in args if m in MODELS]
        invalid = [m for m in args if m not in MODELS]
        if invalid:
            print(f"  Model tidak dikenal: {', '.join(invalid)}")
            print(f"  Tersedia: {', '.join(MODELS.keys())}")
        if not to_download:
            print("  Tidak ada model valid.")
            return

    print()

    success = 0
    for model_size in to_download:
        if use_mlx:
            if download_model_mlx(model_size):
                success += 1
        else:
            if download_model_fw(model_size):
                success += 1
        print()

    print("=" * 58)
    print(f"  Selesai! {success}/{len(to_download)} model berhasil.")
    print("=" * 58)
    show_status()

    print("  Sekarang jalankan: ./venv/bin/python3 app.py")
    print("  Buka: http://localhost:8080")
    print()


if __name__ == '__main__':
    main()
