import os

test_file = r"D:\True Tone SIH\ASVspoof2021_DF_eval\flac\DF_E_2000011.flac"

# Test 1: torchaudio
try:
    import torchaudio
    data, sr = torchaudio.load(test_file)
    print("torchaudio SUCCESS:", data.shape, sr)
except Exception as e:
    print("torchaudio FAILED:", type(e).__name__, e)

# Test 2: librosa
try:
    import librosa
    y, sr = librosa.load(test_file, sr=None)
    print("librosa SUCCESS:", y.shape, sr)
except Exception as e:
    print("librosa FAILED:", type(e).__name__, e)

# Test 3: ffmpeg / ffprobe via subprocess
try:
    import subprocess
    res = subprocess.run(["ffprobe", "-version"], capture_output=True, text=True)
    print("ffprobe available:", res.returncode == 0)
except Exception as e:
    print("ffprobe not available:", e)
