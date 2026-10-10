import os
import soundfile as sf

flac_dir = r"D:\True Tone SIH\ASVspoof2021_DF_eval\flac"
files = [os.path.join(flac_dir, f) for f in os.listdir(flac_dir)[:30]]

for f in files:
    size = os.path.getsize(f)
    print(f"File: {os.path.basename(f)}, Size: {size} bytes", end=" -> ")
    try:
        data, sr = sf.read(f)
        print(f"OK (sr={sr}, len={len(data)})")
    except Exception as e:
        print(f"FAIL: {type(e).__name__}: {e}")
