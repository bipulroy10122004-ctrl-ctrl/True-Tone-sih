import os
import sys
import time
import soundfile as sf
import numpy as np

sys.path.insert(0, r"c:\Users\bipul\OneDrive\Documents\GitHub\True Tone SIH\True-Tone-sih\src\audio")
from features import extract_lfcc

flac_dir = r"D:\True Tone SIH\ASVspoof2021_DF_eval\flac"
files = [os.path.join(flac_dir, f) for f in os.listdir(flac_dir)[:20]]

t0 = time.time()
for f in files:
    feat = extract_lfcc(f)
t1 = time.time()

elapsed = t1 - t0
rate = len(files) / elapsed
print(f"Processed {len(files)} files in {elapsed:.2f}s ({rate:.1f} files/sec per thread)")
