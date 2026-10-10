"""
True Tone SIH - High-Speed Feature Caching Script
Pre-extracts and saves LFCC (400, 60) tensors onto Drive D:.
Uses ThreadPoolExecutor and single-thread BLAS to prevent Windows memory exhaustion.
"""
import os
import sys

# Critical for Windows: set thread limit before importing numpy/scipy
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_lfcc

DATA_ROOT = "D:/True Tone SIH/data/LA"
CACHE_ROOT = "D:/True Tone SIH/data/cached_features"

def process_item(item):
    """Worker task for thread execution."""
    flac_path, save_path, label = item
    if os.path.exists(save_path):
        return True
    try:
        feat = extract_lfcc(flac_path)
        torch.save({"feat": torch.from_numpy(feat), "label": label}, save_path)
        return True
    except Exception as e:
        return False

def cache_split(split_name, protocol_file, flac_dir, limit=None, num_workers=4):
    os.makedirs(f"{CACHE_ROOT}/{split_name}", exist_ok=True)
    proto_path = f"{DATA_ROOT}/ASVspoof2019_LA_cm_protocols/{protocol_file}"

    cols = ["speaker_id", "filename", "system_id", "null", "label"]
    df = pd.read_csv(proto_path, sep=" ", names=cols)

    if limit and limit > 0:
        df = df.head(limit)
        print(f"[{split_name}] Quick-mode enabled: caching first {limit} samples.")
    else:
        print(f"[{split_name}] Full caching: {len(df)} samples.")

    tasks = []
    for _, row in df.iterrows():
        flac_path = f"{DATA_ROOT}/{flac_dir}/{row['filename']}.flac"
        save_path = f"{CACHE_ROOT}/{split_name}/{row['filename']}.pt"
        label = 1 if row["label"] == "spoof" else 0  # 1 = Spoof (High Risk), 0 = Bonafide (Safe)
        tasks.append((flac_path, save_path, label))

    successful = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_item, item): item for item in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc=f"Caching {split_name}"):
            if future.result():
                successful += 1

    print(f"[{split_name}] Finished: {successful}/{len(tasks)} tensors cached at {CACHE_ROOT}/{split_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-extract LFCC features to Drive D:")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of audio files to cache (for quick testing)")
    parser.add_argument("--workers", type=int, default=4, help="Number of thread workers (default: 4)")
    args = parser.parse_args()

    print(f"=== True Tone Pre-Processing Feature Cache ===")
    print(f"Source: {DATA_ROOT}")
    print(f"Cache Target: {CACHE_ROOT}")
    print(f"Workers: {args.workers} threads")

    cache_split("train", "ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train/flac", limit=args.limit, num_workers=args.workers)
    cache_split("dev", "ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev/flac", limit=args.limit, num_workers=args.workers)
    print("=== Feature Caching Complete ===")
