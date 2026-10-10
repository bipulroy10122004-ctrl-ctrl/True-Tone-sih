"""
True Tone SIH - Custom Audio Preprocessing & Feature Caching Utility
Converts custom WAV/FLAC audio files (WaveFake, ElevenLabs, LJSpeech, etc.)
into cached LFCC tensors ready for training with train.py.
"""

import os
import sys
import glob
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from features import extract_lfcc

SUPPORTED_EXTENSIONS = ("*.wav", "*.flac", "*.ogg", "*.WAV", "*.FLAC", "*.OGG")

def find_audio_files(directory: str):
    """Recursively search for all supported audio files."""
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return sorted(list(set(files)))

def process_and_cache(item):
    """Extracts LFCC and saves .pt tensor file."""
    audio_path, out_pt_path, label = item
    if os.path.exists(out_pt_path) and os.path.getsize(out_pt_path) > 1000:
        return True
    try:
        feat = extract_lfcc(audio_path)
        torch.save({"feat": torch.from_numpy(feat), "label": label}, out_pt_path)
        return True
    except Exception as e:
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess custom audio datasets (e.g. WaveFake + Real audio) into cached LFCC tensors"
    )
    parser.add_argument(
        "--spoof_dir",
        type=str,
        required=True,
        help="Directory containing AI-generated / spoof audio files (.wav, .flac)"
    )
    parser.add_argument(
        "--real_dir",
        type=str,
        required=True,
        help="Directory containing genuine / real human audio files (.wav, .flac)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "custom_cached_features"),
        help="Output directory to store train and dev cached tensors"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of dataset to use for validation (default: 0.2 = 20 percent)"
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=None,
        help="Optional max samples to process per class (for quick testing)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker threads (default: 4)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val splits"
    )

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 70)
    print("  TRUE TONE SIH: CUSTOM AUDIO DATASET PREPROCESSING")
    print("=" * 70)
    print(f"Spoof (AI) Directory:  {args.spoof_dir}")
    print(f"Real (Human) Directory: {args.real_dir}")
    print(f"Output Cache:           {args.output_dir}")
    print(f"Validation Ratio:       {args.val_ratio * 100:.0f}%")
    print("=" * 70)

    # 1. Discover audio files
    spoof_files = find_audio_files(args.spoof_dir)
    real_files = find_audio_files(args.real_dir)

    print(f"Found {len(spoof_files):,} spoof audio files.")
    print(f"Found {len(real_files):,} real human audio files.")

    if len(spoof_files) == 0:
        print(f"Error: No audio files (.wav, .flac) found in spoof directory: {args.spoof_dir}")
        sys.exit(1)
    if len(real_files) == 0:
        print(f"Error: No audio files (.wav, .flac) found in real directory: {args.real_dir}")
        sys.exit(1)

    # Optional cap
    if args.max_samples_per_class:
        random.shuffle(spoof_files)
        random.shuffle(real_files)
        spoof_files = spoof_files[:args.max_samples_per_class]
        real_files = real_files[:args.max_samples_per_class]
        print(f"Limited to {len(spoof_files)} spoof and {len(real_files)} real samples.")

    # 2. Split train / validation
    random.shuffle(spoof_files)
    random.shuffle(real_files)

    n_val_spoof = max(1, int(len(spoof_files) * args.val_ratio)) if len(spoof_files) >= 2 else 0
    n_val_real = max(1, int(len(real_files) * args.val_ratio)) if len(real_files) >= 2 else 0

    val_spoof = spoof_files[:n_val_spoof]
    train_spoof = spoof_files[n_val_spoof:] if len(spoof_files) > n_val_spoof else spoof_files

    val_real = real_files[:n_val_real]
    train_real = real_files[n_val_real:] if len(real_files) > n_val_real else real_files

    train_dir = os.path.join(args.output_dir, "train")
    dev_dir = os.path.join(args.output_dir, "dev")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)

    print(f"\nPlanned Dataset Splits:")
    print(f"  Training Set:   {len(train_spoof):,} Spoof  |  {len(train_real):,} Real  (Total: {len(train_spoof) + len(train_real):,})")
    print(f"  Validation Set: {len(val_spoof):,} Spoof  |  {len(val_real):,} Real  (Total: {len(val_spoof) + len(val_real):,})")

    # 3. Build Task List: (audio_path, output_pt_path, label: 1=spoof, 0=real)
    tasks = []

    for i, p in enumerate(train_spoof):
        out_pt = os.path.join(train_dir, f"spoof_{i:06d}.pt")
        tasks.append((p, out_pt, 1))

    for i, p in enumerate(train_real):
        out_pt = os.path.join(train_dir, f"real_{i:06d}.pt")
        tasks.append((p, out_pt, 0))

    for i, p in enumerate(val_spoof):
        out_pt = os.path.join(dev_dir, f"spoof_{i:06d}.pt")
        tasks.append((p, out_pt, 1))

    for i, p in enumerate(val_real):
        out_pt = os.path.join(dev_dir, f"real_{i:06d}.pt")
        tasks.append((p, out_pt, 0))

    print(f"\nExtracting LFCC features and caching {len(tasks):,} tensors...")
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_and_cache, t): t for t in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Processing"):
            if future.result():
                successful += 1
            else:
                failed += 1

    print("\n" + "=" * 70)
    print(f"Feature Caching Complete!")
    print(f"  Successfully cached: {successful:,} tensors")
    if failed > 0:
        print(f"  Failed to read:      {failed:,} files (check file corruption)")
    print(f"  Train cache location: {train_dir}")
    print(f"  Dev cache location:   {dev_dir}")
    print("=" * 70)
    print("\nNext Step: Run model training using the command below:\n")
    print(f'python src/audio/train.py --train_cache "{train_dir}" --dev_cache "{dev_dir}" --epochs 15 --batch_size 32')
    print("=" * 70)

if __name__ == "__main__":
    main()
