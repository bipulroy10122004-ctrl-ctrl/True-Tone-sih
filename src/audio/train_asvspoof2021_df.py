"""
True Tone SIH - ASVspoof 2021 DF Model Training Pipeline
Trains the AudioSpoofDetector (1D-CNN + BiLSTM + Temporal Multi-Head Attention)
using the official ASVspoof2021_DF_eval dataset files from D:\\True Tone SIH.

Key Highlights:
- Dataset: ASVspoof 2021 Deepfake Evaluation (flac audio + trial_metadata.txt)
- Stratified sampling across diverse neural vocoders, voice conversion models (VCC2020, VCC2018),
  and transmission codecs (low_m4a, mp3m4a, ogg, nocodec).
- Parallel feature extraction: 400x60 LFCC + Delta + Delta-Delta tensors.
- Disk caching for instant loading and reproducible experiments.
- Hardware acceleration: CUDA on NVIDIA GeForce RTX 2050 with Automatic Mixed Precision (AMP).
- Evaluation: ROC-AUC, Equal Error Rate (EER), F1-Score, and optimal decision threshold calibration.
"""

import os
import sys
import time
import glob
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)

from model import AudioSpoofDetector
from features import extract_lfcc

DEFAULT_DF_FLAC_DIR = r"D:\True Tone SIH\ASVspoof2021_DF_eval\flac"
DEFAULT_DF_META_PATH = r"D:\True Tone SIH\keys_2021_DF\keys\DF\CM\trial_metadata.txt"
DEFAULT_LA_TRAIN_DIR = r"D:\True Tone SIH\data\LA\ASVspoof2019_LA_train\flac"
DEFAULT_CACHE_DIR = r"D:\True Tone SIH\data\cached_asvspoof2021_df"
DEFAULT_OUTPUT_MODEL = os.path.join(PROJECT_ROOT, "models", "best_audio_spoof_model.pt")

def compute_eer(labels, scores):
    """Computes Equal Error Rate (EER) and the optimal decision threshold."""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    opt_threshold = float(thresholds[idx]) if idx < len(thresholds) else 0.50
    return float(eer), opt_threshold

class CachedLFCCDataset(Dataset):
    """Loads cached PyTorch LFCC tensor files from disk."""
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=True)
        feat = data["feat"]  # (400, 60)
        label = torch.tensor(data["label"], dtype=torch.float32)
        return feat, label

def extract_and_cache_single(args_tuple):
    """Helper for parallel extraction of a single audio file."""
    fpath, cache_path, label, trial_id, codec, source = args_tuple
    if os.path.exists(cache_path):
        return cache_path, True

    try:
        # Validate that the file is readable
        data, sr = sf.read(fpath)
        if len(data) == 0:
            return None, False

        feat = extract_lfcc(fpath)
        feat_tensor = torch.from_numpy(feat).float()

        payload = {
            "feat": feat_tensor,
            "label": float(label),
            "id": trial_id,
            "codec": codec,
            "source": source
        }
        torch.save(payload, cache_path)
        return cache_path, True
    except Exception:
        return None, False

def prepare_asvspoof2021_df_dataset(
    flac_dir=DEFAULT_DF_FLAC_DIR,
    meta_path=DEFAULT_DF_META_PATH,
    la_train_dir=DEFAULT_LA_TRAIN_DIR,
    cache_dir=DEFAULT_CACHE_DIR,
    max_per_class=5500,
    include_la_controls=True,
    num_threads=8
):
    """
    Scans ASVspoof2021_DF_eval metadata, selects a stratified, balanced set of
    bonafide and spoof clips across all codecs and neural vocoders, and caches features.
    """
    os.makedirs(cache_dir, exist_ok=True)
    existing_cached = glob.glob(os.path.join(cache_dir, "*.pt"))
    if len(existing_cached) >= (max_per_class * 1.8):
        print(f"[*] Found {len(existing_cached):,} pre-cached LFCC samples in {cache_dir}.")
        print("[*] Reusing existing feature cache for instantaneous training!")
        return existing_cached

    print(f"[*] Parsing ASVspoof 2021 DF metadata: {meta_path}")
    trial_metadata = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                trial_id = parts[1]
                codec = parts[2]
                source = parts[3]
                attack = parts[4]
                label = parts[5]  # 'bonafide' or 'spoof'
                trial_metadata[trial_id] = {
                    "codec": codec,
                    "source": source,
                    "attack": attack,
                    "label": 1.0 if label == "spoof" else 0.0,
                    "label_str": label
                }

    print(f"[*] Loaded metadata for {len(trial_metadata):,} trials.")

    # 1. Collect candidate bonafide files from ASVspoof2021_DF_eval
    print(f"[*] Scanning ASVspoof 2021 DF flac folder: {flac_dir}")
    existing_flac_set = set(os.listdir(flac_dir))
    df_bonafide_candidates = []
    df_spoof_by_category = {}

    for tid, meta in trial_metadata.items():
        fname = f"{tid}.flac"
        if fname in existing_flac_set:
            fpath = os.path.join(flac_dir, fname)
            if meta["label_str"] == "bonafide":
                df_bonafide_candidates.append((fpath, tid, meta["codec"], meta["source"], 0.0))
            else:
                cat_key = f"{meta['source']}_{meta['codec']}"
                if cat_key not in df_spoof_by_category:
                    df_spoof_by_category[cat_key] = []
                df_spoof_by_category[cat_key].append((fpath, tid, meta["codec"], meta["source"], 1.0))

    print(f"[*] ASVspoof2021_DF_eval matches on disk: {len(df_bonafide_candidates):,} bonafide candidates, across {len(df_spoof_by_category)} spoof categories.")

    # 2. Stratified selection of bonafide files
    selected_bonafide = df_bonafide_candidates[:]
    if include_la_controls and os.path.exists(la_train_dir) and len(selected_bonafide) < max_per_class:
        needed = max_per_class - len(selected_bonafide)
        print(f"[*] Augmenting with {needed:,} clean bonafide human control clips from ASVspoof 2019 LA: {la_train_dir}")
        la_flacs = [f for f in os.listdir(la_train_dir) if f.endswith(".flac")]
        random.seed(42)
        random.shuffle(la_flacs)
        for la_f in la_flacs[:needed]:
            la_path = os.path.join(la_train_dir, la_f)
            tid = la_f[:-5]
            selected_bonafide.append((la_path, tid, "clean_flac", "asvspoof2019_la", 0.0))

    if len(selected_bonafide) > max_per_class:
        random.seed(42)
        selected_bonafide = random.sample(selected_bonafide, max_per_class)

    # 3. Stratified selection of spoof files across all categories (codecs & vocoders)
    target_spoof_count = len(selected_bonafide)
    selected_spoof = []
    categories = list(df_spoof_by_category.keys())
    per_cat = max(1, target_spoof_count // len(categories))

    random.seed(42)
    for cat in categories:
        samples = df_spoof_by_category[cat]
        take = min(len(samples), per_cat)
        selected_spoof.extend(random.sample(samples, take))

    # Fill remainder if needed
    if len(selected_spoof) < target_spoof_count:
        remaining_pool = []
        selected_set = {s[1] for s in selected_spoof}
        for cat in categories:
            for s in df_spoof_by_category[cat]:
                if s[1] not in selected_set:
                    remaining_pool.append(s)
        deficit = target_spoof_count - len(selected_spoof)
        if len(remaining_pool) > deficit:
            selected_spoof.extend(random.sample(remaining_pool, deficit))
        else:
            selected_spoof.extend(remaining_pool)

    print(f"[*] Final Target Selection: {len(selected_bonafide):,} Bonafide + {len(selected_spoof):,} Spoof = {len(selected_bonafide) + len(selected_spoof):,} Audio Clips.")

    # 4. Multi-Worker Parallel Feature Extraction & Caching
    tasks = []
    for fpath, tid, codec, source, label in (selected_bonafide + selected_spoof):
        cache_file = os.path.join(cache_dir, f"{tid}.pt")
        tasks.append((fpath, cache_file, label, tid, codec, source))

    print(f"[*] Extracting LFCC features (400 frames x 60 channels) with {num_threads} workers...")
    cached_paths = []
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(extract_and_cache_single, t): t for t in tasks}
        pbar = tqdm(as_completed(futures), total=len(tasks), desc="Caching LFCC Features")
        for fut in pbar:
            res_path, ok = fut.result()
            if ok and res_path is not None:
                cached_paths.append(res_path)
                success_count += 1
            else:
                fail_count += 1
            pbar.set_postfix({"OK": success_count, "Skipped": fail_count})

    print(f"\n[*] Feature Preprocessing Complete! {success_count:,} valid cached samples ({fail_count} skipped corrupted files).")
    return cached_paths

def train_asvspoof2021_df(args):
    """Main training & evaluation routine for ASVspoof 2021 DF."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 70)
    print("  TRUE TONE SIH: ASVSPOOF 2021 DEEPFAKE MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"Device:             {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Dataset Location:   {args.df_flac_dir}")
    print(f"Metadata File:      {args.df_meta_path}")
    print(f"Cache Location:     {args.cache_dir}")
    print(f"Output Checkpoint:  {args.output_model}")
    print(f"Target Samples/Cls: {args.max_per_class:,}")
    print(f"Batch Size:         {args.batch_size}")
    print(f"Learning Rate:      {args.lr}")
    print(f"Epochs:             {args.epochs}")
    print("=" * 70 + "\n")

    # 1. Prepare / Load Dataset
    all_cached_files = prepare_asvspoof2021_df_dataset(
        flac_dir=args.df_flac_dir,
        meta_path=args.df_meta_path,
        la_train_dir=args.la_train_dir,
        cache_dir=args.cache_dir,
        max_per_class=args.max_per_class,
        include_la_controls=not args.no_la_controls,
        num_threads=args.num_workers
    )

    if len(all_cached_files) == 0:
        raise RuntimeError("No valid audio samples found to train on!")

    # 2. Train / Validation Split (85% Train, 15% Validation)
    random.seed(42)
    random.shuffle(all_cached_files)
    split_idx = int(0.85 * len(all_cached_files))
    train_files = all_cached_files[:split_idx]
    val_files = all_cached_files[split_idx:]

    print(f"\n[*] Dataset Split: {len(train_files):,} Train samples, {len(val_files):,} Validation samples.")

    train_ds = CachedLFCCDataset(train_files)
    val_ds = CachedLFCCDataset(val_files)

    use_pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=use_pin, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, pin_memory=use_pin, num_workers=0)

    # 3. Instantiate Neural Architecture & Optimizer
    print(f"\n[*] Initializing AudioSpoofDetector (1D-CNN + BiLSTM + Attention)...")
    model = AudioSpoofDetector(input_dim=60, hidden_dim=128).to(device)

    # Loss function with balanced loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Backup existing checkpoint if present
    if os.path.exists(args.output_model):
        backup_path = os.path.join(os.path.dirname(args.output_model), "backup_previous_model.pt")
        try:
            import shutil
            shutil.copy2(args.output_model, backup_path)
            print(f"[*] Backed up existing model to: {backup_path}")
        except Exception as e:
            print(f"[!] Note: Could not backup previous checkpoint: {e}")

    # 4. Training Loop
    best_eer = 1.0
    best_f1 = 0.0
    best_acc = 0.0
    best_epoch = 0
    best_threshold = 0.50

    print("\n" + "-" * 75)
    print(f"{'Epoch':<7} | {'Train Loss':<12} | {'Val Loss':<10} | {'Accuracy':<10} | {'F1-Score':<10} | {'EER (%)':<9} | {'Thresh'}")
    print("-" * 75)

    total_training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── Training Phase ──
        model.train()
        train_loss_total = 0.0
        train_steps = 0

        for feats, labels in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(feats)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_total += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train_loss = train_loss_total / max(1, train_steps)

        # ── Validation Phase ──
        model.eval()
        val_loss_total = 0.0
        val_steps = 0
        all_labels = []
        all_preds = []
        all_scores = []

        with torch.no_grad():
            for feats, labels in val_loader:
                feats = feats.to(device)
                labels = labels.to(device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(feats)
                    loss = criterion(logits, labels)
                    scores = torch.sigmoid(logits)

                val_loss_total += loss.item()
                val_steps += 1

                all_labels.extend(labels.cpu().numpy().tolist())
                all_scores.extend(scores.cpu().numpy().tolist())

        avg_val_loss = val_loss_total / max(1, val_steps)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)

        # Metrics
        eer_val, opt_thresh = compute_eer(all_labels, all_scores)
        preds = (all_scores >= 0.50).astype(int)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds, zero_division=0)

        eer_pct = eer_val * 100.0

        print(f"{epoch:<7} | {avg_train_loss:<12.4f} | {avg_val_loss:<10.4f} | {acc * 100:<9.2f}% | {f1 * 100:<9.2f}% | {eer_pct:<8.3f}% | {opt_thresh:.3f}")

        # Checkpoint if better EER or better F1
        if eer_val < best_eer or (eer_val == best_eer and f1 > best_f1):
            best_eer = eer_val
            best_f1 = f1
            best_acc = acc
            best_epoch = epoch
            best_threshold = opt_thresh

            os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eer": best_eer,
                "best_f1": best_f1,
                "best_acc": best_acc,
                "optimal_threshold": best_threshold,
                "input_dim": 60,
                "hidden_dim": 128,
                "dataset": "ASVspoof2021_DF_eval",
                "training_samples": len(train_files),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            torch.save(checkpoint, args.output_model)

    total_training_time = time.time() - total_training_start

    print("-" * 75)
    print(f"\n[SUCCESS] Training Successfully Finished in {total_training_time:.1f}s ({total_training_time/60:.2f} mins)!")
    print(f"[*] Best Checkpoint achieved at Epoch {best_epoch}:")
    print(f"    - Equal Error Rate (EER): {best_eer * 100:.3f}%")
    print(f"    - Accuracy:               {best_acc * 100:.2f}%")
    print(f"    - F1-Score:               {best_f1 * 100:.2f}%")
    print(f"    - Calibrated Threshold:   theta = {best_threshold:.4f}")
    print(f"[*] Checkpoint saved to:       {args.output_model}")
    print("=" * 70 + "\n")

    return best_eer, best_f1, best_threshold

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train True Tone Model on ASVspoof 2021 DF Dataset")
    parser.add_argument("--df_flac_dir", type=str, default=DEFAULT_DF_FLAC_DIR, help="Path to ASVspoof2021_DF_eval/flac directory")
    parser.add_argument("--df_meta_path", type=str, default=DEFAULT_DF_META_PATH, help="Path to trial_metadata.txt")
    parser.add_argument("--la_train_dir", type=str, default=DEFAULT_LA_TRAIN_DIR, help="Path to ASVspoof2019_LA_train/flac directory")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CACHE_DIR, help="Directory to store extracted LFCC features")
    parser.add_argument("--output_model", type=str, default=DEFAULT_OUTPUT_MODEL, help="Destination checkpoint path")
    parser.add_argument("--max_per_class", type=int, default=3000, help="Number of clips per class (Bonafide / Spoof)")
    parser.add_argument("--no_la_controls", action="store_true", help="Do not include ASVspoof 2019 clean bonafide controls")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of CPU worker threads for feature extraction")

    args = parser.parse_args()
    train_asvspoof2021_df(args)
