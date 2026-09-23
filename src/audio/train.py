"""
True Tone SIH - Training Script for Audio Spoof Detection
Trains CNN-BiLSTM-Attention on cached LFCC features with CUDA mixed precision (AMP).
Evaluates using Equal Error Rate (EER) and saves optimal model checkpoints.
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
from tqdm import tqdm

# Add current dir to path and resolve project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from model import AudioSpoofDetector

DEFAULT_CHECKPOINT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "models"))

class CachedAudioDataset(Dataset):
    """Loads pre-cached LFCC tensor files from disk."""
    def __init__(self, cache_dir):
        self.files = glob.glob(f"{cache_dir}/*.pt")
        if len(self.files) == 0:
            raise RuntimeError(f"No cached .pt files found in {cache_dir}. Run preprocess_cache.py first!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)
        return data["feat"], torch.tensor(data["label"], dtype=torch.float32)

def compute_eer(labels, scores):
    """
    Computes Equal Error Rate (EER) where False Acceptance Rate equals False Rejection Rate.
    Labels: 1 = Spoof (Target), 0 = Bonafide (Non-target)
    Scores: Probability or logit output (higher = more likely spoof)
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    opt_threshold = thresholds[idx]
    return float(eer), float(opt_threshold)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60, flush=True)
    print("True Tone Audio Spoof Detector Training Pipeline", flush=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Train Cache: {args.train_cache}", flush=True)
    print(f"Dev Cache:   {args.dev_cache}", flush=True)
    print(f"Checkpoints: {args.checkpoint_dir}", flush=True)
    print("=" * 60, flush=True)

    # 1. Load Data
    train_ds = CachedAudioDataset(args.train_cache)
    dev_ds = CachedAudioDataset(args.dev_cache)
    print(f"Loaded {len(train_ds):,} training samples, {len(dev_ds):,} dev samples.", flush=True)

    # In Windows, num_workers=0 avoids pagefile IPC bottlenecks
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size * 2, shuffle=False, pin_memory=True)

    # 2. Instantiate Model & Optimizer
    model = AudioSpoofDetector(input_dim=60, hidden_dim=128).to(device)

    # In ASVspoof, spoof samples outnumber genuine by ~9:1
    # pos_weight penalizes false alarms on bonafide calls
    pos_weight = torch.tensor([args.pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_eer = 1.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_audio_spoof_model.pt")
    start_epoch = 1

    if args.resume and os.path.exists(best_checkpoint_path):
        print(f"Resuming training from checkpoint: {best_checkpoint_path}", flush=True)
        ckpt = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass
        start_epoch = ckpt.get("epoch", 0) + 1
        best_eer = ckpt.get("best_eer", 1.0)
        print(f"Resumed at Epoch {start_epoch}/{args.epochs}, Current Best Dev EER: {best_eer * 100:.4f}%", flush=True)

    if start_epoch > args.epochs:
        print(f"Model checkpoint has already reached Epoch {start_epoch - 1} (target was {args.epochs} epochs).", flush=True)
        print("To continue training, run with a higher --epochs value (e.g., --epochs 20 --resume).", flush=True)
        return

    # 3. Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}] Training", leave=False)
        for feats, labels in pbar:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(feats)
                loss = criterion(logits, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({"loss": f"{total_loss / batch_count:.4f}"})

        scheduler.step()
        avg_train_loss = total_loss / max(1, batch_count)

        # 4. Validation & EER Evaluation
        model.eval()
        dev_labels = []
        dev_scores = []

        with torch.no_grad():
            dev_pbar = tqdm(dev_loader, desc=f"Epoch [{epoch:02d}/{args.epochs:02d}] Dev Eval", leave=False)
            for feats, labels in dev_pbar:
                feats = feats.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(feats)
                    probs = torch.sigmoid(logits)

                dev_scores.extend(probs.cpu().numpy().tolist())
                dev_labels.extend(labels.numpy().tolist())

        eer, threshold = compute_eer(np.array(dev_labels), np.array(dev_scores))

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"| Train Loss: {avg_train_loss:.4f} "
              f"| Dev EER: {eer * 100:.2f}% "
              f"| Threshold: {threshold:.4f}", flush=True)

        # Save Best Model
        if eer < best_eer:
            best_eer = eer
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_eer": best_eer,
                "optimal_threshold": threshold,
                "input_dim": 60,
                "hidden_dim": 128
            }, best_checkpoint_path)
            print(f"  >>> New Best Model Saved! Dev EER: {best_eer * 100:.2f}%", flush=True)

    print("=" * 60, flush=True)
    print(f"Training Complete! Best Dev EER: {best_eer * 100:.2f}%", flush=True)
    print(f"Model saved at: {best_checkpoint_path}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Audio Spoof Detector on RTX 2050 GPU")
    parser.add_argument("--train_cache", type=str, default="D:/True Tone SIH/data/cached_features/train", help="Path to train cached tensors")
    parser.add_argument("--dev_cache", type=str, default="D:/True Tone SIH/data/cached_features/dev", help="Path to dev cached tensors")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate")
    parser.add_argument("--pos_weight", type=float, default=0.25, help="Weight penalty for spoof class")
    parser.add_argument("--checkpoint_dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Directory to save model checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume training from previous best checkpoint")
    args = parser.parse_args()

    train(args)
