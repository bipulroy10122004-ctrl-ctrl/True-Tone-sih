"""
True Tone SIH - Single Audio Inference & Testing Utility
Tests an individual FLAC or WAV file against the trained anti-spoofing detector.
Outputs classification verdict, spoof probability percentage, and inference latency.
"""
import os
import sys
import time
import argparse
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from features import extract_lfcc
from model import AudioSpoofDetector

DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "models", "best_audio_spoof_model.pt"))

def predict_single_audio(audio_path, model_path=DEFAULT_MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on: {device}...")

    # Load Model
    model = AudioSpoofDetector(input_dim=60, hidden_dim=128).to(device)
    threshold = 0.50

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        threshold = checkpoint.get("optimal_threshold", 0.50)
        print(f"Loaded trained checkpoint: {model_path} (Opt Threshold: {threshold:.4f})")
    else:
        print(f"Warning: Checkpoint {model_path} not found. Running with initialized weights for demonstration.")

    model.eval()

    # 1. Measure Feature Extraction Latency
    t0 = time.perf_counter()
    feat = extract_lfcc(audio_path)
    feat_time = (time.perf_counter() - t0) * 1000.0

    # 2. Measure Model Forward Latency
    tensor_in = torch.from_numpy(feat).unsqueeze(0).to(device)
    t1 = time.perf_counter()
    with torch.no_grad():
        logit = model(tensor_in)
        prob = torch.sigmoid(logit).item()
    forward_time = (time.perf_counter() - t1) * 1000.0
    total_latency = feat_time + forward_time

    # 3. Verdict
    is_spoof = prob >= threshold
    verdict = "SYNTHETIC AI VOICE / SPOOF" if is_spoof else "AUTHENTIC HUMAN / BONAFIDE"
    color = "\033[91m" if is_spoof else "\033[92m"
    reset = "\033[0m"

    print("=" * 65)
    print(f"File: {audio_path}")
    print(f"Feature Extraction Latency: {feat_time:.2f} ms")
    print(f"Model Forward Latency:      {forward_time:.2f} ms")
    print(f"Total Inference Latency:    {total_latency:.2f} ms")
    print("-" * 65)
    print(f"Spoof Risk Probability:     {prob * 100:.2f}%")
    print(f"Decision Threshold:         {threshold:.4f}")
    print(f"Final Verdict:              {color}{verdict}{reset}")
    print("=" * 65)

    return {"prob": prob, "is_spoof": is_spoof, "latency_ms": total_latency}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a single audio file for synthetic voice detection")
    parser.add_argument("--audio", type=str, required=True, help="Path to .flac or .wav audio file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model checkpoint")
    args = parser.parse_args()

    predict_single_audio(args.audio, args.model)
