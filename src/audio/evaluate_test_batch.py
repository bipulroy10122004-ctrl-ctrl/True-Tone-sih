"""
True Tone SIH - Comprehensive Model Evaluation & Benchmark Test
Evaluates the trained AudioSpoofDetector on a balanced sample of unseen ASVspoof files.
Computes Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Latency breakdown across attacks.
"""
import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from detector_service import AudioSpoofInferenceEngine

DATA_ROOT = "D:/True Tone SIH/data/LA"
DEV_PROTO = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")
DEV_FLAC = os.path.join(DATA_ROOT, "ASVspoof2019_LA_dev", "flac")

def run_evaluation(num_samples_per_class=25):
    print("=" * 70)
    print("TRUE TONE SIH: COMPREHENSIVE MODEL EVALUATION BENCHMARK")
    print("=" * 70)

    if not os.path.exists(DEV_PROTO):
        print(f"Error: Protocol file not found at {DEV_PROTO}")
        return

    cols = ["speaker_id", "filename", "system_id", "null", "label"]
    df = pd.read_csv(DEV_PROTO, sep=" ", names=cols)

    # Sample balanced set
    bonafide_df = df[df["label"] == "bonafide"].head(num_samples_per_class)
    spoof_df = df[df["label"] == "spoof"].head(num_samples_per_class)
    test_df = pd.concat([bonafide_df, spoof_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Evaluating {len(test_df)} samples ({len(bonafide_df)} Bonafide, {len(spoof_df)} Spoof)...")
    print("Loading inference engine...")
    detector = AudioSpoofInferenceEngine()

    y_true = []
    y_pred = []
    probabilities = []
    feature_latencies = []
    forward_latencies = []
    total_latencies = []
    attack_results = {}

    for idx, row in test_df.iterrows():
        flac_path = os.path.join(DEV_FLAC, f"{row['filename']}.flac")
        if not os.path.exists(flac_path):
            continue

        true_label = 1 if row["label"] == "spoof" else 0
        attack_type = row["system_id"]

        res = detector.predict(flac_path)
        pred_label = 1 if res["is_spoof"] else 0

        y_true.append(true_label)
        y_pred.append(pred_label)
        probabilities.append(res["spoof_probability"])
        feature_latencies.append(res["feature_latency_ms"])
        forward_latencies.append(res["forward_latency_ms"])
        total_latencies.append(res["total_latency_ms"])

        if attack_type not in attack_results:
            attack_results[attack_type] = {"total": 0, "correct": 0}
        attack_results[attack_type]["total"] += 1
        if pred_label == true_label:
            attack_results[attack_type]["correct"] += 1

    # Metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred) # [[TN, FP], [FN, TP]]

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    print("\n" + "-" * 70)
    print("  EVALUATION RESULTS & PERFORMANCE REPORT")
    print("-" * 70)
    print(f"Total Samples Evaluated:    {len(y_true)}")
    print(f"Overall Accuracy:           {acc * 100:.2f}%")
    print(f"Precision (Spoof):          {prec * 100:.2f}%")
    print(f"Recall / Detection Rate:    {rec * 100:.2f}%")
    print(f"F1-Score:                   {f1 * 100:.2f}%")
    print(f"Optimal Threshold:          {detector.threshold:.4f}")
    print("\nConfusion Matrix:")
    print(f"  True Authentic (TN):      {tn} (Correctly classified as Real Human)")
    print(f"  False Alarms (FP):        {fp} (Real Human falsely flagged as Spoof)")
    print(f"  Missed Spoofs (FN):       {fn} (AI Deepfake missed as Real Human)")
    print(f"  Neutralized Spoofs (TP):  {tp} (AI Deepfake correctly blocked)")
    print("\nLatency Benchmarks:")
    print(f"  Mean Feature Extraction:  {np.mean(feature_latencies):.2f} ms")
    print(f"  Mean Neural Forward Pass: {np.mean(forward_latencies):.2f} ms")
    print(f"  Mean Total Call Latency:  {np.mean(total_latencies):.2f} ms (95th percentile: {np.percentile(total_latencies, 95):.2f} ms)")

    print("\nAttack-Specific Accuracy Breakdown:")
    for att, data in sorted(attack_results.items()):
        att_acc = (data["correct"] / data["total"]) * 100
        desc = "Authentic Voice" if att == "-" else f"Synthesis Attack {att}"
        print(f"  {att:5s} ({desc:22s}): {att_acc:6.2f}% ({data['correct']}/{data['total']})")

    print("=" * 70)

if __name__ == "__main__":
    run_evaluation(num_samples_per_class=30)
