"""
True Tone SIH - Multi-Dataset Comprehensive Benchmark Suite
Evaluates the trained AudioSpoofDetector across 4 diverse datasets:
1. ASVspoof 2019 Dev Set (Known Attacks: A01 - A06 + Bonafide Human)
2. ASVspoof 2019 Eval Set (Unseen Attacks: A07 - A19 + Bonafide Human)
3. WaveFake Neural Vocoder Dataset (MelGAN, Parallel WaveGAN, HiFi-GAN, WaveGlow)
4. Streaming In-Call Sliding Window Benchmark (2-second live VoIP frames)
"""
import os
import sys
import io
import time
import json
import numpy as np
import pandas as pd
import soundfile as sf
import pyarrow.parquet as pq
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from detector_service import AudioSpoofInferenceEngine
from predict import predict

DATA_ROOT = "D:/True Tone SIH/data/LA"
DEV_PROTO = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")
DEV_FLAC = os.path.join(DATA_ROOT, "ASVspoof2019_LA_dev", "flac")
EVAL_PROTO = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt")
EVAL_FLAC = os.path.join(DATA_ROOT, "ASVspoof2019_LA_eval", "flac")
PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "wavefake_samples", "partition0.parquet")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

VOCODER_NAMES = {
    "WF1": "MelGAN (Neural Vocoder)",
    "WF2": "Parallel WaveGAN (Neural)",
    "WF3": "Multi-band MelGAN (Neural)",
    "WF4": "HiFi-GAN (State-of-the-Art)",
    "WF5": "WaveGlow (Flow-based)",
    "WF6": "FullBand-MelGAN",
    "R":   "LJSpeech (Bonafide Human)"
}

ATTACK_NAMES = {
    "A01": "Neural TTS (WaveNet)",
    "A02": "Neural TTS (Tacotron)",
    "A03": "Neural TTS (WaveGlow)",
    "A04": "Voice Conversion (CycleGAN)",
    "A05": "Voice Conversion (VAW-GAN)",
    "A06": "Waveform Concatenation",
    "A07": "Unseen Neural Vocoder (WaveNet v2)",
    "A08": "Unseen Neural Vocoder (WaveRNN)",
    "A09": "Unseen Voice Conversion (Flow-TTS)",
    "A10": "Unseen Voice Conversion (StarGAN)",
    "A11": "Unseen Transfer Learning TTS",
    "A12": "Unseen Multi-Speaker Tacotron",
    "A13": "Unseen Harmonic Voice Conversion",
    "A14": "Unseen Deep Neural Vocoder",
    "A15": "Unseen Continuous Pitch Synthesizer",
    "A16": "Unseen GAN-based Voice Clone",
    "A17": "Unseen Waveform Diffusion Model",
    "A18": "Unseen Cross-Lingual Voice Clone",
    "A19": "Unseen Hybrid Concatenative TTS"
}

def print_metrics(benchmark_name, y_true, y_pred, probs, latencies, attack_dict=None):
    acc = accuracy_score(y_true, y_pred) * 100.0
    prec = precision_score(y_true, y_pred, zero_division=0) * 100.0
    rec = recall_score(y_true, y_pred, zero_division=0) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    spec = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    mean_lat = np.mean(latencies) if latencies else 0.0

    print("\n" + "=" * 76)
    print(f"  BENCHMARK REPORT: {benchmark_name}")
    print("=" * 76)
    print(f"Total Samples Evaluated:    {len(y_true)}")
    print(f"Overall Accuracy:           {acc:.2f}%")
    print(f"AI Spoof Recall (Detection Rate): {rec:.2f}% ({tp}/{tp+fn})")
    print(f"Human Voice Pass Rate (Specificity): {spec:.2f}% ({tn}/{tn+fp})")
    print(f"Precision (Spoof):          {prec:.2f}%")
    print(f"False Alarm Rate (FAR):     {(100.0 - spec):.2f}% ({fp} false alarms)")
    print(f"F1-Score:                   {f1:.2f}%")
    print(f"Average Inference Latency:  {mean_lat:.2f} ms")
    print(f"Confusion Matrix:           [TN={tn}, FP={fp}] | [FN={fn}, TP={tp}]")

    if attack_dict:
        print("\nAttack / Architecture Breakdown:")
        print(f"{'Attack ID / Name':40s} | {'Samples':>7s} | {'Detected':>8s} | {'Recall':>9s}")
        print("-" * 72)
        for att, stats in sorted(attack_dict.items(), key=lambda x: str(x[0])):
            n = stats["total"]
            c = stats["correct"]
            r = (c / n * 100.0) if n > 0 else 0.0
            print(f"{str(att):40s} | {n:7d} | {c:8d} | {r:>8.1f}%")

    return {
        "benchmark": benchmark_name,
        "samples": len(y_true),
        "accuracy": round(acc, 2),
        "recall": round(rec, 2),
        "specificity": round(spec, 2),
        "f1": round(f1, 2),
        "latency_ms": round(mean_lat, 2)
    }

def run_all_benchmarks():
    print("#" * 80)
    print("  TRUE TONE SIH: MULTI-DATASET MODEL EVALUATION SUITE")
    print("  Testing Robustness & Cross-Dataset Generalization")
    print("#" * 80)

    detector = AudioSpoofInferenceEngine()
    summary = []

    # =========================================================================
    # BENCHMARK 1: ASVspoof 2019 Dev Set (Known Attacks A01 - A06)
    # =========================================================================
    if os.path.exists(DEV_PROTO) and os.path.exists(DEV_FLAC):
        print("\n[1/4] Loading ASVspoof 2019 Dev Set...")
        cols = ["speaker_id", "filename", "environment", "system_id", "label"]
        df_dev = pd.read_csv(DEV_PROTO, sep=" ", names=cols, keep_default_na=False)
        
        # Sample 30 Bonafide, and 5 per attack A01-A06 (30 spoof) = 60 total
        bonafide_dev = df_dev[df_dev["label"] == "bonafide"].sample(n=30, random_state=42)
        dev_spoof_list = [g.sample(min(len(g), 5), random_state=42) for _, g in df_dev[df_dev["label"] == "spoof"].groupby("system_id")]
        spoof_dev = pd.concat(dev_spoof_list)
        test_dev = pd.concat([bonafide_dev, spoof_dev]).sample(frac=1.0, random_state=42).reset_index(drop=True)

        y_true, y_pred, probs, lats = [], [], [], []
        attack_stats = defaultdict(lambda: {"total": 0, "correct": 0})

        for _, row in test_dev.iterrows():
            fpath = os.path.join(DEV_FLAC, f"{row['filename']}.flac")
            if not os.path.exists(fpath): continue
            t_label = 1 if row["label"] == "spoof" else 0
            sys_id = str(row["system_id"])
            att_name = f"{sys_id}: {ATTACK_NAMES.get(sys_id, sys_id)}" if t_label else "Authentic Human Voice"

            res = detector.predict(fpath)
            p_label = 1 if res["is_spoof"] else 0
            
            y_true.append(t_label)
            y_pred.append(p_label)
            probs.append(res["spoof_probability"])
            lats.append(res["total_latency_ms"])

            attack_stats[att_name]["total"] += 1
            if p_label == t_label:
                attack_stats[att_name]["correct"] += 1

        s1 = print_metrics("ASVspoof 2019 Dev Set (Known Attacks A01-A06)", y_true, y_pred, probs, lats, attack_stats)
        summary.append(s1)

    # =========================================================================
    # BENCHMARK 2: ASVspoof 2019 Eval Set (Unseen Attacks A07 - A19)
    # =========================================================================
    if os.path.exists(EVAL_PROTO) and os.path.exists(EVAL_FLAC):
        print("\n[2/4] Loading ASVspoof 2019 Eval Set (Zero-Shot Generalization)...")
        cols = ["speaker_id", "filename", "environment", "system_id", "label"]
        df_eval = pd.read_csv(EVAL_PROTO, sep=" ", names=cols, keep_default_na=False)
        
        # Sample 30 Bonafide, and 3 per unseen attack A07-A19 (13 attacks * 3 = 39 spoof) = 69 total
        bonafide_eval = df_eval[df_eval["label"] == "bonafide"].sample(n=30, random_state=42)
        eval_spoof_list = [g.sample(min(len(g), 3), random_state=42) for _, g in df_eval[df_eval["label"] == "spoof"].groupby("system_id")]
        spoof_eval = pd.concat(eval_spoof_list)
        test_eval = pd.concat([bonafide_eval, spoof_eval]).sample(frac=1.0, random_state=42).reset_index(drop=True)

        y_true, y_pred, probs, lats = [], [], [], []
        attack_stats = defaultdict(lambda: {"total": 0, "correct": 0})

        for _, row in test_eval.iterrows():
            fpath = os.path.join(EVAL_FLAC, f"{row['filename']}.flac")
            if not os.path.exists(fpath): continue
            t_label = 1 if row["label"] == "spoof" else 0
            sys_id = str(row["system_id"])
            att_name = f"{sys_id}: {ATTACK_NAMES.get(sys_id, sys_id)}" if t_label else "Authentic Human Voice"

            res = detector.predict(fpath)
            p_label = 1 if res["is_spoof"] else 0
            
            y_true.append(t_label)
            y_pred.append(p_label)
            probs.append(res["spoof_probability"])
            lats.append(res["total_latency_ms"])

            attack_stats[att_name]["total"] += 1
            if p_label == t_label:
                attack_stats[att_name]["correct"] += 1

        s2 = print_metrics("ASVspoof 2019 Eval Set (Unseen Attacks A07-A19)", y_true, y_pred, probs, lats, attack_stats)
        summary.append(s2)

    # =========================================================================
    # BENCHMARK 3: WaveFake Neural Vocoder Benchmark (Parquet partition)
    # =========================================================================
    if os.path.exists(PARQUET_PATH):
        print("\n[3/4] Loading WaveFake Neural Vocoders (MelGAN, Parallel WaveGAN, HiFi-GAN, WaveGlow)...")
        pfile = pq.ParquetFile(PARQUET_PATH)
        table = pfile.read_row_group(0) # First row group ~ 60-100 samples
        n_samples = min(len(table), 60)

        y_true, y_pred, probs, lats = [], [], [], []
        vocoder_stats = defaultdict(lambda: {"total": 0, "correct": 0})

        for i in range(n_samples):
            label_code = table["real_or_fake"][i].as_py()
            v_name = VOCODER_NAMES.get(label_code, f"Vocoder {label_code}")
            t_label = 0 if label_code == "R" else 1

            audio_entry = table["audio"][i].as_py()
            audio_bytes = audio_entry["bytes"]

            t0 = time.perf_counter()
            res = predict(audio_bytes)
            lat = (time.perf_counter() - t0) * 1000.0

            p_label = 1 if res["is_spoof"] else 0
            y_true.append(t_label)
            y_pred.append(p_label)
            probs.append(res["spoof_probability"])
            lats.append(lat)

            vocoder_stats[v_name]["total"] += 1
            if p_label == t_label:
                vocoder_stats[v_name]["correct"] += 1

        s3 = print_metrics("WaveFake Neural Vocoder Benchmark", y_true, y_pred, probs, lats, vocoder_stats)
        summary.append(s3)

    # =========================================================================
    # BENCHMARK 4: Streaming In-Call Sliding Window Benchmark (2s Frames)
    # =========================================================================
    print("\n[4/4] Evaluating Real-time Streaming VoIP Frames (2-Second Sliding Windows)...")
    human_fpath = os.path.join(SAMPLES_DIR, "sample_bonafide_human_voice.flac")
    spoof_fpath = os.path.join(SAMPLES_DIR, "sample_spoof_ai_voice.flac")

    y_true, y_pred, probs, lats = [], [], [], []
    stream_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    if os.path.exists(human_fpath):
        y, sr = sf.read(human_fpath)
        win = int(sr * 2.0)
        step = int(sr * 1.0)
        for s in range(0, len(y) - win + 1, step):
            buf = io.BytesIO()
            sf.write(buf, y[s:s+win], sr, format="WAV")
            buf.seek(0)
            res = predict(buf.getvalue())
            t_label = 0
            p_label = 1 if res["is_spoof"] else 0
            y_true.append(t_label)
            y_pred.append(p_label)
            probs.append(res["spoof_probability"])
            lats.append(res["total_latency_ms"])
            stream_stats["Continuous Human Speech (2s)"]["total"] += 1
            if p_label == t_label: stream_stats["Continuous Human Speech (2s)"]["correct"] += 1

    if os.path.exists(spoof_fpath):
        y, sr = sf.read(spoof_fpath)
        pad_y = np.pad(y, (0, max(0, int(sr*2.0) - len(y))), mode='constant')
        buf = io.BytesIO()
        sf.write(buf, pad_y, sr, format="WAV")
        buf.seek(0)
        res = predict(buf.getvalue())
        t_label = 1
        p_label = 1 if res["is_spoof"] else 0
        y_true.append(t_label)
        y_pred.append(p_label)
        probs.append(res["spoof_probability"])
        lats.append(res["total_latency_ms"])
        stream_stats["Synthetic AI Deepfake Speech (2s)"]["total"] += 1
        if p_label == t_label: stream_stats["Synthetic AI Deepfake Speech (2s)"]["correct"] += 1

    s4 = print_metrics("Real-Time Streaming VoIP Sliding Frames (2s Windows)", y_true, y_pred, probs, lats, stream_stats)
    summary.append(s4)

    # =========================================================================
    # EXECUTIVE SUMMARY TABLE
    # =========================================================================
    print("\n" + "#" * 80)
    print("  EXECUTIVE CROSS-DATASET BENCHMARK SUMMARY")
    print("#" * 80)
    print(f"{'Dataset / Benchmark':48s} | {'Samples':>7s} | {'Accuracy':>9s} | {'AI Recall':>9s} | {'Human Spec':>10s} | {'Latency':>8s}")
    print("-" * 102)
    for s in summary:
        print(f"{s['benchmark']:48s} | {s['samples']:7d} | {s['accuracy']:>8.2f}% | {s['recall']:>8.2f}% | {s['specificity']:>9.2f}% | {s['latency_ms']:>6.1f}ms")
    print("-" * 102)

if __name__ == "__main__":
    run_all_benchmarks()
