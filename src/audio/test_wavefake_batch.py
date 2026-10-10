"""
True Tone SIH - WaveFake Batch Benchmark Suite
Evaluates the deep learning anti-spoofing model against hundreds of samples
from the official WaveFake dataset across all 6+ neural vocoders.
"""

import os
import sys
import io
import time
import json
import collections
import soundfile as sf
import scipy.signal
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from predict import predict

PARQUET_PATH = os.path.join(PROJECT_ROOT, "data", "wavefake_samples", "partition0.parquet")
HF_URL = "https://huggingface.co/datasets/ajaykarthick/wavefake-audio/resolve/main/data/partition0-00000-of-00001.parquet"

VOCODER_NAMES = {
    "WF1": "MelGAN",
    "WF2": "Parallel WaveGAN",
    "WF3": "Multi-band MelGAN",
    "WF4": "HiFi-GAN",
    "WF5": "WaveGlow",
    "WF6": "FullBand-MelGAN",
    "WF7": "Conformer / Other",
    "R":   "LJSpeech (Bonafide Human)"
}

def evaluate_wavefake_batch(max_samples=100):
    import pyarrow.parquet as pq

    print("=" * 80)
    print(f"TRUE TONE SIH: WAVEFAKE EXTENDED BATCH BENCHMARK (N={max_samples})")
    print("=" * 80)

    # Check local parquet or stream via fsspec
    if os.path.exists(PARQUET_PATH) and os.path.getsize(PARQUET_PATH) > 10 * 1024 * 1024:
        print(f"Reading from local parquet file: {PARQUET_PATH}")
        pfile = pq.ParquetFile(PARQUET_PATH)
    else:
        print(f"Local file not ready yet. Streaming row groups directly over HTTP...")
        import fsspec
        fs = fsspec.filesystem("http")
        f = fs.open(HF_URL, "rb")
        pfile = pq.ParquetFile(f)

    total_available = pfile.metadata.num_rows
    print(f"Total available samples in partition: {total_available}")
    print(f"Evaluating first {max_samples} samples across all vocoder architectures...\n")

    results_by_vocoder = collections.defaultdict(list)
    latencies = []
    processed = 0

    print(f"{'Idx':>4s} | {'Audio ID':12s} | {'Architecture':22s} | {'GT':8s} | {'Risk':>7s} | {'Verdict':16s} | {'Latency':>8s}")
    print("-" * 88)

    for rg_idx in range(pfile.num_row_groups):
        if processed >= max_samples:
            break
        table = pfile.read_row_group(rg_idx)
        n_rows = len(table)

        for i in range(n_rows):
            if processed >= max_samples:
                break

            audio_id = table["audio_id"][i].as_py()
            label_code = table["real_or_fake"][i].as_py()
            vocoder_label = VOCODER_NAMES.get(label_code, label_code)
            is_bonafide = (label_code == "R")
            gt_text = "HUMAN" if is_bonafide else "SPOOF"

            audio_entry = table["audio"][i].as_py()
            audio_bytes = audio_entry["bytes"]

            # Resample 22050 -> 16000
            t0 = time.perf_counter()
            y, sr = sf.read(io.BytesIO(audio_bytes))
            if sr != 16000:
                num_16k = int(len(y) * 16000 / sr)
                y = scipy.signal.resample(y, num_16k).astype(np.float32)
                bio = io.BytesIO()
                sf.write(bio, y, 16000, format="WAV")
                bio.seek(0)
                pred_res = predict(bio.getvalue())
            else:
                pred_res = predict(audio_bytes)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            is_pred_spoof = pred_res["is_spoof"]
            spoof_prob = pred_res["spoof_percentage"]
            verdict_short = "SPOOF" if is_pred_spoof else "BONAFIDE"

            results_by_vocoder[label_code].append({
                "audio_id": audio_id,
                "is_bonafide": is_bonafide,
                "is_pred_spoof": is_pred_spoof,
                "prob": spoof_prob,
                "latency_ms": latency_ms
            })

            processed += 1
            color = "\033[91m" if is_pred_spoof else "\033[92m"
            reset = "\033[0m"
            print(f"{processed:4d} | {audio_id:12s} | {vocoder_label:22s} | {gt_text:8s} | {spoof_prob:>6.2f}% | {color}{verdict_short:16s}{reset} | {latency_ms:>6.2f} ms")

    # Aggregate Statistics
    print("-" * 88)
    print("\n" + "=" * 80)
    print("DETAILED RESULTS BY NEURAL VOCODER ARCHITECTURE")
    print("=" * 80)
    print(f"{'Vocoder / Source':26s} | {'Samples':>7s} | {'Detected':>8s} | {'Accuracy/Recall':>15s} | {'Mean Spoof Prob':>15s}")
    print("-" * 80)

    total_spoof_samples = 0
    total_spoof_detected = 0

    for code in sorted(results_by_vocoder.keys()):
        items = results_by_vocoder[code]
        v_name = VOCODER_NAMES.get(code, code)
        n = len(items)
        mean_p = float(np.mean([x["prob"] for x in items]))

        if code == "R":
            correct = sum(1 for x in items if not x["is_pred_spoof"])
            rate = (correct / n) * 100.0
            print(f"{v_name:26s} | {n:7d} | {correct:8d} | {rate:>14.2f}% | {mean_p:>14.2f}%")
        else:
            detected = sum(1 for x in items if x["is_pred_spoof"])
            total_spoof_samples += n
            total_spoof_detected += detected
            recall = (detected / n) * 100.0
            print(f"{v_name:26s} | {n:7d} | {detected:8d} | {recall:>14.2f}% | {mean_p:>14.2f}%")

    overall_spoof_recall = (total_spoof_detected / total_spoof_samples) * 100.0 if total_spoof_samples else 0
    print("-" * 80)
    print(f"{'OVERALL NEURAL VOCODER RECALL':26s} | {total_spoof_samples:7d} | {total_spoof_detected:8d} | {overall_spoof_recall:>14.2f}% |")
    print(f"Mean Latency per Audio File: {np.mean(latencies):.2f} ms (p95: {np.percentage(latencies, 95) if hasattr(np, 'percentage') else np.percentile(latencies, 95):.2f} ms)")
    print("=" * 80)

    # Save summary report
    summary_path = os.path.join(PROJECT_ROOT, "data", "wavefake_samples", "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "samples_tested": processed,
            "overall_spoof_recall_pct": overall_spoof_recall,
            "mean_latency_ms": float(np.mean(latencies)),
            "per_vocoder": {
                VOCODER_NAMES.get(code, code): {
                    "count": len(items),
                    "mean_prob": float(np.mean([x["prob"] for x in items])),
                    "detected_spoofs": sum(1 for x in items if x["is_pred_spoof"])
                } for code, items in results_by_vocoder.items()
            }
        }, f, indent=2)
    print(f"\nDetailed metrics saved to: {summary_path}")

if __name__ == "__main__":
    n = 100
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            pass
    evaluate_wavefake_batch(n)
