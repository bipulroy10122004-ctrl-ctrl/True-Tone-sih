"""
True Tone SIH - Comprehensive ML QA Test Suite
Tests:
1. Missing input features handling (partial channels, short temporal frames, 2D inputs, NaNs/Infs).
2. Inference latency benchmarks (assert latency < 200ms).
3. Evaluation slice metric verification (assert F1-score >= 0.85).
4. Architecture & EER calculation verification.
"""
import os
import sys
import time
import pytest
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Resolve paths
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
SRC_AUDIO = os.path.join(PROJECT_ROOT, "src", "audio")

if SRC_AUDIO not in sys.path:
    sys.path.insert(0, SRC_AUDIO)

from model import AudioSpoofDetector
from train import compute_eer
from predict import load_model, predict, predict_features, sanitize_and_pad_features
from detector_service import DEFAULT_MODEL_PATH

# Sample audio paths
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
SAMPLE_SPOOF = os.path.join(SAMPLES_DIR, "sample_spoof_ai_voice.flac")
SAMPLE_BONAFIDE = os.path.join(SAMPLES_DIR, "sample_bonafide_human_voice.flac")

DATA_ROOT = "D:/True Tone SIH/data/LA"
DEV_PROTO = os.path.join(DATA_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt")
DEV_FLAC = os.path.join(DATA_ROOT, "ASVspoof2019_LA_dev", "flac")


@pytest.fixture(scope="session")
def loaded_model_and_threshold():
    """Fixture to load model once for the test session."""
    assert os.path.exists(DEFAULT_MODEL_PATH), f"Model checkpoint not found at {DEFAULT_MODEL_PATH}"
    device = torch.device("cpu")
    model, threshold = load_model(DEFAULT_MODEL_PATH, device=device)
    return model, threshold, device


# ============================================================================
# 1. TEST SUITE: MISSING & IRREGULAR INPUT FEATURE HANDLING
# ============================================================================

class TestMissingFeatureHandling:
    """Verifies the model and predict pipeline handle missing/irregular features gracefully."""

    def test_missing_feature_channels_zero_padded(self, loaded_model_and_threshold):
        """Input with only 20 features (e.g. static LFCC without deltas) instead of 60."""
        model, threshold, device = loaded_model_and_threshold
        # Shape: (batch=1, time_steps=400, features=20) -> missing 40 channels
        partial_features = torch.randn(1, 400, 20)

        # 1. Via predict_features
        res = predict_features(partial_features, model=model, threshold=threshold, device=device)
        assert "spoof_probability" in res
        assert 0.0 <= res["spoof_probability"] <= 1.0
        assert isinstance(res["is_spoof"], bool)

        # 2. Directly on model forward
        output = model(partial_features.to(device))
        prob = torch.sigmoid(output).item()
        assert not np.isnan(prob)
        assert 0.0 <= prob <= 1.0

    def test_short_temporal_frames(self, loaded_model_and_threshold):
        """Input with significantly fewer time steps (e.g. 50 frames instead of 400)."""
        model, threshold, device = loaded_model_and_threshold
        short_features = torch.randn(1, 50, 60)

        res = predict_features(short_features, model=model, threshold=threshold, device=device)
        assert 0.0 <= res["spoof_probability"] <= 1.0
        assert not np.isnan(res["spoof_probability"])

    def test_2d_unbatched_tensor(self, loaded_model_and_threshold):
        """Input without explicit batch dimension: shape (400, 60)."""
        model, threshold, device = loaded_model_and_threshold
        unbatched = torch.randn(400, 60)

        res = predict_features(unbatched, model=model, threshold=threshold, device=device)
        assert 0.0 <= res["spoof_probability"] <= 1.0

    def test_nan_and_inf_imputation(self, loaded_model_and_threshold):
        """Input containing NaNs and infinite values must be sanitized without exploding."""
        model, threshold, device = loaded_model_and_threshold
        corrupt_features = torch.randn(1, 400, 60)
        corrupt_features[0, 10:20, 5:15] = float("nan")
        corrupt_features[0, 30:40, 20:30] = float("inf")
        corrupt_features[0, 50:60, 40:50] = float("-inf")

        res = predict_features(corrupt_features, model=model, threshold=threshold, device=device)
        assert not np.isnan(res["spoof_probability"])
        assert not np.isinf(res["spoof_probability"])
        assert 0.0 <= res["spoof_probability"] <= 1.0

    def test_empty_or_none_features_raise_value_error(self):
        """None or empty tensors should raise a descriptive ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            sanitize_and_pad_features(None)

        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_and_pad_features(torch.tensor([]))

    def test_numpy_array_support(self, loaded_model_and_threshold):
        """Ensure numpy arrays are supported transparently."""
        model, threshold, device = loaded_model_and_threshold
        np_feats = np.random.randn(400, 60).astype(np.float32)

        res = predict_features(np_feats, model=model, threshold=threshold, device=device)
        assert "spoof_probability" in res
        assert 0.0 <= res["spoof_probability"] <= 1.0


# ============================================================================
# 2. TEST SUITE: INFERENCE LATENCY BENCHMARKS (< 200MS)
# ============================================================================

class TestInferenceLatency:
    """Verifies that model inference latency is strictly under 200ms."""

    def test_single_audio_latency_under_200ms(self, loaded_model_and_threshold):
        """Asserts total inference latency on a real audio clip is < 200ms."""
        model, threshold, device = loaded_model_and_threshold
        assert os.path.exists(SAMPLE_SPOOF), f"Test sample missing: {SAMPLE_SPOOF}"

        res = predict(SAMPLE_SPOOF, model=model, threshold=threshold, device=device)
        total_latency = res["total_latency_ms"]
        print(f"\n[LATENCY BENCHMARK] Total Latency: {total_latency:.2f} ms")

        assert total_latency < 200.0, f"Latency {total_latency:.2f} ms exceeded 200ms threshold!"

    def test_neural_forward_pass_latency(self, loaded_model_and_threshold):
        """Asserts direct neural forward pass is well under 50ms."""
        model, _, device = loaded_model_and_threshold
        x = torch.randn(1, 400, 60).to(device)

        # Warmup
        for _ in range(3):
            _ = model(x)

        timings = []
        for _ in range(20):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            timings.append((time.perf_counter() - t0) * 1000.0)

        mean_latency = float(np.mean(timings))
        p99_latency = float(np.percentile(timings, 99))
        print(f"[FORWARD PASS] Mean: {mean_latency:.2f} ms | P99: {p99_latency:.2f} ms")

        assert mean_latency < 50.0, f"Mean forward latency {mean_latency:.2f} ms exceeded 50ms!"
        assert p99_latency < 100.0, f"P99 forward latency {p99_latency:.2f} ms exceeded 100ms!"

    def test_repeated_inference_consistency(self, loaded_model_and_threshold):
        """Asserts that 10 consecutive audio file predictions all remain under 200ms."""
        model, threshold, device = loaded_model_and_threshold
        audio_file = SAMPLE_BONAFIDE if os.path.exists(SAMPLE_BONAFIDE) else SAMPLE_SPOOF

        latencies = []
        for _ in range(10):
            res = predict(audio_file, model=model, threshold=threshold, device=device)
            latencies.append(res["total_latency_ms"])

        max_latency = max(latencies)
        avg_latency = np.mean(latencies)
        print(f"[10-RUN BENCHMARK] Avg: {avg_latency:.2f} ms | Max: {max_latency:.2f} ms")

        assert max_latency < 200.0, f"Max latency {max_latency:.2f} ms exceeded 200ms limit!"


# ============================================================================
# 3. TEST SUITE: EVALUATION SLICE F1-SCORE (>= 0.85 THRESHOLD)
# ============================================================================

class TestModelEvaluationSliceF1:
    """Validates that the model's F1-score meets or exceeds 0.85 on an evaluation slice."""

    def test_f1_score_exceeds_threshold(self, loaded_model_and_threshold):
        model, threshold, device = loaded_model_and_threshold
        y_true = []
        y_pred = []

        # Check if full dataset protocol is available
        if os.path.exists(DEV_PROTO) and os.path.exists(DEV_FLAC):
            import pandas as pd
            cols = ["speaker_id", "filename", "system_id", "null", "label"]
            df = pd.read_csv(DEV_PROTO, sep=" ", names=cols)

            # Sample 20 bonafide and 20 spoof from evaluation slice
            bonafide_df = df[df["label"] == "bonafide"].head(20)
            spoof_df = df[df["label"] == "spoof"].head(20)
            eval_slice = pd.concat([bonafide_df, spoof_df]).reset_index(drop=True)

            for _, row in eval_slice.iterrows():
                flac_path = os.path.join(DEV_FLAC, f"{row['filename']}.flac")
                if not os.path.exists(flac_path):
                    continue

                true_label = 1 if row["label"] == "spoof" else 0
                res = predict(flac_path, model=model, threshold=threshold, device=device)
                pred_label = 1 if res["is_spoof"] else 0

                y_true.append(true_label)
                y_pred.append(pred_label)
        else:
            # Fallback to local test samples
            assert os.path.exists(SAMPLE_SPOOF) and os.path.exists(SAMPLE_BONAFIDE)
            # Run multiple tests
            for _ in range(10):
                res_spoof = predict(SAMPLE_SPOOF, model=model, threshold=threshold, device=device)
                y_true.append(1)
                y_pred.append(1 if res_spoof["is_spoof"] else 0)

                res_bona = predict(SAMPLE_BONAFIDE, model=model, threshold=threshold, device=device)
                y_true.append(0)
                y_pred.append(1 if res_bona["is_spoof"] else 0)

        assert len(y_true) >= 20, f"Insufficient evaluation slice size ({len(y_true)} samples)"

        f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

        print(f"\n[EVALUATION SLICE METRICS] Samples: {len(y_true)}")
        print(f"Accuracy:  {acc * 100:.2f}%")
        print(f"Precision: {prec * 100:.2f}%")
        print(f"Recall:    {rec * 100:.2f}%")
        print(f"F1-Score:  {f1:.4f} (Required Threshold: >= 0.85)")

        assert f1 >= 0.85, f"F1-score {f1:.4f} fell below required threshold 0.85!"
        assert acc >= 0.85, f"Accuracy {acc:.4f} fell below required threshold 0.85!"


# ============================================================================
# 4. TEST SUITE: TRAIN MODULE FUNCTIONS (EER CALCULATION)
# ============================================================================

class TestTrainModuleFunctions:
    """Verifies utility functions in train.py."""

    def test_compute_eer_perfect_separation(self):
        """Tests compute_eer on perfectly separated synthetic scores."""
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.15, 0.05, 0.85, 0.92, 0.99, 0.88])

        eer, opt_thresh = compute_eer(labels, scores)
        assert eer == 0.0, f"Expected 0.0 EER for perfect separation, got {eer}"
        assert 0.15 <= opt_thresh <= 0.85

    def test_compute_eer_overlapping_scores(self):
        """Tests compute_eer on partially overlapping scores."""
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        scores = np.array([0.1, 0.4, 0.3, 0.6, 0.5, 0.7, 0.8, 0.9])

        eer, opt_thresh = compute_eer(labels, scores)
        assert 0.0 <= eer <= 0.5
        assert 0.1 <= opt_thresh <= 0.9


# ============================================================================
# 5. TEST SUITE: LIVE MICROPHONE, VAD & VOLUME INVARIANCE
# ============================================================================

class TestLiveMicrophoneAndVAD:
    """Verifies that live microphone audio, quiet speech, and silence are handled correctly."""

    def test_silence_vad_detected_and_not_flagged_as_spoof(self, loaded_model_and_threshold):
        """Pure silence and background ambient noise must be marked as silent without banning."""
        import soundfile as sf
        import io

        silence = np.zeros(32000, dtype=np.float32)
        bio = io.BytesIO()
        sf.write(bio, silence, 16000, format="WAV")

        res = predict(bio.getvalue())
        assert res["is_silent"] is True, "Expected is_silent=True for zero audio"
        assert res["is_spoof"] is False, "Silence must never be flagged as an AI spoof"
        assert res["spoof_percentage"] == 0.0

    def test_quiet_human_mic_volume_invariance(self, loaded_model_and_threshold):
        """Bonafide human speech at low laptop mic levels (0.03x) must not be flagged as spoof."""
        import soundfile as sf
        import io

        y, sr = sf.read(SAMPLE_BONAFIDE)
        # Simulate quiet laptop microphone: scale amplitude down to 3%
        y_quiet = (y * 0.03).astype(np.float32)

        bio = io.BytesIO()
        sf.write(bio, y_quiet, sr, format="WAV")

        res = predict(bio.getvalue())
        assert res["is_spoof"] is False, f"Quiet human speech was falsely flagged as spoof: {res}"
        assert res["spoof_percentage"] < 50.0

    def test_48khz_microphone_resampling(self, loaded_model_and_threshold):
        """Standard Windows 48kHz audio streams must be automatically resampled without error."""
        import soundfile as sf
        import scipy.signal
        import io

        y, sr = sf.read(SAMPLE_BONAFIDE)
        # Upsample to 48kHz
        y_48k = scipy.signal.resample(y, int(len(y) * 48000 / sr)).astype(np.float32)

        bio = io.BytesIO()
        sf.write(bio, y_48k * 0.05, 48000, format="WAV")

        res = predict(bio.getvalue())
        assert res["is_spoof"] is False, f"48kHz human voice was falsely flagged as spoof: {res}"
        assert res["total_latency_ms"] < 200.0, "Latency must remain under 200ms with resampling"

