"""
Tests for Telephony Codecs, VAD, and Forensic DSP Feature Extraction
"""

import numpy as np
import pytest
from true_tone.dsp import (
    encode_mulaw,
    decode_mulaw,
    encode_alaw,
    decode_alaw,
    simulate_telephony_channel,
    EnergyVAD,
    extract_lfcc,
    extract_phase_features,
    extract_prosody_features,
    extract_full_features
)


def test_mulaw_compression_expansion():
    # Test linear PCM preservation through mu-law
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    orig = 0.8 * np.sin(2 * np.pi * 440 * t)
    
    encoded = encode_mulaw(orig)
    assert encoded.dtype == np.uint8
    assert len(encoded) == len(orig)
    
    decoded = decode_mulaw(encoded)
    assert decoded.dtype == np.float32
    
    # Calculate Reconstruction SNR (should be > 28 dB for 8-bit companded audio)
    noise = orig - decoded
    snr = 10 * np.log10(np.mean(orig ** 2) / np.mean(noise ** 2))
    assert snr > 25.0


def test_alaw_compression_expansion():
    t = np.linspace(0, 0.5, 8000, endpoint=False)
    orig = 0.7 * np.sin(2 * np.pi * 500 * t)
    
    encoded = encode_alaw(orig)
    decoded = decode_alaw(encoded)
    noise = orig - decoded
    snr = 10 * np.log10(np.mean(orig ** 2) / np.mean(noise ** 2))
    assert snr > 25.0


def test_simulate_telephony_channel():
    x = np.random.randn(16000).astype(np.float32) * 0.5
    x_tel, out_sr = simulate_telephony_channel(x, sr=16000, target_sr=8000, codec="mulaw")
    assert out_sr == 8000
    assert len(x_tel) == 8000
    assert not np.isnan(x_tel).any()
    assert np.max(np.abs(x_tel)) <= 1.0


def test_energy_vad():
    vad = EnergyVAD(sr=16000, energy_threshold=0.01)
    
    # Silent frame
    silence = np.zeros(int(16000 * 0.05), dtype=np.float32)
    assert not vad.is_speech_frame(silence)
    
    # Speech-like burst
    t = np.linspace(0, 0.05, len(silence), endpoint=False)
    speech_frame = 0.5 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    assert vad.is_speech_frame(speech_frame)


def test_lfcc_extraction():
    x = np.random.randn(16000).astype(np.float32) * 0.3
    lfcc = extract_lfcc(x, sr=16000, n_ceps=20, include_deltas=True)
    # 20 ceps * 3 (static + delta + delta-delta) = 60 rows
    assert lfcc.shape[0] == 60
    assert lfcc.shape[1] > 10
    assert not np.isnan(lfcc).any()


def test_full_features_extraction():
    x = np.random.randn(16000).astype(np.float32) * 0.3
    res = extract_full_features(x, sr=16000)
    
    assert "feature_vector" in res
    assert "phase_metrics" in res
    assert "prosody_metrics" in res
    assert len(res["feature_vector"]) == 133
    assert not np.isnan(res["feature_vector"]).any()
