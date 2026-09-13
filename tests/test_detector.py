"""
Tests for TrueToneDetector & Temporal Scoring Engine
"""

import os
import numpy as np
import pytest
from true_tone.models import TrueToneDetector, RiskLevel


def test_detector_initialization():
    detector = TrueToneDetector(sr=16000)
    assert detector.low_threshold == 0.40
    assert detector.high_threshold == 0.70


def test_detector_silence_handling():
    detector = TrueToneDetector(sr=16000)
    silence = np.zeros(16000 * 2, dtype=np.float32)
    
    res = detector.analyze_chunk(silence, session_id="test_silence")
    assert not res.is_speech
    assert res.risk_level == RiskLevel.SAFE
    assert res.score <= 0.10


def test_detector_on_synthetic_samples():
    detector = TrueToneDetector(model_path="models/true_tone_classifier.joblib", sr=16000)
    
    # Bonafide signal
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    human_sig = np.sin(2 * np.pi * 150 * t) + 0.4 * np.sin(2 * np.pi * 300 * t)
    human_sig += np.random.normal(0, 0.03, len(t))
    
    res_human = detector.analyze_chunk(human_sig.astype(np.float32), session_id="test_human")
    assert res_human.is_speech
    assert res_human.latency_ms > 0
    assert 0.0 <= res_human.score <= 1.0


def test_session_state_ema():
    detector = TrueToneDetector(sr=16000)
    session = detector.get_or_create_session("session_ema")
    
    s1 = session.update(0.8, is_speech=True)
    assert s1 == 0.8
    
    # Second score should be smoothed with alpha = 0.35
    s2 = session.update(0.2, is_speech=True)
    expected = 0.35 * 0.2 + (1 - 0.35) * 0.8
    assert abs(s2 - expected) < 1e-4
