"""
True Tone Audio Deepfake Detection & Temporal Scoring Engine

Performs real-time, low-latency deepfake detection on telephony speech chunks.
Integrates LFCC spectral analysis, phase inconsistency forensic scoring,
biological prosody verification, and exponential moving average (EMA) temporal smoothing.
"""

import time
import enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np

from true_tone.dsp import extract_full_features, EnergyVAD


class RiskLevel(str, enum.Enum):
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    CRITICAL = "CRITICAL"


@dataclass
class DetectionResult:
    score: float                         # Instantaneous chunk score [0.0 - 1.0]
    smoothed_score: float                # Temporal smoothed session score [0.0 - 1.0]
    risk_level: RiskLevel                # SAFE, SUSPICIOUS, CRITICAL
    action: str                          # Recommended operational action
    is_speech: bool                      # Whether active speech was present
    latency_ms: float                    # Time taken to process chunk
    forensics: Dict[str, Any]            # Detailed sub-scores and forensic metrics
    timestamp: float = 0.0


class SessionState:
    """Tracks state across chunks for an active cellular/SIP call."""
    def __init__(self, session_id: str, alpha: float = 0.35, max_history: int = 50):
        self.session_id = session_id
        self.alpha = alpha  # EMA smoothing factor
        self.max_history = max_history
        self.history: List[float] = []
        self.current_smoothed_score: float = 0.10
        self.total_chunks: int = 0
        self.speech_chunks: int = 0
        self.high_risk_streak: int = 0

    def update(self, raw_score: float, is_speech: bool) -> float:
        self.total_chunks += 1
        if not is_speech:
            # During silence, retain previous smoothed score with slight decay towards baseline
            self.current_smoothed_score = 0.95 * self.current_smoothed_score + 0.05 * 0.10
            return self.current_smoothed_score
            
        self.speech_chunks += 1
        self.history.append(raw_score)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        # Exponential Moving Average update
        if self.speech_chunks == 1:
            self.current_smoothed_score = raw_score
        else:
            self.current_smoothed_score = (self.alpha * raw_score) + ((1.0 - self.alpha) * self.current_smoothed_score)
            
        if self.current_smoothed_score >= 0.70:
            self.high_risk_streak += 1
        else:
            self.high_risk_streak = max(0, self.high_risk_streak - 1)
            
        return self.current_smoothed_score


class TrueToneDetector:
    """
    Forensic Deepfake Detection Model for SIM-to-SIM Telephony Audio.
    Evaluates acoustic, phase, and prosodic cues to determine if speech is AI-generated.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        sr: int = 16000,
        low_threshold: float = 0.40,
        high_threshold: float = 0.70
    ):
        self.sr = sr
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.vad = EnergyVAD(sr=sr)
        self.sessions: Dict[str, SessionState] = {}
        self.trained_classifier = None
        
        if model_path:
            self._load_model(model_path)
            
    def _load_model(self, model_path: str):
        try:
            import joblib
            self.trained_classifier = joblib.load(model_path)
        except Exception as e:
            print(f"[TrueToneDetector] Notice: Could not load {model_path} ({e}). Using forensic expert ensemble.")
            self.trained_classifier = None

    def get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id)
        return self.sessions[session_id]

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _evaluate_forensics(self, feats: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
        """
        Calibrated forensic expert scoring combining:
        1. LFCC spectral statistics (40% weight): vocoder spectral smoothing & envelope slope
        2. Phase group delay & Instantaneous Frequency (35% weight): vocoder phase incoherence
        3. Prosody & Biological micro-variations (25% weight): micro-jitter deficit and pitch robotics
        """
        lfcc_vec = feats["feature_vector"][:120]  # LFCC mean + std
        phase = feats["phase_metrics"]
        prosody = feats["prosody_metrics"]
        
        # 1. Acoustic / Spectral Score
        # Neural TTS typically has lower LFCC variance in high cepstral bins (spectral over-smoothing)
        high_cepstral_variance = float(np.mean(feats["feature_vector"][80:120]))
        spectral_tilt = float(phase["spectral_tilt_ratio"])
        
        # Natural human speech has high dynamic variance across speech frames
        # Synthesized speech often has uniform/low variance across cepstral acceleration
        acoustic_indicator = 1.0 / (1.0 + np.exp(3.0 * (high_cepstral_variance - 0.45)))
        acoustic_score = np.clip(acoustic_indicator * 0.8 + (1.0 if spectral_tilt < 0.05 else 0.2) * 0.2, 0.02, 0.98)
        
        # 2. Phase Inconsistency Score
        # Neural vocoders (HiFi-GAN, WaveGlow, Diffusion) produce phase incoherence
        # measured by group delay entropy and instantaneous frequency skewness
        gd_entropy = phase["gd_entropy"]
        if_skew = abs(phase["if_skew"])
        phase_anomaly = np.clip(0.35 * (gd_entropy / 3.0) + 0.35 * (if_skew / 2.5), 0.03, 0.97)
        
        # 3. Prosody & Biological Score
        # AI voices often lack natural cycle-to-cycle micro-jitter (jitter < 0.008 is typical of TTS)
        # and natural pitch drifts (f0_std / f0_mean < 0.08)
        jitter = prosody["jitter"]
        shimmer = prosody["shimmer"]
        f0_std = prosody["f0_std"]
        f0_mean = prosody["f0_mean"]
        
        pitch_variability = (f0_std / (f0_mean + 1e-8)) if f0_mean > 50.0 else 0.15
        
        # Low jitter + low pitch variability strongly indicates synthetic speech
        jitter_penalty = 1.0 if (jitter < 0.005 and f0_mean > 60.0) else (0.7 if jitter < 0.012 else 0.1)
        pitch_penalty = 1.0 if pitch_variability < 0.06 else (0.5 if pitch_variability < 0.10 else 0.1)
        prosody_score = np.clip(0.6 * jitter_penalty + 0.4 * pitch_penalty, 0.02, 0.98)
        
        # Composite Multi-Branch Late Fusion
        composite_score = float(0.40 * acoustic_score + 0.35 * phase_anomaly + 0.25 * prosody_score)
        composite_score = np.clip(composite_score, 0.01, 0.99)
        
        breakdown = {
            "acoustic_score": round(float(acoustic_score), 4),
            "phase_score": round(float(phase_anomaly), 4),
            "prosody_score": round(float(prosody_score), 4),
            "jitter": round(float(jitter), 5),
            "shimmer": round(float(shimmer), 5),
            "f0_mean": round(float(f0_mean), 2),
            "gd_entropy": round(float(gd_entropy), 4)
        }
        
        return composite_score, breakdown

    def analyze_chunk(
        self,
        audio_chunk: np.ndarray,
        session_id: str = "default_session",
        sr: Optional[int] = None
    ) -> DetectionResult:
        """
        Analyzes a single streaming audio chunk (typically 1.0s to 3.0s).
        Returns a DetectionResult with instantaneous score, smoothed score, and risk action.
        """
        t0 = time.perf_counter()
        target_sr = sr or self.sr
        
        # 1. Voice Activity Detection
        active_audio, speech_ratio = self.vad.filter_active_speech(audio_chunk)
        is_speech = (speech_ratio >= 0.20) and (len(active_audio) >= int(target_sr * 0.3))
        
        session = self.get_or_create_session(session_id)
        
        if not is_speech:
            smoothed = session.update(raw_score=0.05, is_speech=False)
            latency = (time.perf_counter() - t0) * 1000.0
            return DetectionResult(
                score=0.05,
                smoothed_score=round(smoothed, 4),
                risk_level=RiskLevel.SAFE,
                action="SILENCE / INACTIVE AUDIO • CONTINUE MONITORING",
                is_speech=False,
                latency_ms=round(latency, 2),
                forensics={"note": "Non-speech or silence chunk"},
                timestamp=time.time()
            )
            
        # 2. Extract full forensic features
        feats = extract_full_features(active_audio, sr=target_sr)
        
        # 3. Model Inference
        if self.trained_classifier is not None:
            feat_vec = feats["feature_vector"].reshape(1, -1)
            raw_score = float(self.trained_classifier.predict_proba(feat_vec)[0, 1])
            composite_score, forensics = self._evaluate_forensics(feats)
            # Ensemble trained model with heuristic forensic checks
            final_raw = 0.7 * raw_score + 0.3 * composite_score
            forensics["classifier_raw_score"] = round(raw_score, 4)
        else:
            final_raw, forensics = self._evaluate_forensics(feats)
            
        # 4. Temporal Smoothing
        smoothed_score = session.update(raw_score=final_raw, is_speech=True)
        
        # 5. Risk Assessment & Decision Logic
        if smoothed_score >= self.high_threshold:
            risk = RiskLevel.CRITICAL
            action = "CRITICAL SPOOF DETECTED • TRIGGER USER IN-CALL WARNING / BLOCK TRANSACTION"
        elif smoothed_score >= self.low_threshold:
            risk = RiskLevel.SUSPICIOUS
            action = "SUSPICIOUS VOICE DETECTED • SILENT SECURITY TELEMETRY ALERT"
        else:
            risk = RiskLevel.SAFE
            action = "VERIFIED BONAFIDE HUMAN CALLER • ALLOW STREAM"
            
        latency = (time.perf_counter() - t0) * 1000.0
        
        return DetectionResult(
            score=round(final_raw, 4),
            smoothed_score=round(smoothed_score, 4),
            risk_level=risk,
            action=action,
            is_speech=True,
            latency_ms=round(latency, 2),
            forensics=forensics,
            timestamp=time.time()
        )
