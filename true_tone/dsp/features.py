"""
Forensic Audio Feature Extraction for Deepfake Voice Detection

Extracts:
1. Linear Frequency Cepstral Coefficients (LFCC) + Delta + Delta-Delta
2. Spectral Phase & Modified Group Delay (MGD) Anomaly Metrics
3. Prosodic & Biological Features (F0 contour, Jitter, Shimmer, HNR, Micro-pauses)
"""

import numpy as np
from scipy import signal
from scipy.fft import dct
import librosa


def linear_filterbank(sr: int, n_fft: int = 512, n_filters: int = 40, fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """
    Constructs a triangular filterbank with linearly spaced center frequencies.
    Key difference from Mel: maintains uniform frequency resolution across the full band,
    preserving vocoder periodic artifacts that Mel-scaling compresses away.
    """
    if fmax is None:
        fmax = sr / 2.0
        
    all_freqs = np.linspace(0, sr / 2.0, 1 + n_fft // 2)
    filter_freqs = np.linspace(fmin, fmax, n_filters + 2)
    
    bank = np.zeros((n_filters, 1 + n_fft // 2))
    for i in range(n_filters):
        left, center, right = filter_freqs[i], filter_freqs[i + 1], filter_freqs[i + 2]
        
        # Upward slope
        up_mask = (all_freqs >= left) & (all_freqs <= center)
        if center > left:
            bank[i, up_mask] = (all_freqs[up_mask] - left) / (center - left)
            
        # Downward slope
        down_mask = (all_freqs >= center) & (all_freqs <= right)
        if right > center:
            bank[i, down_mask] = (right - all_freqs[down_mask]) / (right - center)
            
    return bank


def extract_lfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 160,
    n_filters: int = 40,
    n_ceps: int = 20,
    include_deltas: bool = True
) -> np.ndarray:
    """
    Extracts Linear Frequency Cepstral Coefficients (LFCC) with Delta & Delta-Delta.
    Returns: (n_ceps * 3, n_frames) matrix.
    """
    if len(audio) < n_fft:
        # Pad with zeros if short
        audio = np.pad(audio, (0, n_fft - len(audio)))
        
    # 1. Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hamming')
    power_spec = np.abs(stft) ** 2
    
    # 2. Linear Filterbank
    fb = linear_filterbank(sr=sr, n_fft=n_fft, n_filters=n_filters, fmin=0.0, fmax=sr / 2.0)
    filter_energies = np.dot(fb, power_spec)
    filter_energies = np.maximum(filter_energies, 1e-12)
    log_energies = np.log10(filter_energies)
    
    # 3. Discrete Cosine Transform (DCT-II)
    cepstra = dct(log_energies, type=2, axis=0, norm='ortho')[:n_ceps]
    
    if not include_deltas:
        return cepstra
        
    # 4. Compute Delta and Delta-Delta (Velocity & Acceleration)
    delta1 = librosa.feature.delta(cepstra, order=1)
    delta2 = librosa.feature.delta(cepstra, order=2)
    
    lfcc_full = np.vstack([cepstra, delta1, delta2])
    return lfcc_full


def extract_phase_features(audio: np.ndarray, sr: int = 16000, n_fft: int = 512, hop_length: int = 160) -> dict:
    """
    Forensic Phase Analysis: Neural vocoders (HiFi-GAN, WaveGlow, Diffusion) produce
    phase inconsistencies across time frames compared to human vocal cords.
    Calculates:
    - Phase Derivative / Instantaneous Frequency (IF) variance
    - Modified Group Delay (MGD) entropy
    - Spectral phase coherence
    """
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
        
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, window='hann')
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Instantaneous frequency deviation (phase unwrapping across time)
    unwrapped_phase = np.unwrap(phase, axis=1)
    phase_derivative = np.diff(unwrapped_phase, axis=1)
    
    if phase_derivative.size > 0:
        if_mean = float(np.mean(phase_derivative))
        if_std = float(np.std(phase_derivative))
        if_skew = float(np.mean(((phase_derivative - if_mean) / (if_std + 1e-8)) ** 3))
    else:
        if_mean, if_std, if_skew = 0.0, 0.0, 0.0
        
    # Group delay across frequency axis
    unwrapped_freq = np.unwrap(phase, axis=0)
    group_delay = -np.diff(unwrapped_freq, axis=0)
    
    if group_delay.size > 0:
        gd_entropy = float(-np.sum(np.abs(group_delay) * np.log(np.abs(group_delay) + 1e-10)) / (group_delay.size + 1e-8))
        gd_variance = float(np.var(group_delay))
    else:
        gd_entropy, gd_variance = 0.0, 0.0
        
    # High-frequency phase noise ratio
    half_bin = n_fft // 4
    high_freq_mag = np.mean(magnitude[half_bin:, :])
    low_freq_mag = np.mean(magnitude[:half_bin, :]) + 1e-8
    spectral_tilt_ratio = float(high_freq_mag / low_freq_mag)
    
    return {
        "if_mean": if_mean,
        "if_std": if_std,
        "if_skew": if_skew,
        "gd_entropy": gd_entropy,
        "gd_variance": gd_variance,
        "spectral_tilt_ratio": spectral_tilt_ratio
    }


def extract_prosody_features(audio: np.ndarray, sr: int = 16000) -> dict:
    """
    Biological Prosody & Vocal Tract Micro-Perturbation Extraction:
    - Pitch (F0) tracking & contour statistics
    - Jitter: cycle-to-cycle perturbation in pitch period
    - Shimmer: cycle-to-cycle perturbation in amplitude
    - Harmonicity (Harmonics-to-Noise Ratio)
    - Speech rhythm and micro-pause regularity
    """
    if len(audio) < sr * 0.2:  # Less than 200ms
        return {
            "f0_mean": 0.0, "f0_std": 0.0, "f0_range": 0.0,
            "jitter": 0.0, "shimmer": 0.0, "hnr": 0.0, "pause_ratio": 0.0
        }
        
    # Pitch extraction using librosa pyin (probabilistic YIN)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz
        fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
        sr=sr
    )
    
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
    
    if len(voiced_f0) > 3:
        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_range = float(np.ptp(voiced_f0))
        
        # Jitter: Relative average perturbation in consecutive pitch periods
        periods = 1.0 / (voiced_f0 + 1e-8)
        period_diffs = np.abs(np.diff(periods))
        jitter = float(np.mean(period_diffs) / (np.mean(periods) + 1e-8))
    else:
        f0_mean, f0_std, f0_range, jitter = 0.0, 0.0, 0.0, 0.0
        
    # Shimmer: Amplitude perturbation on short-time RMS
    frame_length = int(sr * 0.025)
    hop_length = int(sr * 0.010)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    
    if len(rms) > 3:
        rms_diffs = np.abs(np.diff(rms))
        shimmer = float(np.mean(rms_diffs) / (np.mean(rms) + 1e-8))
    else:
        shimmer = 0.0
        
    # Harmonic-to-Noise Ratio (HNR) approximation using harmonic-percussive separation
    y_harm, y_perc = librosa.effects.hpss(audio)
    harm_energy = np.mean(y_harm ** 2) + 1e-12
    perc_energy = np.mean(y_perc ** 2) + 1e-12
    hnr = float(10.0 * np.log10(harm_energy / perc_energy))
    
    # Pause ratio (fraction of unvoiced/quiet frames in active speech)
    pause_ratio = float(np.mean(rms < (np.mean(rms) * 0.2))) if len(rms) > 0 else 0.0
    
    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_range": f0_range,
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr": hnr,
        "pause_ratio": pause_ratio
    }


def extract_full_features(audio: np.ndarray, sr: int = 16000) -> dict:
    """
    Aggregates LFCC statistics, phase descriptors, and prosody metrics into
    a single flat numerical feature vector (ideal for sub-20ms inference).
    """
    # 1. LFCC
    lfcc = extract_lfcc(audio, sr=sr, n_ceps=20, include_deltas=True)  # (60, frames)
    lfcc_mean = np.mean(lfcc, axis=1)  # 60 dims
    lfcc_std = np.std(lfcc, axis=1)    # 60 dims
    
    # 2. Phase features
    phase_feats = extract_phase_features(audio, sr=sr)
    
    # 3. Prosody features
    prosody_feats = extract_prosody_features(audio, sr=sr)
    
    # Assemble structured 1D feature vector
    flat_vector = np.concatenate([
        lfcc_mean,
        lfcc_std,
        np.array([
            phase_feats["if_mean"],
            phase_feats["if_std"],
            phase_feats["if_skew"],
            phase_feats["gd_entropy"],
            phase_feats["gd_variance"],
            phase_feats["spectral_tilt_ratio"],
            prosody_feats["f0_mean"],
            prosody_feats["f0_std"],
            prosody_feats["f0_range"],
            prosody_feats["jitter"],
            prosody_feats["shimmer"],
            prosody_feats["hnr"],
            prosody_feats["pause_ratio"]
        ], dtype=np.float32)
    ])
    
    return {
        "feature_vector": flat_vector.astype(np.float32),
        "lfcc_matrix": lfcc.astype(np.float32),
        "phase_metrics": phase_feats,
        "prosody_metrics": prosody_feats
    }
