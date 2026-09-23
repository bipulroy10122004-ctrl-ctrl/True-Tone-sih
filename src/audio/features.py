"""
True Tone SIH - Audio Feature Extraction Module
Linear Frequency Cepstral Coefficients (LFCC) + Delta + Delta-Delta
Optimized for Anti-Spoofing and Voice Deepfake Detection
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import scipy.fftpack
from scipy.signal import lfilter
import soundfile as sf

def pre_emphasis(signal, coeff=0.97):
    """Applies pre-emphasis filter to amplify high-frequency vocoder cues."""
    return lfilter([1, -coeff], [1], signal)

def linear_filter_bank(n_filters=40, n_fft=512, sample_rate=16000, low_freq=0, high_freq=8000):
    """
    Generates linearly spaced triangular filterbank across 0 - 8000 Hz.
    Unlike Mel-scale, linear spacing preserves high-frequency synthetic artifacts.
    """
    linear_freqs = np.linspace(low_freq, high_freq, n_filters + 2)
    fft_bins = np.floor((n_fft + 1) * linear_freqs / sample_rate).astype(int)
    filter_bank = np.zeros((n_filters, int(n_fft // 2 + 1)))

    for i in range(n_filters):
        f_left = fft_bins[i]
        f_center = fft_bins[i + 1]
        f_right = fft_bins[i + 2]

        denom_left = max(1, f_center - f_left)
        denom_right = max(1, f_right - f_center)

        for j in range(f_left, f_center):
            filter_bank[i, j] = (j - f_left) / denom_left
        for j in range(f_center, f_right):
            filter_bank[i, j] = (f_right - j) / denom_right

    return filter_bank

def compute_deltas(feat, order=2):
    """
    Computes 1st (velocity) and 2nd (acceleration) order temporal derivatives.
    Formula: d_t = (c_{t+1} - c_{t-1} + 2*(c_{t+2} - c_{t-2})) / 10
    """
    deltas = [feat]
    for _ in range(order):
        prev = deltas[-1]
        d = np.zeros_like(prev)
        t_len = prev.shape[0]
        for t in range(2, t_len - 2):
            d[t] = (prev[t + 1] - prev[t - 1] + 2.0 * (prev[t + 2] - prev[t - 2])) / 10.0
        deltas.append(d)
    return np.concatenate(deltas, axis=1)

def check_audio_energy(y: np.ndarray, amp_thresh: float = 0.005, rms_thresh: float = 0.001):
    """
    Computes RMS and Peak amplitude to identify ambient silence or background idle state.
    Returns (is_silent, max_amp, rms)
    """
    if y is None or len(y) == 0:
        return True, 0.0, 0.0
    max_amp = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    is_silent = bool(max_amp < amp_thresh and rms < rms_thresh)
    return is_silent, max_amp, rms

def extract_lfcc(file_path, max_frames=400, n_lfcc=20):
    """
    Reads an audio file, automatically resamples to 16kHz, applies peak normalization,
    and computes LFCC + Delta + Delta-Delta features.
    Output:
        Tensor of shape (max_frames, n_lfcc * 3) -> Default (400, 60)
    """
    import scipy.signal
    y, sr = sf.read(file_path)

    # Ensure 1D mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Automatically resample to 16,000 Hz using polyphase FIR filtering
    if sr != 16000 and len(y) > 0:
        from math import gcd
        g = gcd(int(sr), 16000)
        up = 16000 // g
        down = int(sr) // g
        y = scipy.signal.resample_poly(y, up, down).astype(np.float32)
        sr = 16000

    # Dynamic range normalization: gently normalize speech without amplifying quiet pauses or noise
    max_amp = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) > 0 else 0.0
    if max_amp >= 0.10 and rms >= 0.02:
        norm_gain = min(2.5, 0.90 / max_amp)
        y = (y * norm_gain).astype(np.float32)

    # Pre-emphasis
    y = pre_emphasis(y)

    frame_len = int(0.025 * sr)   # 25 ms frame length (400 samples at 16kHz)
    frame_step = int(0.010 * sr)  # 10 ms hop size (160 samples at 16kHz)
    n_fft = 512

    # Split into overlapping frames
    num_frames = max(1, 1 + int(np.floor((len(y) - frame_len) / frame_step)))
    frames = np.zeros((num_frames, frame_len))
    for i in range(num_frames):
        start = i * frame_step
        frames[i] = y[start : start + frame_len]

    # Hamming Window + Short-Time FFT
    window = np.hamming(frame_len)
    spec = np.abs(np.fft.rfft(frames * window, n=n_fft))

    # Apply Linear Filterbank
    fb = linear_filter_bank(n_filters=40, n_fft=n_fft, sample_rate=sr)
    energy = np.maximum(np.dot(spec, fb.T), 1e-12)
    log_energy = np.log(energy)

    # DCT Type-II to obtain Cepstral Coefficients
    lfcc_base = scipy.fftpack.dct(log_energy, type=2, axis=1, norm='ortho')[:, :n_lfcc]

    # Compute Delta and Delta-Delta (Velocity + Acceleration)
    full_feat = compute_deltas(lfcc_base, order=2)

    # Fixed-length alignment: repeat-pad or truncate to max_frames (4.0 seconds)
    t_frames = full_feat.shape[0]
    if t_frames < max_frames:
        pad_width = max_frames - t_frames
        full_feat = np.pad(full_feat, ((0, pad_width), (0, 0)), mode='edge')
    else:
        full_feat = full_feat[:max_frames, :]

    return full_feat.astype(np.float32)

