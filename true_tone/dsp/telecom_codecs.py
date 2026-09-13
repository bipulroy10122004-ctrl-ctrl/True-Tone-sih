"""
Telecom Codecs & Cellular Channel Simulation Module

Simulates real-world cellular and telephony constraints:
1. G.711 μ-law (PCMU) and A-law (PCMA) companding
2. 300 Hz - 3400 Hz Telephone Bandpass (narrowband POTS/2G)
3. 50 Hz - 7000 Hz Wideband Bandpass (VoLTE/AMR-WB)
4. Packet Loss Concealment (PLC) & Jitter Drop simulation
5. Additive cellular channel noise at specified SNR
"""

import numpy as np
from scipy import signal


def encode_mulaw(x: np.ndarray, mu: int = 255) -> np.ndarray:
    """
    Compress linear PCM [-1.0, 1.0] to 8-bit G.711 mu-law.
    """
    x = np.clip(x, -1.0, 1.0)
    # mu-law formula: sgn(x) * ln(1 + mu*|x|) / ln(1 + mu)
    companded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    # Quantize to 8-bit unsigned integer (0..255)
    quantized = np.round((companded + 1.0) / 2.0 * 255.0).astype(np.uint8)
    return quantized


def decode_mulaw(y: np.ndarray, mu: int = 255) -> np.ndarray:
    """
    Expand 8-bit G.711 mu-law back to linear float PCM [-1.0, 1.0].
    """
    norm = (y.astype(np.float32) / 255.0) * 2.0 - 1.0
    expanded = np.sign(norm) * (1.0 / mu) * ((1.0 + mu) ** np.abs(norm) - 1.0)
    return np.clip(expanded, -1.0, 1.0)


def encode_alaw(x: np.ndarray, a: float = 87.6) -> np.ndarray:
    """
    Compress linear PCM [-1.0, 1.0] to 8-bit G.711 A-law.
    """
    x = np.clip(x, -1.0, 1.0)
    abs_x = np.abs(x)
    sign_x = np.sign(x)
    
    companded = np.zeros_like(x)
    mask = abs_x < (1.0 / a)
    companded[mask] = (a * abs_x[mask]) / (1.0 + np.log(a))
    companded[~mask] = (1.0 + np.log(a * abs_x[~mask])) / (1.0 + np.log(a))
    companded = sign_x * companded
    
    quantized = np.round((companded + 1.0) / 2.0 * 255.0).astype(np.uint8)
    return quantized


def decode_alaw(y: np.ndarray, a: float = 87.6) -> np.ndarray:
    """
    Expand 8-bit G.711 A-law back to linear float PCM [-1.0, 1.0].
    """
    norm = (y.astype(np.float32) / 255.0) * 2.0 - 1.0
    abs_y = np.abs(norm)
    sign_y = np.sign(norm)
    
    expanded = np.zeros_like(norm)
    threshold = 1.0 / (1.0 + np.log(a))
    mask = abs_y < threshold
    expanded[mask] = (abs_y[mask] * (1.0 + np.log(a))) / a
    expanded[~mask] = np.exp(abs_y[~mask] * (1.0 + np.log(a)) - 1.0) / a
    expanded = sign_y * expanded
    return np.clip(expanded, -1.0, 1.0)


def apply_bandpass(audio: np.ndarray, sr: int, lowcut: float = 300.0, highcut: float = 3400.0, order: int = 5) -> np.ndarray:
    """
    Applies Butterworth bandpass filter simulating telephone acoustic transmission.
    For standard PSTN / 2G narrowband: 300 Hz - 3400 Hz.
    For VoLTE / AMR-WB: 50 Hz - 7000 Hz.
    """
    nyquist = 0.5 * sr
    low = max(10.0, lowcut) / nyquist
    high = min(nyquist - 10.0, highcut) / nyquist
    
    if low >= high:
        return audio
        
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.lfilter(b, a, audio)
    return filtered


def simulate_packet_loss(audio: np.ndarray, sr: int, packet_loss_rate: float = 0.03, packet_duration_ms: float = 20.0) -> np.ndarray:
    """
    Simulates RTP packet loss and basic Packet Loss Concealment (PLC - zero-fill or repeating).
    """
    if packet_loss_rate <= 0.0:
        return audio
        
    packet_samples = int(sr * (packet_duration_ms / 1000.0))
    total_packets = len(audio) // packet_samples
    if total_packets <= 0:
        return audio
        
    output = audio.copy()
    loss_mask = np.random.rand(total_packets) < packet_loss_rate
    
    for i in range(total_packets):
        if loss_mask[i]:
            start = i * packet_samples
            end = start + packet_samples
            if i > 0:
                # Basic concealment: attenuate previous packet
                prev_start = (i - 1) * packet_samples
                output[start:end] = output[prev_start:prev_start + packet_samples] * 0.4
            else:
                output[start:end] = 0.0
                
    return output


def add_channel_noise(audio: np.ndarray, snr_db: float = 25.0) -> np.ndarray:
    """
    Adds Gaussian channel noise at the target Signal-to-Noise Ratio (SNR in dB).
    """
    signal_power = np.mean(audio ** 2)
    if signal_power <= 1e-12:
        return audio
        
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), size=len(audio))
    return audio + noise


def simulate_telephony_channel(
    audio: np.ndarray,
    sr: int = 16000,
    target_sr: int = 8000,
    codec: str = "mulaw",
    packet_loss_rate: float = 0.02,
    snr_db: float = 28.0
) -> tuple[np.ndarray, int]:
    """
    Full end-to-end cellular call degradation pipeline:
    1. Resample to telephony sampling rate (8kHz for narrowband, 16kHz for VoLTE)
    2. Apply acoustic bandpass (300Hz-3400Hz for 8kHz, 50Hz-7000Hz for 16kHz)
    3. Simulate RTP packet loss & jitter
    4. Apply G.711 μ-law / A-law companding & quantization
    5. Add channel background noise
    """
    # 1. Resample if necessary
    if sr != target_sr:
        num_samples = int(len(audio) * float(target_sr) / sr)
        audio = signal.resample(audio, num_samples)
        sr = target_sr
        
    # 2. Bandpass filtering
    if target_sr == 8000:
        audio = apply_bandpass(audio, sr, lowcut=300.0, highcut=3400.0)
    else:
        audio = apply_bandpass(audio, sr, lowcut=50.0, highcut=7000.0)
        
    # 3. Packet loss
    audio = simulate_packet_loss(audio, sr, packet_loss_rate=packet_loss_rate)
    
    # 4. Codec quantization
    if codec == "mulaw":
        quantized = encode_mulaw(audio)
        audio = decode_mulaw(quantized)
    elif codec == "alaw":
        quantized = encode_alaw(audio)
        audio = decode_alaw(quantized)
        
    # 5. Add channel noise
    audio = add_channel_noise(audio, snr_db=snr_db)
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95
        
    return audio.astype(np.float32), sr
