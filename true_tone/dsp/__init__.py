"""
True Tone DSP Module
Provides telephony audio codecs simulation, Voice Activity Detection (VAD),
Linear Frequency Cepstral Coefficients (LFCC), phase anomaly detection,
and prosody feature extractors.
"""

from .telecom_codecs import simulate_telephony_channel, encode_mulaw, decode_mulaw, encode_alaw, decode_alaw
from .vad import EnergyVAD
from .features import extract_lfcc, extract_phase_features, extract_prosody_features, extract_full_features

__all__ = [
    "simulate_telephony_channel",
    "encode_mulaw",
    "decode_mulaw",
    "encode_alaw",
    "decode_alaw",
    "EnergyVAD",
    "extract_lfcc",
    "extract_phase_features",
    "extract_prosody_features",
    "extract_full_features",
]
