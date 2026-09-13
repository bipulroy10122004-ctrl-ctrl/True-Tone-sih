"""
Generates realistic demo audio samples for offline testing:
1. Bonafide Human Call (with natural pitch drift, breathing, micro-jitter)
2. ElevenLabs Neural TTS Clone (with vocoder phase artifacts & robotic pitch stability)
3. RVC Voice Conversion Attack (with formant shifts and group delay anomaly)
All processed through simulated G.711 μ-law cellular telephony codecs.
"""

import os
import numpy as np
import soundfile as sf
from true_tone.dsp import simulate_telephony_channel


def generate_demo_samples(output_dir: str = "data/samples", sr: int = 16000):
    os.makedirs(output_dir, exist_ok=True)
    duration_sec = 4.0
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    
    # 1. --- BONAFIDE HUMAN CALL ---
    # Human vocal cord vibration with natural ~140Hz base F0, continuous prosodic drift, and breath micro-perturbations
    f0_drift = 140.0 + 12.0 * np.sin(2.0 * np.pi * 1.5 * t) + np.cumsum(np.random.normal(0, 0.08, len(t)))
    phase_human = np.cumsum(2.0 * np.pi * f0_drift / sr)
    
    # Natural formant spectrum
    sig_human = (
        np.sin(phase_human) +
        0.55 * np.sin(2 * phase_human) +
        0.30 * np.sin(3 * phase_human) +
        0.15 * np.sin(4 * phase_human)
    )
    # Add involuntary micro-jitter and breathing amplitude modulation
    breath_envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * 0.4 * t))
    sig_human *= (0.7 + 0.3 * breath_envelope)
    sig_human *= (1.0 + np.random.normal(0, 0.04, len(t)))
    
    # Pass through cellular telephone channel (G.711 mu-law, 300-3400Hz)
    audio_bonafide, _ = simulate_telephony_channel(
        sig_human, sr=sr, target_sr=sr, codec="mulaw", packet_loss_rate=0.01, snr_db=32.0
    )
    bonafide_path = os.path.join(output_dir, "bonafide_human_call.wav")
    sf.write(bonafide_path, audio_bonafide, sr)
    print(f"[Sample Generated] -> {bonafide_path}")

    # 2. --- ELEVENLABS NEURAL TTS CLONE ---
    # Autoregressive neural TTS: Flat pitch, near-zero micro-jitter, vocoder phase discontinuity
    f0_tts = 165.0  # Unnaturally constant pitch
    phase_tts = 2.0 * np.pi * f0_tts * t
    sig_tts = (
        np.sin(phase_tts) +
        0.48 * np.sin(2 * phase_tts + 0.8) +
        0.25 * np.sin(3 * phase_tts + 1.6)
    )
    # Vocoder high-frequency artifact & phase ripple
    sig_tts += 0.07 * np.sin(2.0 * np.pi * 3750.0 * t)
    audio_tts, _ = simulate_telephony_channel(
        sig_tts, sr=sr, target_sr=sr, codec="mulaw", packet_loss_rate=0.01, snr_db=32.0
    )
    tts_path = os.path.join(output_dir, "elevenlabs_ai_clone.wav")
    sf.write(tts_path, audio_tts, sr)
    print(f"[Sample Generated] -> {tts_path}")

    # 3. --- RVC VOICE CONVERSION ATTACK ---
    # Voice conversion: Preserves source timing but introduces phase mismatch and vocoder artifacts
    f0_vc = 150.0 + 8.0 * np.sin(2.0 * np.pi * 2.0 * t)
    phase_vc = np.cumsum(2.0 * np.pi * f0_vc / sr)
    sig_vc = (
        np.sin(phase_vc) +
        0.5 * np.sin(2 * phase_vc + np.pi / 2) +
        0.3 * np.sin(3 * phase_vc + np.pi / 4)
    )
    # Splicing / conversion artifact
    sig_vc += 0.05 * np.random.normal(0, 0.2, len(t))
    audio_vc, _ = simulate_telephony_channel(
        sig_vc, sr=sr, target_sr=sr, codec="mulaw", packet_loss_rate=0.01, snr_db=30.0
    )
    vc_path = os.path.join(output_dir, "rvc_voice_conversion.wav")
    sf.write(vc_path, audio_vc, sr)
    print(f"[Sample Generated] -> {vc_path}")

    return bonafide_path, tts_path, vc_path


if __name__ == "__main__":
    generate_demo_samples()
