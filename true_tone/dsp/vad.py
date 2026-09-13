"""
Voice Activity Detection (VAD) for Real-Time Streaming Audio
Filters out silence, dial-tones, line noise, and background static.
"""

import numpy as np


class EnergyVAD:
    """
    Lightweight, ultra-low-latency Voice Activity Detector with adaptive noise tracking.
    Ideal for real-time streaming (<0.5ms computation per frame).
    """
    def __init__(
        self,
        sr: int = 16000,
        frame_duration_ms: float = 25.0,
        energy_threshold: float = 0.015,
        zcr_threshold: float = 0.005,
        hangover_frames: int = 4
    ):
        self.sr = sr
        self.frame_size = int(sr * (frame_duration_ms / 1000.0))
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.hangover_frames = hangover_frames
        self._hangover_counter = 0
        self._noise_energy = 0.005
        
    def is_speech_frame(self, frame: np.ndarray) -> bool:
        """
        Determines if a single short frame contains voiced/unvoiced speech.
        """
        if len(frame) == 0:
            return False
            
        # Calculate Root-Mean-Square (RMS) Energy
        rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
        
        # Calculate Zero Crossing Rate (ZCR)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * len(frame))
        
        # Dynamic energy threshold relative to estimated background noise
        dynamic_threshold = max(self.energy_threshold, self._noise_energy * 2.5)
        
        is_speech = (rms > dynamic_threshold) and (zero_crossings > self.zcr_threshold)
        
        if is_speech:
            self._hangover_counter = self.hangover_frames
            return True
        elif self._hangover_counter > 0:
            self._hangover_counter -= 1
            return True
        else:
            # Update background noise estimate during quiet periods
            self._noise_energy = 0.95 * self._noise_energy + 0.05 * rms
            return False

    def filter_active_speech(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Processes a full audio segment and returns only active speech frames
        along with the speech ratio (active_speech_duration / total_duration).
        """
        if len(audio) < self.frame_size:
            return audio, 1.0
            
        num_frames = len(audio) // self.frame_size
        speech_mask = np.zeros(len(audio), dtype=bool)
        active_frame_count = 0
        
        for i in range(num_frames):
            start = i * self.frame_size
            end = start + self.frame_size
            frame = audio[start:end]
            
            if self.is_speech_frame(frame):
                speech_mask[start:end] = True
                active_frame_count += 1
                
        active_audio = audio[speech_mask]
        speech_ratio = float(active_frame_count) / max(1, num_frames)
        
        return active_audio, speech_ratio
