"""
True Tone SIH - System Audio Loopback Analyzer
Directly accesses Windows system audio (WASAPI loopback) to capture incoming caller
voice from web-to-web IP calls (WhatsApp Web, Google Meet, Teams, Discord, Zoom)
and analyzes it in real-time using the trained PyTorch anti-spoofing detector.
"""
import os
import sys
import time
import io
import threading
import queue
from typing import Optional, Callable, Dict, Any, List
import numpy as np
import soundfile as sf
import soundcard as sc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from audio.detector_service import get_inference_engine, detect_voice_activity

class SystemAudioAnalyzer:
    """
    Real-Time System Audio Loopback Capture & Deepfake Detection Engine.
    Taps into the active audio output stream (what the user hears from the caller)
    via Windows WASAPI Loopback without requiring stereo mix or virtual cables.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.engine = get_inference_engine(model_path=model_path) if model_path else get_inference_engine()
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._lock = threading.Lock()
        
        # Telemetry & Status
        self.active_device_name = "Default System Audio"
        self.active_device_id = None
        self.latest_result: Dict[str, Any] = {
            "status": "STANDBY",
            "is_capturing": False,
            "is_speech": False,
            "is_spoof": False,
            "spoof_percentage": 0.0,
            "verdict": "STANDBY / WAITING TO START",
            "risk_level": "IDLE",
            "audio_level_rms": 0.0,
            "max_amplitude": 0.0,
            "total_latency_ms": 0.0,
            "timestamp": time.time()
        }

    @staticmethod
    def get_available_output_devices() -> List[Dict[str, Any]]:
        """
        Enumerates all physical and virtual audio output devices on the system
        that can be captured via loopback.
        """
        devices = []
        try:
            default_sp = sc.default_speaker()
            default_id = default_sp.id if default_sp else None
        except Exception:
            default_id = None

        try:
            speakers = sc.all_speakers()
            for idx, sp in enumerate(speakers):
                is_def = (sp.id == default_id)
                devices.append({
                    "index": idx,
                    "id": sp.id,
                    "name": sp.name,
                    "is_default": is_def,
                    "channels": getattr(sp, "channels", 2)
                })
        except Exception as e:
            print(f"[SystemAudioAnalyzer] Error querying speakers: {e}")

        return devices

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Registers a listener for live detection results."""
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Unregisters a listener."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def _notify_callbacks(self, data: Dict[str, Any]):
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(data)
            except Exception as e:
                print(f"[SystemAudioAnalyzer] Callback error: {e}")

    def start_capture(
        self,
        device_id: Optional[str] = None,
        chunk_seconds: float = 1.5,
        overlap_seconds: float = 0.5,
        sample_rate: int = 16000
    ) -> bool:
        """
        Starts the continuous system audio capture and analysis loop in a background thread.
        """
        if self._is_running:
            print("[SystemAudioAnalyzer] Capture is already active.")
            return True

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_worker,
            args=(device_id, chunk_seconds, overlap_seconds, sample_rate),
            name="SystemAudioLoopbackThread",
            daemon=True
        )
        self._thread.start()
        self._is_running = True
        return True

    def stop_capture(self, timeout: float = 3.0):
        """Stops the audio capture loop cleanly."""
        if not self._is_running:
            return

        print("[SystemAudioAnalyzer] Stopping system audio capture...")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._is_running = False

        with self._lock:
            self.latest_result["is_capturing"] = False
            self.latest_result["status"] = "STOPPED"
            self.latest_result["verdict"] = "SYSTEM AUDIO MONITOR STOPPED"

    @property
    def is_capturing(self) -> bool:
        return self._is_running

    def get_latest_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.latest_result)

    def _capture_worker(
        self,
        device_id: Optional[str],
        chunk_seconds: float,
        overlap_seconds: float,
        sample_rate: int
    ):
        """Worker thread that executes the continuous WASAPI loopback capture & ML inference."""
        target_sp = None
        speakers = sc.all_speakers()

        if device_id:
            for sp in speakers:
                if sp.id == device_id or sp.name == device_id:
                    target_sp = sp
                    break

        if target_sp is None:
            target_sp = sc.default_speaker()

        self.active_device_name = target_sp.name
        self.active_device_id = target_sp.id

        print(f"[SystemAudioAnalyzer] Binding to loopback on: '{target_sp.name}' ({sample_rate} Hz, mono)")

        try:
            loopback_mic = sc.get_microphone(id=target_sp.id, include_loopback=True)
        except Exception as e:
            err_msg = f"Failed to acquire loopback mic for {target_sp.name}: {e}"
            print(f"[SystemAudioAnalyzer] ERROR: {err_msg}")
            with self._lock:
                self.latest_result["status"] = "ERROR"
                self.latest_result["error"] = err_msg
            self._is_running = False
            return

        chunk_samples = int(chunk_seconds * sample_rate)
        # Step size in samples between sliding evaluation windows
        step_seconds = max(0.2, chunk_seconds - overlap_seconds)
        step_samples = int(step_seconds * sample_rate)

        rolling_buffer = np.zeros(0, dtype=np.float32)
        chunk_counter = 0

        with self._lock:
            self.latest_result["is_capturing"] = True
            self.latest_result["status"] = "MONITORING_ACTIVE"
            self.latest_result["device_name"] = target_sp.name

        try:
            # We record in small increments (e.g. 0.3s) and maintain a rolling window of chunk_samples
            step_record_frames = step_samples
            with loopback_mic.recorder(samplerate=sample_rate, channels=1) as rec:
                print(f"[SystemAudioAnalyzer] Live loopback active. Monitoring caller voice...")

                while not self._stop_event.is_set():
                    raw_data = rec.record(numframes=step_record_frames)
                    if raw_data is None or len(raw_data) == 0:
                        time.sleep(0.05)
                        continue

                    raw_flat = raw_data.flatten()
                    rolling_buffer = np.concatenate([rolling_buffer, raw_flat])

                    # Wait until we have accumulated at least a full chunk window (e.g. 1.5s)
                    if len(rolling_buffer) < chunk_samples:
                        continue

                    # Slice the current chunk window and retain the overlap tail in rolling_buffer
                    eval_chunk = rolling_buffer[:chunk_samples]
                    rolling_buffer = rolling_buffer[step_samples:]

                    chunk_counter += 1
                    t0 = time.perf_counter()

                    # Quick acoustic telemetry
                    max_amp = float(np.max(np.abs(eval_chunk)))
                    rms = float(np.sqrt(np.mean(eval_chunk ** 2)))

                    # 1. Voice Activity Detection (VAD)
                    is_speech, vad_details = detect_voice_activity(eval_chunk)

                    if not is_speech:
                        # Caller is not speaking (silence or ambient background)
                        metric_data = {
                            "status": "AMBIENT_LISTENING",
                            "chunk_id": chunk_counter,
                            "is_capturing": True,
                            "is_speech": False,
                            "is_silent": True,
                            "is_spoof": False,
                            "spoof_percentage": 0.0,
                            "confidence_score": 1.0,
                            "verdict": "AMBIENT SILENCE / LISTENING FOR CALLER",
                            "risk_level": "IDLE",
                            "audio_level_rms": round(rms, 5),
                            "max_amplitude": round(max_amp, 5),
                            "total_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                            "vad_metrics": vad_details,
                            "device_name": target_sp.name,
                            "timestamp": time.time()
                        }
                    else:
                        # 2. Caller Voice Detected -> Run Deepfake Anti-Spoofing Model
                        bio = io.BytesIO()
                        sf.write(bio, eval_chunk, sample_rate, format='WAV')
                        pred = self.engine.predict(bio.getvalue())

                        inference_latency = (time.perf_counter() - t0) * 1000.0
                        metric_data = {
                            "status": "AI_INFERENCE_COMPLETE",
                            "chunk_id": chunk_counter,
                            "is_capturing": True,
                            "is_speech": True,
                            "is_silent": False,
                            "is_spoof": pred["is_spoof"],
                            "spoof_percentage": pred["spoof_percentage"],
                            "confidence_score": pred["confidence_score"],
                            "verdict": pred["verdict"],
                            "risk_level": pred["risk_level"],
                            "audio_level_rms": round(rms, 5),
                            "max_amplitude": round(max_amp, 5),
                            "total_latency_ms": round(inference_latency, 2),
                            "feature_latency_ms": pred.get("feature_latency_ms", 0.0),
                            "forward_latency_ms": pred.get("forward_latency_ms", 0.0),
                            "vad_metrics": vad_details,
                            "device_name": target_sp.name,
                            "timestamp": time.time()
                        }

                    with self._lock:
                        self.latest_result = metric_data

                    self._notify_callbacks(metric_data)

        except Exception as e:
            print(f"[SystemAudioAnalyzer] Loopback error encountered: {e}")
            with self._lock:
                self.latest_result["status"] = "ERROR"
                self.latest_result["error"] = str(e)
                self.latest_result["is_capturing"] = False
        finally:
            self._is_running = False
            print("[SystemAudioAnalyzer] Loopback audio capture terminated.")


# Global Singleton for System Audio Analyzer
_system_analyzer_instance: Optional[SystemAudioAnalyzer] = None

def get_system_audio_analyzer() -> SystemAudioAnalyzer:
    global _system_analyzer_instance
    if _system_analyzer_instance is None:
        _system_analyzer_instance = SystemAudioAnalyzer()
    return _system_analyzer_instance


def render_cli_hud(data: Dict[str, Any], caller_ip: str = "Unknown"):
    """Renders a real-time cybersecurity HUD dashboard in the console."""
    is_speech = data.get("is_speech", False)
    is_spoof = data.get("is_spoof", False)
    pct = data.get("spoof_percentage", 0.0)
    verdict = data.get("verdict", "LISTENING")
    risk = data.get("risk_level", "IDLE")
    rms = data.get("audio_level_rms", 0.0)
    latency = data.get("total_latency_ms", 0.0)
    chunk_id = data.get("chunk_id", 0)

    # ANSI Color codes
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    # Volume VU bar (20 chars)
    vol_bars = min(20, int(rms * 120))
    vu_meter = "█" * vol_bars + "░" * (20 - vol_bars)

    if not is_speech:
        status_badge = f"{DIM}[ ⏸️  AMBIENT SILENCE / LISTENING ]{RESET}"
        risk_color = DIM
    elif is_spoof:
        status_badge = f"{RED}{BOLD}[ 🚨 DEEPFAKE SYNTHETIC AI VOICE DETECTED ]{RESET}"
        risk_color = RED
    else:
        status_badge = f"{GREEN}{BOLD}[ ✅ AUTHENTIC HUMAN VOICE VERIFIED ]{RESET}"
        risk_color = GREEN

    # Format HUD row
    print(f"\r{CYAN}Chunk #{chunk_id:04d}{RESET} | "
          f"Vol: [{CYAN}{vu_meter}{RESET}] {rms:.4f} | "
          f"{status_badge} | "
          f"Risk: {risk_color}{risk:<9}{RESET} ({pct:5.1f}%) | "
          f"{DIM}{latency:4.1f}ms{RESET}", end="", flush=True)

    if is_speech and is_spoof:
        print()  # Print newline to preserve critical alert
        print(f"  {RED}{BOLD}>>> ALERT: Caller on {caller_ip} is using Synthetic AI Voice! Verdict: {verdict}{RESET}")
    elif is_speech and not is_spoof and chunk_id % 10 == 0:
        print()  # Periodic confirmation newline


def run_interactive_monitor(caller_ip: str = "198.51.100.110", device_id: Optional[str] = None):
    """Runs interactive terminal monitoring loop."""
    print("=" * 75)
    print("  TRUE TONE SIH: SYSTEM AUDIO REAL-TIME CALL ANALYZER")
    print("  Capturing Remote Caller Audio from WhatsApp Web / IP Call Streams")
    print("=" * 75)

    analyzer = get_system_audio_analyzer()
    devices = analyzer.get_available_output_devices()

    print("\nDetected System Audio Output Devices:")
    for d in devices:
        def_flag = " (DEFAULT)" if d["is_default"] else ""
        print(f"  [{d['index']}] {d['name']}{def_flag}")

    analyzer.register_callback(lambda d: render_cli_hud(d, caller_ip=caller_ip))

    print(f"\nTarget Caller IP: {caller_ip}")
    print("Starting Windows WASAPI Loopback Capture (Press Ctrl+C to stop)...")
    print("-" * 75)

    analyzer.start_capture(device_id=device_id)

    try:
        while analyzer.is_capturing:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nUser interrupted monitoring. Shutting down...")
    finally:
        analyzer.stop_capture()
        print("\n[OK] System audio monitor finished cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="True Tone Real-time System Audio Deepfake Call Analyzer")
    parser.add_argument("--ip", type=str, default="198.51.100.110", help="Caller IP address to associate with call session")
    parser.add_argument("--device", type=str, default=None, help="Specific audio output device name or index")
    args = parser.parse_args()

    run_interactive_monitor(caller_ip=args.ip, device_id=args.device)
