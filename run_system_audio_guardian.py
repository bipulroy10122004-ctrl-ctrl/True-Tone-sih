"""
True Tone SIH - System Audio Live Call Defense Guardian
Command-line runner to monitor system audio during WhatsApp Web / VoIP calls,
analyze incoming caller voice in real-time, and trigger automatic firewall mitigation.
"""
import os
import sys
import time
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from audio.system_audio_analyzer import get_system_audio_analyzer, render_cli_hud
from security.ip_blocker import get_ip_blocker

def main():
    parser = argparse.ArgumentParser(
        description="True Tone SIH - System Audio Guardian (WhatsApp Web & IP Call Defense)"
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="198.51.100.110",
        help="IP address of the remote caller (default: 198.51.100.110)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Specific audio output device name or index (defaults to active system speaker/headphones)"
    )
    parser.add_argument(
        "--auto-ban",
        action="store_true",
        default=True,
        help="Automatically blacklist and ban caller IP if synthetic deepfake voice is detected"
    )
    args = parser.parse_args()

    blocker = get_ip_blocker()
    analyzer = get_system_audio_analyzer()

    print("=" * 75)
    print("  🛡️  TRUE TONE SIH: SYSTEM AUDIO REAL-TIME CALL GUARDIAN")
    print("  Deepfake Voice Interceptor for WhatsApp Web & Web-to-Web IP Calls")
    print("=" * 75)
    print(f"  Target Caller IP:      {args.ip}")
    print(f"  Firewall Auto-Ban:     {'ENABLED' if args.auto_ban else 'DISABLED'}")

    devices = analyzer.get_available_output_devices()
    print("\n  Available Audio Playback Devices:")
    for d in devices:
        def_tag = " [ACTIVE DEFAULT]" if d["is_default"] else ""
        print(f"    [{d['index']}] {d['name']}{def_tag}")
    print("-" * 75)

    # Check if IP is already banned
    is_blocked, ban_rec = blocker.is_blocked(args.ip)
    if is_blocked:
        print(f"\n⚠️ WARNING: Caller IP '{args.ip}' is ALREADY BANNED in the True Tone Firewall!")
        print(f"Reason: {ban_rec.get('reason')} (Banned at: {ban_rec.get('blocked_at')})")
        print("To unblock this IP before testing, run via dashboard or restart with a different IP.")
        print("-" * 75)

    consecutive_spoofs = [0]

    def on_audio_evaluated(data):
        render_cli_hud(data, caller_ip=args.ip)

        if data.get("is_speech") and data.get("is_spoof"):
            consecutive_spoofs[0] += 1
            if args.auto_ban and consecutive_spoofs[0] >= 1:
                # Trigger IP Firewall mitigation
                threat_res = blocker.process_ai_detection_result(
                    ip=args.ip,
                    detection_result=data,
                    simulate_ip=True
                )
                if threat_res.get("blocked"):
                    print(f"\n\n🚨 [FIREWALL ACTION ENFORCED] Caller IP '{args.ip}' has been QUARANTINED & BLOCKED!")
                    print(f"   Reason: {threat_res.get('reason')}")
                    print(f"   Action: {threat_res.get('action')}\n")
        elif data.get("is_speech") and not data.get("is_spoof"):
            consecutive_spoofs[0] = 0

    analyzer.register_callback(on_audio_evaluated)

    print("\n[STARTED] System Audio Loopback Tap is Active. Make or receive your WhatsApp Web call now.")
    print("Listening to system speaker output... (Press Ctrl+C to terminate session)\n")

    analyzer.start_capture(device_id=args.device)

    try:
        while analyzer.is_capturing:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nSession terminated by operator.")
    finally:
        analyzer.stop_capture()
        print("[SHUTDOWN] System audio listener stopped safely.")

if __name__ == "__main__":
    main()
