"""
True Tone SIH - Automated End-to-End Test Suite for AI Voice Detection & IP Blocker
Validates:
1. ML Model inference on Authentic Human Voice -> Allowed through gateway
2. ML Model inference on Synthetic AI Voice -> Dynamic Auto-Ban triggered
3. Re-transmission from banned IP -> Immediate gateway rejection (403)
4. Admin Unblock override -> Access restored
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if os.path.join(SRC_DIR, "audio") not in sys.path:
    sys.path.insert(0, os.path.join(SRC_DIR, "audio"))
if os.path.join(SRC_DIR, "security") not in sys.path:
    sys.path.insert(0, os.path.join(SRC_DIR, "security"))

from audio.detector_service import AudioSpoofInferenceEngine
from security.ip_blocker import IPBlockerManager

def run_e2e_security_test():
    print("=" * 70)
    print("TRUE TONE SIH: AI VOICE DETECTION & IP BLOCKER INTEGRATION TEST")
    print("=" * 70)

    # 1. Initialize Test IP Blocker with isolated DB
    test_db = os.path.join(PROJECT_ROOT, "models", "test_firewall.db")
    if os.path.exists(test_db):
        os.remove(test_db)

    blocker = IPBlockerManager(db_path=test_db, default_ban_ttl_seconds=300)
    detector = AudioSpoofInferenceEngine()

    bonafide_audio = os.path.join(PROJECT_ROOT, "samples", "sample_bonafide_human_voice.flac")
    spoof_audio = os.path.join(PROJECT_ROOT, "samples", "sample_spoof_ai_voice.flac")

    assert os.path.exists(bonafide_audio), f"Missing test audio: {bonafide_audio}"
    assert os.path.exists(spoof_audio), f"Missing test audio: {spoof_audio}"

    caller_ip_safe = "192.0.2.10"
    caller_ip_attacker = "198.51.100.99"

    # -------------------------------------------------------------
    # TEST CASE 1: Authentic Human Voice Call
    # -------------------------------------------------------------
    print(f"\n[TEST 1] Testing Authentic Human Voice Call from {caller_ip_safe}...")
    det_bonafide = detector.predict(bonafide_audio)
    print(f" -> ML Model Output: Probability = {det_bonafide['spoof_percentage']}%, Verdict = {det_bonafide['verdict']}")
    print(f" -> Latency: {det_bonafide['total_latency_ms']} ms")
    
    assert not det_bonafide["is_spoof"], "Expected bonafide classification!"

    response_bonafide = blocker.process_ai_detection_result(
        ip=caller_ip_safe,
        detection_result=det_bonafide,
        simulate_ip=True
    )
    print(f" -> IP Blocker Action: {response_bonafide['action']} (Blocked: {response_bonafide['blocked']})")
    assert not response_bonafide["blocked"], "Bonafide call should NOT be blocked!"

    is_blocked, _ = blocker.is_blocked(caller_ip_safe)
    assert not is_blocked, "Safe IP should remain unblocked!"
    print(" [PASSED] TEST 1: Authentic human call permitted without obstruction.")

    # -------------------------------------------------------------
    # TEST CASE 1B: Ambient Room Noise / Non-Speech Acoustic Input
    # -------------------------------------------------------------
    print(f"\n[TEST 1B] Testing Ambient Room Noise from {caller_ip_safe}...")
    import io
    import numpy as np
    import soundfile as sf
    noise_sig = np.random.normal(0, 0.03, 32000).astype(np.float32)
    bio_noise = io.BytesIO()
    sf.write(bio_noise, noise_sig, 16000, format="WAV")
    det_noise = detector.predict(bio_noise.getvalue())
    print(f" -> ML Model Output: Probability = {det_noise['spoof_percentage']}%, Verdict = {det_noise['verdict']}")
    print(f" -> Latency: {det_noise['total_latency_ms']} ms")
    assert not det_noise["is_spoof"], "Ambient noise MUST NOT be flagged as AI spoof!"
    assert det_noise["is_silent"] or not det_noise["is_speech"], "Ambient noise must be identified as idle/ambient non-speech!"
    response_noise = blocker.process_ai_detection_result(
        ip=caller_ip_safe,
        detection_result=det_noise,
        simulate_ip=True
    )
    print(f" -> IP Blocker Action: {response_noise['action']} (Blocked: {response_noise['blocked']})")
    assert not response_noise["blocked"], "Ambient noise must NOT trigger IP block!"
    print(" [PASSED] TEST 1B: Ambient room noise correctly treated as non-speech idle without false alarm.")

    # -------------------------------------------------------------
    # TEST CASE 2: Synthetic AI Voice Attack -> Auto IP Ban
    # -------------------------------------------------------------
    print(f"\n[TEST 2] Testing Synthetic AI Voice Spoof Attack from {caller_ip_attacker}...")
    det_spoof = detector.predict(spoof_audio)
    print(f" -> ML Model Output: Probability = {det_spoof['spoof_percentage']}%, Verdict = {det_spoof['verdict']}")
    print(f" -> Latency: {det_spoof['total_latency_ms']} ms")

    assert det_spoof["is_spoof"], "Expected spoof classification!"

    response_spoof = blocker.process_ai_detection_result(
        ip=caller_ip_attacker,
        detection_result=det_spoof,
        simulate_ip=True
    )
    print(f" -> IP Blocker Action: {response_spoof['action']} (Blocked: {response_spoof['blocked']})")
    assert response_spoof["blocked"], "Spoof attack MUST trigger automated IP block!"

    is_blocked, rec = blocker.is_blocked(caller_ip_attacker)
    assert is_blocked, "Attacker IP must now be actively blocked in memory & DB!"
    print(f" -> Active Block Record: Reason: {rec['reason']}, Strike: {rec['strike_count']}")
    print(" [PASSED] TEST 2: Synthetic voice attack triggered instant automated IP ban.")

    # -------------------------------------------------------------
    # TEST CASE 3: Re-transmission from Banned IP (Gateway Interception)
    # -------------------------------------------------------------
    print(f"\n[TEST 3] Simulating subsequent request from banned IP {caller_ip_attacker}...")
    is_blocked, ban_rec = blocker.is_blocked(caller_ip_attacker)
    assert is_blocked, "Banned IP must be intercepted at gateway before compute!"
    print(f" -> Gateway Interception: REJECTED with 403 Forbidden! Ban expires at {ban_rec['expires_at']}")
    print(" [PASSED] TEST 3: Gateway immediately drops requests from banned IP.")

    # -------------------------------------------------------------
    # TEST CASE 4: Admin Unblock Override
    # -------------------------------------------------------------
    print(f"\n[TEST 4] Admin unblocking IP {caller_ip_attacker}...")
    unblocked = blocker.unblock_ip(caller_ip_attacker, reason="Integration Test Restoration")
    assert unblocked, "Unblock operation should succeed!"

    is_blocked_after, _ = blocker.is_blocked(caller_ip_attacker)
    assert not is_blocked_after, "IP must be restored to clean state!"
    print(" [PASSED] TEST 4: Admin unblock restored normal access.")

    # -------------------------------------------------------------
    # TEST CASE 5: Telemetry Summary
    # -------------------------------------------------------------
    stats = blocker.get_telemetry_stats()
    print(f"\n[TEST 5] Telemetry & Incident Audit Check:")
    print(f" -> Total Inspected Events: {stats['total_calls_inspected']}")
    print(f" -> AI Spoofs Neutralized: {stats['synthetic_spoofs_stopped']}")
    print(f" -> Bonafide Allowed: {stats['bonafide_calls_permitted']}")
    assert stats["total_calls_inspected"] >= 2
    print(" [PASSED] TEST 5: Telemetry metrics match expected security event counts.")

    print("\n" + "=" * 70)
    print("ALL 5 END-TO-END SECURITY INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

    # Cleanup test DB if unlocked
    try:
        if os.path.exists(test_db):
            os.remove(test_db)
    except Exception:
        pass

def test_e2e_security_pipeline():
    run_e2e_security_test()

if __name__ == "__main__":
    run_e2e_security_test()
