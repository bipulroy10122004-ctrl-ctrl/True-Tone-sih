"""
True Tone SIH - AI Voice Detection & Automated IP Blocker API
Exposes endpoints for real-time audio spoof detection, IP threat mitigation,
incident audit trail, and interactive cyber defense testing.
"""
import os
import sys
import shutil
import json
import asyncio
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, Header, Request, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if os.path.join(SRC_DIR, "audio") not in sys.path:
    sys.path.insert(0, os.path.join(SRC_DIR, "audio"))
if os.path.join(SRC_DIR, "security") not in sys.path:
    sys.path.insert(0, os.path.join(SRC_DIR, "security"))

from audio.detector_service import get_inference_engine
from security.ip_blocker import get_ip_blocker
from security.middleware import IPBlockerSecurityMiddleware, extract_client_ip

app = FastAPI(
    title="True Tone SIH - AI Voice Detection & IP Blocker",
    description="Real-time Anti-Spoofing AI Audio Detector with Automated Network IP Mitigation",
    version="1.0.0"
)

# Attach Security Middleware
app.add_middleware(IPBlockerSecurityMiddleware)

# Static files and Templates
WEB_DIR = os.path.join(SRC_DIR, "web")
STATIC_DIR = os.path.join(WEB_DIR, "static")
TEMPLATES_DIR = os.path.join(WEB_DIR, "templates")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Pydantic Schemas
class UnblockRequest(BaseModel):
    ip: str
    reason: Optional[str] = "Manual Admin Override via Command Center"

class ManualBlockRequest(BaseModel):
    ip: str
    reason: str = "Admin Manual Quarantine"
    duration_seconds: Optional[int] = 3600
    is_permanent: bool = False

class SimulateCallRequest(BaseModel):
    caller_ip: str
    sample_type: str  # "spoof" or "bonafide"

# --- Web UI Routes ---

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/roadmap", response_class=FileResponse)
async def serve_roadmap():
    roadmap_path = os.path.join(PROJECT_ROOT, "TRUE_TONE_ROADMAP.html")
    if os.path.exists(roadmap_path):
        return FileResponse(roadmap_path)
    raise HTTPException(status_code=404, detail="Roadmap file not found")

@app.get("/documentation", response_class=FileResponse)
async def serve_documentation():
    doc_path = os.path.join(PROJECT_ROOT, "DOCUMENTATION.html")
    if os.path.exists(doc_path):
        return FileResponse(doc_path)
    raise HTTPException(status_code=404, detail="Documentation file not found")

# --- Audio Sample Serving ---

@app.get("/api/samples")
async def list_sample_audio():
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for fname in os.listdir(SAMPLES_DIR):
            if fname.endswith((".flac", ".wav", ".mp3")):
                is_spoof = "spoof" in fname.lower()
                samples.append({
                    "filename": fname,
                    "label": "AI Voice (Spoof)" if is_spoof else "Human Voice (Bonafide)",
                    "type": "spoof" if is_spoof else "bonafide",
                    "url": f"/api/samples/download/{fname}"
                })
    return {"samples": samples}

@app.get("/api/samples/download/{filename}")
async def download_sample(filename: str):
    file_path = os.path.join(SAMPLES_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Sample audio file not found")

# --- Core Detection & IP Blocker Endpoints ---

@app.post("/api/detect")
async def detect_audio_and_mitigate(
    request: Request,
    file: UploadFile = File(...),
    caller_ip: Optional[str] = Form(None),
    x_caller_ip: Optional[str] = Header(None)
):
    """
    Core AI Voice Detection & IP Firewall Endpoint.
    1. Extracts Caller IP.
    2. Checks if Caller IP is already banned.
    3. Runs AI ML Model inference on the audio.
    4. Automatically blocks the IP if deepfake/spoof detected.
    """
    blocker = get_ip_blocker()
    detector = get_inference_engine()

    # Determine originating IP
    target_ip = caller_ip or x_caller_ip or extract_client_ip(request)

    # 1. Check if IP is already banned
    is_banned, ban_rec = blocker.is_blocked(target_ip)
    if is_banned:
        return JSONResponse(
            status_code=403,
            content={
                "status": "FORBIDDEN",
                "action": "CONNECTION_TERMINATED",
                "message": f"Caller IP '{target_ip}' is already banned by True Tone Firewall.",
                "ip": target_ip,
                "block_details": ban_rec
            }
        )

    # 2. Read and analyze audio
    audio_bytes = await file.read()
    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty or corrupted.")

    try:
        detection_result = detector.predict(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio inference error: {str(e)}")

    # 3. Process AI detection result through IP Blocker
    is_simulated = bool(caller_ip or x_caller_ip)
    threat_response = blocker.process_ai_detection_result(
        ip=target_ip,
        detection_result=detection_result,
        simulate_ip=is_simulated
    )

    return {
        "status": "SUCCESS",
        "caller_ip": target_ip,
        "is_simulated_ip": is_simulated,
        "action_enforced": threat_response["action"],
        "is_ip_blocked": threat_response["blocked"],
        "reason": threat_response["reason"],
        "block_record": threat_response.get("block_record"),
        "ai_detection": detection_result
    }


@app.post("/api/simulate-call")
async def simulate_voice_call(req: SimulateCallRequest):
    """
    High-level test utility to simulate an incoming VoIP/Call session
    from a specified IP with sample spoof or bonafide audio.
    """
    blocker = get_ip_blocker()
    detector = get_inference_engine()

    # 1. Check if caller IP is already banned
    is_banned, ban_rec = blocker.is_blocked(req.caller_ip)
    if is_banned:
        return JSONResponse(
            status_code=403,
            content={
                "status": "FORBIDDEN",
                "action": "CALL_REJECTED",
                "message": f"Incoming call dropped: IP '{req.caller_ip}' is currently banned.",
                "caller_ip": req.caller_ip,
                "block_details": ban_rec
            }
        )

    # 2. Locate requested sample audio
    sample_file = "sample_spoof_ai_voice.flac" if req.sample_type == "spoof" else "sample_bonafide_human_voice.flac"
    sample_path = os.path.join(SAMPLES_DIR, sample_file)

    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Sample audio '{sample_file}' not found.")

    # 3. Predict with ML Model
    detection_result = detector.predict(sample_path)

    # 4. Enforce threat mitigation
    threat_response = blocker.process_ai_detection_result(
        ip=req.caller_ip,
        detection_result=detection_result,
        simulate_ip=True
    )

    return {
        "status": "SUCCESS",
        "caller_ip": req.caller_ip,
        "sample_type": req.sample_type,
        "action_enforced": threat_response["action"],
        "is_ip_blocked": threat_response["blocked"],
        "reason": threat_response["reason"],
        "block_record": threat_response.get("block_record"),
        "ai_detection": detection_result
    }

# --- Real-Time Live IP Call Gateway (WebSocket & Streaming) ---

@app.websocket("/ws/live-call")
async def websocket_live_call(websocket: WebSocket, caller_ip: Optional[str] = "198.51.100.55"):
    """
    Real-Time VoIP / IP Call Security Gateway.
    Monitors live audio chunks from the caller. If AI-generated deepfake voice
    is identified mid-call, the session is terminated and the caller IP is instantly banned.
    """
    await websocket.accept()
    blocker = get_ip_blocker()
    detector = get_inference_engine()

    # 1. Gateway Check: Is caller IP already blacklisted?
    is_banned, ban_rec = blocker.is_blocked(caller_ip)
    if is_banned:
        await websocket.send_json({
            "event": "CALL_REJECTED",
            "status": "CONNECTION_TERMINATED",
            "message": f"Incoming IP Call Dropped: Caller IP '{caller_ip}' is actively banned by True Tone Firewall.",
            "caller_ip": caller_ip,
            "block_record": ban_rec
        })
        await websocket.close(code=4403, reason="Caller IP is Banned")
        return

    # 2. Establish active monitoring session
    await websocket.send_json({
        "event": "CALL_CONNECTED",
        "status": "MONITORING_ACTIVE",
        "message": f"Call established with {caller_ip}. AI Deepfake Defense Gateway is actively analyzing acoustic phonemes.",
        "caller_ip": caller_ip
    })

    audio_buffer = bytearray()
    chunk_counter = 0
    consecutive_spoof_hits = 0

    try:
        while True:
            # Receive either binary audio chunk or JSON control payload
            message = await websocket.receive()
            if "bytes" in message and message["bytes"]:
                chunk = message["bytes"]

                # If self-contained WAV buffer from Web Audio API
                target_audio = None
                if len(chunk) > 1000 and chunk.startswith(b"RIFF"):
                    target_audio = bytes(chunk)
                else:
                    audio_buffer.extend(chunk)
                    if len(audio_buffer) >= 32000:
                        target_audio = bytes(audio_buffer)
                        audio_buffer.clear()

                if target_audio is not None:
                    try:
                        detection = detector.predict(target_audio)
                        chunk_counter += 1
                        latency_ms = detection.get("total_latency_ms", detection.get("latency_ms", 0.0))
                        print(f"[Live Call Gateway] Chunk #{chunk_counter} from {caller_ip} (bytes: {len(target_audio)}): "
                              f"Silent={detection.get('is_silent')}, Spoof={detection.get('is_spoof')}, Score={detection.get('spoof_percentage')}%, Latency={latency_ms:.1f}ms")
                        if detection.get("is_silent"):
                            # Ambient silence or inter-speech pause - decay suspicious frame count, do not ban or kill call!
                            consecutive_spoof_hits = max(0, consecutive_spoof_hits - 1)
                            await websocket.send_json({
                                "event": "AMBIENT_LISTENING",
                                "status": "LISTENING_ACTIVE",
                                "verdict": "AMBIENT SILENCE / WAITING FOR VOICE",
                                "spoof_percentage": 0.0,
                                "caller_ip": caller_ip,
                                "message": f"Ambient background / inter-speech pause on {caller_ip}. True Tone Guardian listening for voice...",
                                "detection": detection
                            })
                        elif detection["is_spoof"]:
                            consecutive_spoof_hits += 1
                            # Enforce threat neutralization ONLY on sustained synthetic speech (>= 2 consecutive speech frames)
                            # to prevent isolated ambient transient spikes from falsely dropping legitimate calls.
                            if consecutive_spoof_hits >= 2:
                                threat_resp = blocker.process_ai_detection_result(
                                    ip=caller_ip,
                                    detection_result=detection,
                                    simulate_ip=True
                                )
                                await websocket.send_json({
                                    "event": "CALL_TERMINATED_IP_BANNED",
                                    "status": "THREAT_NEUTRALIZED",
                                    "verdict": "SYNTHETIC AI VOICE / SPOOF",
                                    "spoof_percentage": detection["spoof_percentage"],
                                    "caller_ip": caller_ip,
                                    "message": f"DEEPFAKE ALERT! Sustained synthetic AI voice detected on {caller_ip} ({detection['spoof_percentage']}%). Call terminated & IP quarantined.",
                                    "detection": detection,
                                    "block_record": threat_resp.get("block_record")
                                })
                                await websocket.close(code=4403, reason="AI Deepfake Detected: Caller IP Banned")
                                break
                            else:
                                await websocket.send_json({
                                    "event": "THREAT_SUSPECTED",
                                    "status": "ANALYZING_PHONEMES",
                                    "verdict": "SUSPECTED SYNTHETIC PHONEMES",
                                    "spoof_percentage": detection["spoof_percentage"],
                                    "caller_ip": caller_ip,
                                    "message": f"Elevated synthetic acoustic signature on {caller_ip} ({detection['spoof_percentage']}%). Cross-verifying across adjacent speech frames...",
                                    "detection": detection
                                })
                        else:
                            consecutive_spoof_hits = 0
                            # Authentic voice: keep call alive
                            await websocket.send_json({
                                "event": "VOICE_VERIFIED_AUTHENTIC",
                                "status": "CALL_PERMITTED",
                                "verdict": "AUTHENTIC HUMAN / BONAFIDE",
                                "spoof_percentage": detection["spoof_percentage"],
                                "caller_ip": caller_ip,
                                "message": f"Caller acoustics verified as natural human vocal tract ({detection['spoof_percentage']}% risk). Call remains active.",
                                "detection": detection
                            })
                    except Exception as err:
                        print(f"[WebSocket] Error predicting audio chunk: {err}")
                        pass

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                cmd = data.get("command")
                if cmd == "simulate_chunk":
                    sample_type = data.get("sample_type", "spoof")
                    sample_file = "sample_spoof_ai_voice.flac" if sample_type == "spoof" else "sample_bonafide_human_voice.flac"
                    sample_path = os.path.join(SAMPLES_DIR, sample_file)

                    detection = detector.predict(sample_path)
                    threat_resp = blocker.process_ai_detection_result(
                        ip=caller_ip,
                        detection_result=detection,
                        simulate_ip=True
                    )

                    if detection["is_spoof"]:
                        await websocket.send_json({
                            "event": "CALL_TERMINATED_IP_BANNED",
                            "status": "THREAT_NEUTRALIZED",
                            "verdict": "SYNTHETIC AI VOICE / SPOOF",
                            "spoof_percentage": detection["spoof_percentage"],
                            "caller_ip": caller_ip,
                            "message": f"DEEPFAKE ALERT! Synthetic AI voice detected on {caller_ip}. Call terminated & IP quarantined.",
                            "detection": detection,
                            "block_record": threat_resp.get("block_record")
                        })
                        await websocket.close(code=4403, reason="AI Deepfake Detected: Caller IP Banned")
                        break
                    else:
                        await websocket.send_json({
                            "event": "VOICE_VERIFIED_AUTHENTIC",
                            "status": "CALL_PERMITTED",
                            "verdict": "AUTHENTIC HUMAN / BONAFIDE",
                            "spoof_percentage": detection["spoof_percentage"],
                            "caller_ip": caller_ip,
                            "message": f"Caller acoustics verified as natural human vocal tract. Call remains active.",
                            "detection": detection
                        })
                elif cmd == "end_call":
                    await websocket.send_json({"event": "CALL_ENDED_BY_USER", "message": "Call session ended gracefully."})
                    await websocket.close()
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close()
        except Exception:
            pass

# --- Firewall & Blocklist Administration ---

@app.get("/api/blocked-ips")
async def get_blocked_ips():
    blocker = get_ip_blocker()
    return {"blocked_ips": blocker.get_all_blocked_ips()}

@app.post("/api/unblock")
async def unblock_ip(req: UnblockRequest):
    blocker = get_ip_blocker()
    clean_ip = req.ip.strip()
    success = blocker.unblock_ip(clean_ip, reason=req.reason)
    return {
        "status": "SUCCESS",
        "ip": clean_ip,
        "message": f"IP {clean_ip} unblocked successfully."
    }

@app.post("/api/unblock-all")
async def unblock_all_ips():
    blocker = get_ip_blocker()
    count = blocker.unblock_all(reason="Admin Manual Clear All")
    return {
        "status": "SUCCESS",
        "unblocked_count": count,
        "message": f"Successfully unblocked all {count} IP addresses."
    }


@app.post("/api/block-manual")
async def block_ip_manual(req: ManualBlockRequest):
    blocker = get_ip_blocker()
    record = blocker.block_ip(
        ip=req.ip,
        reason=req.reason,
        threat_score=1.0,
        duration_seconds=req.duration_seconds,
        is_permanent=req.is_permanent
    )
    blocker.log_incident(
        ip=req.ip,
        event_type="MANUAL_ADMIN_BLOCK",
        spoof_probability=1.0,
        verdict="MANUAL_BAN",
        action_taken=f"Banned manually: {req.reason}",
        details={"block_record": record}
    )
    return {"status": "SUCCESS", "block_record": record}

@app.get("/api/incidents")
async def get_incidents(limit: int = 50):
    blocker = get_ip_blocker()
    return {"incidents": blocker.get_recent_incidents(limit=limit)}

@app.get("/api/stats")
async def get_stats():
    blocker = get_ip_blocker()
    detector = get_inference_engine()
    stats = blocker.get_telemetry_stats()
    stats.update({
        "ml_model_epoch": detector.epoch,
        "ml_model_eer": f"{detector.best_eer * 100:.4f}%",
        "ml_decision_threshold": detector.threshold,
        "compute_device": str(detector.device)
    })
    return stats


@app.get("/api/benchmark/wavefake")
async def get_wavefake_benchmark():
    summary_path = os.path.join(PROJECT_ROOT, "data", "wavefake_samples", "benchmark_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            data = json.load(f)
        return {"status": "SUCCESS", "benchmark": data}
    return {"status": "NOT_RUN", "message": "Benchmark summary not found."}

