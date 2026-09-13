"""
FastAPI Real-Time Telephony Deepfake Detection Server ("True Tone")

Endpoints:
- GET  /health: Health check and engine telemetry
- POST /v1/analyze/file: Batch audio file upload and forensic report
- WS   /v1/stream/analyze: High-speed WebSocket streaming for live SIM calls
- WS   /v1/stream/twilio: Live Twilio Media Streams interception endpoint (G.711 μ-law)
- POST /v1/telephony/twilio/voice: TwiML webhook to tap cellular phone calls
- GET  /: Real-time browser-based forensic dashboard & audio inspector
"""

import os
import io
import json
import base64
import time
from typing import Optional
import numpy as np
import soundfile as sf
import librosa

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from true_tone.models import TrueToneDetector, RiskLevel
from true_tone.dsp import decode_mulaw, simulate_telephony_channel

app = FastAPI(
    title="True Tone - SIM-to-SIM Call Deepfake Voice Detection API",
    description="Real-time forensic verification engine for cellular telephony streams.",
    version="1.0.0"
)

# Enable CORS for web clients and dashboard integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector with trained model if exists
MODEL_PATH = "models/true_tone_classifier.joblib"
detector = TrueToneDetector(
    model_path=MODEL_PATH if os.path.exists(MODEL_PATH) else None,
    sr=16000,
    low_threshold=0.40,
    high_threshold=0.70
)


@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "engine": "True Tone Forensic Engine v1.0",
        "model_loaded": detector.trained_classifier is not None,
        "sample_rate": detector.sr,
        "thresholds": {
            "low_threshold": detector.low_threshold,
            "high_threshold": detector.high_threshold
        },
        "active_call_sessions": len(detector.sessions)
    }


@app.post("/v1/analyze/file")
async def analyze_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form("batch_upload"),
    simulate_telecom: bool = Form(True)
):
    """
    Analyzes an uploaded audio recording (WAV, MP3, FLAC, OGG).
    Optionally applies cellular telephony channel degradation (G.711/AMR).
    """
    try:
        contents = await file.read()
        audio_bytes = io.BytesIO(contents)
        audio, sr = sf.read(audio_bytes)
        
        # Convert stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        audio = audio.astype(np.float32)
        
        # Resample to 16kHz if different
        if sr != detector.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=detector.sr)
            sr = detector.sr
            
        if simulate_telecom:
            audio, sr = simulate_telephony_channel(audio, sr=sr, target_sr=sr, codec="mulaw")
            
        # Analyze full audio or segment into 2.0s chunks
        chunk_size = int(sr * 2.0)
        results = []
        
        if len(audio) <= chunk_size:
            res = detector.analyze_chunk(audio, session_id=session_id, sr=sr)
            results.append(res)
        else:
            num_chunks = len(audio) // chunk_size
            for i in range(num_chunks):
                chunk = audio[i * chunk_size : (i + 1) * chunk_size]
                res = detector.analyze_chunk(chunk, session_id=session_id, sr=sr)
                results.append(res)
                
        final_res = results[-1]
        
        return {
            "filename": file.filename,
            "duration_sec": round(len(audio) / float(sr), 2),
            "simulated_cellular_channel": simulate_telecom,
            "overall_risk_level": final_res.risk_level.value,
            "deepfake_probability": final_res.smoothed_score,
            "recommended_action": final_res.action,
            "total_chunks_processed": len(results),
            "forensic_breakdown": final_res.forensics,
            "chunk_timeline": [
                {
                    "chunk_index": idx + 1,
                    "instantaneous_score": r.score,
                    "smoothed_score": r.smoothed_score,
                    "risk_level": r.risk_level.value,
                    "is_speech": r.is_speech,
                    "latency_ms": r.latency_ms
                }
                for idx, r in enumerate(results)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing audio: {str(e)}")


@app.websocket("/v1/stream/analyze")
async def websocket_stream_analyze(websocket: WebSocket):
    """
    Real-Time Audio Stream Inspection WebSocket.
    Accepts:
    1. JSON init: {"type": "init", "session_id": "call_123", "sr": 16000}
    2. Binary PCM 16-bit / float32 audio bytes OR JSON {"type": "audio", "data": "<base64>"}
    Emits real-time verification scores and risk alerts.
    """
    await websocket.accept()
    session_id = f"stream_{int(time.time() * 1000)}"
    sr = 16000
    audio_buffer = np.array([], dtype=np.float32)
    chunk_samples = int(sr * 1.5)  # 1.5 second analysis window
    hop_samples = int(sr * 0.5)    # 0.5 second hop for low latency updates
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                # Assume 16-bit signed PCM
                chunk_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, chunk_np])
            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                msg_type = data.get("type", "")
                
                if msg_type == "init":
                    session_id = data.get("session_id", session_id)
                    sr = data.get("sr", sr)
                    chunk_samples = int(sr * 1.5)
                    hop_samples = int(sr * 0.5)
                    await websocket.send_json({"type": "ready", "session_id": session_id, "sr": sr})
                    continue
                elif msg_type == "audio":
                    b64_str = data.get("data", "")
                    raw_bytes = base64.b64decode(b64_str)
                    chunk_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_buffer = np.concatenate([audio_buffer, chunk_np])
                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                    
            # When buffer reaches chunk_samples, run detector
            while len(audio_buffer) >= chunk_samples:
                analysis_window = audio_buffer[:chunk_samples]
                audio_buffer = audio_buffer[hop_samples:]  # advance by hop
                
                res = detector.analyze_chunk(analysis_window, session_id=session_id, sr=sr)
                
                await websocket.send_json({
                    "type": "result",
                    "session_id": session_id,
                    "score": res.score,
                    "smoothed_score": res.smoothed_score,
                    "risk_level": res.risk_level.value,
                    "action": res.action,
                    "is_speech": res.is_speech,
                    "latency_ms": res.latency_ms,
                    "forensics": res.forensics,
                    "timestamp": res.timestamp
                })
    except WebSocketDisconnect:
        detector.clear_session(session_id)
    except Exception as e:
        await websocket.close(code=1011, reason=str(e))
        detector.clear_session(session_id)


@app.post("/v1/telephony/twilio/voice")
async def twilio_voice_webhook():
    """
    TwiML Webhook for live Twilio SIP/Cellular call interception.
    Returns TwiML directing the call audio stream to /v1/stream/twilio.
    """
    twiml_response = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Aditi">True Tone Security Gatekeeper is actively verifying this call.</Say>
    <Connect>
        <Stream url="wss://true-tone.internal/v1/stream/twilio" />
    </Connect>
    <Dial timeout="30" callerId="{{To}}">
        <Number>+91XXXXXXXXXX</Number>
    </Dial>
</Response>"""
    return HTMLResponse(content=twiml_response, media_type="application/xml")


@app.websocket("/v1/stream/twilio")
async def twilio_media_stream(websocket: WebSocket):
    """
    Dedicated handler for Twilio Media Streams.
    Twilio sends 8000 Hz, 8-bit G.711 μ-law PCM payloads in base64.
    """
    await websocket.accept()
    session_id = "twilio_call"
    audio_buffer = np.array([], dtype=np.float32)
    sr = 8000
    chunk_samples = int(sr * 1.5)  # 1.5s
    hop_samples = int(sr * 0.5)
    
    try:
        while True:
            msg_text = await websocket.receive_text()
            data = json.loads(msg_text)
            event = data.get("event")
            
            if event == "start":
                session_id = data.get("start", {}).get("streamSid", "twilio_stream")
                print(f"[Twilio Stream] Started call analysis: {session_id}")
            elif event == "media":
                payload = data.get("media", {}).get("payload", "")
                raw_bytes = base64.b64decode(payload)
                # Decode 8-bit G.711 mu-law into linear float32
                mu_uint8 = np.frombuffer(raw_bytes, dtype=np.uint8)
                pcm_float = decode_mulaw(mu_uint8)
                audio_buffer = np.concatenate([audio_buffer, pcm_float])
                
                # Check if enough audio accumulated
                if len(audio_buffer) >= chunk_samples:
                    window = audio_buffer[:chunk_samples]
                    audio_buffer = audio_buffer[hop_samples:]
                    
                    res = detector.analyze_chunk(window, session_id=session_id, sr=sr)
                    
                    # If spoof detected on live call, log alert
                    if res.risk_level == RiskLevel.CRITICAL:
                        print(f"[ALERT - CRITICAL SPOOF DETECTED] Session: {session_id} | Score: {res.smoothed_score}")
            elif event == "stop":
                detector.clear_session(session_id)
                break
    except WebSocketDisconnect:
        detector.clear_session(session_id)


# Serve Interactive Dashboard
DASHBOARD_HTML_PATH = "true_tone/dashboard/index.html"

@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    if os.path.exists(DASHBOARD_HTML_PATH):
        with open(DASHBOARD_HTML_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <html>
        <head><title>True Tone Engine</title></head>
        <body style="font-family:sans-serif; background:#0f172a; color:#f8fafc; padding:40px;">
            <h1>True Tone Deepfake Voice Interception Engine</h1>
            <p>API is active. Visit <code>/health</code> or <code>/docs</code> for Swagger UI.</p>
        </body>
    </html>
    """
