"""
WebSocket Streaming Integration Test for True Tone Server
"""

import json
import time
import numpy as np
import pytest
from starlette.testclient import TestClient
from true_tone.server.app import app

client = TestClient(app)


def test_websocket_stream_flow():
    with client.websocket_connect("/v1/stream/analyze") as websocket:
        # 1. Send Init
        websocket.send_text(json.dumps({
            "type": "init",
            "session_id": "test_ws_session",
            "sr": 16000
        }))
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert ready["session_id"] == "test_ws_session"
        
        # 2. Stream 2 seconds of 16-bit PCM audio
        sr = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        sig = (0.7 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        pcm_bytes = (sig * 32767.0).astype(np.int16).tobytes()
        
        # Send raw bytes
        websocket.send_bytes(pcm_bytes)
        
        # Receive result packet
        result = websocket.receive_json()
        assert result["type"] == "result"
        assert "score" in result
        assert "smoothed_score" in result
        assert "risk_level" in result
        assert "latency_ms" in result
        assert result["session_id"] == "test_ws_session"
