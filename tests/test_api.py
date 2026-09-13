"""
Tests for FastAPI REST Endpoints
"""

import os
import pytest
from starlette.testclient import TestClient
from true_tone.server.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "thresholds" in data


def test_dashboard_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "TRUE TONE" in response.text


def test_analyze_file_endpoint():
    sample_file = "data/samples/bonafide_human_call.wav"
    if not os.path.exists(sample_file):
        pytest.skip("Sample audio file not found")
        
    with open(sample_file, "rb") as f:
        response = client.post(
            "/v1/analyze/file",
            files={"file": ("bonafide.wav", f, "audio/wav")},
            data={"session_id": "test_upload", "simulate_telecom": "true"}
        )
        
    assert response.status_code == 200
    data = response.json()
    assert "overall_risk_level" in data
    assert "deepfake_probability" in data
    assert "chunk_timeline" in data
    assert len(data["chunk_timeline"]) >= 1
