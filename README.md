# True Tone: Real-Time SIM-to-SIM Call Deepfake Voice Detection Framework

> **AI-Powered Cellular Call Verification Gatekeeper**
> Developed for Smart India Hackathon (SIH) — Detecting AI-generated, synthetic, and cloned voices across cellular (VoLTE/VoNR), SIP, and telephony communications in real time.

---

## 📌 Problem & Telecom Architecture Overview

Voice cloning and neural Speech Synthesis (TTS / Voice Conversion) have enabled malicious actors to impersonate trusted individuals, executives, and family members over direct phone calls to execute financial fraud.

Standard audio deepfake detectors fail on phone calls because cellular networks transmit audio using narrowband or wideband lossy codecs (**AMR-NB @ 8kHz**, **AMR-WB @ 16kHz**, or **G.711 μ-law/A-law**). These codecs discard high frequencies (>3.4kHz or >7kHz) where conventional neural vocoder artifacts reside.

**True Tone** addresses this by analyzing:
1. **Linear Frequency Cepstral Coefficients (LFCC)**: Preserves linear filterbank resolution to detect vocoder spectral over-smoothing.
2. **Phase Inconsistency & Group Delay Entropy**: Captures subtle phase incoherencies left by neural vocoders in the sub-4kHz band.
3. **Biological Prosody & Micro-Jitter**: Evaluates natural pitch drift and involuntary vocal fold micro-tremors absent in synthetic speech.
4. **Sub-50ms Real-Time Inference**: Processes streaming 1.5s sliding-window chunks with Exponential Moving Average (EMA) temporal smoothing.

```
[ SIM A: Caller ] 
       │ (VoLTE/VoNR / 4G/5G)
       ▼
[ Carrier IMS Core / SBC / Mobile Gateway ]
       │
       ├───► [ Receiving Handset (SIM B) ]
       │
       ▼ (SIPREC / RTP Mirroring / WebSocket)
[ True Tone Ingestion Gateway ]
       │
       ├──► [ Voice Activity Detection (VAD) ]
       ├──► [ Telephony Codec Normalization (8kHz/16kHz) ]
       ├──► [ Feature Extraction (133-dim: LFCC + Phase + Prosody) ]
       ├──► [ Calibrated Ensemble Classifier ]
       └──► [ Temporal Session Smoothing (EMA) ]
              │
              ▼
    [ Real-Time Security Alert ]
    • P(Deepfake) > 0.70  ->  CRITICAL SPOOF: In-Call HUD Alert / Block Transaction
    • P(Deepfake) 0.4-0.7 ->  SUSPICIOUS: Silent Telemetry / Step-Up MFA
    • P(Deepfake) < 0.40  ->  SAFE: Bonafide Human Caller Verified
```

---

## 🚀 Quick Start

### 1. Installation & Environment Setup

Ensure you have Python 3.10+ installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

*(Or individually: `pip install numpy scipy librosa soundfile fastapi uvicorn websockets python-multipart scikit-learn joblib pytest`)*

---

### 2. Launch the Real-Time Verification Server & Cockpit

Start the FastAPI server:

```bash
python -m uvicorn true_tone.server.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser and navigate to:
- **Interactive Verification Cockpit**: [http://localhost:8000/](http://localhost:8000/)
- **Swagger REST API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Engine Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

### 3. Run the Live Telephony Call Streaming Simulator

Simulate a live SIM-to-SIM cellular call streaming audio into the verification gateway:

```bash
# Test with a genuine human phone call
python -m true_tone.stream_client_sim --audio data/samples/bonafide_human_call.wav

# Test with an AI voice clone call (ElevenLabs simulation)
python -m true_tone.stream_client_sim --audio data/samples/elevenlabs_ai_clone.wav

# Test with a voice conversion attack call (RVC simulation)
python -m true_tone.stream_client_sim --audio data/samples/rvc_voice_conversion.wav
```

---

### 4. Train / Re-calibrate the Forensic Model

Train the calibrated Gradient Boosting classifier on telephony-augmented acoustic features:

```bash
python -m true_tone.models.train --samples 120 --output models/true_tone_classifier.joblib
```

---

### 5. Run the Automated Test Suite

Verify all DSP modules, codecs, VAD, detector logic, and WebSocket endpoints:

```bash
python -m pytest tests/ -v
```

---

## 📊 Recommended Datasets Directory

| Dataset | URL / Reference | Purpose & Key Features |
| :--- | :--- | :--- |
| **ASVspoof 2021 (DF)** | [ASVspoof 2021](https://www.asvspoof.org/) / Zenodo: `4835108` | **Essential**: Audio compressed with lossy codecs (G.711, AAC, MP3) simulating telecom networks. |
| **ASVspoof 5 (2024)** | [ASVspoof 2024 Challenge](https://www.asvspoof.org/asvspoof5) | Latest challenge dataset featuring modern diffusion TTS and multilingual telephony audio. |
| **WaveFake** | [WaveFake Zenodo](https://zenodo.org/record/5642694) | 100,000+ generated clips across 6 neural vocoders (HiFi-GAN, MelGAN, WaveGlow). |
| **AI4Bharat IndicSpeech** | [AI4Bharat Shrutilipi](https://ai4bharat.iitm.ac.in/shrutilipi) | 6,400+ hours of natural Indian accented speech across 13 languages (Hindi, Tamil, Telugu, etc.). |
| **PartialSpoof** | [PartialSpoof GitHub](https://github.com/piotrkawa/partialspoof) | Partially manipulated speech (detects spliced words/PINs in legitimate calls). |
| **VoxCeleb 1 & 2** | [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) | 1M+ utterances from 6,000+ speakers in noisy real-world acoustic conditions. |

To view the interactive CLI guide:
```bash
python -m true_tone.data.download_datasets
```

---

## 🔌 API Reference

### WebSocket Streaming: `/v1/stream/analyze`
Bidirectional WebSocket for live audio tapping.
- **Client sends**: 
  - `{"type": "init", "session_id": "call_987", "sr": 16000}`
  - Binary 16-bit linear PCM audio chunks or `{"type": "audio", "data": "<base64>"}`
- **Server emits**:
  ```json
  {
    "type": "result",
    "session_id": "call_987",
    "score": 0.942,
    "smoothed_score": 0.915,
    "risk_level": "CRITICAL",
    "action": "CRITICAL SPOOF DETECTED • TRIGGER IN-CALL USER ALERT & BLOCK TRANSACTION",
    "is_speech": true,
    "latency_ms": 28.4,
    "forensics": {
      "acoustic_score": 0.96,
      "phase_score": 0.93,
      "prosody_score": 0.88,
      "jitter": 0.0031,
      "f0_mean": 165.2,
      "gd_entropy": 0.912
    }
  }
  ```

### Batch File Upload: `POST /v1/analyze/file`
Upload any audio recording (`.wav`, `.mp3`, `.ogg`, `.flac`) for full forensic inspection.
- Parameters:
  - `file`: Audio file
  - `simulate_telecom`: `true` (evaluates under cellular G.711 channel degradation)

### Twilio Cellular Call Tap: `POST /v1/telephony/twilio/voice`
Returns TwiML instructing Twilio to stream live phone call audio via WebSocket to `/v1/stream/twilio`.

---

## 🛡️ License & Compliance
This project is architected in alignment with the **Digital Personal Data Protection (DPDP) Act 2023** and **TRAI Cyber Security Guidelines**:
- In-memory stream processing (raw audio is not stored on disk unless explicitly configured for audit trails).
- Feature-only logging (storing mathematical embeddings and score telemetry).