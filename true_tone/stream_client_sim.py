"""
Live SIM-to-SIM Call Streaming Client & Interception Simulator

Simulates an active telephone call audio stream intercepted by an IMS gateway
or mobile application, chunking audio and streaming it to the True Tone WebSocket API.
Displays real-time forensic detection telemetry, latency, and threat actions in the console.
"""

import sys
import os
import time
import json
import asyncio
import argparse
import soundfile as sf
import websockets
import numpy as np


async def stream_audio_call(
    audio_path: str,
    server_url: str = "ws://127.0.0.1:8000/v1/stream/analyze",
    chunk_duration_sec: float = 0.5
):
    """
    Streams audio to the True Tone server in real-time chunks, mimicking an active phone call.
    """
    if not os.path.exists(audio_path):
        print(f"[Error] Audio file not found: {audio_path}")
        return

    print(f"\n[Simulator] Reading call audio: {audio_path}")
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    duration_sec = len(audio) / float(sr)
    print(f"[Simulator] Call Duration: {duration_sec:.2f}s | Sample Rate: {sr} Hz")
    print(f"[Simulator] Connecting to True Tone gateway at: {server_url} ...")

    try:
        async with websockets.connect(server_url) as ws:
            print("[Connected] WebSocket established. Initializing call session...\n")
            
            # Send init message
            await ws.send(json.dumps({
                "type": "init",
                "session_id": f"sim_call_{int(time.time())}",
                "sr": sr
            }))
            ready_msg = await ws.recv()
            print(f"[Server Handshake] {ready_msg}\n")
            
            # Start background listener for results
            async def receive_results():
                try:
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        if data.get("type") == "result":
                            risk = data.get("risk_level", "SAFE")
                            score = data.get("smoothed_score", 0.0)
                            latency = data.get("latency_ms", 0.0)
                            action = data.get("action", "")
                            
                            # Terminal color formatting
                            if risk == "CRITICAL":
                                color_code = "\033[91m"  # Red
                                tag = "[ CRITICAL DEEPFAKE ALERT ]"
                            elif risk == "SUSPICIOUS":
                                color_code = "\033[93m"  # Yellow
                                tag = "[ SUSPICIOUS VOICE ]"
                            else:
                                color_code = "\033[92m"  # Green
                                tag = "[ BONAFIDE HUMAN CALLER ]"
                                
                            reset_code = "\033[0m"
                            
                            print(f"{color_code}{tag} Score: {score:.3f} | Latency: {latency}ms | Action: {action}{reset_code}")
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    pass

            rx_task = asyncio.create_task(receive_results())
            
            # Stream audio chunks at real-time rate
            chunk_samples = int(sr * chunk_duration_sec)
            num_chunks = int(np.ceil(len(audio) / chunk_samples))
            
            print("=" * 70)
            print("  STARTING LIVE TELEPHONY CALL AUDIO STREAM")
            print("=" * 70)
            
            for i in range(num_chunks):
                start = i * chunk_samples
                end = min(len(audio), start + chunk_samples)
                chunk = audio[start:end]
                
                # Convert float32 [-1.0, 1.0] to 16-bit PCM bytes
                pcm_16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
                raw_bytes = pcm_16.tobytes()
                
                # Send raw binary stream
                await ws.send(raw_bytes)
                
                # Sleep to mimic real-time audio playback
                await asyncio.sleep(chunk_duration_sec)
                
            # Allow trailing packets to be processed
            await asyncio.sleep(1.0)
            rx_task.cancel()
            print("\n" + "=" * 70)
            print("  CALL AUDIO STREAM COMPLETED")
            print("=" * 70 + "\n")
            
    except ConnectionRefusedError:
        print(f"\n[Error] Could not connect to {server_url}.")
        print("Make sure the True Tone server is running: python -m uvicorn true_tone.server.app:app --port 8000\n")
    except Exception as e:
        print(f"\n[Error] Streaming exception: {e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="True Tone Telephony Stream Simulator")
    parser.add_argument("--audio", default="data/samples/bonafide_human_call.wav", help="Path to audio file to stream")
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/stream/analyze", help="WebSocket server URL")
    parser.add_argument("--chunk", type=float, default=0.5, help="Chunk duration in seconds")
    args = parser.parse_args()

    asyncio.run(stream_audio_call(audio_path=args.audio, server_url=args.url, chunk_duration_sec=args.chunk))
