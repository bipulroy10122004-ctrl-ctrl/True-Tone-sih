"""
True Tone SIH - Cyber Defense System Launcher
Launches the AI Voice Detection API, IP Firewall Middleware, and Web Command Center.
"""
import os
import sys
import uvicorn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

def main():
    print("=" * 65)
    print("  TRUE TONE SIH: AI VOICE DETECTION & IP FIREWALL DEFENSE")
    print("=" * 65)
    print("Starting FastAPI Gateway & Command Center Dashboard...")
    print("Access the Web Dashboard at: http://localhost:8000")
    print("API Documentation at:        http://localhost:8000/docs")
    print("=" * 65)

    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        app_dir=SCRIPT_DIR
    )

if __name__ == "__main__":
    main()
