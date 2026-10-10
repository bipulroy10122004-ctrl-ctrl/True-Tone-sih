"""
True Tone SIH - Network Gateway Security Middleware
Intercepts all incoming HTTP/WebSocket traffic at the application boundary.
Inspects client IP against the real-time active blocklist and terminates requests
from blacklisted sources before allocating compute resources.
"""
import os
import sys
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ip_blocker import get_ip_blocker

# Endpoints exempt from IP block enforcement (e.g. admin unblock, static UI, health check)
EXEMPT_PATHS = {
    "/api/unblock",
    "/api/unblock-all",
    "/api/blocked-ips",
    "/api/stats",
    "/api/incidents",
    "/api/health",
    "/docs",
    "/openapi.json"
}

def extract_client_ip(request: Request) -> str:
    """Extracts genuine client IP from proxy headers or direct socket connection."""
    # Check standard reverse-proxy headers
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # First IP in list is original client
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    if request.client and request.client.host:
        return request.client.host

    return "127.0.0.1"


class IPBlockerSecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, ip_blocker=None):
        super().__init__(app)
        self.ip_blocker = ip_blocker or get_ip_blocker()

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # 1. Allow static files, dashboard UI, and exempt admin routes
        if path.startswith("/static") or path == "/" or any(path.startswith(ep) for ep in EXEMPT_PATHS):
            return await call_next(request)

        # 2. Extract Client IP
        client_ip = extract_client_ip(request)

        # 3. Check for simulated caller IP header (for developer / dashboard testing)
        simulated_ip = request.headers.get("X-Caller-IP")
        ip_to_check = simulated_ip if simulated_ip else client_ip

        # 4. Check Blocklist
        is_blocked, block_record = self.ip_blocker.is_blocked(ip_to_check)
        if is_blocked:
            return JSONResponse(
                status_code=403,
                content={
                    "status": "FORBIDDEN_BLOCKED",
                    "error_code": "IP_BLACKLISTED_BY_TRUE_TONE_FIREWALL",
                    "message": f"Connection terminated. The IP '{ip_to_check}' is banned due to synthetic voice attack.",
                    "ip": ip_to_check,
                    "reason": block_record.get("reason", "AI Voice Spoofing Violation"),
                    "threat_score": block_record.get("threat_score", 1.0),
                    "blocked_at": block_record.get("blocked_at"),
                    "expires_at": block_record.get("expires_at")
                }
            )

        # 5. Pass request forward
        response = await call_next(request)
        return response
