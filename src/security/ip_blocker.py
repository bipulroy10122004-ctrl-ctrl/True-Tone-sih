"""
True Tone SIH - Automated IP Blocker & Threat Mitigation Engine
Provides high-performance in-memory caching with SQLite persistence, dynamic threat scoring,
strike-based auto-blocking, audit trail logging, and optional OS-level firewall integration.
"""
import os
import sqlite3
import datetime
import subprocess
import json
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "models", "security_firewall.db")
)

# Standard trusted addresses to prevent accidental lockout
DEFAULT_ALLOWLIST = {"127.0.0.1", "::1", "localhost"}

class IPBlockerManager:
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        default_ban_ttl_seconds: int = 3600,
        max_strikes_before_ban: int = 1,
        enable_os_firewall: bool = False,
        allowlist: Optional[set] = None
    ):
        self.db_path = db_path
        self.default_ban_ttl = default_ban_ttl_seconds
        self.max_strikes = max_strikes_before_ban
        self.enable_os_firewall = enable_os_firewall
        self.allowlist = set(allowlist) if allowlist else set(DEFAULT_ALLOWLIST)

        # In-memory fast cache for microsecond lookup: {ip: expire_datetime_or_None}
        self._active_blocklist_cache: Dict[str, Optional[datetime.datetime]] = {}

        self._init_database()
        self._load_cache_from_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Blocked IPs Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS blocked_ips (
                    ip TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    threat_score REAL NOT NULL,
                    strike_count INTEGER DEFAULT 1,
                    blocked_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP,
                    is_permanent INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)

            # 2. Incident Audit Logs Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incident_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    spoof_probability REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    details TEXT
                )
            """)

            # 3. IP Reputation & Telemetry
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ip_reputation (
                    ip TEXT PRIMARY KEY,
                    total_calls INTEGER DEFAULT 0,
                    spoof_calls INTEGER DEFAULT 0,
                    bonafide_calls INTEGER DEFAULT 0,
                    strikes INTEGER DEFAULT 0,
                    last_seen TIMESTAMP NOT NULL
                )
            """)
            conn.commit()

    def _load_cache_from_db(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        self._active_blocklist_cache.clear()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ip, expires_at, is_permanent FROM blocked_ips")
            rows = cursor.fetchall()
            for r in rows:
                ip = r["ip"]
                is_perm = bool(r["is_permanent"])
                exp_str = r["expires_at"]
                if is_perm or not exp_str:
                    self._active_blocklist_cache[ip] = None
                else:
                    try:
                        exp_dt = datetime.datetime.fromisoformat(exp_str)
                        if exp_dt > now:
                            self._active_blocklist_cache[ip] = exp_dt
                    except Exception:
                        self._active_blocklist_cache[ip] = None

    def is_blocked(self, ip: str) -> Tuple[bool, Optional[Dict]]:
        """
        Fast $O(1)$ in-memory check to see if an IP is actively banned.
        Returns: (is_blocked: bool, block_record_or_None: dict)
        """
        if ip in self.allowlist:
            return False, None

        now = datetime.datetime.now(datetime.timezone.utc)

        if ip in self._active_blocklist_cache:
            exp = self._active_blocklist_cache[ip]
            if exp is None or exp > now:
                # Fetch full record
                with self._get_connection() as conn:
                    c = conn.cursor()
                    c.execute("SELECT * FROM blocked_ips WHERE ip = ?", (ip,))
                    row = c.fetchone()
                    return True, dict(row) if row else {"ip": ip, "reason": "Active Block"}
            else:
                # Expired -> remove from cache
                self.unblock_ip(ip, reason="Automatic TTL expiration")

        return False, None

    def block_ip(
        self,
        ip: str,
        reason: str,
        threat_score: float = 1.0,
        duration_seconds: Optional[int] = None,
        is_permanent: bool = False,
        metadata: Optional[dict] = None
    ) -> Dict:
        """Blocks an IP immediately, updates database and in-memory cache, and syncs OS firewall if enabled."""
        now = datetime.datetime.now(datetime.timezone.utc)
        ttl = duration_seconds if duration_seconds is not None else self.default_ban_ttl
        expires_at = (now + datetime.timedelta(seconds=ttl)) if not is_permanent else None
        exp_iso = expires_at.isoformat() if expires_at else None

        meta_json = json.dumps(metadata or {})

        with self._get_connection() as conn:
            c = conn.cursor()
            # Upsert into blocked_ips
            c.execute("""
                INSERT INTO blocked_ips (ip, reason, threat_score, strike_count, blocked_at, expires_at, is_permanent, metadata)
                VALUES (?, ?, ?, 1, ?, ?, ?, ?)
                ON CONFLICT(ip) DO UPDATE SET
                    reason = excluded.reason,
                    threat_score = excluded.threat_score,
                    strike_count = blocked_ips.strike_count + 1,
                    blocked_at = excluded.blocked_at,
                    expires_at = excluded.expires_at,
                    is_permanent = excluded.is_permanent,
                    metadata = excluded.metadata
            """, (ip, reason, threat_score, now.isoformat(), exp_iso, 1 if is_permanent else 0, meta_json))
            conn.commit()

        # Update cache
        self._active_blocklist_cache[ip] = expires_at

        # Sync OS firewall if activated
        if self.enable_os_firewall:
            self._sync_os_firewall_rule(ip, action="block")

        record = {
            "ip": ip,
            "reason": reason,
            "threat_score": threat_score,
            "blocked_at": now.isoformat(),
            "expires_at": exp_iso,
            "is_permanent": is_permanent
        }
        return record

    def unblock_ip(self, ip: str, reason: str = "Manual Admin Override") -> bool:
        """Removes an IP from the active blocklist and in-memory cache."""
        ip = str(ip).strip()
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM blocked_ips WHERE ip = ?", (ip,))
            affected = c.rowcount > 0
            conn.commit()

        cached_had = ip in self._active_blocklist_cache
        if cached_had:
            self._active_blocklist_cache.pop(ip, None)

        if self.enable_os_firewall:
            self._sync_os_firewall_rule(ip, action="unblock")

        self.log_incident(
            ip=ip,
            event_type="UNBLOCK_OVERRIDE",
            spoof_probability=0.0,
            verdict="UNBLOCKED",
            action_taken=f"IP Restored: {reason}",
            details={"reason": reason}
        )
        return True

    def unblock_all(self, reason: str = "Admin Manual Clear All") -> int:
        """Flushes all blocked IPs from both database and in-memory cache."""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT count(*) as cnt FROM blocked_ips")
            cnt = c.fetchone()["cnt"]
            c.execute("DELETE FROM blocked_ips")
            conn.commit()

        self._active_blocklist_cache.clear()

        self.log_incident(
            ip="0.0.0.0",
            event_type="UNBLOCK_ALL_OVERRIDE",
            spoof_probability=0.0,
            verdict="ALL_UNBLOCKED",
            action_taken=f"All {cnt} Blocked IPs Restored: {reason}",
            details={"count": cnt, "reason": reason}
        )
        return cnt


    def log_incident(
        self,
        ip: str,
        event_type: str,
        spoof_probability: float,
        verdict: str,
        action_taken: str,
        details: Optional[dict] = None
    ):
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO incident_logs (ip, event_type, spoof_probability, verdict, action_taken, timestamp, details)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ip, event_type, spoof_probability, verdict, action_taken, now, json.dumps(details or {})))

            # Update IP reputation
            is_spoof = spoof_probability >= 0.5
            c.execute("""
                INSERT INTO ip_reputation (ip, total_calls, spoof_calls, bonafide_calls, strikes, last_seen)
                VALUES (?, 1, ?, ?, ?, ?)
                ON CONFLICT(ip) DO UPDATE SET
                    total_calls = ip_reputation.total_calls + 1,
                    spoof_calls = ip_reputation.spoof_calls + excluded.spoof_calls,
                    bonafide_calls = ip_reputation.bonafide_calls + excluded.bonafide_calls,
                    strikes = ip_reputation.strikes + excluded.strikes,
                    last_seen = excluded.last_seen
            """, (ip, 1 if is_spoof else 0, 0 if is_spoof else 1, 1 if is_spoof else 0, now))
            conn.commit()

    def process_ai_detection_result(
        self,
        ip: str,
        detection_result: dict,
        simulate_ip: bool = False
    ) -> Dict:
        """
        Takes output from AudioSpoofInferenceEngine, evaluates threat,
        and enforces automated IP blocking if spoof detected.
        """
        is_spoof = detection_result["is_spoof"]
        prob = detection_result["spoof_probability"]
        verdict = detection_result["verdict"]

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # If IP is in allowlist and not simulating mock caller IP
        if ip in self.allowlist and not simulate_ip:
            self.log_incident(
                ip=ip,
                event_type="CALL_INSPECTED",
                spoof_probability=prob,
                verdict=verdict,
                action_taken="ALLOWED (Trusted IP Whitelisted)",
                details={"metrics": detection_result}
            )
            return {
                "blocked": False,
                "action": "ALLOWED",
                "reason": "Origin IP is whitelisted",
                "detection": detection_result
            }

        if is_spoof:
            # Trigger dynamic auto-ban
            reason = f"AI Voice Spoofing Violation (Confidence: {detection_result['spoof_percentage']}%, Latency: {detection_result['total_latency_ms']}ms)"
            block_rec = self.block_ip(
                ip=ip,
                reason=reason,
                threat_score=prob,
                duration_seconds=self.default_ban_ttl,
                is_permanent=False,
                metadata={"detection": detection_result}
            )
            self.log_incident(
                ip=ip,
                event_type="AI_SPOOF_AUTO_BLOCK",
                spoof_probability=prob,
                verdict=verdict,
                action_taken=f"BLOCKED at Gateway & Firewall (TTL: {self.default_ban_ttl}s)",
                details={"detection": detection_result, "block_record": block_rec}
            )
            return {
                "blocked": True,
                "action": "BLOCKED",
                "reason": reason,
                "block_record": block_rec,
                "detection": detection_result
            }
        elif detection_result.get("is_silent") or not detection_result.get("is_speech", True):
            # Ambient noise / inter-speech silence - non-vocal acoustic sound
            self.log_incident(
                ip=ip,
                event_type="AMBIENT_NOISE_PASSED",
                spoof_probability=prob,
                verdict=verdict,
                action_taken="MONITORED (Ambient Sound / Waiting for Voice)",
                details={"detection": detection_result}
            )
            return {
                "blocked": False,
                "action": "PERMITTED",
                "reason": "Ambient Sound / Waiting for Voice (No Vocal Threat)",
                "detection": detection_result
            }
        else:
            # Bonafide / Authentic call
            self.log_incident(
                ip=ip,
                event_type="BONAFIDE_CALL_PASSED",
                spoof_probability=prob,
                verdict=verdict,
                action_taken="PERMITTED (Authentic Human Voice)",
                details={"detection": detection_result}
            )
            return {
                "blocked": False,
                "action": "PERMITTED",
                "reason": "Authentic Human Voice Verified",
                "detection": detection_result
            }

    def get_all_blocked_ips(self) -> List[Dict]:
        """Retrieves list of all currently recorded blocked IPs with calculated remaining TTL."""
        now = datetime.datetime.now(datetime.timezone.utc)
        results = []
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM blocked_ips ORDER BY blocked_at DESC")
            rows = c.fetchall()
            for r in rows:
                item = dict(r)
                if item["expires_at"]:
                    try:
                        exp = datetime.datetime.fromisoformat(item["expires_at"])
                        remaining_seconds = max(0, int((exp - now).total_seconds()))
                        item["remaining_ttl_seconds"] = remaining_seconds
                        item["is_expired"] = remaining_seconds == 0
                    except Exception:
                        item["remaining_ttl_seconds"] = None
                        item["is_expired"] = False
                else:
                    item["remaining_ttl_seconds"] = None
                    item["is_expired"] = False

                try:
                    item["metadata"] = json.loads(item["metadata"]) if item["metadata"] else {}
                except Exception:
                    pass
                results.append(item)
        return results

    def get_recent_incidents(self, limit: int = 50) -> List[Dict]:
        """Retrieves audit trail of recent security events."""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM incident_logs ORDER BY id DESC LIMIT ?", (limit,))
            rows = c.fetchall()
            logs = []
            for r in rows:
                item = dict(r)
                try:
                    item["details"] = json.loads(item["details"]) if item["details"] else {}
                except Exception:
                    pass
                logs.append(item)
            return logs

    def get_telemetry_stats(self) -> Dict:
        """Returns real-time dashboard analytics counters."""
        with self._get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM blocked_ips")
            total_active_bans = c.fetchone()[0]

            c.execute("SELECT COUNT(*) FROM incident_logs")
            total_incidents = c.fetchone()[0]

            c.execute("SELECT COUNT(*) FROM incident_logs WHERE event_type LIKE '%SPOOF%'")
            total_spoofs_stopped = c.fetchone()[0]

            c.execute("SELECT COUNT(*) FROM incident_logs WHERE event_type LIKE '%BONAFIDE%'")
            total_bonafide_calls = c.fetchone()[0]

            c.execute("SELECT COUNT(DISTINCT ip) FROM ip_reputation")
            total_unique_ips = c.fetchone()[0]

            return {
                "active_bans_count": total_active_bans,
                "total_calls_inspected": total_incidents,
                "synthetic_spoofs_stopped": total_spoofs_stopped,
                "bonafide_calls_permitted": total_bonafide_calls,
                "unique_ips_tracked": total_unique_ips,
                "in_memory_cached_bans": len(self._active_blocklist_cache),
                "os_firewall_sync_enabled": self.enable_os_firewall
            }

    def _sync_os_firewall_rule(self, ip: str, action: str = "block"):
        """
        Synchronizes IP rule with Windows Firewall using netsh advfirewall.
        (Only invoked when enable_os_firewall is explicitly set to True).
        """
        rule_name = f"TrueTone_Block_{ip.replace(':', '_').replace('.', '_')}"
        try:
            if action == "block":
                cmd = f'netsh advfirewall firewall add rule name="{rule_name}" dir=in action=block remoteip={ip}'
                subprocess.run(cmd, shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif action == "unblock":
                cmd = f'netsh advfirewall firewall delete rule name="{rule_name}"'
                subprocess.run(cmd, shell=True, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[IPBlockerManager] Warning: OS firewall sync failed for {ip}: {e}")


# Global singleton instance
_ip_manager_instance = None

def get_ip_blocker(enable_os_firewall: bool = False) -> IPBlockerManager:
    global _ip_manager_instance
    if _ip_manager_instance is None:
        _ip_manager_instance = IPBlockerManager(enable_os_firewall=enable_os_firewall)
    return _ip_manager_instance
