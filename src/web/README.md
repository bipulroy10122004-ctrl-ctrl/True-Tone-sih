# 🌐 True Tone SIH — Web Architecture & Frontend Structure

This directory houses the Cyber Defense Web Command Center for the **True Tone SIH** anti-spoofing platform.

---

## 📁 Directory Layout

```
src/web/
├── static/
│   ├── assets/              # Icons, badges, visual cues, logos
│   │
│   ├── css/                 # Modular CSS architecture
│   │   ├── variables.css    # Design tokens: cyber dark palette, fonts, glassmorphism, spacing
│   │   ├── base.css         # CSS Reset, typography base, scrollbars, micro-animations
│   │   ├── layout.css       # Main grid system, navigation, header, telemetry summary
│   │   ├── components/      # Component-scoped styling
│   │   │   ├── call_gateway.css   # Live VoIP visualizer, VU meter, frequency HUD
│   │   │   ├── upload_panel.css   # Audio dropzone, drag-and-drop, sample selectors
│   │   │   ├── verdict_box.css    # Threat probability gauge, risk level pills
│   │   │   ├── firewall_table.css # Quarantined IP ledger, action controls
│   │   │   ├── incident_feed.css  # Live threat event log & timeline
│   │   │   └── modals.css         # Dialogs, unblock modals, and toast alerts
│   │   └── style.css        # Central stylesheet entry point (imports modules)
│   │
│   └── js/                  # Modular JavaScript
│       ├── modules/
│       │   ├── state.js           # Centralized UI & audio streaming state
│       │   ├── api.js             # REST API client (/api/detect, /api/blocked-ips, etc.)
│       │   ├── audio_visualizer.js# WebRTC AudioContext, Canvas waveform & EQ visualizer
│       │   ├── live_call.js       # WebSocket VoIP streaming, hysteresis strike logic
│       │   ├── firewall.js        # IP blocklist DOM rendering, TTL timers, unblocking
│       │   ├── incident_log.js    # Live audit feed polling & rendering
│       │   └── ui.js              # Tab toggles, modals, notifications/toasts
│       └── dashboard.js     # Main app orchestrator
│
└── templates/
    ├── base.html            # Shared HTML skeleton (head, Google fonts, CSS/JS links)
    ├── index.html           # Main command center layout (assembles components)
    ├── components/          # Reusable Jinja2 template components
    │   ├── header.html      # Top nav bar, status badges, model health
    │   ├── telemetry.html   # High-level stats cards (active bans, spoofs stopped)
    │   ├── live_call.html   # Live VoIP / WebRTC audio inspection panel & visualizer
    │   ├── simulator.html   # Quick synthetic vs bonafide attack simulation
    │   ├── upload.html      # Drag-and-drop custom audio file upload panel
    │   ├── verdict.html     # Threat detection verdict card & confidence meter
    │   ├── firewall.html    # Active IP firewall quarantine table
    │   ├── incidents.html   # Real-time incident audit feed
    │   └── modals.html      # Quarantine modals, unblock dialogs, helper popups
    └── pages/
        ├── roadmap.html     # Interactive architecture roadmap & technical specs
        └── flowchart.html   # Complete end-to-end flowchart
```

---

## 🛠️ How Frontend & Backend Connect

1. **Jinja2 Rendering**: FastAPI serves `templates/index.html` at `GET /`.
2. **Static Asset Mounting**: FastAPI mounts `static/` at `/static`.
3. **Real-Time WebSockets**: Connects to `ws://localhost:8000/ws/live-call` for live binary audio PCM chunking and instant threat telemetry.
4. **REST Endpoints**:
   - `POST /api/detect`: Multipart audio analysis and caller IP mitigation.
   - `GET /api/blocked-ips`: Retrieves active quarantine blocklist.
   - `POST /api/block-manual`: Manually quarantines an IP address.
   - `POST /api/unblock`: Restores / whitelists an IP.
   - `GET /api/incidents`: Audit ledger of recent threat detections.
   - `GET /api/stats`: Telemetry stats (active bans, spoofs stopped, etc.).
