from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["Web Dashboard"])

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sparsh CCTV - AI Surveillance & Emergency Response</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0f1117;
            --card-bg: #181b24;
            --card-border: #262a38;
            --accent-blue: #2563eb;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-orange: #f59e0b;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-dark);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        header {
            background-color: var(--card-bg);
            border-bottom: 1px solid var(--card-border);
            padding: 16px 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #ef4444, #3b82f6);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        }
        .brand h1 { font-size: 18px; font-weight: 700; letter-spacing: 0.5px; }
        .brand p { font-size: 12px; color: var(--text-muted); }
        .header-stats {
            display: flex;
            gap: 16px;
            align-items: center;
            font-size: 13px;
        }
        .badge {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-blue { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
        .badge-orange { background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }

        .container {
            display: grid;
            grid-template-columns: 1fr 480px;
            gap: 20px;
            padding: 20px;
            flex: 1;
        }
        @media (max-width: 1024px) {
            .container { grid-template-columns: 1fr; }
        }

        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .card-header {
            padding: 14px 18px;
            border-bottom: 1px solid var(--card-border);
            font-weight: 600;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .video-wrapper {
            position: relative;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 480px;
        }
        .video-wrapper img {
            width: 100%;
            height: auto;
            max-height: 600px;
            object-fit: contain;
            display: block;
        }
        .controls-bar {
            padding: 14px 18px;
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            background: #141720;
            border-top: 1px solid var(--card-border);
        }
        button {
            cursor: pointer;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            border: none;
            transition: all 0.15s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .btn-primary { background: var(--accent-blue); color: #fff; }
        .btn-primary:hover { background: #1d4ed8; }
        .btn-success { background: var(--accent-green); color: #fff; }
        .btn-success:hover { background: #059669; }
        .btn-danger { background: var(--accent-red); color: #fff; }
        .btn-danger:hover { background: #dc2626; }
        .btn-accident { background: #ea580c; color: #fff; }
        .btn-accident:hover { background: #c2410c; }
        .btn-fire { background: #dc2626; color: #fff; }
        .btn-fire:hover { background: #b91c1c; }
        .btn-secondary { background: #2b3040; color: #e5e7eb; border: 1px solid #3b4256; }
        .btn-secondary:hover { background: #373e52; }

        .telemetry-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            padding: 16px;
        }
        .telemetry-box {
            background: #141720;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 12px;
        }
        .telemetry-title { font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; }
        .telemetry-value { font-size: 20px; font-weight: 700; margin-top: 4px; font-family: 'JetBrains Mono', monospace; }

        /* Right Panel Tabs */
        .tabs {
            display: flex;
            border-bottom: 1px solid var(--card-border);
            background: #141720;
        }
        .tab-btn {
            background: none;
            color: var(--text-muted);
            border-radius: 0;
            padding: 12px 18px;
            font-size: 13px;
            border-bottom: 2px solid transparent;
        }
        .tab-btn.active {
            color: #fff;
            border-bottom-color: var(--accent-blue);
            background: rgba(37, 99, 235, 0.08);
        }
        .tab-content {
            padding: 16px;
            overflow-y: auto;
            max-height: calc(100vh - 160px);
        }

        /* Form elements */
        .form-group { margin-bottom: 14px; }
        label { display: block; font-size: 12px; font-weight: 600; margin-bottom: 6px; color: var(--text-muted); }
        input, select {
            width: 100%;
            padding: 8px 12px;
            background: #12141c;
            border: 1px solid #2d3345;
            border-radius: 6px;
            color: #fff;
            font-size: 13px;
            font-family: inherit;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--accent-blue);
        }

        /* Logs table */
        table { width: 100%; border-collapse: collapse; font-size: 12px; }
        th, td { padding: 8px 10px; border-bottom: 1px solid var(--card-border); text-align: left; }
        th { color: var(--text-muted); font-weight: 600; background: #141720; }
        tr:hover { background: rgba(255, 255, 255, 0.02); }

        .facility-item {
            background: #141720;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="brand-icon">📹</div>
            <div>
                <h1>Sparsh CCTV Agentic Surveillance</h1>
                <p>FastAPI Real-Time Accident Severity & Fire Detection Gateway</p>
            </div>
        </div>
        <div class="header-stats">
            <span id="headerStatusBadge" class="badge badge-green">● IDLE</span>
            <span id="headerCamText" style="color: var(--text-muted);">CAM-01 (Sector 25 Dwarka)</span>
            <a href="/docs" target="_blank" style="color: var(--accent-blue); text-decoration: none; font-weight: 600;">Swagger Docs &rarr;</a>
        </div>
    </header>

    <div class="container">
        <!-- Left Panel: Live CCTV Video Player & Stream Controls -->
        <div style="display: flex; flex-direction: column; gap: 20px;">
            <div class="card">
                <div class="card-header">
                    <span>LIVE SURVEILLANCE FEED (15 FPS MJPEG)</span>
                    <span id="clockTime" style="font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--text-muted);">--:--:--</span>
                </div>
                <!-- Dynamic Video Source Selector Bar -->
                <div style="background: #141720; border-bottom: 1px solid var(--card-border); padding: 10px 18px; display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap;">
                    <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                        <span style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px;">FEED SOURCE:</span>
                        
                        <button id="btnQuickRTSP" class="btn-secondary" style="padding: 5px 12px; font-size: 12px;" onclick="quickSelectSource('rtsp')">
                            📡 Live RTSP
                        </button>

                        <select id="feedSourceSelect" onchange="applySourceSwitch(this.value)" style="background: #1e2230; color: #f3f4f6; border: 1px solid var(--card-border); border-radius: 6px; padding: 6px 12px; font-size: 13px; font-weight: 500; cursor: pointer; min-width: 270px;">
                            <optgroup label="🔴 Live Hardware Stream">
                                <option value="rtsp" selected>📡 Live Sparsh RTSP Camera (Default)</option>
                            </optgroup>
                            <optgroup id="feedTestVideosOptgroup" label="🎞️ Recorded Incident Clips (data/videos/test/)">
                                <option value="bikeacc.mp4">🛵 bikeacc.mp4</option>
                                <option value="blast.mp4">💥 blast.mp4</option>
                                <option value="crash.mp4">💥 crash.mp4</option>
                                <option value="fire.mp4">🔥 fire.mp4</option>
                                <option value="truck_acc.mp4">🚛 truck_acc.mp4</option>
                            </optgroup>
                        </select>

                        <button class="btn-primary" style="padding: 6px 14px; font-size: 12px;" onclick="applySourceSwitch()">
                            ▶ Switch Feed
                        </button>
                    </div>
                    <div>
                        <span id="activeSourceBadge" class="badge badge-orange">● Connecting...</span>
                    </div>
                </div>
                <div class="video-wrapper" style="position: relative; background: #0b0d13; min-height: 480px; display: flex; align-items: center; justify-content: center; overflow: hidden; border-radius: 8px;">
                    <img id="liveStreamImg" src="/api/surveillance/stream" alt="Live Surveillance Stream" style="width: 100%; height: 100%; object-fit: contain; display: block;" onerror="handleStreamError()" />
                    <div id="streamNotice" style="position: absolute; bottom: 12px; right: 12px; pointer-events: none;">
                        <span id="streamNoticeBadge" class="badge badge-green" style="background: rgba(16, 185, 129, 0.85);">● STREAM READY</span>
                    </div>
                </div>
                <div class="controls-bar">
                    <button class="btn-success" onclick="startStream()">▶ Start Stream</button>
                    <button class="btn-danger" onclick="stopStream()">⏹ Stop Stream</button>
                    <button class="btn-secondary" onclick="pauseStream()">⏸ Pause</button>
                    <button class="btn-secondary" onclick="resumeStream()">⏯ Resume</button>
                    <button class="btn-primary" onclick="takeSnapshot()"> Snapshot</button>
                    <button class="btn-accident" onclick="triggerAccidentAlert()">💥 Trigger Accident Alert</button>
                    <button class="btn-fire" onclick="triggerFireAlert()">🔥 Trigger Fire Alert</button>
                </div>
            </div>

            <!-- Telemetry Stats Cards -->
            <div class="telemetry-grid">
                <div class="telemetry-box">
                    <div class="telemetry-title">Stream Rate</div>
                    <div class="telemetry-value" id="fpsVal">0.0 FPS</div>
                </div>
                <div class="telemetry-box">
                    <div class="telemetry-title">Vehicles Tracked</div>
                    <div class="telemetry-value" id="vehCountVal">0</div>
                </div>
                <div class="telemetry-box">
                    <div class="telemetry-title">Accident Severity</div>
                    <div class="telemetry-value" id="sevVal" style="color: var(--accent-green);">CLEAR</div>
                </div>
                <div class="telemetry-box">
                    <div class="telemetry-title">Fire / Smoke</div>
                    <div class="telemetry-value" id="fireVal" style="color: var(--accent-green);">NONE</div>
                </div>
            </div>
        </div>

        <!-- Right Panel: Dynamic Tabs (Telemetry, Dynamic Settings, Incident Logs) -->
        <div class="card">
            <div class="tabs">
                <button class="tab-btn active" onclick="switchTab('tabDetections', this)">Live Alerts</button>
                <button class="tab-btn" onclick="switchTab('tabSettings', this)">Dynamic Settings</button>
                <button class="tab-btn" onclick="switchTab('tabIncidents', this)">Incident Audit</button>
            </div>

            <!-- Tab 1: Live Alerts & Nearby Responders -->
            <div id="tabDetections" class="tab-content">
                <h3 style="font-size: 14px; margin-bottom: 12px;">Active Camera Geolocation</h3>
                <div id="locationInfoBox" style="background: #141720; padding: 12px; border-radius: 8px; margin-bottom: 16px; font-size: 12px; line-height: 1.6;">
                    Loading location...
                </div>

                <h3 style="font-size: 14px; margin-bottom: 12px;">3km Emergency Responders</h3>
                <div id="facilitiesContainer">
                    Loading facilities...
                </div>
            </div>

            <!-- Tab 2: Dynamic System Settings (View & Configure at Runtime) -->
            <div id="tabSettings" class="tab-content" style="display: none;">
                <h3 style="font-size: 14px; margin-bottom: 14px; color: var(--accent-blue); display: flex; align-items: center; justify-content: space-between;">
                    <span>Dynamic Runtime Settings (No Restart Required)</span>
                    <span id="settingsStatusBadge" class="badge badge-green">● Loaded</span>
                </h3>
                
                <div class="form-group">
                    <label>Camera ID</label>
                    <input id="set_camera_id" type="text" />
                </div>
                <div class="form-group">
                    <label>Camera Location String / Plus Code</label>
                    <input id="set_camera_location" type="text" />
                </div>
                <div class="form-group">
                    <label>Active Video Stream Source</label>
                    <select id="set_video_source">
                        <option value="rtsp"> Live Sparsh RTSP Camera</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Crash Detector Confidence Threshold (0.1 - 1.0)</label>
                    <input id="set_crash_conf" type="number" step="0.05" min="0.1" max="1.0" />
                </div>
                <div class="form-group">
                    <label>Fire Detector Confidence Threshold (0.1 - 1.0)</label>
                    <input id="set_fire_conf" type="number" step="0.05" min="0.1" max="1.0" />
                </div>
                <div class="form-group">
                    <label>Alert Snapshot Cooldown (seconds)</label>
                    <input id="set_cooldown" type="number" step="1" min="1" />
                </div>
                <div class="form-group">
                    <label>Emergency Search Radius (km)</label>
                    <input id="set_radius" type="number" step="0.5" min="0.5" />
                </div>
                <div class="form-group">
                    <label>Default Department Alert Email</label>
                    <input id="set_email" type="email" />
                </div>
                <div class="form-group">
                    <label>RTSP Camera IP</label>
                    <input id="set_rtsp_ip" type="text" />
                </div>
                <div class="form-group">
                    <label>Emergency Phone</label>
                    <input id="set_twilio_phone" type="text" />
                </div>

                <button class="btn-primary" style="width: 100%; margin-top: 10px; justify-content: center;" onclick="saveDynamicSettings()">
                     Save & Apply Dynamic Settings
                </button>
            </div>

            <!-- Tab 3: Incidents & Emergency Dispatches Audit Log -->
            <div id="tabIncidents" class="tab-content" style="display: none;">
                <h3 style="font-size: 14px; margin-bottom: 10px;">Recent Incident Events</h3>
                <div style="max-height: 250px; overflow-y: auto; margin-bottom: 20px;">
                    <table>
                        <thead>
                            <tr><th>Time</th><th>Event</th><th>Severity</th><th>Snap</th></tr>
                        </thead>
                        <tbody id="incidentsTableBody">
                            <tr><td colspan="4">Loading incidents...</td></tr>
                        </tbody>
                    </table>
                </div>

                <h3 style="font-size: 14px; margin-bottom: 10px;">Recent Emergency Dispatches</h3>
                <div style="max-height: 250px; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr><th>Dispatch ID</th><th>Event</th><th>Services</th></tr>
                        </thead>
                        <tbody id="dispatchesTableBody">
                            <tr><td colspan="3">Loading dispatches...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <script>
        function switchTab(tabId, btn) {
            document.querySelectorAll('.tab-content').forEach(el => el.style.display = 'none');
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).style.display = 'block';
            btn.classList.add('active');
            if (tabId === 'tabSettings') loadCurrentSettings();
            if (tabId === 'tabIncidents') loadLogs();
        }

        let availableSourcesData = null;

        function quickSelectSource(src) {
            const sel = document.getElementById('feedSourceSelect');
            if (sel) {
                sel.value = src;
                applySourceSwitch(src);
            }
        }

        async function loadAvailableSources() {
            try {
                console.log('[Dashboard] loadAvailableSources starting fetch...');
                const res = await fetch('/api/surveillance/available-sources');
                console.log('[Dashboard] loadAvailableSources res status:', res.status);
                if (!res.ok) throw new Error('Failed to fetch available sources: ' + res.status);
                const data = await res.json();
                console.log('[Dashboard] loadAvailableSources data:', data);
                availableSourcesData = data;

                // 1. Populate Dashboard Bar Dropdown Optgroup
                const optgroup = document.getElementById('feedTestVideosOptgroup');
                if (optgroup) {
                    optgroup.innerHTML = '';
                    (data.test_videos || []).forEach(v => {
                        const opt = document.createElement('option');
                        opt.value = v.filename;
                        const icon = v.filename.toLowerCase().includes('fire') ? '🔥' : (v.filename.toLowerCase().includes('blast') ? '💥' : '🎞️');
                        opt.innerText = `${icon} ${v.filename} (${v.size_mb} MB)`;
                        optgroup.appendChild(opt);
                    });
                }

                // Update selected value in dashboard dropdown
                const feedSelect = document.getElementById('feedSourceSelect');
                if (feedSelect) {
                    if (data.is_rtsp) {
                        feedSelect.value = 'rtsp';
                    } else if (data.current_source) {
                        const cur = data.current_source.toLowerCase();
                        for (let i = 0; i < feedSelect.options.length; i++) {
                            const val = feedSelect.options[i].value.toLowerCase();
                            if (val !== 'rtsp' && cur.includes(val)) {
                                feedSelect.selectedIndex = i;
                                break;
                            }
                        }
                    }
                }

                // 2. Populate Settings Tab Dropdown
                const setSelect = document.getElementById('set_video_source');
                if (setSelect) {
                    setSelect.innerHTML = '<option value="rtsp">📡 Live Sparsh RTSP Camera</option>';
                    (data.test_videos || []).forEach(v => {
                        const opt = document.createElement('option');
                        opt.value = v.path;
                        opt.innerText = `🎞️ ${v.filename} (${v.size_mb} MB)`;
                        setSelect.appendChild(opt);
                    });
                }

                // 3. Update Active Source Badge
                updateActiveSourceBadge(data);
            } catch (e) {
                console.error('Failed to load available sources:', e);
            }
        }

        function updateActiveSourceBadge(data) {
            const badge = document.getElementById('activeSourceBadge');
            if (!badge) return;
            if (data.is_rtsp) {
                if (data.stream_connected) {
                    badge.className = 'badge badge-green';
                    badge.innerText = '● Live RTSP Feed';
                } else {
                    badge.className = 'badge badge-orange';
                    badge.innerText = '● RTSP Offline (Slate Active)';
                }
            } else {
                badge.className = 'badge badge-blue';
                badge.innerText = `● Test Video: ${data.current_source_label || 'Active'}`;
            }
        }

        async function applySourceSwitch(targetSource) {
            let sourceToSwitch = targetSource;
            if (!sourceToSwitch) {
                const feedSelect = document.getElementById('feedSourceSelect');
                sourceToSwitch = feedSelect ? feedSelect.value : 'rtsp';
            }
            if (!sourceToSwitch) sourceToSwitch = 'rtsp';

            const badge = document.getElementById('activeSourceBadge');
            if (badge) {
                badge.className = 'badge badge-orange';
                badge.innerText = 'Switching feed...';
            }

            try {
                const res = await fetch('/api/surveillance/switch-source', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source: sourceToSwitch })
                });
                const data = await res.json();
                
                // Refresh live stream image element
                const streamImg = document.getElementById('liveStreamImg');
                if (streamImg) {
                    streamImg.style.display = 'block';
                    streamImg.src = '/api/surveillance/stream?t=' + Date.now();
                }

                await loadAvailableSources();
                fetchStatus();
            } catch (err) {
                console.error('Error switching video source:', err);
                await loadAvailableSources();
            }
        }

        function handleStreamError() {
            console.warn('Stream interrupted, attempting to reconnect...');
            const notice = document.getElementById('streamNoticeBadge');
            if (notice) {
                notice.className = 'badge badge-orange';
                notice.innerText = '● RECONNECTING STREAM...';
            }
            setTimeout(() => {
                const streamImg = document.getElementById('liveStreamImg');
                if (streamImg) streamImg.src = '/api/surveillance/stream?t=' + Date.now();
                if (notice) {
                    notice.className = 'badge badge-green';
                    notice.innerText = '● STREAM READY';
                }
            }, 2000);
        }

        async function fetchStatus() {
            try {
                const res = await fetch('/api/surveillance/status');
                const data = await res.json();
                
                document.getElementById('fpsVal').innerText = data.fps + ' FPS';
                document.getElementById('vehCountVal').innerText = data.monitored_vehicles;
                
                const sevEl = document.getElementById('sevVal');
                sevEl.innerText = data.highest_severity_label;
                sevEl.style.color = (data.current_status === 'NO_STREAM') ? '#9ca3af' : ((data.highest_severity >= 2) ? '#ef4444' : '#10b981');

                const fireEl = document.getElementById('fireVal');
                fireEl.innerText = data.fire_status_label;
                fireEl.style.color = (data.current_status === 'NO_STREAM') ? '#9ca3af' : ((data.has_fire || data.has_smoke) ? '#ef4444' : '#10b981');

                const badge = document.getElementById('headerStatusBadge');
                if (data.current_status === 'NO_STREAM' || !data.stream_connected) {
                    badge.className = 'badge badge-orange';
                    badge.innerText = '● NO LIVE STREAM';
                } else if (data.is_running) {
                    badge.className = (data.current_status === 'INCIDENT') ? 'badge badge-red' : 'badge badge-green';
                    badge.innerText = (data.current_status === 'INCIDENT') ? '🚨 INCIDENT DETECTED' : '● RUNNING (' + data.fps + ' FPS)';
                } else {
                    badge.className = 'badge badge-blue';
                    badge.innerText = '● STANDBY';
                }
            } catch (err) {
                console.error(err);
            }
        }

        async function startStream() {
            await fetch('/api/surveillance/start', { method: 'POST' });
            fetchStatus();
        }
        async function stopStream() {
            await fetch('/api/surveillance/stop', { method: 'POST' });
            fetchStatus();
        }
        async function pauseStream() {
            await fetch('/api/surveillance/pause', { method: 'POST' });
            fetchStatus();
        }
        async function resumeStream() {
            await fetch('/api/surveillance/resume', { method: 'POST' });
            fetchStatus();
        }
        async function takeSnapshot() {
            const res = await fetch('/api/surveillance/snapshot', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ tag: 'manual' })
            });
            const data = await res.json();
            alert('Snapshot saved: ' + data.filename);
        }
        async function triggerAccidentAlert() {
            if (confirm(`🚨 DISPATCH EMERGENCY ACCIDENT ALERT?\n\nThis will send high-priority incident notifications to Police, Hospitals, and Traffic Control.`)) {
                try {
                    const res = await fetch('/api/surveillance/trigger-accident-alert', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            severity_label: 'Severe Accident (Manual Dispatch)',
                            confidence: 0.98,
                            dispatch_emergency: true
                        })
                    });
                    const data = await res.json();
                    if (res.ok) {
                        alert(`✅ Accident emergency alert dispatched successfully!\n\nDepartments notified: ` + (data.departments_notified || []).join(', ') + `\nSnapshot: ` + (data.snapshot?.filename || 'Saved'));
                        loadLogs();
                    } else {
                        alert(`❌ Dispatch failed: ` + (data.detail || data.message || 'Unknown error'));
                    }
                } catch (err) {
                    alert(`Network error dispatching alert: ` + err);
                }
            }
        }

        async function triggerFireAlert() {
            if (confirm(`🔥 DISPATCH EMERGENCY FIRE ALERT?\n\nThis will send high-priority fire hazard notifications to Fire Stations, Hospitals, and Police.`)) {
                try {
                    const res = await fetch('/api/surveillance/trigger-fire-alert', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            severity_label: 'Active Fire Hazard (Manual Dispatch)',
                            confidence: 0.99,
                            dispatch_emergency: true
                        })
                    });
                    const data = await res.json();
                    if (res.ok) {
                        alert(`✅ Fire emergency alert dispatched successfully!\n\nDepartments notified: ` + (data.departments_notified || []).join(', ') + `\nSnapshot: ` + (data.snapshot?.filename || 'Saved'));
                        loadLogs();
                    } else {
                        alert(`❌ Dispatch failed: ` + (data.detail || data.message || 'Unknown error'));
                    }
                } catch (err) {
                    alert(`Network error dispatching alert: ` + err);
                }
            }
        }

        async function triggerAlertModal() {
            return triggerAccidentAlert();
        }

        async function loadLocationAndFacilities() {
            try {
                const locRes = await fetch('/api/emergency/location');
                if (!locRes.ok) throw new Error('Location fetch failed');
                const loc = await locRes.json();
                document.getElementById('locationInfoBox').innerHTML = `
                    <strong>Camera:</strong> ${loc.camera_id || 'CAM-01'}<br>
                    <strong>Address:</strong> ${loc.formatted_address || loc.camera_location || 'Configured Location'}<br>
                    <strong>GPS:</strong> ${loc.latitude ?? 'N/A'}, ${loc.longitude ?? 'N/A'}<br>
                    <a href="${loc.maps_url || '#'}" target="_blank" style="color: var(--accent-blue); text-decoration: none; font-weight: bold; margin-top: 4px; display: inline-block;">📍 View on Google Maps</a>
                `;

                const facRes = await fetch('/api/emergency/facilities?limit=3');
                if (facRes.ok) {
                    const fac = await facRes.json();
                    let html = '';
                    (fac.hospitals || []).forEach(h => {
                        html += `<div class="facility-item"><div><strong>🏥 ${h.name}</strong><br><span style="color: var(--text-muted); font-size: 11px;">${h.distance_km} km away | Tel: ${h.phone}</span></div></div>`;
                    });
                    (fac.police_stations || []).forEach(p => {
                        html += `<div class="facility-item"><div><strong>👮 ${p.name}</strong><br><span style="color: var(--text-muted); font-size: 11px;">${p.distance_km} km away | Tel: ${p.phone}</span></div></div>`;
                    });
                    (fac.fire_stations || []).forEach(f => {
                        html += `<div class="facility-item"><div><strong>🚒 ${f.name}</strong><br><span style="color: var(--text-muted); font-size: 11px;">${f.distance_km} km away | Tel: ${f.phone}</span></div></div>`;
                    });
                    document.getElementById('facilitiesContainer').innerHTML = html || '<div style="color: var(--text-muted); font-size: 12px;">No emergency facilities found in range.</div>';
                }
            } catch(e) {
                console.error('Error loading location/facilities:', e);
                document.getElementById('locationInfoBox').innerHTML = '<span style="color: var(--accent-orange);">📍 Camera location loaded from default configuration.</span>';
            }
        }

        async function loadCurrentSettings() {
            try {
                const res = await fetch('/api/settings');
                if (!res.ok) throw new Error('Settings fetch failed');
                const s = await res.json();
                if (s.camera) {
                    document.getElementById('set_camera_id').value = s.camera.camera_id || '';
                    document.getElementById('set_camera_location').value = s.camera.camera_location || '';
                    document.getElementById('set_rtsp_ip').value = s.camera.rtsp_camera_ip || '';
                }
                if (s.detectors) {
                    document.getElementById('set_crash_conf').value = s.detectors.crash_conf ?? 0.25;
                    document.getElementById('set_fire_conf').value = s.detectors.fire_conf ?? 0.40;
                }
                if (s.alerts) {
                    document.getElementById('set_cooldown').value = s.alerts.alert_cooldown_sec ?? 10;
                    document.getElementById('set_radius').value = s.alerts.emergency_search_radius_km ?? 3.0;
                }
                if (s.email) {
                    document.getElementById('set_email').value = s.email.default_recipient_email || '';
                }
                if (s.twilio) {
                    document.getElementById('set_twilio_phone').value = s.twilio.emergency_dispatch_phone || '';
                }

                const curSource = s.camera?.video_source;
                const setSelect = document.getElementById('set_video_source');
                if (setSelect && curSource) {
                    for (let i = 0; i < setSelect.options.length; i++) {
                        const optVal = setSelect.options[i].value;
                        if (optVal === curSource || (curSource === 'rtsp' && optVal === 'rtsp') || curSource.endsWith(optVal)) {
                            setSelect.selectedIndex = i;
                            break;
                        }
                    }
                }
                const badge = document.getElementById('settingsStatusBadge');
                if (badge) {
                    badge.className = 'badge badge-green';
                    badge.innerText = '● Synced';
                }
            } catch(e) {
                console.error('Failed to load settings:', e);
                const badge = document.getElementById('settingsStatusBadge');
                if (badge) {
                    badge.className = 'badge badge-orange';
                    badge.innerText = '● Sync Pending';
                }
            }
        }

        async function saveDynamicSettings() {
            const badge = document.getElementById('settingsStatusBadge');
            if (badge) {
                badge.className = 'badge badge-orange';
                badge.innerText = '● Saving...';
            }
            const updates = {
                camera_id: document.getElementById('set_camera_id').value,
                camera_location: document.getElementById('set_camera_location').value,
                crash_conf: parseFloat(document.getElementById('set_crash_conf').value),
                fire_conf: parseFloat(document.getElementById('set_fire_conf').value),
                alert_cooldown_sec: parseFloat(document.getElementById('set_cooldown').value),
                emergency_search_radius_km: parseFloat(document.getElementById('set_radius').value),
                default_recipient_email: document.getElementById('set_email').value,
                rtsp_camera_ip: document.getElementById('set_rtsp_ip').value,
                emergency_dispatch_phone: document.getElementById('set_twilio_phone').value,
                video_source: document.getElementById('set_video_source') ? document.getElementById('set_video_source').value : undefined
            };

            try {
                const res = await fetch('/api/settings', {
                    method: 'PATCH',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(updates)
                });
                const data = await res.json();
                if (res.ok) {
                    if (badge) {
                        badge.className = 'badge badge-green';
                        badge.innerText = '● Synced';
                    }
                    alert('Settings updated dynamically at runtime!');
                    await loadAvailableSources();
                    const streamImg = document.getElementById('liveStreamImg');
                    if (streamImg && streamImg.src) streamImg.src = '/api/surveillance/stream?t=' + Date.now();
                    loadLocationAndFacilities();
                } else {
                    if (badge) {
                        badge.className = 'badge badge-red';
                        badge.innerText = '● Error';
                    }
                    alert('Save failed: ' + (data.detail || data.message || 'Unknown error'));
                }
            } catch (err) {
                console.error(err);
                if (badge) {
                    badge.className = 'badge badge-red';
                    badge.innerText = '● Error';
                }
                alert('Network error saving settings: ' + err);
            }
        }

        async function loadLogs() {
            try {
                const incRes = await fetch('/api/incidents?limit=10');
                const inc = await incRes.json();
                let incHtml = '';
                (inc.incidents || []).forEach(r => {
                    incHtml += `<tr><td>${r.Timestamp || ''}</td><td>${r.Event_Type || ''}</td><td>${r.Max_Accident_Severity || ''}</td><td><a href="/api/snapshots/${r.Snapshot_Path ? r.Snapshot_Path.split(/[\\\\/]/).pop() : ''}" target="_blank" style="color:var(--accent-blue)">View</a></td></tr>`;
                });
                document.getElementById('incidentsTableBody').innerHTML = incHtml || '<tr><td colspan="4">No incidents logged yet.</td></tr>';

                const dispRes = await fetch('/api/dispatches?limit=10');
                const disp = await dispRes.json();
                let dispHtml = '';
                (disp.dispatches || []).forEach(d => {
                    const depts = Object.keys(d.dispatched_services || {}).join(', ');
                    dispHtml += `<tr><td>${d.dispatch_id}</td><td>${d.incident ? d.incident.type : ''}</td><td>${depts}</td></tr>`;
                });
                document.getElementById('dispatchesTableBody').innerHTML = dispHtml || '<tr><td colspan="3">No dispatches recorded yet.</td></tr>';
            } catch(e) {
                console.error(e);
            }
        }

        async function initDashboard() {
            console.log('[Dashboard] initDashboard starting...');
            const clock = document.getElementById('clockTime');
            if (clock) clock.innerText = new Date().toLocaleTimeString();

            // 1. First fetch sources so dropdown optgroups are constructed
            await loadAvailableSources();

            // 2. Then load current settings (so set_video_source option matches), location, status, logs
            await Promise.allSettled([
                fetchStatus(),
                loadCurrentSettings(),
                loadLocationAndFacilities(),
                loadLogs()
            ]);
        }

        setInterval(fetchStatus, 2000);
        setInterval(() => {
            const clock = document.getElementById('clockTime');
            if (clock) clock.innerText = new Date().toLocaleTimeString();
        }, 1000);

        initDashboard();
    </script>
</body>
</html>
"""


@router.get(
    "/", response_class=HTMLResponse, summary="Sparsh CCTV Surveillance Web Dashboard"
)
@router.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Sparsh CCTV Surveillance Web Dashboard",
)
def get_dashboard():
    """Serves the single-page responsive web surveillance control center."""
    return HTMLResponse(content=DASHBOARD_HTML)
