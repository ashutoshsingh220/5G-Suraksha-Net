import React, { useState, useEffect, useRef } from 'react';
import {
  Camera,
  Maximize2,
  Minimize2,
  Users,
  Crosshair,
  Gauge,
  Zap,
  ChevronDown,
  Layers,
  Flame,
  RefreshCw,
  AlertTriangle,
} from 'lucide-react';
import { mockVideoSources, type VideoSourceItem } from '../../data/mockData';

export type VideoStreamStatus = 'CONNECTING' | 'CONNECTED' | 'DISCONNECTED' | 'ERROR' | 'UNAVAILABLE';

interface LiveVideoPanelProps {
  selectedSourceId?: string;
  onSelectSource?: (id: string) => void;
}

export const LiveVideoPanel: React.FC<LiveVideoPanelProps> = ({
  selectedSourceId = 'drone-cam',
  onSelectSource,
}) => {
  const panelRef = useRef<HTMLDivElement>(null);
  const videoImgRef = useRef<HTMLImageElement>(null);

  const [streamStatus, setStreamStatus] = useState<VideoStreamStatus>('CONNECTING');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [showOverlays, setShowOverlays] = useState(true);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [snapshotFlash, setSnapshotFlash] = useState(false);
  const [retryKey, setRetryKey] = useState(0);

  // Live telemetry state directly fed from real-time pipeline detection
  const [telemetry, setTelemetry] = useState({
    persons: 0,
    crowdDensity: 0.0,
    weapons: 0,
    fight: 0,
    fps: 0.0,
    latencyMs: 0,
  });

  const sources: VideoSourceItem[] = mockVideoSources;
  const currentSource = sources.find((s) => s.id === selectedSourceId) || sources[0];

  // Backend stream endpoint
  const backendStreamUrl = `http://localhost:8100/video/stream?t=${retryKey}`;

  // Handle source status: all sources connect to the live backend stream
  const isSourceLive = true;

  useEffect(() => {
    setStreamStatus('CONNECTING');
  }, [selectedSourceId]);

  // Fullscreen change listener
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange);
    };
  }, []);

  // Poll video status from backend when source is active (1-second responsive interval)
  useEffect(() => {
    if (!isSourceLive) return;

    let isMounted = true;
    const pollBackendStatus = async () => {
      try {
        const res = await fetch('http://localhost:8100/video/status');
        if (res.ok && isMounted) {
          const data = await res.json();
          setTelemetry({
            persons: typeof data.persons === 'number' ? data.persons : 0,
            crowdDensity: typeof data.crowd_density === 'number' ? data.crowd_density : 0.0,
            weapons: typeof data.weapons === 'number' ? data.weapons : 0,
            fight: typeof data.fights === 'number' ? data.fights : 0,
            fps: typeof data.effective_fps === 'number' ? data.effective_fps : 0.0,
            latencyMs: typeof data.latency_ms === 'number' ? data.latency_ms : 0,
          });
          if (data.has_live_frame || data.effective_fps > 0) {
            setStreamStatus('CONNECTED');
          }
        } else if (isMounted) {
          setStreamStatus('DISCONNECTED');
        }
      } catch {
        if (isMounted) {
          setStreamStatus('DISCONNECTED');
        }
      }
    };

    pollBackendStatus();
    const interval = setInterval(pollBackendStatus, 1000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [isSourceLive, retryKey]);

  // Fullscreen toggle handler
  const handleToggleFullscreen = () => {
    if (!panelRef.current) return;
    if (!document.fullscreenElement) {
      panelRef.current.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  };

  // Client-side snapshot capture
  const handleCaptureSnapshot = () => {
    setSnapshotFlash(true);
    setTimeout(() => setSnapshotFlash(false), 300);

    // If stream image is rendered, capture to canvas and download safely
    const img = videoImgRef.current;
    if (img && img.complete && img.naturalWidth > 0) {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(img, 0, 0);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.92);
          const a = document.createElement('a');
          a.href = dataUrl;
          a.download = `suraksha_snapshot_${Date.now()}.jpg`;
          a.click();
        }
      } catch {
        // Cross-origin canvas security fallback
      }
    }
  };

  const handleRetry = () => {
    setStreamStatus('CONNECTING');
    setRetryKey((prev) => prev + 1);
  };

  const handleSelectSource = async (id: string) => {
    if (onSelectSource) {
      onSelectSource(id);
    }
    setIsDropdownOpen(false);
    setStreamStatus('CONNECTING');
    try {
      if (id === 'drone-cam') {
        await fetch('http://localhost:8100/video/source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_type: 'rtsp',
            url: 'rtsp://10.254.18.48:8554/drone',
          }),
        });
      } else if (id === 'laptop-cam') {
        await fetch('http://localhost:8100/video/source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_type: 'webcam',
            camera_index: 0,
          }),
        });
      } else if (id === 'recorded-vid') {
        await fetch('http://localhost:8100/video/source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_type: 'file',
            url: 'datasets/videos/test/synthetic_cctv.mp4',
          }),
        });
      }
    } catch {
      // Backend not reached or starting up
    }
    setRetryKey((prev) => prev + 1);
  };

  const getStatusBadge = () => {
    switch (streamStatus) {
      case 'CONNECTED':
        return (
          <div className="flex items-center gap-1.5 bg-[#151E28] border border-[#2BC48A]/40 rounded-full px-2 py-0.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#2BC48A]" />
            <span className="text-[9px] font-mono font-medium text-[#2BC48A] tracking-wider">
              CONNECTED
            </span>
          </div>
        );
      case 'CONNECTING':
        return (
          <div className="flex items-center gap-1.5 bg-[#151E28] border border-[#E7A83B]/40 rounded-full px-2 py-0.5">
            <RefreshCw className="w-2.5 h-2.5 text-[#E7A83B] animate-spin" />
            <span className="text-[9px] font-mono font-medium text-[#E7A83B] tracking-wider">
              CONNECTING
            </span>
          </div>
        );
      case 'UNAVAILABLE':
        return (
          <div className="flex items-center gap-1.5 bg-[#151E28] border border-[#263341] rounded-full px-2 py-0.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#687585]" />
            <span className="text-[9px] font-mono font-medium text-[#98A6B5] tracking-wider">
              OFFLINE
            </span>
          </div>
        );
      case 'ERROR':
      case 'DISCONNECTED':
      default:
        return (
          <div className="flex items-center gap-1.5 bg-[#151E28] border border-[#E05252]/40 rounded-full px-2 py-0.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#E05252]" />
            <span className="text-[9px] font-mono font-medium text-[#E05252] tracking-wider">
              DISCONNECTED
            </span>
          </div>
        );
    }
  };

  return (
    <div
      ref={panelRef}
      className={`suraksha-panel flex flex-col h-full overflow-hidden transition-all ${
        isFullscreen ? 'fixed inset-0 z-50 rounded-none bg-[#0B0F14]' : ''
      }`}
    >
      {/* Top Video Header / Tactical Toolbar */}
      <div className="suraksha-panel-header px-3 py-1.5 flex items-center justify-between flex-shrink-0 z-10">
        <div className="flex items-center gap-2.5">
          {/* LIVE Pill with pulsing red dot when actively streaming */}
          <div
            className={`flex items-center gap-1.5 rounded-full px-2 py-0.5 border ${
              streamStatus === 'CONNECTED'
                ? 'bg-[#151E28] border-[#E05252]/40'
                : 'bg-[#151E28] border-[#263341]'
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                streamStatus === 'CONNECTED' ? 'bg-[#E05252] animate-pulse' : 'bg-[#687585]'
              }`}
            />
            <span
              className={`text-[10px] font-mono font-semibold tracking-wider ${
                streamStatus === 'CONNECTED' ? 'text-[#E05252]' : 'text-[#687585]'
              }`}
            >
              LIVE
            </span>
          </div>

          {/* Stream Selector Dropdown */}
          <div className="relative">
            <button
              onClick={() => setIsDropdownOpen(!isDropdownOpen)}
              className="flex items-center gap-1.5 bg-[#111821] hover:bg-[#151E28] border border-[#263341] px-2 py-0.5 rounded-[6px] text-xs font-mono text-[#E8EDF3] transition-colors"
            >
              <span className="text-[#3B9EFF] font-semibold">{currentSource.name}</span>
              <ChevronDown className="w-3 h-3 text-[#98A6B5]" />
            </button>

            {isDropdownOpen && (
              <div className="absolute left-0 mt-1 w-60 bg-[#111821] border border-[#263341] rounded-[8px] shadow-2xl py-1 z-30 font-mono text-xs">
                {sources.map((s) => (
                  <button
                    key={s.id}
                    onClick={() => handleSelectSource(s.id)}
                    className={`w-full text-left px-3 py-1.5 hover:bg-[#151E28] transition-colors flex items-center justify-between ${
                      s.id === selectedSourceId
                        ? 'text-[#3B9EFF] font-semibold bg-[#151E28]'
                        : 'text-[#E8EDF3]'
                    }`}
                  >
                    <div className="flex items-center gap-1.5">
                      <span>{s.name}</span>
                    </div>
                    {s.status === 'STANDBY' ? (
                      <span className="text-[8.5px] px-1.5 py-0.2 rounded bg-[#151E28] text-[#98A6B5] border border-[#263341]">
                        OFFLINE
                      </span>
                    ) : (
                      <span className="text-[8.5px] px-1.5 py-0.2 rounded bg-[#151E28] text-[#2BC48A] border border-[#2BC48A]/40">
                        READY
                      </span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Connection Status Pill */}
          {getStatusBadge()}
        </div>

        {/* Video Controls (Right) */}
        <div className="flex items-center gap-1.5 text-[#98A6B5]">
          <button
            onClick={() => setShowOverlays(!showOverlays)}
            title={showOverlays ? 'Hide AI Overlays' : 'Show AI Overlays'}
            className={`px-2 py-0.5 rounded-[6px] border text-[10px] font-mono flex items-center gap-1 transition-colors ${
              showOverlays
                ? 'bg-[#151E28] border-[#3B9EFF] text-[#3B9EFF]'
                : 'bg-[#151E28] border-[#263341] text-[#98A6B5] hover:text-[#E8EDF3]'
            }`}
          >
            <Layers className="w-3 h-3" />
            <span className="hidden md:inline">HUD</span>
          </button>
          <button
            onClick={handleCaptureSnapshot}
            title="Capture Shutter Snapshot"
            className="p-1 rounded-[6px] border border-[#263341] bg-[#151E28] hover:bg-[#263341] hover:text-[#E8EDF3] transition-colors"
          >
            <Camera className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleToggleFullscreen}
            title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
            className="p-1 rounded-[6px] border border-[#263341] bg-[#151E28] hover:bg-[#263341] hover:text-[#E8EDF3] transition-colors"
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Main Video Viewport (Generous 16:9 Canvas) */}
      <div className="relative flex-1 min-h-[460px] xl:min-h-[520px] bg-[#0B0F14] overflow-hidden flex items-center justify-center">
        {/* Shutter Flash Animation */}
        {snapshotFlash && (
          <div className="absolute inset-0 bg-white/30 z-30 pointer-events-none transition-opacity duration-200" />
        )}

        {/* Case 1: Active Stream (Laptop Webcam or MP4 video) */}
        {isSourceLive && (
          <div className="relative w-full h-full flex items-center justify-center bg-black">
            {/* Live MJPEG Stream Image with automated fallback */}
            <img
              ref={videoImgRef}
              src={backendStreamUrl}
              alt="Live 5G Suraksha-Net CCTV Stream"
              className="w-full h-full object-contain"
              onLoad={() => setStreamStatus('CONNECTED')}
              onError={() => {
                // If stream is still loading or backend is momentarily starting up
                if (streamStatus !== 'DISCONNECTED') {
                  setStreamStatus('DISCONNECTED');
                }
              }}
            />

            {/* Offline / Disconnected Overlay State */}
            {streamStatus === 'DISCONNECTED' && (
              <div className="absolute inset-0 bg-[#0B0F14]/95 flex flex-col items-center justify-center p-6 text-center z-20">
                <div className="w-12 h-12 rounded-full bg-[#151E28] border border-[#E7A83B]/40 flex items-center justify-center text-[#E7A83B] mb-3">
                  <AlertTriangle className="w-6 h-6" />
                </div>
                <h3 className="text-sm font-mono font-bold text-[#E8EDF3] tracking-wider uppercase mb-1">
                  BACKEND STREAM OFFLINE
                </h3>
                <p className="text-xs font-mono text-[#98A6B5] max-w-md mb-4 leading-relaxed">
                  FastAPI video stream at <code className="text-[#3B9EFF]">http://localhost:8100/video/stream</code> is not publishing frames yet.
                  Start the backend pipeline via <code className="text-[#E7A83B]">python scripts/run_api.py</code>.
                </p>
                <div className="flex items-center gap-3">
                  <button
                    onClick={handleRetry}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-[6px] bg-[#3B9EFF] hover:bg-[#2e82d3] text-white font-mono text-xs font-medium transition-colors shadow-sm"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>RETRY STREAM</span>
                  </button>
                  <button
                    onClick={() => {
                      if (videoImgRef.current) {
                        videoImgRef.current.src = '/cctv_live_stream.png';
                        setStreamStatus('CONNECTED');
                      }
                    }}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-[6px] bg-[#151E28] hover:bg-[#263341] border border-[#263341] text-[#E8EDF3] font-mono text-xs transition-colors"
                  >
                    <span>VIEW STORED SAMPLE</span>
                  </button>
                </div>
              </div>
            )}

            {/* Live Watermark Overlay */}
            {showOverlays && streamStatus === 'CONNECTED' && (
              <div className="absolute bottom-2 right-2 text-[9px] font-mono text-[#98A6B5] bg-[#0B0F14]/90 px-2 py-1 rounded-[6px] border border-[#263341] pointer-events-none flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-[#2BC48A] animate-pulse" />
                <span>
                  {currentSource.name.toUpperCase()} • 5G SURAKSHA-NET • YOLO11s + BYTETRACK
                </span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Telemetry Strip Directly Under Video */}
      <div className="grid grid-cols-6 gap-2 p-2 bg-[#111821] border-t border-[#263341] flex-shrink-0">
        {/* Persons */}
        <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 flex items-center gap-2.5">
          <div className="p-1.5 rounded-[4px] bg-[#111821] border border-[#263341] text-[#98A6B5]">
            <Users className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-mono uppercase text-[#98A6B5] leading-none">Persons</div>
            <div className="text-base font-mono font-semibold text-[#E8EDF3] leading-tight mt-1">{telemetry.persons}</div>
          </div>
        </div>

        {/* Crowd Density */}
        <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 flex items-center gap-2.5">
          <div className="p-1.5 rounded-[4px] bg-[#111821] border border-[#263341] text-[#E7A83B]">
            <Users className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-mono uppercase text-[#98A6B5] leading-none">Crowd Density</div>
            <div className="text-base font-mono font-semibold text-[#E8EDF3] leading-tight mt-1">{telemetry.crowdDensity.toFixed(2)}</div>
          </div>
        </div>

        {/* Weapons */}
        <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 flex items-center gap-2.5">
          <div className="p-1.5 rounded-[4px] bg-[#111821] border border-[#263341] text-[#E05252]">
            <Crosshair className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-mono uppercase text-[#98A6B5] leading-none">Weapons</div>
            <div className={`text-base font-mono font-semibold leading-tight mt-1 ${telemetry.weapons > 0 ? 'text-[#E05252]' : 'text-[#E8EDF3]'}`}>{telemetry.weapons}</div>
          </div>
        </div>

        {/* Fight */}
        <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 flex items-center gap-2.5">
          <div className="p-1.5 rounded-[4px] bg-[#111821] border border-[#263341] text-[#E05252]">
            <Flame className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-mono uppercase text-[#98A6B5] leading-none">Fight</div>
            <div className={`text-base font-mono font-semibold leading-tight mt-1 ${telemetry.fight > 0 ? 'text-[#E05252]' : 'text-[#E8EDF3]'}`}>{telemetry.fight}</div>
          </div>
        </div>

        {/* FPS */}
        <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 flex items-center gap-2.5">
          <div className="p-1.5 rounded-[4px] bg-[#111821] border border-[#263341] text-[#49C6D9]">
            <Gauge className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-mono uppercase text-[#98A6B5] leading-none">FPS</div>
            <div className="text-base font-mono font-semibold text-[#49C6D9] leading-tight mt-1">{telemetry.fps.toFixed(1)}</div>
          </div>
        </div>

        {/* Latency */}
        <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 flex items-center gap-2.5">
          <div className="p-1.5 rounded-[4px] bg-[#111821] border border-[#263341] text-[#3B9EFF]">
            <Zap className="w-4 h-4" />
          </div>
          <div>
            <div className="text-[11px] font-mono uppercase text-[#98A6B5] leading-none">Latency</div>
            <div className="text-base font-mono font-semibold text-[#E8EDF3] leading-tight mt-1">{telemetry.latencyMs} ms</div>
          </div>
        </div>
      </div>
    </div>
  );
};
