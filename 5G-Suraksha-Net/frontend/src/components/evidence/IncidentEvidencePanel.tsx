import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Camera,
  Video,
  Maximize2,
  Download,
  AlertTriangle,
  CheckCircle2,
  Clock,
  X,
  RefreshCw,
  Eye,
  Film,
} from 'lucide-react';
import { useIncidents } from '../../context/IncidentContext';

export const IncidentEvidencePanel: React.FC = () => {
  const { activeIncident } = useIncidents();

  const [activeTab, setActiveTab] = useState<'snapshot' | 'clip'>('snapshot');
  const [snapshotLoaded, setSnapshotLoaded] = useState<boolean>(false);
  const [snapshotError, setSnapshotError] = useState<boolean>(false);
  const [clipError, setClipError] = useState<boolean>(false);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [clipFinalized, setClipFinalized] = useState<boolean>(false);
  const [retryCount, setRetryCount] = useState<number>(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const modalVideoRef = useRef<HTMLVideoElement | null>(null);

  const incidentId = activeIncident?.incident_id;
  const isRecordingPostEvent = activeIncident?.status === 'recording_post_event';

  const querySuffix = retryCount > 0 ? `?t=${retryCount}` : '';
  const snapshotUrl = incidentId
    ? `http://localhost:8100/incidents/${incidentId}/snapshot${querySuffix}`
    : '';
  const clipUrl = incidentId
    ? `http://localhost:8100/incidents/${incidentId}/clip${querySuffix}`
    : '';

  // Reset media states when selected incident changes
  useEffect(() => {
    setSnapshotLoaded(false);
    setSnapshotError(false);
    setClipError(false);
    setRetryCount(0);
    setClipFinalized(activeIncident?.status === 'finalized');
  }, [incidentId, activeIncident?.status]);

  // Bounded check for post-event clip finalization if arriving during recording_post_event
  useEffect(() => {
    if (!incidentId || !isRecordingPostEvent) return;

    let attempts = 0;
    const maxAttempts = 6; // 6 * 2.5s = 15s max bounded check
    const interval = setInterval(() => {
      attempts++;
      fetch(`http://localhost:8100/incidents/${incidentId}/clip`, { method: 'HEAD' })
        .then((res) => {
          if (res.ok) {
            setClipFinalized(true);
            setClipError(false);
            clearInterval(interval);
          }
        })
        .catch(() => {});

      if (attempts >= maxAttempts) {
        clearInterval(interval);
      }
    }, 2500);

    return () => clearInterval(interval);
  }, [incidentId, isRecordingPostEvent]);

  // Close modal on ESC key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setIsModalOpen(false);
      }
    };
    if (isModalOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isModalOpen]);

  const handleDownload = useCallback((url: string, filename: string) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.target = '_blank';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  // Determine Overall Evidence Status Badge
  const getEvidenceStatusBadge = () => {
    if (!activeIncident) {
      return (
        <span className="text-[11px] font-mono font-medium text-[#687585] bg-[#151E28] border border-[#263341] px-2 py-0.5 rounded-[4px]">
          STANDBY
        </span>
      );
    }
    if (isRecordingPostEvent && !clipFinalized) {
      return (
        <span className="text-[11px] font-mono font-medium text-[#E7A83B] bg-[#151E28] border border-[#E7A83B]/50 px-2 py-0.5 rounded-[4px] flex items-center gap-1">
          <Clock className="w-3 h-3 animate-spin" />
          <span>EVIDENCE RECORDING</span>
        </span>
      );
    }
    if (snapshotError && clipError) {
      return (
        <span className="text-[11px] font-mono font-medium text-[#E05252] bg-[#151E28] border border-[#E05252]/50 px-2 py-0.5 rounded-[4px]">
          EVIDENCE UNAVAILABLE
        </span>
      );
    }
    return (
      <span className="text-[11px] font-mono font-medium text-[#2BC48A] bg-[#151E28] border border-[#2BC48A]/50 px-2 py-0.5 rounded-[4px] flex items-center gap-1">
        <CheckCircle2 className="w-3 h-3" />
        <span>EVIDENCE AVAILABLE</span>
      </span>
    );
  };

  return (
    <div className="suraksha-panel flex flex-col h-full overflow-hidden border border-[#263341]">
      {/* 1. Header */}
      <div className="suraksha-panel-header px-3 py-1.5 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <Film className="w-3.5 h-3.5 text-[#3B9EFF]" />
          <h2 className="text-xs font-semibold font-sans tracking-wider text-[#E8EDF3] uppercase">
            INCIDENT EVIDENCE & FORENSIC REVIEW
          </h2>
        </div>

        <div className="flex items-center gap-2">
          {/* Toggle Tabs: Snapshot vs Video Clip */}
          {activeIncident && (
            <div className="flex items-center bg-[#0B0F14] border border-[#263341] rounded-[4px] p-0.5">
              <button
                onClick={() => setActiveTab('snapshot')}
                className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded-[2px] text-xs font-mono transition-colors cursor-pointer ${
                  activeTab === 'snapshot'
                    ? 'bg-[#151E28] text-[#3B9EFF] font-medium border border-[#3B9EFF]/40'
                    : 'text-[#98A6B5] hover:text-[#E8EDF3]'
                }`}
              >
                <Camera className="w-3.5 h-3.5" />
                <span>Snapshot</span>
              </button>
              <button
                onClick={() => setActiveTab('clip')}
                className={`flex items-center gap-1.5 px-2.5 py-0.5 rounded-[2px] text-xs font-mono transition-colors cursor-pointer ${
                  activeTab === 'clip'
                    ? 'bg-[#151E28] text-[#3B9EFF] font-medium border border-[#3B9EFF]/40'
                    : 'text-[#98A6B5] hover:text-[#E8EDF3]'
                }`}
              >
                <Video className="w-3.5 h-3.5" />
                <span>Evidence Clip</span>
              </button>
            </div>
          )}

          {getEvidenceStatusBadge()}
        </div>
      </div>

      {/* 2. Main Body Content */}
      <div className="p-2 flex-1 flex flex-col gap-2 min-h-0 overflow-hidden">
        {/* State A: No incident selected */}
        {!activeIncident ? (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-4 gap-2">
            <Film className="w-8 h-8 text-[#687585] stroke-[1.5]" />
            <div className="text-xs font-sans font-semibold text-[#E8EDF3] uppercase tracking-wide">
              NO INCIDENT SELECTED
            </div>
            <p className="text-[10.5px] font-sans text-[#98A6B5] max-w-[280px]">
              Select an incident from the Incident Feed or Active Incident Card to inspect forensic snapshots and evidence clips.
            </p>
            <div className="text-[9px] font-mono text-[#687585] mt-2 border border-[#263341] bg-[#111821] px-2 py-0.5 rounded-[4px]">
              SYSTEM MONITORING &bull; ZERO SIMULATED EVIDENCE
            </div>
          </div>
        ) : (
          /* State B: Incident is active */
          <div className="flex-1 flex flex-col gap-2 min-h-0 overflow-hidden">
            {/* Media Viewer Area */}
            <div className="relative flex-1 min-h-[220px] bg-[#0B0F14] border border-[#263341] rounded-[6px] overflow-hidden flex items-center justify-center">
              {activeTab === 'snapshot' ? (
                /* Tab 1: Snapshot View */
                <div className="relative w-full h-full flex items-center justify-center overflow-hidden">
                  {!snapshotError ? (
                    <img
                      src={snapshotUrl}
                      alt={`Incident ${incidentId} Snapshot`}
                      onLoad={() => {
                        setSnapshotLoaded(true);
                        setSnapshotError(false);
                      }}
                      onError={() => {
                        setSnapshotLoaded(false);
                        setSnapshotError(true);
                      }}
                      className="max-h-full max-w-full object-contain cursor-pointer select-none"
                      onClick={() => setIsModalOpen(true)}
                    />
                  ) : (
                    <div className="flex flex-col items-center justify-center text-center p-4 gap-1.5 text-[#98A6B5]">
                      <AlertTriangle className="w-6 h-6 text-[#E7A83B]" />
                      <div className="text-xs font-mono font-semibold text-[#E8EDF3]">
                        SNAPSHOT UNAVAILABLE
                      </div>
                      <div className="text-[10px] font-mono text-[#687585]">
                        File not present on server or still buffering.
                      </div>
                      <button
                        onClick={() => {
                          setSnapshotError(false);
                          setSnapshotLoaded(false);
                          setRetryCount((c) => c + 1);
                        }}
                        className="mt-2 inline-flex items-center gap-1 text-[10px] font-mono text-[#3B9EFF] hover:text-[#2e82d3] bg-[#111821] border border-[#263341] px-2 py-0.5 rounded-[4px] cursor-pointer"
                      >
                        <RefreshCw className="w-3 h-3" />
                        <span>Retry</span>
                      </button>
                    </div>
                  )}

                  {/* Top-Right Quick Expand Action */}
                  {snapshotLoaded && !snapshotError && (
                    <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-[#111821]/90 border border-[#263341] p-1 rounded-[4px] backdrop-blur-xs">
                      <button
                        onClick={() => setIsModalOpen(true)}
                        className="p-1 hover:bg-[#151E28] text-[#98A6B5] hover:text-[#E8EDF3] rounded transition-colors cursor-pointer"
                        title="View Full Resolution Snapshot"
                      >
                        <Maximize2 className="w-3.5 h-3.5" />
                      </button>
                      <button
                        onClick={() => handleDownload(snapshotUrl, `${incidentId}_snapshot.jpg`)}
                        className="p-1 hover:bg-[#151E28] text-[#3B9EFF] hover:text-[#2e82d3] rounded transition-colors cursor-pointer"
                        title="Download Original Snapshot"
                      >
                        <Download className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}
                </div>
              ) : (
                /* Tab 2: Evidence Video Clip View */
                <div className="relative w-full h-full flex items-center justify-center overflow-hidden bg-black">
                  {isRecordingPostEvent && !clipFinalized ? (
                    <div className="flex flex-col items-center justify-center text-center p-4 gap-2">
                      <Clock className="w-7 h-7 text-[#E7A83B] animate-spin" />
                      <div className="text-xs font-mono font-semibold text-[#E7A83B] uppercase">
                        RECORDING POST-EVENT EVIDENCE CLIP
                      </div>
                      <p className="text-[10px] font-mono text-[#98A6B5] max-w-[280px]">
                        Capturing T-5s to T+5s temporal evidence window. Clip will auto-refresh upon encoding completion.
                      </p>
                    </div>
                  ) : !clipError ? (
                    <video
                      key={clipUrl}
                      ref={videoRef}
                      src={clipUrl}
                      controls
                      playsInline
                      preload="auto"
                      onError={() => setClipError(true)}
                      className="max-h-full max-w-full object-contain"
                    >
                      <source src={clipUrl} type="video/mp4" />
                      Your browser does not support HTML5 video playback.
                    </video>
                  ) : (
                    <div className="flex flex-col items-center justify-center text-center p-4 gap-1.5 text-[#98A6B5]">
                      <AlertTriangle className="w-6 h-6 text-[#E7A83B]" />
                      <div className="text-xs font-mono font-semibold text-[#E8EDF3]">
                        VIDEO EVIDENCE UNAVAILABLE
                      </div>
                      <div className="text-[10px] font-mono text-[#687585]">
                        Clip not recorded or undergoing compilation.
                      </div>
                      <button
                        onClick={() => {
                          setClipError(false);
                          setRetryCount((c) => c + 1);
                        }}
                        className="mt-2 inline-flex items-center gap-1 text-[10px] font-mono text-[#3B9EFF] hover:text-[#2e82d3] bg-[#111821] border border-[#263341] px-2 py-0.5 rounded-[4px] cursor-pointer"
                      >
                        <RefreshCw className="w-3 h-3" />
                        <span>Retry</span>
                      </button>
                    </div>
                  )}

                  {/* Top-Right Clip Download Action */}
                  {!clipError && (!isRecordingPostEvent || clipFinalized) && (
                    <div className="absolute top-2 right-2 flex items-center gap-1.5 bg-[#111821]/90 border border-[#263341] p-1 rounded-[4px] backdrop-blur-xs">
                      <button
                        onClick={() => handleDownload(clipUrl, `${incidentId}_clip.mp4`)}
                        className="p-1 hover:bg-[#151E28] text-[#3B9EFF] hover:text-[#2e82d3] rounded transition-colors"
                        title="Download Original MP4 Clip"
                      >
                        <Download className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Forensic Metadata Strip & Action Bar */}
            <div className="bg-[#151E28] border border-[#263341] p-2 rounded-[6px] flex flex-col gap-1.5 text-xs font-mono">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 truncate">
                  <span className="text-[#687585]">ID:</span>
                  <span className="text-[#E8EDF3] font-semibold truncate max-w-[130px]">{activeIncident.incident_id}</span>
                  <span className="text-[#263341]">&bull;</span>
                  <span className="text-[#98A6B5] font-medium uppercase">{activeIncident.incident_type.replace(/_/g, ' ')}</span>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <span
                    className={`font-semibold px-2 py-0.5 rounded-[3px] border text-[10.5px] uppercase ${
                      activeIncident.severity === 'critical'
                        ? 'bg-[#E05252] text-white border-[#E05252]'
                        : activeIncident.severity === 'high'
                        ? 'bg-[#E7A83B] text-white border-[#E7A83B]'
                        : 'bg-[#111821] text-[#E7A83B] border-[#E7A83B]/50'
                    }`}
                  >
                    {activeIncident.severity}
                  </span>
                  <span className="text-[#3B9EFF] font-semibold text-xs">
                    {(activeIncident.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              </div>

              <div className="flex items-center justify-between text-[11px] text-[#98A6B5] pt-1 border-t border-[#263341]">
                <div className="truncate">
                  <span className="text-[#687585]">Camera: </span>
                  <span className="text-[#E8EDF3]">{activeIncident.camera_id}</span>
                  {activeIncident.evidence?.frame_idx !== undefined && activeIncident.evidence?.frame_idx !== null && (
                    <span className="text-[#687585] ml-2">
                      Frame #{activeIncident.evidence.frame_idx}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setIsModalOpen(true)}
                    className="inline-flex items-center gap-1.5 text-[#3B9EFF] hover:text-[#2e82d3] font-medium cursor-pointer"
                  >
                    <Eye className="w-3.5 h-3.5" />
                    <span>View Fullscreen</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 3. Investigation Fullscreen Modal Overlay */}
      {isModalOpen && activeIncident && (
        <div
          className="fixed inset-0 z-50 bg-[#0B0F14]/90 backdrop-blur-sm flex items-center justify-center p-4 select-none"
          onClick={() => setIsModalOpen(false)}
        >
          <div
            className="relative w-full max-w-5xl max-h-[92vh] bg-[#111821] border border-[#263341] rounded-[8px] shadow-2xl flex flex-col overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div className="px-4 py-2.5 bg-[#151E28] border-b border-[#263341] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Film className="w-4 h-4 text-[#3B9EFF]" />
                <span className="text-xs font-mono font-semibold text-[#E8EDF3] uppercase tracking-wider">
                  FORENSIC INVESTIGATION VIEWER &bull; {activeIncident.incident_id}
                </span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-[10px] font-mono text-[#98A6B5]">
                  Press <kbd className="px-1 py-0.5 bg-[#111821] border border-[#263341] rounded text-[#E8EDF3]">ESC</kbd> to close
                </span>
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="p-1 hover:bg-[#263341] text-[#98A6B5] hover:text-[#E8EDF3] rounded transition-colors cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Modal Body */}
            <div className="flex-1 p-3 flex flex-col md:flex-row gap-3 min-h-0 overflow-hidden">
              {/* Media Container (70% width) */}
              <div className="flex-1 bg-black rounded-[6px] border border-[#263341] overflow-hidden flex items-center justify-center relative min-h-[360px]">
                {activeTab === 'snapshot' ? (
                  !snapshotError ? (
                    <img
                      src={snapshotUrl}
                      alt={`Full Resolution Snapshot ${incidentId}`}
                      className="max-h-full max-w-full object-contain"
                    />
                  ) : (
                    <div className="text-center p-4 text-[#98A6B5] font-mono text-xs">
                      Snapshot unavailable.
                    </div>
                  )
                ) : !clipError && (!isRecordingPostEvent || clipFinalized) ? (
                  <video
                    key={clipUrl}
                    ref={modalVideoRef}
                    src={clipUrl}
                    controls
                    autoPlay
                    playsInline
                    className="max-h-full max-w-full object-contain"
                  >
                    Your browser does not support HTML5 video playback.
                  </video>
                ) : (
                  <div className="text-center p-4 text-[#98A6B5] font-mono text-xs">
                    {isRecordingPostEvent ? 'Recording in progress...' : 'Video clip unavailable.'}
                  </div>
                )}
              </div>

              {/* Forensic Details Sidebar (30% width) */}
              <div className="w-full md:w-80 flex flex-col gap-2.5 text-xs font-mono bg-[#151E28] border border-[#263341] p-3 rounded-[6px] overflow-y-auto">
                <div className="text-[11px] font-semibold text-[#3B9EFF] uppercase border-b border-[#263341] pb-1">
                  Incident Forensics
                </div>

                <div className="space-y-1.5">
                  <div className="flex justify-between py-1 border-b border-[#263341]">
                    <span className="text-[#98A6B5]">Incident ID</span>
                    <span className="text-[#E8EDF3] font-semibold">{activeIncident.incident_id}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#263341]">
                    <span className="text-[#98A6B5]">Type</span>
                    <span className="text-[#E8EDF3] font-semibold uppercase">{activeIncident.incident_type}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#263341]">
                    <span className="text-[#98A6B5]">Severity</span>
                    <span className="text-[#E05252] font-semibold uppercase">{activeIncident.severity}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#263341]">
                    <span className="text-[#98A6B5]">Confidence</span>
                    <span className="text-[#2BC48A] font-semibold">{(activeIncident.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#263341]">
                    <span className="text-[#98A6B5]">Camera Source</span>
                    <span className="text-[#E8EDF3] font-semibold">{activeIncident.camera_id}</span>
                  </div>
                  <div className="flex justify-between py-1 border-b border-[#263341]">
                    <span className="text-[#98A6B5]">Start Time</span>
                    <span className="text-[#E8EDF3] text-[10px]">{activeIncident.start_time}</span>
                  </div>
                  {activeIncident.location && (
                    <div className="flex flex-col py-1 border-b border-[#263341]">
                      <span className="text-[#98A6B5]">Resolved Location</span>
                      <span className="text-[#E8EDF3] text-[10px] mt-0.5">{activeIncident.location.name}</span>
                    </div>
                  )}
                  {activeIncident.evidence?.frame_idx !== undefined && (
                    <div className="flex justify-between py-1 border-b border-[#263341]">
                      <span className="text-[#98A6B5]">Frame Index</span>
                      <span className="text-[#49C6D9] font-semibold">#{activeIncident.evidence.frame_idx}</span>
                    </div>
                  )}
                </div>

                {/* Artifact Downloads in Modal */}
                <div className="mt-auto pt-2 space-y-1.5">
                  <button
                    onClick={() => handleDownload(snapshotUrl, `${incidentId}_snapshot.jpg`)}
                    className="w-full flex items-center justify-center gap-1.5 py-1.5 px-3 bg-[#111821] hover:bg-[#263341] text-[#3B9EFF] border border-[#263341] rounded-[6px] text-[11px] font-mono font-medium transition-colors cursor-pointer"
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download Snapshot (JPEG)</span>
                  </button>
                  <button
                    onClick={() => handleDownload(clipUrl, `${incidentId}_clip.mp4`)}
                    disabled={isRecordingPostEvent && !clipFinalized}
                    className={`w-full flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-[6px] text-[11px] font-mono font-medium transition-colors border ${
                      isRecordingPostEvent && !clipFinalized
                        ? 'bg-[#111821] border-[#263341] text-[#687585] cursor-not-allowed'
                        : 'bg-[#111821] hover:bg-[#263341] text-[#3B9EFF] border-[#263341] cursor-pointer'
                    }`}
                  >
                    <Download className="w-3.5 h-3.5" />
                    <span>Download Evidence Clip (MP4)</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
