import React, { useState } from 'react';
import {
  AlertTriangle,
  Eye,
  Video,
  ShieldCheck,
  Radar,
  Brain,
} from 'lucide-react';
import { useIncidents } from '../../context/IncidentContext';
import { useSystemStatus } from '../../hooks/useSystemStatus';
import type { IncidentSeverity, IncidentType } from '../../types/incidents';
import { AgentAssessmentCard } from '../agents/AgentAssessmentCard';

export const IncidentDetectedCard: React.FC = () => {
  const { activeIncident, newIncidentHighlight, connectionStatus } = useIncidents();
  const { status: sysStatus } = useSystemStatus({ pollingIntervalMs: 1500 });
  const [showAgentBriefing, setShowAgentBriefing] = useState<boolean>(false);
  const [showVideoModal, setShowVideoModal] = useState<boolean>(false);
  const [snapshotError, setSnapshotError] = useState<boolean>(false);

  React.useEffect(() => {
    setSnapshotError(false);
  }, [activeIncident?.incident_id]);

  // Format incident type for clean display
  const formatIncidentType = (type: IncidentType): string => {
    switch (type) {
      case 'armed_fight':
        return 'ARMED FIGHT';
      case 'weapon':
        return 'WEAPON DETECTED';
      case 'fight':
        return 'FIGHT / VIOLENCE';
      case 'crowd_panic':
        return 'CROWD PANIC';
      case 'crowd_density_critical':
        return 'CRITICAL DENSITY';
      case 'crowd_density_high':
        return 'HIGH CROWD DENSITY';
      case 'crowd_rapid_growth':
        return 'RAPID CROWD SURGE';
      default:
        return String(type).replace(/_/g, ' ').toUpperCase();
    }
  };

  // Semantic severity badges (no neon, strictly professional)
  const getSeverityBadge = (sev: IncidentSeverity) => {
    const s = String(sev).toLowerCase();
    switch (s) {
      case 'critical':
        return (
          <span className="bg-[#E05252] text-white px-2 py-0.5 rounded-[4px] text-[9.5px] font-mono font-semibold tracking-wider uppercase">
            CRITICAL
          </span>
        );
      case 'high':
        return (
          <span className="bg-[#E7A83B] text-white px-2 py-0.5 rounded-[4px] text-[9.5px] font-mono font-semibold tracking-wider uppercase">
            HIGH
          </span>
        );
      case 'medium':
      case 'moderate':
        return (
          <span className="bg-[#151E28] border border-[#E7A83B]/50 text-[#E7A83B] px-2 py-0.5 rounded-[4px] text-[9.5px] font-mono font-semibold tracking-wider uppercase">
            {s.toUpperCase()}
          </span>
        );
      case 'low':
      default:
        return (
          <span className="bg-[#151E28] border border-[#263341] text-[#98A6B5] px-2 py-0.5 rounded-[4px] text-[9.5px] font-mono font-semibold tracking-wider uppercase">
            LOW
          </span>
        );
    }
  };

  // Format ISO timestamp
  const formatTime = (isoString?: string | null): string => {
    if (!isoString) return '--:--:--';
    try {
      const d = new Date(isoString);
      return d.toLocaleTimeString('en-IN', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      }) + ' IST';
    } catch {
      return isoString;
    }
  };

  // ---------------------------------------------------------------------------
  // 1. NO ACTIVE INCIDENT (PROFESSIONAL OPERATIONAL MONITORING STATE)
  // ---------------------------------------------------------------------------
  if (!activeIncident) {
    return (
      <div className="suraksha-panel flex flex-col overflow-hidden transition-all duration-300">
        {/* Header */}
        <div className="suraksha-panel-header px-3 py-1.5 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-1.5 text-[#E8EDF3]">
            <ShieldCheck className="w-3.5 h-3.5 text-[#3B9EFF]" />
            <h2 className="text-[11px] font-semibold font-sans tracking-wider uppercase">
              SYSTEM MONITORING
            </h2>
          </div>
          <div className="flex items-center gap-1.5 bg-[#151E28] border border-[#2BC48A]/40 rounded-full px-2 py-0.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#2BC48A]" />
            <span className="text-[9px] font-mono font-medium text-[#2BC48A]">
              AI Operational
            </span>
          </div>
        </div>

        <div className="p-3 flex flex-col gap-2.5">
          {/* Radar / Surveillance Graphic */}
          <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2.5 flex items-center gap-3">
            <div className="relative w-12 h-12 rounded-full bg-[#111821] border border-[#263341] flex items-center justify-center flex-shrink-0">
              <Radar className="w-6 h-6 text-[#3B9EFF]/80 animate-spin" style={{ animationDuration: '6s' }} />
              <div className="absolute inset-0 rounded-full border border-[#3B9EFF]/20" />
              <div className="w-1.5 h-1.5 rounded-full bg-[#2BC48A]" />
            </div>

            <div className="flex-1 min-w-0">
              <div className="text-xs font-semibold font-sans text-[#E8EDF3] uppercase tracking-wide">
                No Active Incidents
              </div>
              <div className="text-[10px] font-sans text-[#98A6B5] leading-relaxed mt-0.5">
                AI inference actively monitoring live camera feeds. Multi-modal fusion standing by.
              </div>
            </div>
          </div>

          {/* Subsystem Pipeline Readiness Grid */}
          {(() => {
            const isStreamLive = sysStatus?.video_source?.status === 'LIVE' || (sysStatus?.video_source?.effective_fps ?? 0) > 0;
            const detState = (sysStatus as any)?.detection_state;
            const weaponsCount = detState?.weapons ?? 0;
            const fightsCount = detState?.fights ?? 0;
            const crowdDensity = detState?.crowd_density ?? 0.0;

            return (
              <div className="grid grid-cols-2 gap-1.5 text-[10px] font-mono">
                <div className="bg-[#151E28] border border-[#263341] rounded-[6px] px-2 py-1 flex items-center justify-between">
                  <span className="text-[#98A6B5]">Weapon Detection</span>
                  {weaponsCount > 0 ? (
                    <span className="text-[#E05252] font-semibold animate-pulse">ALERT ({weaponsCount})</span>
                  ) : isStreamLive ? (
                    <span className="text-[#2BC48A] font-medium">ACTIVE (MONITORING)</span>
                  ) : (
                    <span className="text-[#98A6B5] font-medium">STANDBY</span>
                  )}
                </div>
                <div className="bg-[#151E28] border border-[#263341] rounded-[6px] px-2 py-1 flex items-center justify-between">
                  <span className="text-[#98A6B5]">Fight Recognizer</span>
                  {fightsCount > 0 ? (
                    <span className="text-[#E05252] font-semibold animate-pulse">ALERT ({fightsCount})</span>
                  ) : isStreamLive ? (
                    <span className="text-[#2BC48A] font-medium">ACTIVE (MONITORING)</span>
                  ) : (
                    <span className="text-[#98A6B5] font-medium">STANDBY</span>
                  )}
                </div>
                <div className="bg-[#151E28] border border-[#263341] rounded-[6px] px-2 py-1 flex items-center justify-between">
                  <span className="text-[#98A6B5]">Crowd Dynamics</span>
                  {crowdDensity >= 0.55 ? (
                    <span className="text-[#E05252] font-semibold">CRITICAL</span>
                  ) : crowdDensity >= 0.35 ? (
                    <span className="text-[#E7A83B] font-semibold">ELEVATED</span>
                  ) : (
                    <span className="text-[#2BC48A] font-medium">NORMAL</span>
                  )}
                </div>
                <div className="bg-[#151E28] border border-[#263341] rounded-[6px] px-2 py-1 flex items-center justify-between">
                  <span className="text-[#98A6B5]">Event Stream</span>
                  <span className={connectionStatus === 'CONNECTED' ? 'text-[#2BC48A] font-medium' : 'text-[#E7A83B] font-medium'}>
                    {connectionStatus}
                  </span>
                </div>
                <div className="bg-[#151E28] border border-[#263341] rounded-[6px] px-2 py-1 flex items-center justify-between col-span-2">
                  <span className="text-[#98A6B5]">Network Priority</span>
                  <span className="text-[#2BC48A] font-medium">ROUTINE (BEST-EFFORT)</span>
                </div>
              </div>
            );
          })()}
        </div>
      </div>
    );
  }

  // ---------------------------------------------------------------------------
  // 2. ACTIVE INCIDENT DETECTED (REAL BACKEND DATA)
  // ---------------------------------------------------------------------------
  const hasSnapshot = Boolean(activeIncident.incident_id);
  const snapshotUrl = `http://localhost:8100/incidents/${activeIncident.incident_id}/snapshot?t=${activeIncident.start_time || activeIncident.incident_id}`;
  const clipUrl = `http://localhost:8100/incidents/${activeIncident.incident_id}/clip?t=${activeIncident.start_time || activeIncident.incident_id}`;

  return (
    <div
      className={`suraksha-panel-emergency flex flex-col overflow-hidden transition-all duration-300 ${
        newIncidentHighlight ? 'ring-2 ring-[#E05252]' : ''
      }`}
    >
      {/* Header */}
      <div className="px-3.5 py-2 border-b border-[#263341] bg-[#151E28] flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 text-[#E05252]" />
          <h2 className="text-xs font-semibold font-sans tracking-wider text-[#E05252] uppercase">
            INCIDENT DETECTED
          </h2>
        </div>

        <div className="flex items-center gap-2">
          {activeIncident.source_mode === 'DEMO' && (
            <span className="bg-[#151E28] border border-[#E7A83B]/50 text-[#E7A83B] px-2 py-0.5 rounded-[4px] text-xs font-semibold font-mono tracking-wider uppercase">
              DEMO
            </span>
          )}
          <span className="text-xs font-mono text-[#98A6B5]">
            #{activeIncident.incident_id}
          </span>
          <div className="flex items-center gap-1.5 bg-[#151E28] border border-[#E05252]/40 rounded-full px-2.5 py-0.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#E05252] animate-pulse" />
            <span className="text-xs font-mono font-medium text-[#E05252] uppercase">
              {activeIncident.status}
            </span>
          </div>
        </div>
      </div>

      <div className="p-3 space-y-2.5">
        {activeIncident.source_mode === 'DEMO' && (
          <div className="bg-[#151E28] border border-[#E7A83B]/40 text-[#E7A83B] text-xs font-mono px-2.5 py-1 rounded-[6px] flex items-center justify-between">
            <span className="font-semibold">CONTROLLED DEMO SCENARIO • NON-LIVE</span>
            <span className="text-xs text-[#E7A83B]/80 font-medium">HUMAN APPROVAL MANDATORY</span>
          </div>
        )}

        {/* Main Incident Details Box */}
        <div className="flex gap-3 items-start">
          {/* Target Suspect Crop / ROI Thumbnail */}
          <div className="relative w-22 h-24 rounded-[6px] bg-[#111821] border border-[#263341] overflow-hidden flex-shrink-0 flex items-center justify-center">
            {hasSnapshot && !snapshotError ? (
              <img
                src={snapshotUrl}
                alt="Incident Snapshot"
                className="w-full h-full object-cover"
                onError={() => {
                  setSnapshotError(true);
                }}
              />
            ) : (
              <svg viewBox="0 0 100 110" className="w-full h-full">
                <rect width="100" height="110" fill="#111821" />
                <circle cx="50" cy="28" r="12" fill="#151E28" />
                <path d="M32 46h36l-3 55h-30z" fill="#151E28" />
                {activeIncident.incident_type === 'weapon' || activeIncident.incident_type === 'armed_fight' ? (
                  <>
                    <rect x="64" y="52" width="16" height="5" rx="1" fill="#E05252" />
                    <rect x="64" y="52" width="5" height="11" rx="1" fill="#991b1b" />
                  </>
                ) : (
                  <circle cx="50" cy="50" r="10" fill="#E05252" fillOpacity="0.4" />
                )}
                {/* Crosshair Overlay */}
                <circle cx="50" cy="50" r="26" fill="none" stroke="#E05252" strokeWidth="1" strokeDasharray="3 3" />
                <line x1="50" y1="18" x2="50" y2="26" stroke="#E05252" strokeWidth="1.5" />
                <line x1="50" y1="74" x2="50" y2="82" stroke="#E05252" strokeWidth="1.5" />
                <line x1="18" y1="50" x2="26" y2="50" stroke="#E05252" strokeWidth="1.5" />
                <line x1="74" y1="50" x2="82" y2="50" stroke="#E05252" strokeWidth="1.5" />
              </svg>
            )}

            <div className="absolute top-1 left-1 bg-[#E05252] text-white text-[9px] font-mono px-1.5 py-0.5 rounded-[2px] font-bold">
              ROI
            </div>
            {activeIncident.bbox && (
              <div className="absolute bottom-1 right-1 bg-[#0B0F14]/90 text-[#98A6B5] text-[9px] font-mono px-1 rounded-[2px]">
                {Math.round(activeIncident.bbox.x2 - activeIncident.bbox.x1)}×
                {Math.round(activeIncident.bbox.y2 - activeIncident.bbox.y1)}
              </div>
            )}
          </div>

          {/* Key-Value Telemetry */}
          <div className="flex-1 space-y-1.5 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-[#98A6B5] font-sans font-medium">Type</span>
              <span className="text-[#E8EDF3] font-mono font-semibold">
                {formatIncidentType(activeIncident.incident_type)}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[#98A6B5] font-sans font-medium">Severity</span>
              {getSeverityBadge(activeIncident.severity)}
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[#98A6B5] font-sans font-medium">Network Priority</span>
              <span className={`px-2 py-0.5 rounded-[4px] text-xs font-mono font-semibold tracking-wider uppercase ${
                String(activeIncident.severity).toLowerCase() === 'critical'
                  ? 'bg-[#E05252] text-white'
                  : 'bg-[#E7A83B] text-white'
              }`}>
                {String(activeIncident.severity).toLowerCase() === 'critical' ? 'CRITICAL' : 'HIGH'}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[#98A6B5] font-sans font-medium">Confidence</span>
              <span className="text-[#E05252] font-mono font-semibold">
                {Math.round(activeIncident.confidence * 100)}%
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-[#98A6B5] font-sans font-medium">Time</span>
              <span className="text-[#98A6B5] font-mono text-xs">
                {formatTime(activeIncident.start_time)}
              </span>
            </div>

            {/* Camera / Sector info */}
            <div className="flex items-center justify-between text-xs text-[#98A6B5] pt-1 border-t border-[#263341]">
              <span className="font-sans font-medium">Camera</span>
              <span className="text-[#E8EDF3] font-mono">
                {activeIncident.camera_id}
              </span>
            </div>
          </div>
        </div>

        {/* Action Buttons: View Snapshot & View 10s Clip */}
        <div className="grid grid-cols-2 gap-2 pt-0.5">
          <a
            href={hasSnapshot ? snapshotUrl : '#'}
            target={hasSnapshot ? '_blank' : undefined}
            rel="noreferrer"
            className={`flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-[6px] text-xs font-sans font-medium transition-colors ${
              hasSnapshot
                ? 'bg-[#3B9EFF] hover:bg-[#2e82d3] text-white cursor-pointer shadow-sm'
                : 'bg-[#151E28] text-[#687585] cursor-not-allowed border border-[#263341]'
            }`}
            onClick={(e) => {
              if (!hasSnapshot) e.preventDefault();
            }}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>{hasSnapshot ? 'View Snapshot' : 'No Snapshot'}</span>
          </a>

          <button
            onClick={() => setShowVideoModal(true)}
            className="flex items-center justify-center gap-1.5 py-1.5 px-3 rounded-[6px] text-xs font-sans font-medium transition-colors bg-[#151E28] hover:bg-[#263341] text-[#E8EDF3] border border-[#263341] cursor-pointer shadow-sm"
          >
            <Video className="w-3.5 h-3.5 text-[#3B9EFF]" />
            <span>View 10s Clip</span>
          </button>
        </div>

        {/* Agentic Briefing Button */}
        <button
          onClick={() => setShowAgentBriefing(true)}
          className="w-full py-2 px-3 rounded-[6px] bg-[#151E28] hover:bg-[#263341] text-[#3B9EFF] border border-[#263341] flex items-center justify-center gap-2 text-xs font-sans font-semibold tracking-wide transition-colors cursor-pointer shadow-sm"
        >
          <Brain className="w-4 h-4 text-[#3B9EFF]" />
          <span>AGENTIC BRIEFING & COORDINATION</span>
        </button>
      </div>

      {/* Agentic Briefing Modal Dialog */}
      {showAgentBriefing && (
        <div className="fixed inset-0 z-50 bg-[#0B0F14]/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#111821] border border-[#263341] rounded-[8px] max-w-3xl w-full h-[85vh] flex flex-col shadow-2xl overflow-hidden">
            <div className="p-2.5 border-b border-[#263341] flex items-center justify-between bg-[#151E28]">
              <div className="flex items-center gap-2 text-[#E8EDF3] font-semibold text-xs uppercase font-sans">
                <Brain className="w-4 h-4 text-[#3B9EFF]" />
                <span>Decision Support Briefing — Incident #{activeIncident.incident_id}</span>
              </div>
              <button
                onClick={() => setShowAgentBriefing(false)}
                className="text-[#98A6B5] hover:text-[#E8EDF3] px-2 py-0.5 rounded text-sm font-mono cursor-pointer"
              >
                ✕
              </button>
            </div>
            <div className="flex-1 overflow-hidden">
              <AgentAssessmentCard incidentId={activeIncident.incident_id} />
            </div>
          </div>
        </div>
      )}

      {/* Video Clip Modal Dialog */}
      {showVideoModal && (
        <div className="fixed inset-0 z-50 bg-[#0B0F14]/85 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#111821] border border-[#263341] rounded-[8px] max-w-2xl w-full flex flex-col shadow-2xl overflow-hidden">
            <div className="p-2.5 border-b border-[#263341] flex items-center justify-between bg-[#151E28]">
              <div className="flex items-center gap-2 text-[#E8EDF3] font-semibold text-xs uppercase font-sans">
                <Video className="w-4 h-4 text-[#3B9EFF]" />
                <span>Forensic 10s Video Clip — #{activeIncident.incident_id}</span>
              </div>
              <button
                onClick={() => setShowVideoModal(false)}
                className="text-[#98A6B5] hover:text-[#E8EDF3] px-2 py-0.5 rounded text-sm font-mono cursor-pointer"
              >
                ✕
              </button>
            </div>
            <div className="p-3 bg-black flex items-center justify-center min-h-[300px]">
              <video
                key={clipUrl}
                src={clipUrl}
                controls
                autoPlay
                playsInline
                className="max-h-[60vh] max-w-full rounded-[6px] border border-[#263341]"
              >
                Your browser does not support HTML5 video playback.
              </video>
            </div>
            <div className="p-2.5 border-t border-[#263341] bg-[#111821] flex items-center justify-between text-[11px] font-mono">
              <span className="text-[#98A6B5]">Temporal Window: T-5s to T+5s &bull; Format: H.264 MP4</span>
              <a
                href={clipUrl}
                download={`${activeIncident.incident_id}_clip.mp4`}
                target="_blank"
                rel="noreferrer"
                className="px-2.5 py-1 rounded-[6px] bg-[#3B9EFF] hover:bg-[#2e82d3] text-white font-mono font-medium shadow-sm"
              >
                Download Clip
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
