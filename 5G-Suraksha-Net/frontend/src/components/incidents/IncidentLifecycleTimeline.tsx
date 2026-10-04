import React from 'react';
import {
  CheckCircle2,
  Clock,
  Video,
  ShieldAlert,
  FileCheck,
  Radio,
} from 'lucide-react';
import type { IncidentReport } from '../../types/incidents';

interface IncidentLifecycleTimelineProps {
  incident: IncidentReport;
}

export const IncidentLifecycleTimeline: React.FC<IncidentLifecycleTimelineProps> = ({
  incident,
}) => {
  const formatTime = (isoString?: string | null): string | null => {
    if (!isoString) return null;
    try {
      const d = new Date(isoString);
      return (
        d.toLocaleTimeString('en-IN', {
          hour12: false,
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }) + ' IST'
      );
    } catch {
      return null;
    }
  };

  const startTimeStr = formatTime(incident.start_time);
  const finalizedTimeStr = formatTime(incident.finalized_at);

  const isFinalized = incident.status === 'finalized';
  const isRecording = incident.status === 'recording_post_event' || isFinalized;
  const isVerified =
    incident.status === 'verified' || isRecording || isFinalized;

  const timelineSteps = [
    {
      id: 'detected',
      title: 'DETECTED',
      subtitle: `AI temporal trigger on ${incident.camera_id}`,
      timestamp: startTimeStr,
      completed: true,
      active: incident.status === 'candidate',
      icon: <Radio className="w-3.5 h-3.5" />,
      detail:
        incident.track_ids.length > 0
          ? `Track IDs: [${incident.track_ids.join(', ')}]`
          : incident.person_count
          ? `Crowd count: ${incident.person_count} persons`
          : 'Frame detection triggered',
    },
    {
      id: 'verified',
      title: 'VERIFIED',
      subtitle: 'Multi-frame verification & confidence validation',
      timestamp: null, // Truthfully unrecorded separate verification timestamp; no fabrication
      completed: isVerified,
      active: incident.status === 'verified',
      icon: <CheckCircle2 className="w-3.5 h-3.5" />,
      detail: `Confidence: ${(incident.confidence * 100).toFixed(1)}% • Severity: ${incident.severity.toUpperCase()}`,
    },
    {
      id: 'recording',
      title: 'EVIDENCE RECORDING',
      subtitle: 'Temporal circular buffer & post-event recording',
      timestamp: null,
      completed: isRecording,
      active: incident.status === 'recording_post_event',
      icon: <Video className="w-3.5 h-3.5" />,
      detail:
        incident.status === 'recording_post_event'
          ? 'Recording active in background...'
          : Boolean(incident.evidence?.clip_path)
          ? 'Evidence clip stored'
          : 'Evidence recording stage',
    },
    {
      id: 'finalized',
      title: 'FINALIZED',
      subtitle: 'Evidence artifacts written to storage',
      timestamp: finalizedTimeStr, // Real finalized_at timestamp if present
      completed: isFinalized,
      active: isFinalized,
      icon: <FileCheck className="w-3.5 h-3.5" />,
      detail: isFinalized
        ? Boolean(incident.evidence?.snapshot_path)
          ? 'Snapshot and video clip locked'
          : 'Metadata package archived'
        : 'Pending post-event window completion',
    },
    {
      id: 'response',
      title: 'RESPONSE DISPATCH',
      subtitle: 'Deterministic emergency recommendation generated',
      timestamp: null,
      completed: isVerified,
      active: false,
      icon: <ShieldAlert className="w-3.5 h-3.5" />,
      detail: 'Human operator authorization required before dispatch',
    },
  ];

  return (
    <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-3 text-xs font-mono">
      <div className="flex items-center justify-between border-b border-[#263341] pb-2 mb-3">
        <div className="flex items-center gap-1.5 text-[#E8EDF3] font-semibold tracking-wider text-[11px] uppercase font-sans">
          <Clock className="w-3.5 h-3.5 text-[#3B9EFF]" />
          <span>Incident Lifecycle Timeline</span>
        </div>
        <span className="text-[10px] text-[#98A6B5] font-mono">
          ID: #{incident.incident_id}
        </span>
      </div>

      <div className="relative pl-3 space-y-4">
        {/* Connecting timeline line */}
        <div className="absolute left-[19px] top-2 bottom-2 w-0.5 bg-[#263341]" />

        {timelineSteps.map((step) => {
          const isDone = step.completed;
          const isActive = step.active;

          return (
            <div key={step.id} className="relative flex items-start gap-3">
              {/* Timeline Node Icon */}
              <div
                className={`relative z-10 w-6 h-6 rounded-full flex items-center justify-center border text-xs flex-shrink-0 ${
                  isActive
                    ? 'bg-[#3B9EFF] border-[#3B9EFF] text-[#0B0F14] animate-pulse'
                    : isDone
                    ? 'bg-[#111821] border-[#3B9EFF]/70 text-[#3B9EFF]'
                    : 'bg-[#111821] border-[#263341] text-[#687585]'
                }`}
              >
                {step.icon}
              </div>

              {/* Step Content */}
              <div className="flex-1 min-w-0 pt-0.5">
                <div className="flex items-center justify-between">
                  <span
                    className={`font-semibold text-[11px] tracking-wide font-sans ${
                      isDone ? 'text-[#E8EDF3]' : 'text-[#687585]'
                    }`}
                  >
                    {step.title}
                  </span>
                  {step.timestamp && (
                    <span className="text-[9.5px] text-[#98A6B5] font-mono">
                      {step.timestamp}
                    </span>
                  )}
                </div>

                <div className="text-[10px] text-[#98A6B5] leading-tight mt-0.5 font-sans">
                  {step.subtitle}
                </div>

                <div className="text-[9.5px] text-[#49C6D9] mt-1 bg-[#111821] border border-[#263341] rounded-[4px] px-1.5 py-0.5 inline-block font-mono">
                  {step.detail}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
