import React from 'react';
import {
  Brain,
  ShieldAlert,
  MapPin,
  FileCheck,
  Wifi,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  HelpCircle,
  Clock,
  Layers,
  FileText,
} from 'lucide-react';
import { useAgentAssessment } from '../../hooks/useAgentAssessment';
import type { ActionStep, ActionStepStatus } from '../../types/agents';

interface AgentAssessmentCardProps {
  incidentId: string | null;
  onRefresh?: () => void;
}

export const AgentAssessmentCard: React.FC<AgentAssessmentCardProps> = ({
  incidentId,
}) => {
  const { assessment, loading, error, refresh } = useAgentAssessment(incidentId);

  const getStepBadge = (status: ActionStepStatus) => {
    switch (status) {
      case 'COMPLETED':
        return (
          <span className="bg-[#151E28] border border-[#2BC48A]/60 text-[#2BC48A] px-2 py-0.5 rounded-[4px] text-xs font-mono font-medium">
            COMPLETED
          </span>
        );
      case 'AWAITING_APPROVAL':
        return (
          <span className="bg-[#151014] border border-[#E05252]/60 text-[#E05252] px-2 py-0.5 rounded-[4px] text-xs font-mono font-medium animate-pulse">
            AWAITING APPROVAL
          </span>
        );
      case 'REQUIRED':
        return (
          <span className="bg-[#151E28] border border-[#E7A83B]/60 text-[#E7A83B] px-2 py-0.5 rounded-[4px] text-xs font-mono font-medium">
            REQUIRED
          </span>
        );
      case 'ADVISORY':
      default:
        return (
          <span className="bg-[#151E28] border border-[#263341] text-[#98A6B5] px-2 py-0.5 rounded-[4px] text-xs font-mono font-medium">
            ADVISORY
          </span>
        );
    }
  };

  const getSeverityBadge = (sev: string) => {
    const s = sev.toUpperCase();
    if (s === 'CRITICAL') {
      return <span className="bg-[#E05252] text-white px-2.5 py-0.5 rounded-[4px] text-xs font-mono font-semibold">CRITICAL</span>;
    }
    if (s === 'HIGH') {
      return <span className="bg-[#E7A83B] text-white px-2.5 py-0.5 rounded-[4px] text-xs font-mono font-semibold">HIGH</span>;
    }
    if (s === 'MODERATE' || s === 'MEDIUM') {
      return <span className="bg-[#151E28] border border-[#E7A83B]/50 text-[#E7A83B] px-2.5 py-0.5 rounded-[4px] text-xs font-mono font-semibold">{s}</span>;
    }
    return <span className="bg-[#151E28] border border-[#263341] text-[#98A6B5] px-2.5 py-0.5 rounded-[4px] text-xs font-mono font-semibold">{s}</span>;
  };

  if (!incidentId) {
    return (
      <div className="suraksha-panel p-6 flex flex-col items-center justify-center text-center h-full font-mono text-xs">
        <Brain className="w-8 h-8 text-[#687585] mb-2" />
        <div className="text-[#E8EDF3] font-semibold tracking-wider text-[11px] uppercase font-sans">
          Agentic Orchestration Standby
        </div>
        <div className="text-[#98A6B5] text-[10px] mt-1 max-w-xs font-sans">
          Select or detect an active incident to generate supervisor situation assessment and operational coordination briefing.
        </div>
      </div>
    );
  }

  if (loading && !assessment) {
    return (
      <div className="suraksha-panel p-6 flex flex-col items-center justify-center text-center h-full font-mono text-xs">
        <RefreshCw className="w-6 h-6 text-[#3B9EFF] animate-spin mb-2" />
        <div className="text-[#E8EDF3] font-semibold tracking-wider text-[11px] uppercase font-sans">
          Synthesizing Structured Facts...
        </div>
        <div className="text-[#98A6B5] text-[10px] mt-1 font-sans">
          Evaluating incident #{incidentId}, response policies, geospatial resources, and network state.
        </div>
      </div>
    );
  }

  if (error || !assessment) {
    return (
      <div className="suraksha-panel p-4 flex flex-col items-center justify-center text-center h-full font-mono text-xs">
        <AlertTriangle className="w-6 h-6 text-[#E7A83B] mb-2" />
        <div className="text-[#E7A83B] font-semibold uppercase text-[11px] font-sans">
          Assessment Unavailable
        </div>
        <div className="text-[#98A6B5] text-[10px] mt-1">{error || 'Could not load assessment'}</div>
        <button
          onClick={() => refresh(true)}
          className="mt-3 bg-[#3B9EFF] hover:bg-[#2e82d3] text-white px-3 py-1 rounded-[4px] text-[10px] font-mono font-medium transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="suraksha-panel flex flex-col h-full overflow-hidden text-xs transition-all duration-300">
      {/* 1. Panel Header */}
      <div className="suraksha-panel-header px-3.5 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-[#3B9EFF]" />
          <h2 className="text-xs font-semibold font-sans tracking-wider text-[#E8EDF3] uppercase">
            AGENTIC INTELLIGENCE BRIEFING
          </h2>
        </div>

        <div className="flex items-center gap-2">
          <span className="bg-[#151E28] border border-[#263341] text-[#E7A83B] text-xs px-2 py-0.5 rounded-[4px] font-mono font-semibold uppercase">
            DECISION SUPPORT ONLY
          </span>
          <button
            onClick={() => refresh(true)}
            title="Refresh Agent Assessment"
            className="p-1 hover:bg-[#263341] rounded text-[#98A6B5] hover:text-[#E8EDF3] transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-[#3B9EFF]' : ''}`} />
          </button>
        </div>
      </div>

      {/* 2. Scrollable Assessment Body */}
      <div className="p-3.5 flex-1 flex flex-col gap-3 overflow-y-auto">
        {/* Situation Awareness & Executive Briefing */}
        <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-3 space-y-2">
          <div className="flex items-center justify-between border-b border-[#263341] pb-1.5">
            <div className="flex items-center gap-2 text-[#3B9EFF] font-semibold text-xs uppercase font-sans">
              <FileText className="w-3.5 h-3.5 text-[#3B9EFF]" />
              <span>Situation Awareness & Supervisor Assessment</span>
            </div>
            <span className="text-xs font-mono text-[#98A6B5]">#{assessment.incident_id}</span>
          </div>
          <p className="text-[#E8EDF3] text-xs leading-relaxed font-sans">
            {assessment.situation_summary}
          </p>
        </div>

        {/* Fact Matrix Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 text-xs">
          {/* Severity Tile */}
          <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2.5 flex flex-col justify-between gap-1.5">
            <div className="flex items-center justify-between">
              <span className="text-[#98A6B5] text-xs uppercase font-sans font-medium">Deterministic Severity</span>
              {getSeverityBadge(assessment?.severity_assessment?.severity || 'LOW')}
            </div>
            <div className="text-xs text-[#E8EDF3] mt-1 leading-snug font-sans">
              {assessment?.severity_assessment?.explanation || 'Authoritative severity evaluated by pipeline.'}
            </div>
            <span className="text-xs font-mono text-[#2BC48A] mt-1 font-semibold">
              ✓ AUTHORITATIVE DETERMINISTIC
            </span>
          </div>

          {/* Location Tile */}
          <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2.5 flex flex-col justify-between gap-1.5">
            <div className="flex items-center gap-1.5 text-[#98A6B5] text-xs uppercase font-sans font-medium">
              <MapPin className="w-3.5 h-3.5 text-[#3B9EFF]" />
              <span>Location Context</span>
            </div>
            <div className="text-xs text-[#E8EDF3] font-semibold truncate mt-1 font-sans">
              {assessment?.location_assessment?.location_name || 'Operational Sector'}
            </div>
            <div className="text-xs text-[#98A6B5] font-sans">
              Nearby: {assessment?.location_assessment?.nearby_police_count ?? 0} Police • {assessment?.location_assessment?.nearby_hospital_count ?? 0} Medical
            </div>
            <span className="text-xs font-mono text-[#49C6D9]">
              {assessment?.location_assessment?.gps_fix_available ? 'GPS Fix Confirmed' : 'Configured Sector Baseline'}
            </span>
          </div>

          {/* Forensic Evidence Tile */}
          <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2.5 flex flex-col justify-between gap-1.5">
            <div className="flex items-center gap-1.5 text-[#98A6B5] text-xs uppercase font-sans font-medium">
              <FileCheck className="w-3.5 h-3.5 text-[#3B9EFF]" />
              <span>Forensic Evidence</span>
            </div>
            <div className="space-y-1 mt-1 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-[#98A6B5] font-sans">Snapshot:</span>
                <span className={assessment?.evidence_assessment?.snapshot_available ? 'text-[#2BC48A] font-mono font-semibold' : 'text-[#E7A83B] font-mono'}>
                  {assessment?.evidence_assessment?.snapshot_available ? 'AVAILABLE' : 'PENDING'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#98A6B5] font-sans">10s MP4 Clip:</span>
                <span className={assessment?.evidence_assessment?.video_available ? 'text-[#2BC48A] font-mono font-semibold' : 'text-[#E7A83B] font-mono'}>
                  {assessment?.evidence_assessment?.video_available ? 'AVAILABLE' : 'FINALIZING'}
                </span>
              </div>
            </div>
            <span className="text-xs text-[#687585] font-sans">Immutable Disk Storage</span>
          </div>

          {/* Network Transmission Priority Tile */}
          <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2.5 flex flex-col justify-between gap-1.5">
            <div className="flex items-center gap-1.5 text-[#98A6B5] text-xs uppercase font-sans font-medium">
              <Wifi className="w-3.5 h-3.5 text-[#3B9EFF]" />
              <span>Network Priority</span>
            </div>
            <div className="flex items-center justify-between mt-1">
              <span className="text-[#E8EDF3] font-mono font-semibold">{assessment?.network_assessment?.policy || 'NORMAL'}</span>
              <span className="bg-[#111821] text-[#3B9EFF] border border-[#3B9EFF]/40 px-2 py-0.5 rounded-[4px] text-xs font-mono font-semibold">
                {assessment?.network_assessment?.priority || 'ROUTINE'}
              </span>
            </div>
            <div className="text-xs font-mono text-[#E7A83B] mt-0.5 font-semibold">
              5G CONTROL PLANE: NOT CONNECTED
            </div>
            <span className="text-xs text-[#687585] font-sans">Application-Level Priority</span>
          </div>
        </div>

        {/* Human Operator Gate Banner */}
        <div className="bg-[#151014] border border-[#E05252] rounded-[6px] p-3 flex items-start gap-3">
          <ShieldAlert className="w-5 h-5 text-[#E05252] flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="text-[#E05252] font-semibold text-xs uppercase tracking-wider flex items-center gap-2 font-sans">
              <span>OPERATOR GATE: HUMAN APPROVAL REQUIRED</span>
              <span className="w-2 h-2 rounded-full bg-[#E05252] animate-pulse" />
            </div>
            <p className="text-[#E8EDF3] text-xs leading-relaxed font-sans">
              Autonomous emergency dispatch is strictly prohibited by safety policy. Response actions require human operator confirmation before transmission to ground security or dispatch services.
            </p>
          </div>
        </div>

        {/* Recommended Action Sequence */}
        <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-3 space-y-2.5">
          <div className="flex items-center justify-between border-b border-[#263341] pb-1.5">
            <div className="flex items-center gap-2 text-[#E8EDF3] font-semibold text-xs uppercase font-sans">
              <Clock className="w-3.5 h-3.5 text-[#3B9EFF]" />
              <span>Recommended Action Sequence (Advisory)</span>
            </div>
            <span className="text-xs font-mono text-[#687585]">OPERATIONAL PROTOCOL</span>
          </div>

          <div className="space-y-2">
            {(assessment?.recommended_sequence || []).map((step: ActionStep) => (
              <div
                key={step.step_number}
                className="p-2.5 rounded-[6px] bg-[#111821] border border-[#263341] flex items-start gap-3"
              >
                <span className="w-6 h-6 rounded-full bg-[#151E28] border border-[#263341] text-[#3B9EFF] font-mono font-semibold flex items-center justify-center text-xs flex-shrink-0 mt-0.5">
                  0{step.step_number}
                </span>
                <div className="flex-1 space-y-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-[#E8EDF3] text-xs font-sans">{step.title}</span>
                    {getStepBadge(step.status)}
                  </div>
                  <p className="text-[#98A6B5] text-xs leading-relaxed font-sans">
                    {step.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Missing Information & Uncertainty Gaps */}
        {(assessment?.missing_information?.length || 0) > 0 && (
          <div className="bg-[#151E28] border border-[#E7A83B]/50 rounded-[6px] p-3 space-y-2">
            <div className="flex items-center gap-2 text-[#E7A83B] font-semibold text-xs uppercase font-sans">
              <HelpCircle className="w-3.5 h-3.5 text-[#E7A83B]" />
              <span>Information Gaps & Operational Uncertainties</span>
            </div>
            <ul className="list-disc pl-4 space-y-1 text-xs text-[#E8EDF3] leading-relaxed font-sans">
              {(assessment?.missing_information || []).map((gap: string, i: number) => (
                <li key={i}>{gap}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Traceability Checklist (Data Sources) */}
        <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-3 space-y-2">
          <div className="flex items-center gap-2 text-[#E8EDF3] font-semibold text-xs uppercase font-sans">
            <Layers className="w-3.5 h-3.5 text-[#3B9EFF]" />
            <span>Data Sources Traceability</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs font-sans">
            <div className="flex items-center gap-1.5 text-[#E8EDF3]">
              <CheckCircle2 className="w-3.5 h-3.5 text-[#2BC48A] flex-shrink-0" />
              <span>Incident Report: VERIFIED</span>
            </div>
            <div className="flex items-center gap-1.5 text-[#E8EDF3]">
              <CheckCircle2 className="w-3.5 h-3.5 text-[#2BC48A] flex-shrink-0" />
              <span>Response Planner: VERIFIED</span>
            </div>
            <div className="flex items-center gap-1.5 text-[#E8EDF3]">
              <CheckCircle2 className="w-3.5 h-3.5 text-[#2BC48A] flex-shrink-0" />
              <span>Location Intel: VERIFIED</span>
            </div>
            <div className="flex items-center gap-1.5 text-[#E8EDF3]">
              <CheckCircle2 className="w-3.5 h-3.5 text-[#2BC48A] flex-shrink-0" />
              <span>Evidence Metadata: VERIFIED</span>
            </div>
            <div className="flex items-center gap-1.5 text-[#E8EDF3]">
              <CheckCircle2 className="w-3.5 h-3.5 text-[#2BC48A] flex-shrink-0" />
              <span>Network Policy: VERIFIED</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
