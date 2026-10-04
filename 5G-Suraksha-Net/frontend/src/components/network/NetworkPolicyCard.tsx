import React, { useState } from 'react';
import {
  Wifi,
  Info,
  ChevronDown,
  ChevronUp,
  Clock,
} from 'lucide-react';
import { useNetworkPolicy } from '../../hooks/useNetworkPolicy';
import type {
  NetworkPolicy,
  ApplicationPriority,
  PolicyEventType,
} from '../../types/network';

export const NetworkPolicyCard: React.FC = () => {
  const { policyState, events } = useNetworkPolicy(3000);
  const [showEventLog, setShowEventLog] = useState<boolean>(false);

  // Defaults when offline or loading
  const currentPolicy: NetworkPolicy = policyState?.policy || 'NORMAL';
  const currentPriority: ApplicationPriority = policyState?.priority || 'ROUTINE';
  const reason = policyState?.reason || 'Monitoring camera feed — routine application transmission tier.';
  const activeIncidentId = policyState?.active_incident_id;
  const activeType = policyState?.active_incident_type;
  const activeSev = policyState?.active_severity;

  // Format ISO timestamp
  const formatTime = (isoString?: string | null): string => {
    if (!isoString) return '--:--:--';
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
      return isoString;
    }
  };

  const getPolicyColor = (pol: NetworkPolicy) => {
    switch (pol) {
      case 'CRITICAL':
        return {
          bg: 'bg-[#151014]',
          border: 'border-[#E05252]',
          text: 'text-[#E05252]',
          badgeBg: 'bg-[#E05252]',
          dot: 'bg-[#E05252]',
          ring: 'ring-[#E05252]/40',
        };
      case 'EVENT':
        return {
          bg: 'bg-[#151E28]',
          border: 'border-[#E7A83B]/60',
          text: 'text-[#E7A83B]',
          badgeBg: 'bg-[#E7A83B]',
          dot: 'bg-[#E7A83B]',
          ring: 'ring-[#E7A83B]/40',
        };
      case 'NORMAL':
      default:
        return {
          bg: 'bg-[#151E28]',
          border: 'border-[#263341]',
          text: 'text-[#2BC48A]',
          badgeBg: 'bg-[#2BC48A]',
          dot: 'bg-[#2BC48A]',
          ring: 'ring-[#2BC48A]/30',
        };
    }
  };

  const getEventBadge = (type: PolicyEventType) => {
    switch (type) {
      case 'POLICY_ESCALATED':
        return <span className="bg-[#111821] border border-[#E05252]/60 text-[#E05252] px-1 py-0.2 rounded-[2px] text-[8px] font-mono font-semibold">ESCALATED</span>;
      case 'POLICY_DEESCALATED':
        return <span className="bg-[#111821] border border-[#2BC48A]/60 text-[#2BC48A] px-1 py-0.2 rounded-[2px] text-[8px] font-mono font-semibold">DEESCALATED</span>;
      case 'INCIDENT_PRIORITY_ASSIGNED':
        return <span className="bg-[#111821] border border-[#E7A83B]/60 text-[#E7A83B] px-1 py-0.2 rounded-[2px] text-[8px] font-mono font-semibold">PRIORITY</span>;
      case 'EVIDENCE_READY':
        return <span className="bg-[#111821] border border-[#3B9EFF]/60 text-[#3B9EFF] px-1 py-0.2 rounded-[2px] text-[8px] font-mono font-semibold">EVIDENCE</span>;
      default:
        return <span className="bg-[#111821] border border-[#263341] text-[#98A6B5] px-1 py-0.2 rounded-[2px] text-[8px] font-mono font-semibold">STANDBY</span>;
    }
  };

  const polStyle = getPolicyColor(currentPolicy);

  return (
    <div className="suraksha-panel flex flex-col h-full overflow-hidden transition-all duration-300">
      {/* 1. Header */}
      <div className="suraksha-panel-header px-3 py-1.5 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-1.5">
          <Wifi className="w-3.5 h-3.5 text-[#3B9EFF]" />
          <h2 className="text-[11px] font-semibold font-sans tracking-wider text-[#E8EDF3] uppercase">
            5G & NETWORK INTELLIGENCE
          </h2>
        </div>

        <div className="flex items-center gap-1.5">
          <span className="bg-[#151E28] border border-[#263341] text-[#98A6B5] text-[8.5px] font-mono px-2 py-0.5 rounded-[4px] font-medium uppercase">
            APPLICATION-LEVEL POLICY
          </span>
        </div>
      </div>

      {/* 2. Main Content Body */}
      <div className="p-3 flex-1 flex flex-col gap-2.5 overflow-y-auto font-mono text-xs">
        {/* Policy & Priority Hero Card */}
        <div className={`p-2.5 rounded-[6px] border ${polStyle.bg} ${polStyle.border} flex items-center justify-between`}>
          <div className="space-y-0.5">
            <div className="text-[9px] uppercase tracking-wider text-[#98A6B5] font-sans font-medium">
              Current Transmission Policy
            </div>
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${polStyle.dot} animate-pulse`} />
              <span className={`text-base font-mono font-semibold tracking-wider ${polStyle.text}`}>
                {currentPolicy}
              </span>
            </div>
          </div>

          <div className="text-right space-y-0.5">
            <div className="text-[9px] uppercase tracking-wider text-[#98A6B5] font-sans font-medium">
              Data Priority Tier
            </div>
            <span
              className={`inline-block px-2 py-0.5 rounded-[4px] text-[10.5px] font-mono font-semibold uppercase ${
                currentPriority === 'CRITICAL'
                  ? 'bg-[#E05252] text-white'
                  : currentPriority === 'HIGH'
                  ? 'bg-[#E7A83B] text-white'
                  : 'bg-[#111821] border border-[#2BC48A]/50 text-[#2BC48A]'
              }`}
            >
              {currentPriority}
            </span>
          </div>
        </div>

        {/* Technical Policy Telemetry Grid */}
        <div className="grid grid-cols-2 gap-2 text-[10px]">
          <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 flex flex-col justify-between">
            <span className="text-[#98A6B5] text-[9px] uppercase font-sans font-medium">5G Control Plane</span>
            <div className="flex items-center gap-1.5 mt-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#E7A83B]" />
              <span className="text-[#E7A83B] font-mono font-semibold tracking-tight">NOT CONNECTED</span>
            </div>
            <span className="text-[8px] text-[#687585] mt-0.5 font-mono">No physical slice controller attached</span>
          </div>

          <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 flex flex-col justify-between">
            <span className="text-[#98A6B5] text-[9px] uppercase font-sans font-medium">QoS Enforcement</span>
            <div className="flex items-center gap-1.5 mt-1">
              <span className="w-1.5 h-1.5 rounded-full bg-[#3B9EFF]" />
              <span className="text-[#E8EDF3] font-mono font-semibold">APPLICATION-LEVEL</span>
            </div>
            <span className="text-[8px] text-[#687585] mt-0.5 font-mono">Software priority queuing</span>
          </div>

          <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 flex flex-col justify-between">
            <span className="text-[#98A6B5] text-[9px] uppercase font-sans font-medium">Bandwidth Slice</span>
            <span className="text-[#E8EDF3] font-mono font-semibold mt-1">DEFAULT_BE (BEST-EFFORT)</span>
            <span className="text-[8px] text-[#687585] mt-0.5 font-mono">Local Ethernet / Wi-Fi route</span>
          </div>

          <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 flex flex-col justify-between">
            <span className="text-[#98A6B5] text-[9px] uppercase font-sans font-medium">Active Trigger</span>
            <span className={`mt-1 font-mono font-semibold truncate ${activeIncidentId ? 'text-[#E05252]' : 'text-[#98A6B5]'}`}>
              {activeIncidentId ? `#${activeIncidentId} (${activeType || 'INCIDENT'})` : 'None (Standby)'}
            </span>
            <span className="text-[8px] text-[#687585] mt-0.5 font-mono">
              {activeSev ? `Severity: ${activeSev}` : 'Routine patrol baseline'}
            </span>
          </div>
        </div>

        {/* Policy Reason Narrative */}
        <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 text-[10px] space-y-1">
          <div className="flex items-center gap-1.5 text-[#98A6B5] font-sans font-medium text-[9px] uppercase">
            <Info className="w-3 h-3 text-[#3B9EFF]" />
            <span>Policy Decision Rationale</span>
          </div>
          <p className="text-[#E8EDF3] font-sans leading-relaxed text-[9.5px]">
            {reason}
          </p>
        </div>

        {/* Collapsible Audit Trail & Events */}
        <div className="border border-[#263341] rounded-[6px] bg-[#111821] overflow-hidden">
          <button
            onClick={() => setShowEventLog(!showEventLog)}
            className="w-full px-2.5 py-1.5 flex items-center justify-between text-[10px] text-[#E8EDF3] hover:bg-[#151E28] transition-colors"
          >
            <div className="flex items-center gap-1.5 font-semibold uppercase text-[9px] font-sans text-[#98A6B5]">
              <Clock className="w-3 h-3 text-[#98A6B5]" />
              <span>Policy Audit Events ({events.length})</span>
            </div>
            {showEventLog ? (
              <ChevronUp className="w-3.5 h-3.5 text-[#98A6B5]" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5 text-[#98A6B5]" />
            )}
          </button>

          {showEventLog && (
            <div className="p-2 border-t border-[#263341] max-h-40 overflow-y-auto space-y-1.5">
              {events.length === 0 ? (
                <div className="text-[9px] text-[#687585] text-center py-2 font-mono">
                  No policy transitions recorded yet.
                </div>
              ) : (
                events.map((evt) => (
                  <div
                    key={evt.event_id}
                    className="p-1.5 rounded-[4px] bg-[#151E28] border border-[#263341] text-[9px] space-y-0.5"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        {getEventBadge(evt.event_type)}
                        <span className="text-[#98A6B5] font-mono">
                          {formatTime(evt.timestamp)}
                        </span>
                      </div>
                      <span className="text-[8px] text-[#687585] font-mono">
                        {evt.incident_id ? `#${evt.incident_id}` : 'GLOBAL'}
                      </span>
                    </div>
                    <div className="text-[#E8EDF3] text-[8.5px] leading-tight font-sans">
                      {evt.details}
                    </div>
                  </div>
                ))
              )}
            </div>
          )}
        </div>

        {/* Truthful System Disclaimer */}
        <div className="mt-auto pt-1 text-[8px] text-[#687585] font-mono leading-tight border-t border-[#263341]">
          Strict truthfulness: Policy reflects application software priority tiers. No external 5G slice controller is attached.
        </div>
      </div>
    </div>
  );
};
