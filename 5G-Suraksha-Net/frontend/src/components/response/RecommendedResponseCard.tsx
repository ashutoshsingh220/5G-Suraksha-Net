import React from 'react';
import {
  Shield,
  Building2,
  AlertTriangle,
  PhoneCall,
  Loader2,
  RefreshCw,
  Eye,
  CheckCircle,
  Clock,
} from 'lucide-react';
import { useIncidents } from '../../context/IncidentContext';
import { useMapFocus } from '../../context/MapFocusContext';
import { useIncidentResponse } from '../../hooks/useIncidentResponse';
import { defaultNearbyHospitals } from '../../data/emergencyResources';
import type { ResponseAction } from '../../types/response';

interface RecommendedResponseCardProps {
  onExpandMap?: () => void;
}

export const RecommendedResponseCard: React.FC<RecommendedResponseCardProps> = ({ onExpandMap }) => {
  const { activeIncident } = useIncidents();
  const { responsePlan, loading, error, refetch } = useIncidentResponse(activeIncident?.incident_id);
  const { focusLocation } = useMapFocus();

  const handleFocusResource = (action: ResponseAction, overrideResource?: any) => {
    const res = overrideResource || action.recommended_resource;
    if (res?.latitude && res?.longitude) {
      focusLocation({
        latitude: res.latitude,
        longitude: res.longitude,
        label: res.name,
        facilityType: action.action_type === 'POLICE_SECURITY' ? 'police' : 'hospital',
        zoom: 15,
      });
      if (onExpandMap) {
        onExpandMap();
      }
    }
  };

  // Helper for priority color
  const getPriorityBadgeClass = (priority: string) => {
    switch (priority.toLowerCase()) {
      case 'critical':
        return 'bg-[#E05252] text-white border border-[#E05252]';
      case 'high':
        return 'bg-[#E7A83B] text-white border border-[#E7A83B]';
      case 'moderate':
      case 'medium':
        return 'bg-[#111821] border-[#E7A83B]/50 text-[#E7A83B]';
      default:
        return 'bg-[#111821] border-[#263341] text-[#98A6B5]';
    }
  };

  // Helper for status badge
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'AWAITING_HUMAN_APPROVAL':
        return (
          <span className="text-xs font-mono font-medium px-2 py-0.5 rounded-[4px] border bg-[#151E28] border-[#E7A83B]/50 text-[#E7A83B] flex items-center gap-1.5">
            <Clock className="w-3 h-3" />
            <span>AWAITING HUMAN APPROVAL</span>
          </span>
        );
      case 'RECOMMENDED':
        return (
          <span className="text-xs font-mono font-medium px-2 py-0.5 rounded-[4px] border bg-[#151E28] border-[#2BC48A]/50 text-[#2BC48A]">
            RECOMMENDED
          </span>
        );
      case 'RESOURCE_UNAVAILABLE':
        return (
          <span className="text-xs font-mono font-medium px-2 py-0.5 rounded-[4px] border bg-[#151E28] border-[#E05252]/50 text-[#E05252]">
            RESOURCE UNAVAILABLE
          </span>
        );
      case 'NO_ACTION':
        return (
          <span className="text-xs font-mono font-medium px-2 py-0.5 rounded-[4px] border bg-[#151E28] border-[#263341] text-[#98A6B5]">
            NO ACTION
          </span>
        );
      default:
        return (
          <span className="text-xs font-mono font-medium px-2 py-0.5 rounded-[4px] border bg-[#151E28] border-[#263341] text-[#98A6B5]">
            {status}
          </span>
        );
    }
  };

  return (
    <div className="suraksha-panel flex flex-col h-full overflow-hidden border border-[#263341]">
      {/* Header */}
      <div className="suraksha-panel-header px-3.5 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2 text-[#E8EDF3]">
          <Shield className="w-4 h-4 text-[#3B9EFF]" />
          <h2 className="text-xs font-semibold font-sans tracking-wider uppercase">
            RECOMMENDED RESPONSE
          </h2>
        </div>
        <div className="flex items-center gap-2">
          {responsePlan && getStatusBadge(responsePlan.status)}
          {loading && <Loader2 className="w-3.5 h-3.5 text-[#3B9EFF] animate-spin" />}
        </div>
      </div>

      {/* Safety Notice Strip: Non-autonomous decision support gate */}
      <div className="bg-[#151E28] border-b border-[#263341] px-3.5 py-1.5 flex items-center justify-between text-xs font-sans text-[#E7A83B] flex-shrink-0">
        <div className="flex items-center gap-2">
          <AlertTriangle className="w-3.5 h-3.5 text-[#E7A83B] flex-shrink-0" />
          <span className="font-semibold">DECISION SUPPORT ONLY</span>
          <span className="text-[#687585] hidden sm:inline">&bull;</span>
          <span className="text-[#98A6B5] hidden sm:inline">No automatic emergency dispatch.</span>
        </div>
        <span className="text-xs font-mono font-semibold bg-[#111821] border border-[#E7A83B]/40 px-2 py-0.5 rounded-[4px] uppercase tracking-wider text-[#E7A83B]">
          Human Gate Required
        </span>
      </div>

      {/* Body Content */}
      <div className="p-3 flex-1 flex flex-col gap-2.5 overflow-y-auto">
        {/* Case 1: No active incident */}
        {!activeIncident && (
          <div className="flex-1 flex flex-col items-center justify-center text-center p-6 gap-2.5">
            <Shield className="w-9 h-9 text-[#687585] stroke-[1.5]" />
            <div className="text-sm font-sans font-semibold text-[#E8EDF3] uppercase tracking-wide">
              RESPONSE STANDBY
            </div>
            <p className="text-xs font-sans text-[#98A6B5] max-w-[280px] leading-relaxed">
              No active incident requires emergency response. Continuous autonomous surveillance active.
            </p>
            <div className="text-xs font-sans text-[#687585] mt-2 border border-[#263341] bg-[#111821] px-2.5 py-1 rounded-[4px]">
              DECISION SUPPORT ONLY &bull; Operator approval required
            </div>
          </div>
        )}

        {/* Case 2: Loading response plan */}
        {activeIncident && loading && !responsePlan && (
          <div className="flex-1 flex flex-col items-center justify-center p-4 gap-2 text-center">
            <Loader2 className="w-6 h-6 text-[#3B9EFF] animate-spin" />
            <div className="text-xs font-mono font-semibold text-[#E8EDF3]">RESPONSE ANALYSIS</div>
            <div className="text-[10.5px] font-mono text-[#98A6B5]">
              Loading verified response plan for incident{' '}
              <span className="text-[#3B9EFF]">{activeIncident.incident_id}</span>...
            </div>
          </div>
        )}

        {/* Case 3: Error / Plan unavailable */}
        {activeIncident && !loading && (error || !responsePlan) && (
          <div className="flex-1 flex flex-col p-2.5 gap-2 bg-[#151014] border border-[#E05252]/40 rounded-[6px]">
            <div className="flex items-center justify-between text-[#E05252] text-xs font-mono font-semibold">
              <div className="flex items-center gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 text-[#E05252]" />
                <span>RESPONSE PLAN UNAVAILABLE</span>
              </div>
              <button
                onClick={refetch}
                className="p-1 hover:bg-[#151E28] text-[#E05252] hover:text-[#E8EDF3] rounded transition-colors"
                title="Retry fetching response plan"
              >
                <RefreshCw className="w-3 h-3" />
              </button>
            </div>
            <div className="text-[10px] font-mono text-[#E8EDF3]">
              Retained Incident Context:
              <div className="mt-1 text-[#98A6B5] space-y-0.5">
                <div>Type: <strong className="text-[#E8EDF3] uppercase">{activeIncident.incident_type}</strong></div>
                <div>Severity: <strong className="text-[#E8EDF3] uppercase">{activeIncident.severity}</strong></div>
                <div>Location: <strong className="text-[#E8EDF3]">{activeIncident.location?.name || 'Sector Grid'}</strong></div>
              </div>
            </div>
            <div className="text-[9px] font-mono text-[#E05252]/80 mt-auto">
              {error || 'Unable to connect to response planner engine.'}
            </div>
          </div>
        )}

        {/* Case 4: Plan Loaded Successfully */}
        {activeIncident && responsePlan && (
          <>
            {/* Visual Incident -> Severity -> Location Progression Strip */}
            <div className="bg-[#151E28] border border-[#263341] p-2 rounded-[6px] grid grid-cols-3 gap-2 text-xs font-sans">
              <div className="flex flex-col gap-0.5">
                <span className="text-[10.5px] uppercase font-medium text-[#687585]">Target</span>
                <span className="text-[#E8EDF3] font-mono font-semibold uppercase truncate" title={responsePlan.incident_type}>
                  {responsePlan.incident_type.replace(/_/g, ' ')}
                </span>
              </div>
              <div className="flex flex-col gap-0.5">
                <span className="text-[10.5px] uppercase font-medium text-[#687585]">Severity</span>
                <span className="text-[#E05252] font-mono font-semibold uppercase">
                  {responsePlan.severity}
                </span>
              </div>
              <div className="flex flex-col gap-0.5">
                <span className="text-[10.5px] uppercase font-medium text-[#687585]">Safety Gate</span>
                <span className="text-[#E7A83B] font-mono font-semibold">
                  {responsePlan.human_approval_required ? 'REQUIRED' : 'AUTO'}
                </span>
              </div>
            </div>

            {/* Case 4A: NO_ACTION or empty actions */}
            {(responsePlan.status === 'NO_ACTION' || responsePlan.response_actions.length === 0) ? (
              <div className="bg-[#151E28] border border-[#263341] p-3 rounded-[6px] flex flex-col gap-2">
                <div className="flex items-center gap-2 text-[#E8EDF3] font-semibold text-xs font-sans">
                  <CheckCircle className="w-4 h-4 text-[#3B9EFF]" />
                  <span>NO IMMEDIATE RESPONSE RECOMMENDED</span>
                </div>
                <div className="text-xs font-sans text-[#98A6B5] leading-relaxed">
                  System continues routine monitoring. No emergency escalation required under current deterministic safety policy.
                </div>
                {responsePlan.warning && (
                  <div className="text-xs font-sans text-[#3B9EFF] bg-[#111821] p-2 rounded-[4px] border border-[#263341]">
                    Policy Note: {responsePlan.warning}
                  </div>
                )}
              </div>
            ) : (
              /* Case 4B: Specific Actions (POLICE_SECURITY / MEDICAL_ASSISTANCE) */
              <div className="space-y-2.5">
                {responsePlan.response_actions.map((action, idx) => {
                  const isPolice = action.action_type === 'POLICE_SECURITY';
                  let res = action.recommended_resource;
                  if (!isPolice && (!res || (res.distance_km !== null && res.distance_km !== undefined && res.distance_km > 3.0))) {
                    const localTrauma = defaultNearbyHospitals[0];
                    if (localTrauma) {
                      res = {
                        name: localTrauma.name,
                        category: 'Hospitals & Emergency Trauma Centers',
                        place_type: 'hospital',
                        distance_km: localTrauma.distance_km,
                        distance_m: localTrauma.distance_km * 1000,
                        formatted_address: localTrauma.address,
                        latitude: localTrauma.latitude,
                        longitude: localTrauma.longitude,
                        phone_number: localTrauma.phone,
                        distance_type: 'straight_line',
                      };
                    }
                  }

                  return (
                    <div
                      key={idx}
                      className="bg-[#151E28] border border-[#263341] rounded-[6px] p-3 flex flex-col gap-2 hover:border-[#3B9EFF]/40 transition-colors"
                    >
                      {/* Action Header */}
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {isPolice ? (
                            <Shield className="w-4 h-4 text-[#3B9EFF]" />
                          ) : (
                            <Building2 className="w-4 h-4 text-[#E05252]" />
                          )}
                          <span className="text-sm font-sans font-semibold text-[#E8EDF3]">
                            {isPolice ? 'POLICE SUPPORT' : 'MEDICAL SUPPORT'}
                          </span>
                        </div>
                        <div className="flex items-center gap-1">
                          <span
                            className={`text-xs font-mono font-semibold px-2 py-0.5 rounded-[4px] border uppercase ${getPriorityBadgeClass(
                              action.priority
                            )}`}
                          >
                            {action.priority}
                          </span>
                        </div>
                      </div>

                      {/* Deterministic Policy Reason */}
                      <p className="text-xs font-sans text-[#98A6B5] leading-relaxed">
                        {action.reason}
                      </p>

                      {/* Recommended Resource Details */}
                      {res && (
                        <div className="bg-[#111821] border border-[#263341] p-2.5 rounded-[6px] space-y-1.5">
                          <div className="flex items-start justify-between gap-2">
                            <span
                              className="text-sm font-sans font-semibold text-[#E8EDF3] leading-snug"
                              title={res.name}
                            >
                              {res.name}
                            </span>
                            {res.distance_km !== null && res.distance_km !== undefined && (
                              <span
                                className="text-xs font-mono font-semibold text-[#49C6D9] bg-[#151E28] border border-[#263341] px-2 py-0.5 rounded-[4px] flex-shrink-0"
                                title="Straight-line geographic distance. Not road driving ETA."
                              >
                                {res.distance_km.toFixed(1)} km
                              </span>
                            )}
                          </div>

                          {res.formatted_address && (
                            <div className="text-xs font-sans text-[#98A6B5] leading-snug break-words">
                              {res.formatted_address}
                            </div>
                          )}

                          {/* Contact Info: Genuine 100/112 Helpline + Station Landline */}
                          <div className="pt-1.5 border-t border-[#263341]/60 flex flex-col sm:flex-row sm:items-center justify-between gap-1.5">
                            {isPolice ? (
                              <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs font-mono">
                                <div className="flex items-center gap-1.5 text-[#E8EDF3]">
                                  <PhoneCall className="w-3.5 h-3.5 text-[#3B9EFF] flex-shrink-0" />
                                  <span>Helpline: <strong className="text-[#3B9EFF]">112 / 100</strong></span>
                                </div>
                                <span className="text-[#687585] hidden sm:inline">&bull;</span>
                                <div className="text-[#98A6B5]">
                                  Desk: <span className="text-[#E8EDF3]">011 2805 1585</span>
                                </div>
                              </div>
                            ) : (
                              <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs font-mono">
                                <div className="flex items-center gap-1.5 text-[#E8EDF3]">
                                  <PhoneCall className="w-3.5 h-3.5 text-[#E05252] flex-shrink-0" />
                                  <span>Ambulance: <strong className="text-[#E05252]">102 / 108</strong></span>
                                </div>
                                <span className="text-[#687585] hidden sm:inline">&bull;</span>
                                <div className="text-[#98A6B5]">
                                  Emergency: <span className="text-[#E8EDF3]">{res.phone_number || '011 2089 5000'}</span>
                                </div>
                              </div>
                            )}

                            {res.latitude && res.longitude && (
                              <button
                                onClick={() => handleFocusResource(action, res)}
                                className="inline-flex items-center gap-1.5 text-xs font-sans font-medium text-[#3B9EFF] hover:text-[#49C6D9] cursor-pointer self-end sm:self-auto"
                                title="Focus this facility on the Command Centre Map"
                              >
                                <Eye className="w-3.5 h-3.5" />
                                <span>Focus on Map</span>
                              </button>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
