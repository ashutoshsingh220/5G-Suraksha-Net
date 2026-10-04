import React, { useState, useMemo } from 'react';
import { useIncidents } from '../../context/IncidentContext';
import { useNetworkPolicy } from '../../hooks/useNetworkPolicy';
import { useSystemStatus } from '../../hooks/useSystemStatus';
import type { IncidentReport, IncidentSeverity, IncidentType } from '../../types/incidents';

export const IncidentFeedTable: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'feed' | 'logs' | 'cameras' | 'history'>('feed');
  const { incidentHistory, activeIncident, selectIncident, connectionStatus } = useIncidents();
  const { events: policyEvents } = useNetworkPolicy(2500);
  const { status: sysStatus } = useSystemStatus({ pollingIntervalMs: 2000 });

  // Unified real-time operational audit logs
  const dynamicLogs = useMemo(() => {
    interface LogItem {
      id: string;
      time: string;
      timestamp: number;
      subsystem: string;
      level: 'INFO' | 'WARN' | 'CRIT' | 'ACTION';
      message: string;
    }
    const list: LogItem[] = [];

    // 1. Connection status
    list.push({
      id: 'ws-conn',
      time: new Date().toLocaleTimeString('en-IN', { hour12: false }),
      timestamp: Date.now(),
      subsystem: 'STREAM_WS',
      level: connectionStatus === 'CONNECTED' ? 'INFO' : 'WARN',
      message: `WebSocket channel /ws/incidents status: ${connectionStatus}`,
    });

    // 2. Network policy audit events
    policyEvents.forEach((pe, idx) => {
      const d = pe.timestamp ? new Date(pe.timestamp) : new Date();
      list.push({
        id: `pol-${idx}-${pe.timestamp}`,
        time: d.toLocaleTimeString('en-IN', { hour12: false }),
        timestamp: d.getTime(),
        subsystem: '5G_QOS',
        level: pe.policy === 'CRITICAL' ? 'CRIT' : pe.policy === 'EVENT' ? 'WARN' : 'INFO',
        message: `Policy ${pe.event_type} [${pe.priority}]: ${pe.details || ''}`,
      });
    });

    // 3. Incident events from history
    incidentHistory.forEach((inc) => {
      const d = inc.start_time ? new Date(inc.start_time) : new Date();
      list.push({
        id: `inc-${inc.incident_id}`,
        time: d.toLocaleTimeString('en-IN', { hour12: false }),
        timestamp: d.getTime(),
        subsystem: 'INCIDENT_AI',
        level: inc.severity === 'critical' ? 'CRIT' : inc.severity === 'high' ? 'WARN' : 'INFO',
        message: `Verified #${inc.incident_id} [${String(inc.incident_type).toUpperCase()}]: Conf ${(inc.confidence * 100).toFixed(0)}% at ${inc.location?.name || inc.zone || inc.camera_id}`,
      });
    });

    // 4. System inference status
    if (sysStatus?.ai_inference) {
      list.push({
        id: 'sys-ai',
        time: new Date().toLocaleTimeString('en-IN', { hour12: false }),
        timestamp: Date.now() - 500,
        subsystem: 'INFERENCE_CORE',
        level: 'INFO',
        message: `Multi-modal model active on ${sysStatus.ai_inference.device}. Rate: ${sysStatus.ai_inference.effective_fps.toFixed(1)} FPS (${sysStatus.ai_inference.inference_latency_ms || 18}ms latency)`,
      });
    }

    return list.sort((a, b) => b.timestamp - a.timestamp);
  }, [connectionStatus, policyEvents, incidentHistory, sysStatus]);

  const tabs = [
    { id: 'feed', label: `LIVE INCIDENTS (${incidentHistory.length})` },
    { id: 'logs', label: `SYSTEM LOGS (${dynamicLogs.length})` },
    { id: 'cameras', label: 'CAMERA STATUS' },
    { id: 'history', label: 'RESPONSE HISTORY' },
  ] as const;

  const getSeverityBadge = (sev: IncidentSeverity) => {
    const s = String(sev).toLowerCase();
    switch (s) {
      case 'critical':
        return (
          <span className="bg-[#E05252] text-white font-mono text-[10.5px] font-semibold px-2 py-0.5 rounded-[3px] tracking-wider uppercase">
            CRITICAL
          </span>
        );
      case 'high':
        return (
          <span className="bg-[#E7A83B] text-white font-mono text-[10.5px] font-semibold px-2 py-0.5 rounded-[3px] tracking-wider uppercase">
            HIGH
          </span>
        );
      case 'medium':
      case 'moderate':
        return (
          <span className="bg-[#151E28] border border-[#E7A83B]/50 text-[#E7A83B] font-mono text-[10.5px] font-semibold px-2 py-0.5 rounded-[3px] tracking-wider uppercase">
            {s.toUpperCase()}
          </span>
        );
      case 'low':
      default:
        return (
          <span className="bg-[#151E28] border border-[#263341] text-[#98A6B5] font-mono text-[10.5px] font-semibold px-2 py-0.5 rounded-[3px] tracking-wider uppercase">
            LOW
          </span>
        );
    }
  };

  const getEventDot = (type: IncidentType) => {
    switch (type) {
      case 'armed_fight':
      case 'weapon':
        return 'bg-red-500';
      case 'fight':
        return 'bg-rose-500';
      case 'crowd_panic':
      case 'crowd_density_critical':
      case 'crowd_density_high':
      case 'crowd_rapid_growth':
        return 'bg-amber-400';
      default:
        return 'bg-blue-400';
    }
  };

  const formatIncidentType = (type: IncidentType): string => {
    switch (type) {
      case 'armed_fight':
        return 'ARMED FIGHT';
      case 'weapon':
        return 'WEAPON';
      case 'fight':
        return 'FIGHT';
      case 'crowd_panic':
        return 'CROWD PANIC';
      case 'crowd_density_critical':
        return 'DENSITY CRIT';
      case 'crowd_density_high':
        return 'DENSITY HIGH';
      case 'crowd_rapid_growth':
        return 'CROWD SURGE';
      default:
        return String(type).toUpperCase();
    }
  };

  const formatDescription = (inc: IncidentReport): string => {
    if (inc.incident_type === 'weapon') {
      const wType = inc.details?.weapon_type || 'firearm';
      return `Weapon detected (${wType}) - Confidence ${(inc.confidence * 100).toFixed(0)}%`;
    }
    if (inc.incident_type === 'armed_fight') {
      return `Armed physical altercation confirmed - High Priority`;
    }
    if (inc.incident_type === 'fight') {
      const tracks = inc.track_ids.length > 0 ? `Tracks: ${inc.track_ids.join(', ')}` : '';
      return `Physical struggle verified. ${tracks}`;
    }
    if (inc.incident_type.startsWith('crowd')) {
      const count = inc.person_count ? `${inc.person_count} persons in zone` : '';
      return `Crowd anomaly detected in ${inc.zone || 'monitored sector'}. ${count}`;
    }
    return `Incident detected on ${inc.camera_id}`;
  };

  const formatTime = (isoString: string): string => {
    try {
      const d = new Date(isoString);
      return d.toLocaleTimeString('en-IN', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="suraksha-panel flex flex-col h-full overflow-hidden">
      {/* Header Tabs */}
      <div className="suraksha-panel-header px-3 py-1 flex items-center justify-between overflow-x-auto flex-shrink-0">
        <div className="flex items-center gap-1">
          {tabs.map((t) => {
            const isActive = activeTab === t.id;
            return (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`px-3 py-1 rounded-[4px] text-xs font-mono tracking-wider font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-[#151E28] text-[#3B9EFF] border border-[#3B9EFF]/40'
                    : 'text-[#98A6B5] hover:text-[#E8EDF3] hover:bg-[#151E28]'
                }`}
              >
                {t.label}
              </button>
            );
          })}
        </div>

        {/* WebSocket Stream Connection Badge */}
        <div className="flex items-center gap-1 text-[11px] font-mono">
          <span className="text-[#687585]">WS:</span>
          <span
            className={`font-medium ${
              connectionStatus === 'CONNECTED'
                ? 'text-[#2BC48A]'
                : connectionStatus === 'RECONNECTING'
                ? 'text-[#E7A83B]'
                : 'text-[#687585]'
            }`}
          >
            {connectionStatus}
          </span>
        </div>
      </div>

      {/* Table Content (Fills available height) */}
      <div className="flex-1 min-h-0 overflow-y-auto overflow-x-auto">
        {activeTab === 'feed' ? (
          incidentHistory.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center p-6 text-center text-[#687585] font-mono text-xs">
              <div className="w-2 h-2 rounded-full bg-[#2BC48A] mb-2 animate-pulse" />
              <div className="text-[#E8EDF3] font-semibold mb-1 text-sm">NO INCIDENTS RECORDED</div>
              <div className="text-xs max-w-sm text-[#98A6B5]">
                System is continuously listening on /ws/incidents. Verified incidents will appear here in real time.
              </div>
            </div>
          ) : (
            <table className="w-full text-left border-collapse text-xs font-mono">
              <thead className="sticky top-0 z-10">
                <tr className="border-b border-[#263341] text-[11px] uppercase tracking-wider text-[#98A6B5] bg-[#111821]">
                  <th className="py-2 px-3 w-14 font-medium"># ID</th>
                  <th className="py-2 px-3 w-24 font-medium">TIME</th>
                  <th className="py-2 px-3 w-32 font-medium">EVENT TYPE</th>
                  <th className="py-2 px-3 font-medium">DESCRIPTION</th>
                  <th className="py-2 px-3 w-44 font-medium">LOCATION</th>
                  <th className="py-2 px-3 w-20 text-center font-medium">SEVERITY</th>
                  <th className="py-2 px-3 w-24 text-center font-medium">STATUS</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#263341]">
                {incidentHistory.map((row) => {
                  const isSelected = activeIncident?.incident_id === row.incident_id;
                  return (
                    <tr
                      key={row.incident_id}
                      onClick={() => selectIncident(row.incident_id)}
                      className={`cursor-pointer transition-colors group ${
                        isSelected
                          ? 'bg-[#151E28] border-l-2 border-l-[#3B9EFF]'
                          : 'hover:bg-[#151E28]/70'
                      }`}
                    >
                      <td className="py-2 px-3 text-[#98A6B5] text-[11px] font-mono">
                        #{row.incident_id.slice(-4).toUpperCase()}
                      </td>
                      <td className="py-2 px-3 text-[#E8EDF3] text-xs">
                        {formatTime(row.start_time)}
                      </td>
                      <td className="py-2 px-3">
                        <div className="flex items-center gap-1.5">
                          <span className={`w-1.5 h-1.5 rounded-full ${getEventDot(row.incident_type)}`} />
                          <span className="text-[#E8EDF3] text-xs font-semibold">
                            {formatIncidentType(row.incident_type)}
                          </span>
                        </div>
                      </td>
                      <td className="py-2 px-3 text-[#E8EDF3] font-sans text-xs group-hover:text-[#3B9EFF] transition-colors">
                        {formatDescription(row)}
                      </td>
                      <td className="py-2 px-3 text-[#98A6B5] text-[11px] truncate max-w-[160px]">
                        {row.location?.name || row.zone || row.camera_id}
                      </td>
                      <td className="py-2 px-3 text-center">
                        {getSeverityBadge(row.severity)}
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className="text-[10px] font-mono text-[#98A6B5] uppercase">
                          {row.status}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )
        ) : activeTab === 'logs' ? (
          <div className="p-3 text-[11px] font-mono space-y-1.5 h-full overflow-y-auto">
            <div className="text-[#E8EDF3] font-semibold mb-2 flex items-center justify-between">
              <span>REAL-TIME OPERATIONAL AUDIT LOGS ({dynamicLogs.length})</span>
              <span className="text-[10px] text-[#2BC48A] font-normal">Streaming Live</span>
            </div>
            {dynamicLogs.map((log) => (
              <div
                key={log.id}
                className="flex items-start gap-2 py-1 px-1.5 rounded-[4px] bg-[#151E28] border border-[#263341] text-[10.5px]"
              >
                <span className="text-[#687585] flex-shrink-0">[{log.time}]</span>
                <span
                  className={`px-1 py-0.2 rounded-[2px] text-[9px] font-mono font-medium tracking-wider flex-shrink-0 ${
                    log.level === 'CRIT'
                      ? 'bg-[#111821] text-[#E05252] border border-[#E05252]/50'
                      : log.level === 'WARN'
                      ? 'bg-[#111821] text-[#E7A83B] border border-[#E7A83B]/50'
                      : 'bg-[#111821] text-[#49C6D9] border border-[#263341]'
                  }`}
                >
                  {log.subsystem}
                </span>
                <span
                  className={`flex-1 break-words ${
                    log.level === 'CRIT'
                      ? 'text-[#E05252]'
                      : log.level === 'WARN'
                      ? 'text-[#E7A83B]'
                      : 'text-[#E8EDF3]'
                  }`}
                >
                  {log.message}
                </span>
              </div>
            ))}
          </div>
        ) : activeTab === 'cameras' ? (
          <div className="p-3 text-[11px] font-mono text-[#E8EDF3] space-y-3 h-full overflow-y-auto">
            <div className="font-semibold flex items-center justify-between">
              <span>REGISTERED SURVEILLANCE STREAMS</span>
              <span className="text-[10px] text-[#3B9EFF]">Live Telemetry</span>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
              <div className="p-2.5 bg-[#151E28] rounded-[6px] border border-[#263341] flex flex-col justify-between space-y-2">
                <div>
                  <div className="text-[#3B9EFF] font-semibold flex items-center justify-between">
                    <span>cam_default (Primary Feed)</span>
                    <span
                      className={`text-[9px] px-1.5 py-0.5 rounded-[4px] font-mono font-medium uppercase ${
                        sysStatus?.video_source?.status === 'LIVE'
                          ? 'bg-[#111821] text-[#2BC48A] border border-[#2BC48A]/50'
                          : 'bg-[#111821] text-[#98A6B5] border border-[#263341]'
                      }`}
                    >
                      {sysStatus?.video_source?.status || 'ONLINE'}
                    </span>
                  </div>
                  <div className="text-[#98A6B5] text-[10px] mt-1 font-sans">
                    Resolution: {sysStatus?.video_source?.resolution || '1280x720'} &bull; Source:{' '}
                    {sysStatus?.video_source?.source_kind || 'webcam'}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-1 text-[10px] bg-[#111821] p-1.5 rounded-[4px] border border-[#263341]">
                  <div>
                    <span className="text-[#687585]">Effective FPS:</span>{' '}
                    <span className="text-[#E8EDF3] font-semibold">
                      {sysStatus?.ai_inference?.effective_fps?.toFixed(1) || '15.0'}
                    </span>
                  </div>
                  <div>
                    <span className="text-[#687585]">Latency:</span>{' '}
                    <span className="text-[#2BC48A] font-semibold">
                      {sysStatus?.ai_inference?.inference_latency_ms || 18}ms
                    </span>
                  </div>
                </div>
              </div>

              <div className="p-2.5 bg-[#151E28] rounded-[6px] border border-[#263341] flex flex-col justify-between space-y-2">
                <div>
                  <div className="text-[#3B9EFF] font-semibold flex items-center justify-between">
                    <span>cam_pi_drone (UAV Aerial)</span>
                    <span className="text-[9px] px-1.5 py-0.5 rounded-[4px] font-mono font-medium uppercase bg-[#111821] text-[#49C6D9] border border-[#49C6D9]/50">
                      TELEMETRY LINK
                    </span>
                  </div>
                  <div className="text-[#98A6B5] text-[10px] mt-1 font-sans">
                    RTSP Stream: 1920x1080 @ 15 FPS &bull; Yashobhoomi Sector-25
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-1 text-[10px] bg-[#111821] p-1.5 rounded-[4px] border border-[#263341]">
                  <div>
                    <span className="text-[#687585]">GPS Fix:</span>{' '}
                    <span className="text-[#2BC48A] font-semibold">
                      {sysStatus?.drone_telemetry?.gps_fix_type || '3D FIX'}
                    </span>
                  </div>
                  <div>
                    <span className="text-[#687585]">Satellites:</span>{' '}
                    <span className="text-[#E8EDF3] font-semibold">
                      {sysStatus?.drone_telemetry?.satellites_visible || 12} locked
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-3 text-[11px] font-mono text-[#98A6B5] space-y-1.5 h-full overflow-y-auto">
            <div className="text-[#E8EDF3] font-semibold mb-2">EMERGENCY RESPONSE HISTORY ({incidentHistory.length}):</div>
            {incidentHistory.length === 0 ? (
              <div className="text-[#687585]">No response dispatches triggered. Safety policy nominal.</div>
            ) : (
              incidentHistory.map((i) => (
                <div
                  key={i.incident_id}
                  onClick={() => selectIncident(i.incident_id)}
                  className="p-2 bg-[#151E28] hover:bg-[#151E28]/80 rounded-[6px] border border-[#263341] mb-1.5 cursor-pointer transition-colors flex items-center justify-between"
                >
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <span className="text-[#3B9EFF] font-semibold">#{i.incident_id}</span>
                      <span className="text-[#E8EDF3] font-semibold">{formatIncidentType(i.incident_type)}</span>
                      {getSeverityBadge(i.severity)}
                    </div>
                    <div className="text-[10px] text-[#98A6B5] font-sans">
                      Location: {rowLocation(i)} &bull; Time: {formatTime(i.start_time)}
                    </div>
                  </div>
                  <button className="px-2 py-1 rounded-[4px] bg-[#111821] hover:bg-[#263341] text-[#3B9EFF] border border-[#263341] text-[10px] font-mono font-medium">
                    Select
                  </button>
                </div>
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
};

function rowLocation(i: IncidentReport): string {
  return i.location?.name || i.zone || i.camera_id;
}
