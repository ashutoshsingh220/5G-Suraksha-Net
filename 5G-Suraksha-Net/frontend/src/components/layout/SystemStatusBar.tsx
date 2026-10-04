import React from 'react';
import { useIncidents } from '../../context/IncidentContext';
import { useSystemStatus } from '../../hooks/useSystemStatus';

export const SystemStatusBar: React.FC = () => {
  const { connectionStatus, totalIncidentCount } = useIncidents();
  const { status: sysStatus, loading } = useSystemStatus({ pollingIntervalMs: 3000 });

  const getSystemColor = (status?: string) => {
    switch (status) {
      case 'ONLINE':
        return 'text-[#2BC48A]';
      case 'DEGRADED':
        return 'text-[#E7A83B]';
      case 'OFFLINE':
        return 'text-[#E05252]';
      default:
        return 'text-[#98A6B5]';
    }
  };

  const getStreamColor = (status: string) => {
    switch (status) {
      case 'CONNECTED':
        return 'text-[#2BC48A]';
      case 'CONNECTING':
      case 'RECONNECTING':
        return 'text-[#E7A83B]';
      case 'DISCONNECTED':
      case 'ERROR':
      default:
        return 'text-[#E05252]';
    }
  };

  // Truthful network link state (no fabricated 5G label)
  const isNetworkOnline = sysStatus && sysStatus.status !== 'OFFLINE';
  const networkLabel = isNetworkOnline ? 'ONLINE' : loading ? 'CHECKING' : 'OFFLINE';
  const networkColor = isNetworkOnline ? 'text-[#2BC48A]' : 'text-[#E05252]';

  // AI & Device
  const aiDevice = sysStatus?.ai_inference?.device?.toUpperCase() || 'CPU';
  const aiState = sysStatus?.ai_inference?.status || (loading ? 'INIT' : 'OFFLINE');
  const aiDisplay = `${aiDevice} (${aiState})`;
  const aiColor = aiState === 'ACTIVE' ? 'text-[#49C6D9]' : 'text-[#98A6B5]';

  // Video Source
  const videoKind = sysStatus?.video_source?.source_kind?.toUpperCase() || 'VIDEO';
  const videoState = sysStatus?.video_source?.status || (loading ? 'INIT' : 'OFFLINE');
  const videoDisplay = sysStatus?.video_source?.resolution
    ? `${videoKind} [${sysStatus.video_source.resolution}] (${videoState})`
    : `${videoKind} (${videoState})`;
  const videoColor = videoState === 'LIVE' ? 'text-[#2BC48A]' : 'text-[#98A6B5]';

  // FPS & Latency
  const effectiveFps = sysStatus?.video_source?.effective_fps ?? 0.0;
  const latencyMs = sysStatus?.ai_inference?.inference_latency_ms;
  const latencyDisplay = latencyMs != null && latencyMs > 0 ? `${latencyMs.toFixed(1)} ms` : 'N/A';

  // GPU Memory (truthful: no fabricated % utilization without hardware profiling daemon)
  let gpuDisplay = 'CPU (N/A)';
  if (sysStatus?.gpu?.available) {
    const gpuName = sysStatus.gpu.name || 'CUDA';
    if (sysStatus.gpu.memory_used_mb != null && sysStatus.gpu.memory_total_mb != null) {
      gpuDisplay = `${gpuName} [${sysStatus.gpu.memory_used_mb}/${sysStatus.gpu.memory_total_mb} MB]`;
    } else {
      gpuDisplay = `${gpuName} (ACTIVE)`;
    }
  }

  // Telemetry status
  const droneConnected = sysStatus?.drone_telemetry?.connected;
  const droneSats = sysStatus?.drone_telemetry?.satellites_visible ?? 0;
  const droneDisplay = droneConnected
    ? `MAVLINK (${droneSats} SATS)`
    : 'STANDBY';

  const metrics = [
    {
      label: 'SYSTEM',
      value: sysStatus?.status || (loading ? 'CHECKING' : 'OFFLINE'),
      icon: <span className={`w-1.5 h-1.5 rounded-full ${sysStatus?.status === 'ONLINE' ? 'bg-[#2BC48A]' : 'bg-[#E05252]'}`} />,
      color: getSystemColor(sysStatus?.status),
    },
    {
      label: 'NETWORK LINK',
      value: networkLabel,
      icon: <span className={`w-1.5 h-1.5 rounded-full ${isNetworkOnline ? 'bg-[#2BC48A]' : 'bg-[#E05252]'}`} />,
      color: networkColor,
    },
    {
      label: 'AI ENGINE',
      value: aiDisplay,
      icon: <span className={`w-1.5 h-1.5 rounded-full ${aiState === 'ACTIVE' ? 'bg-[#49C6D9]' : 'bg-[#98A6B5]'}`} />,
      color: aiColor,
    },
    {
      label: 'SOURCE',
      value: videoDisplay,
      icon: <span className={`w-1.5 h-1.5 rounded-full ${videoState === 'LIVE' ? 'bg-[#2BC48A]' : 'bg-[#98A6B5]'}`} />,
      color: videoColor,
    },
    {
      label: 'INCIDENT STREAM',
      value: connectionStatus,
      icon: <span className={`w-1.5 h-1.5 rounded-full ${connectionStatus === 'CONNECTED' ? 'bg-[#2BC48A]' : 'bg-[#E7A83B]'}`} />,
      color: getStreamColor(connectionStatus),
    },
    {
      label: 'INCIDENTS',
      value: String(totalIncidentCount).padStart(2, '0'),
      icon: <span className={`w-1.5 h-1.5 rounded-full ${totalIncidentCount > 0 ? 'bg-[#E05252]' : 'bg-[#98A6B5]'}`} />,
      color: totalIncidentCount > 0 ? 'text-[#E05252]' : 'text-[#98A6B5]',
    },
    {
      label: 'FPS',
      value: effectiveFps.toFixed(1),
      icon: null,
      color: 'text-[#E8EDF3]',
    },
    {
      label: 'LATENCY',
      value: latencyDisplay,
      icon: null,
      color: latencyMs != null && latencyMs < 50 ? 'text-[#2BC48A]' : 'text-[#E7A83B]',
    },
    {
      label: 'ACCELERATOR',
      value: gpuDisplay,
      icon: null,
      color: 'text-[#98A6B5]',
    },
    {
      label: 'DRONE LINK',
      value: droneDisplay,
      icon: <span className={`w-1.5 h-1.5 rounded-full ${droneConnected ? 'bg-[#2BC48A]' : 'bg-[#687585]'}`} />,
      color: droneConnected ? 'text-[#2BC48A]' : 'text-[#98A6B5]',
    },
  ];

  return (
    <div className="h-[34px] bg-[#111821] border-b border-[#263341] px-4 flex items-center justify-between flex-shrink-0 text-xs font-mono select-none overflow-x-auto scrollbar-none">
      <div className="flex items-center gap-5 w-full justify-between">
        {metrics.map((m, idx) => (
          <div key={idx} className="flex items-center gap-1.5 whitespace-nowrap">
            {m.icon}
            <span className="text-xs text-[#98A6B5] tracking-wider uppercase font-sans font-medium">
              {m.label}:
            </span>
            <span className={`text-xs font-mono font-semibold ${m.color}`}>
              {m.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
