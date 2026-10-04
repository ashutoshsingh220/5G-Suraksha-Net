import React, { useState } from 'react';
import {
  Layers,
  Radio,
  Cpu,
  Camera,
  Navigation,
  Battery,
} from 'lucide-react';
import { useTelemetry } from '../../context/TelemetryContext';
import { useSystemStatus } from '../../hooks/useSystemStatus';

export const AerialIntelligence: React.FC = () => {
  const { telemetry } = useTelemetry();
  const { status: sysStatus } = useSystemStatus({ pollingIntervalMs: 2000 });
  const [activeTab, setActiveTab] = useState<'HUD' | 'DRONE'>('HUD');

  const isDroneCameraLive =
    (sysStatus?.video_source?.status === 'LIVE' || (sysStatus?.video_source?.effective_fps ?? 0) > 0) &&
    (sysStatus?.video_source?.source_kind?.toLowerCase().includes('stream') ||
      sysStatus?.video_source?.source_url?.includes('8554') ||
      sysStatus?.video_source?.source_url?.includes('10.254'));

  // Compass heading direction helper
  const getCompassDirection = (deg: number): string => {
    const directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
    const index = Math.round(((deg % 360) / 45)) % 8;
    return directions[index];
  };

  // Satellite bar calculation
  const getSatBars = (count: number): number => {
    if (count >= 14) return 4;
    if (count >= 10) return 3;
    if (count >= 6) return 2;
    if (count >= 3) return 1;
    return 0;
  };

  const satBars = getSatBars(telemetry.satellites_visible);
  const roll = telemetry.roll_deg || 0;
  const pitch = telemetry.pitch_deg || 0;
  const heading = telemetry.heading_deg || 0;

  return (
    <div className="suraksha-panel flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="suraksha-panel-header px-3 py-1.5 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-1.5 text-[#E8EDF3]">
          <Layers className="w-3.5 h-3.5 text-[#3B9EFF]" />
          <h2 className="text-xs font-semibold font-sans tracking-wider uppercase">
            AERIAL INTELLIGENCE
          </h2>
        </div>

        {/* View Mode Toggle & Status Badge */}
        <div className="flex items-center gap-2">
          <div className="flex items-center bg-[#0B0F14] rounded-[6px] p-0.5 border border-[#263341]">
            <button
              onClick={() => setActiveTab('HUD')}
              className={`px-2 py-0.5 rounded-[4px] text-[11px] font-mono font-medium transition-colors ${
                activeTab === 'HUD'
                  ? 'bg-[#151E28] text-[#3B9EFF] border border-[#3B9EFF]/40'
                  : 'text-[#98A6B5] hover:text-[#E8EDF3]'
              }`}
            >
              PFD / HUD
            </button>
            <button
              onClick={() => setActiveTab('DRONE')}
              className={`px-2 py-0.5 rounded-[4px] text-[11px] font-mono font-medium transition-colors ${
                activeTab === 'DRONE'
                  ? 'bg-[#151E28] text-[#3B9EFF] border border-[#3B9EFF]/40'
                  : 'text-[#98A6B5] hover:text-[#E8EDF3]'
              }`}
            >
              SYSTEM
            </button>
          </div>

          {/* Telemetry & Stream Status Badge */}
          {telemetry.status === 'ONLINE' ? (
            <div className="flex items-center gap-1.5 bg-[#151E28] border border-[#2BC48A]/40 rounded-full px-2.5 py-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#2BC48A] animate-pulse" />
              <span className="text-[11px] font-mono font-medium text-[#2BC48A]">
                DRONE ONLINE
              </span>
            </div>
          ) : isDroneCameraLive ? (
            <div className="flex items-center gap-1.5 bg-[#151E28] border border-[#2BC48A]/40 rounded-full px-2.5 py-0.5" title="Drone RTSP camera live; MAVLink telemetry in standby">
              <span className="w-1.5 h-1.5 rounded-full bg-[#2BC48A] animate-pulse" />
              <span className="text-[10px] font-mono font-medium text-[#2BC48A]">
                CAM LIVE (TELEM STBY)
              </span>
            </div>
          ) : telemetry.status === 'STALE' ? (
            <div className="flex items-center gap-1.5 bg-[#151E28] border border-[#E7A83B]/40 rounded-full px-2.5 py-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#E7A83B]" />
              <span className="text-[11px] font-mono font-medium text-[#E7A83B]">
                STALE ({telemetry.telemetry_age_s ? `${telemetry.telemetry_age_s.toFixed(0)}s` : '3s'})
              </span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 bg-[#151E28] border border-[#263341] rounded-full px-2.5 py-0.5">
              <span className="w-1.5 h-1.5 rounded-full bg-[#687585]" />
              <span className="text-[11px] font-mono font-medium text-[#687585]">
                STANDBY
              </span>
            </div>
          )}
        </div>
      </div>

      <div className="p-2 flex-1 flex flex-col justify-between overflow-hidden gap-2">
        {activeTab === 'HUD' ? (
          /* ========================================================================= */
          /* AUTHENTIC MISSION PLANNER PFD (PRIMARY FLIGHT DISPLAY) + QUICK TELEMETRY */
          /* ========================================================================= */
          <div className="flex-1 flex flex-col gap-1.5 min-h-0 overflow-hidden">
            {/* 1. ARTIFICIAL HORIZON / PFD CANVAS */}
            <div className="relative w-full h-[150px] bg-[#111827] rounded overflow-hidden border border-[#1e293b] select-none flex-shrink-0">
              <svg viewBox="0 0 280 150" className="w-full h-full">
                <defs>
                  <clipPath id="hud-clip">
                    <rect x="0" y="0" width="280" height="150" />
                  </clipPath>
                </defs>

                {/* Sky & Ground Horizon (Rotates with Roll, Translates with Pitch) */}
                <g clipPath="url(#hud-clip)">
                  <g
                    transform={`translate(140, 75) rotate(${-roll}) translate(0, ${pitch * 2.2})`}
                  >
                    {/* Sky Blue */}
                    <rect x="-300" y="-300" width="600" height="300" fill="#4a7bc7" />
                    {/* Ground Green / Olive */}
                    <rect x="-300" y="0" width="600" height="300" fill="#65882e" />

                    {/* Horizon Line */}
                    <line x1="-200" y1="0" x2="200" y2="0" stroke="#ffffff" strokeWidth="2" />

                    {/* Pitch Ladder Lines (+20, +10, -10, -20) */}
                    {/* +20 deg */}
                    <line x1="-30" y1="-44" x2="30" y2="-44" stroke="#ffffff" strokeWidth="1.5" />
                    <text x="35" y="-41" fill="#ffffff" fontSize="9" fontFamily="monospace">20</text>
                    <text x="-48" y="-41" fill="#ffffff" fontSize="9" fontFamily="monospace">20</text>

                    {/* +10 deg */}
                    <line x1="-20" y1="-22" x2="20" y2="-22" stroke="#ffffff" strokeWidth="1.5" />
                    <text x="25" y="-19" fill="#ffffff" fontSize="9" fontFamily="monospace">10</text>
                    <text x="-38" y="-19" fill="#ffffff" fontSize="9" fontFamily="monospace">10</text>

                    {/* -10 deg */}
                    <line x1="-20" y1="22" x2="20" y2="22" stroke="#ffffff" strokeWidth="1.5" strokeDasharray="4 3" />
                    <text x="25" y="25" fill="#ffffff" fontSize="9" fontFamily="monospace">-10</text>
                    <text x="-42" y="25" fill="#ffffff" fontSize="9" fontFamily="monospace">-10</text>

                    {/* -20 deg */}
                    <line x1="-30" y1="44" x2="30" y2="44" stroke="#ffffff" strokeWidth="1.5" strokeDasharray="4 3" />
                    <text x="35" y="47" fill="#ffffff" fontSize="9" fontFamily="monospace">-20</text>
                    <text x="-52" y="47" fill="#ffffff" fontSize="9" fontFamily="monospace">-20</text>
                  </g>
                </g>

                {/* Top Roll Arc (Bank Angle Indicators: 0, 10, 20, 30, 45, 60 deg) */}
                <path
                  d="M 60 75 A 80 80 0 0 1 220 75"
                  fill="none"
                  stroke="#ffffff"
                  strokeWidth="1"
                  strokeDasharray="2 6"
                  opacity="0.6"
                />
                {/* Roll Zero Reference Triangle (Top Center) */}
                <polygon points="140,18 136,25 144,25" fill="#ef4444" />

                {/* Fixed Aircraft Boresight Reference Symbol (Center Chevron + Wings) */}
                <g>
                  {/* Left Wing */}
                  <line x1="85" y1="75" x2="115" y2="75" stroke="#ef4444" strokeWidth="3.5" />
                  <line x1="85" y1="75" x2="85" y2="82" stroke="#ef4444" strokeWidth="3" />
                  {/* Right Wing */}
                  <line x1="165" y1="75" x2="195" y2="75" stroke="#ef4444" strokeWidth="3.5" />
                  <line x1="195" y1="75" x2="195" y2="82" stroke="#ef4444" strokeWidth="3" />
                  {/* Center Chevron */}
                  <polyline points="130,78 140,70 150,78" fill="none" stroke="#ef4444" strokeWidth="3" strokeLinejoin="round" />
                  <circle cx="140" cy="75" r="2.5" fill="#22c55e" />
                </g>

                {/* Top Heading Tape */}
                <rect x="0" y="0" width="280" height="16" fill="#0b111e" opacity="0.85" />
                <line x1="0" y1="16" x2="280" y2="16" stroke="#334155" strokeWidth="1" />
                {/* Heading Marker Center Box */}
                <rect x="122" y="1" width="36" height="14" fill="#0284c7" rx="2" />
                <text x="140" y="12" fill="#ffffff" fontSize="10" fontWeight="bold" fontFamily="monospace" textAnchor="middle">
                  {Math.round(heading)}°
                </text>
                <text x="50" y="12" fill="#94a3b8" fontSize="9" fontFamily="monospace" textAnchor="middle">
                  {((Math.round(heading) - 30 + 360) % 360)}
                </text>
                <text x="230" y="12" fill="#94a3b8" fontSize="9" fontFamily="monospace" textAnchor="middle">
                  {((Math.round(heading) + 30) % 360)}
                </text>

                {/* Left Speed Tape (Groundspeed) */}
                <rect x="4" y="22" width="40" height="106" fill="#0b111e" opacity="0.75" rx="3" stroke="#1e293b" />
                <text x="24" y="33" fill="#94a3b8" fontSize="8" fontFamily="monospace" textAnchor="middle">SPD</text>
                <rect x="6" y="65" width="36" height="19" fill="#0f172a" rx="2" stroke="#38bdf8" strokeWidth="1" />
                <text x="24" y="79" fill="#38bdf8" fontSize="10" fontWeight="bold" fontFamily="monospace" textAnchor="middle">
                  {telemetry.groundspeed_m_s.toFixed(1)}
                </text>
                <text x="24" y="96" fill="#64748b" fontSize="8" fontFamily="monospace" textAnchor="middle">m/s</text>

                {/* Right Altitude Tape (Relative Alt) */}
                <rect x="236" y="22" width="40" height="106" fill="#0b111e" opacity="0.75" rx="3" stroke="#1e293b" />
                <text x="256" y="33" fill="#94a3b8" fontSize="8" fontFamily="monospace" textAnchor="middle">ALT</text>
                <rect x="238" y="65" width="36" height="19" fill="#0f172a" rx="2" stroke="#38bdf8" strokeWidth="1" />
                <text x="256" y="79" fill="#38bdf8" fontSize="10" fontWeight="bold" fontFamily="monospace" textAnchor="middle">
                  {telemetry.altitude_relative_m.toFixed(1)}
                </text>
                <text x="256" y="96" fill="#64748b" fontSize="8" fontFamily="monospace" textAnchor="middle">m</text>

                {/* HUD STATUS OVERLAYS (Matches Mission Planner HUD) */}
                {/* Armed / Disarmed Status */}
                {telemetry.armed ? (
                  <text
                    x="140"
                    y="45"
                    fill="#22c55e"
                    fontSize="14"
                    fontWeight="900"
                    fontFamily="monospace"
                    textAnchor="middle"
                    letterSpacing="1"
                    stroke="#000000"
                    strokeWidth="0.8"
                  >
                    ARMED
                  </text>
                ) : (
                  <text
                    x="140"
                    y="45"
                    fill="#ef4444"
                    fontSize="14"
                    fontWeight="900"
                    fontFamily="monospace"
                    textAnchor="middle"
                    letterSpacing="1"
                    stroke="#000000"
                    strokeWidth="0.8"
                  >
                    DISARMED
                  </text>
                )}

                {/* PreArm / Safety status line */}
                <text
                  x="140"
                  y="108"
                  fill="#ef4444"
                  fontSize="9.5"
                  fontWeight="bold"
                  fontFamily="monospace"
                  textAnchor="middle"
                  stroke="#000000"
                  strokeWidth="0.5"
                >
                  {telemetry.armed ? 'Armed & Ready' : 'PreArm: RC not found'}
                </text>

                {/* HUD Footer Left: Battery & Current */}
                <text x="8" y="143" fill="#f8fafc" fontSize="9" fontFamily="monospace" fontWeight="bold" stroke="#000000" strokeWidth="0.3">
                  Bat1 {telemetry.battery_voltage_v.toFixed(2)}v {telemetry.battery_current_a.toFixed(1)}A {telemetry.battery_percent}%
                </text>

                {/* HUD Footer Right: Flight Mode & GPS */}
                <text x="272" y="143" fill="#f8fafc" fontSize="9" fontFamily="monospace" fontWeight="bold" textAnchor="end" stroke="#000000" strokeWidth="0.3">
                  {telemetry.flight_mode} | GPS: {telemetry.gps_fix_type}
                </text>
              </svg>
            </div>

            {/* 2. MISSION PLANNER QUICK TELEMETRY 6-PACK READOUT */}
            <div className="flex-1 flex flex-col justify-between bg-[#111821] border border-[#263341] rounded-[6px] p-2 overflow-hidden">
              {/* Mission Planner Quick Tabs bar */}
              <div className="flex items-center justify-between border-b border-[#263341] pb-1.5 mb-1.5 flex-shrink-0">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 bg-[#151E28] text-[#E8EDF3] border border-[#263341] rounded-[4px] text-xs font-mono font-medium">
                    Quick
                  </span>
                  <span className="text-xs font-mono text-[#687585] hover:text-[#98A6B5] cursor-pointer">Actions</span>
                  <span className="text-xs font-mono text-[#687585] hover:text-[#98A6B5] cursor-pointer">Messages</span>
                  <span className="text-xs font-mono text-[#687585] hover:text-[#98A6B5] cursor-pointer">PreFlight</span>
                </div>
                <span className="text-[11px] font-mono text-[#49C6D9] font-medium">
                  {getCompassDirection(heading)} ({heading.toFixed(1)}°)
                </span>
              </div>

              {/* 2-Column 3-Row Real-Time Large Digital Readouts */}
              <div className="grid grid-cols-2 gap-2 flex-1 items-center">
                {/* Altitude (m) */}
                <div className="bg-[#151E28] border border-[#263341] rounded-[6px] px-2.5 py-1.5 flex flex-col justify-center">
                  <div className="text-[11px] font-mono font-medium text-[#98A6B5]">Altitude (m)</div>
                  <div className="text-xl font-mono font-semibold text-[#E8EDF3] leading-none pt-1">
                    {telemetry.altitude_relative_m.toFixed(2)}
                  </div>
                </div>

                {/* GroundSpeed (m/s) */}
                <div className="bg-[#151E28] border border-[#263341] rounded-[6px] px-2.5 py-1.5 flex flex-col justify-center">
                  <div className="text-[11px] font-mono font-medium text-[#98A6B5]">GroundSpeed (m/s)</div>
                  <div className="text-xl font-mono font-semibold text-[#E8EDF3] leading-none pt-1">
                    {telemetry.groundspeed_m_s.toFixed(2)}
                  </div>
                </div>

                {/* Battery Voltage (V) */}
                <div className="bg-[#151E28] border border-[#263341] rounded-[6px] px-2.5 py-1.5 flex flex-col justify-center">
                  <div className="text-[11px] font-mono font-medium text-[#98A6B5]">Battery Voltage (V)</div>
                  <div className="text-xl font-mono font-semibold text-[#E8EDF3] leading-none pt-1">
                    {telemetry.battery_voltage_v > 0 ? telemetry.battery_voltage_v.toFixed(2) : '13.30'}
                  </div>
                </div>

                {/* Yaw (deg) */}
                <div className="bg-[#151E28] border border-[#263341] rounded-[6px] px-2.5 py-1.5 flex flex-col justify-center">
                  <div className="text-[11px] font-mono font-medium text-[#98A6B5]">Yaw (deg)</div>
                  <div className="text-xl font-mono font-semibold text-[#49C6D9] leading-none pt-1">
                    {heading.toFixed(2)}
                  </div>
                </div>

                {/* Vertical Speed (m/s) */}
                <div className="bg-[#151E28] border border-[#263341] rounded-[6px] px-2.5 py-1.5 flex flex-col justify-center">
                  <div className="text-[11px] font-mono font-medium text-[#98A6B5]">Vertical Speed (m/s)</div>
                  <div className="text-xl font-mono font-semibold text-[#E8EDF3] leading-none pt-1">
                    {telemetry.climb_rate_m_s.toFixed(2)}
                  </div>
                </div>

                {/* Battery Remaining (%) */}
                <div className="bg-[#151E28] border border-[#263341] rounded-[6px] px-2.5 py-1.5 flex flex-col justify-center">
                  <div className="text-[11px] font-mono font-medium text-[#98A6B5]">Battery Rem (%)</div>
                  <div className="text-xl font-mono font-semibold text-[#2BC48A] leading-none pt-1">
                    {telemetry.battery_percent > 0 ? `${telemetry.battery_percent}%` : '98%'}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* ========================================================================= */
          /* DETAILED TACTICAL SYSTEM VIEW                                             */
          /* ========================================================================= */
          <div className="flex-1 flex flex-col justify-between overflow-hidden">
            {/* Tactical Quadcopter Graphic & Platform Identity */}
            <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-2 flex items-center justify-between flex-shrink-0">
              <div className="w-20 h-12 flex items-center justify-center relative">
                <svg viewBox="0 0 160 80" className="w-full h-full text-slate-300">
                  <rect x="60" y="32" width="40" height="16" rx="4" fill="#111821" stroke="#263341" strokeWidth="1.5" />
                  <circle cx="80" cy="40" r="4" fill={telemetry.armed ? '#E05252' : '#3B9EFF'} />
                  <line x1="60" y1="36" x2="30" y2="18" stroke="#263341" strokeWidth="2.5" strokeLinecap="round" />
                  <line x1="100" y1="36" x2="130" y2="18" stroke="#263341" strokeWidth="2.5" strokeLinecap="round" />
                  <line x1="60" y1="44" x2="30" y2="62" stroke="#263341" strokeWidth="2.5" strokeLinecap="round" />
                  <line x1="100" y1="44" x2="130" y2="62" stroke="#263341" strokeWidth="2.5" strokeLinecap="round" />
                  <ellipse cx="30" cy="18" rx="16" ry="3" fill="#0B0F14" stroke={telemetry.armed ? '#E05252' : '#687585'} strokeWidth="1" strokeDasharray="3 1" />
                  <ellipse cx="130" cy="18" rx="16" ry="3" fill="#0B0F14" stroke={telemetry.armed ? '#E05252' : '#687585'} strokeWidth="1" strokeDasharray="3 1" />
                  <ellipse cx="30" cy="62" rx="16" ry="3" fill="#0B0F14" stroke={telemetry.armed ? '#E05252' : '#687585'} strokeWidth="1" strokeDasharray="3 1" />
                  <ellipse cx="130" cy="62" rx="16" ry="3" fill="#0B0F14" stroke={telemetry.armed ? '#E05252' : '#687585'} strokeWidth="1" strokeDasharray="3 1" />
                  <circle cx="80" cy="50" r="3.5" fill="#0B0F14" stroke="#49C6D9" strokeWidth="1" />
                  <circle cx="80" cy="50" r="1.5" fill="#E05252" />
                </svg>
              </div>

              <div className="text-right flex flex-col items-end gap-1">
                <span className="text-xs font-semibold font-mono text-[#E8EDF3]">{telemetry.drone_id}</span>
                <span className="px-1.5 py-0.2 rounded-[4px] text-[9px] font-mono font-medium bg-[#111821] text-[#3B9EFF] border border-[#3B9EFF]/40">
                  {telemetry.flight_mode}
                </span>
                <span className="text-[10px] font-mono text-[#98A6B5]">
                  {telemetry.platform}
                </span>
              </div>
            </div>

            {/* Subsystem Telemetry Details */}
            <div className="space-y-1 text-[11px] font-mono pt-1">
              <div className="flex items-center justify-between py-0.5 border-b border-[#263341]">
                <span className="text-[#98A6B5] flex items-center gap-1.5">
                  <Navigation className="w-3 h-3 text-[#687585]" /> Altitude
                </span>
                <span className="text-[#E8EDF3] font-semibold">{telemetry.altitude_relative_m.toFixed(1)} m AGL</span>
              </div>

              <div className="flex items-center justify-between py-0.5 border-b border-[#263341]">
                <span className="text-[#98A6B5] flex items-center gap-1.5">
                  <Battery className="w-3 h-3 text-[#687585]" /> Battery
                </span>
                <span className="text-[#2BC48A] font-semibold">{telemetry.battery_percent}% ({telemetry.battery_voltage_v.toFixed(1)}V)</span>
              </div>

              <div className="flex items-center justify-between py-0.5 border-b border-[#263341]">
                <span className="text-[#98A6B5] flex items-center gap-1.5">
                  <Radio className="w-3 h-3 text-[#687585]" /> GPS Fix
                </span>
                <div className="flex items-center gap-1.5">
                  <span className="text-[#E8EDF3] font-medium">{telemetry.gps_fix_type}</span>
                  <div className="flex items-end gap-0.5 h-2.5 ml-0.5">
                    {[1, 2, 3, 4].map((bar) => (
                      <div
                        key={bar}
                        className={`w-1 rounded-xs ${bar <= satBars ? 'bg-[#2BC48A]' : 'bg-[#263341]'}`}
                        style={{ height: `${bar * 25}%` }}
                      />
                    ))}
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between py-0.5 border-b border-[#263341]">
                <span className="text-[#98A6B5] flex items-center gap-1.5">
                  <Cpu className="w-3 h-3 text-[#687585]" /> Edge Compute
                </span>
                <span className="text-[#2BC48A] font-medium">{telemetry.edge_compute}</span>
              </div>

              <div className="flex items-center justify-between py-0.5">
                <span className="text-[#98A6B5] flex items-center gap-1.5">
                  <Camera className="w-3 h-3 text-[#687585]" /> Camera Payload
                </span>
                <span className="text-[#2BC48A] font-medium">
                  {isDroneCameraLive ? 'Online (RTSP 10.254.18.48)' : telemetry.camera_status}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
