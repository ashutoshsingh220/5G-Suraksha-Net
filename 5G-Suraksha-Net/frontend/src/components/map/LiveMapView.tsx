import React from 'react';
import {
  ArrowLeft,
  MapPin,
  Navigation,
  AlertTriangle,
  Shield,
  Building2,
  PhoneCall,
} from 'lucide-react';
import { CommandCentreMap, computeHaversineDistanceKm } from './CommandCentreMap';
import { useIncidents } from '../../context/IncidentContext';
import { useTelemetry } from '../../context/TelemetryContext';
import {
  defaultNearbyPolice,
  defaultNearbyHospitals,
} from '../../data/emergencyResources';

interface LiveMapViewProps {
  onBackToCommand?: () => void;
}

export const LiveMapView: React.FC<LiveMapViewProps> = ({ onBackToCommand }) => {
  const { activeIncident } = useIncidents();
  const { telemetry } = useTelemetry();

  // Validate Drone GPS Fix
  const hasValidDroneGps = Boolean(
    telemetry.connected &&
      telemetry.latitude !== null &&
      telemetry.longitude !== null &&
      Math.abs(telemetry.latitude) > 0.001 &&
      Math.abs(telemetry.longitude) > 0.001 &&
      telemetry.gps_fix_type !== 'No GPS' &&
      telemetry.gps_fix_type !== 'No Fix'
  );

  // Validate Incident Location
  const hasIncidentLocation = Boolean(
    activeIncident?.location?.latitude !== null &&
      activeIncident?.location?.latitude !== undefined &&
      activeIncident?.location?.longitude !== null &&
      activeIncident?.location?.longitude !== undefined &&
      Math.abs(activeIncident.location.latitude) > 0.001 &&
      Math.abs(activeIncident.location.longitude) > 0.001
  );

  // Compute straight-line Euclidean distance between drone and incident
  const droneToIncidentDistanceKm =
    hasValidDroneGps && hasIncidentLocation
      ? computeHaversineDistanceKm(
          telemetry.latitude as number,
          telemetry.longitude as number,
          activeIncident!.location!.latitude as number,
          activeIncident!.location!.longitude as number
        )
      : null;

  return (
    <div className="flex-1 flex flex-col p-3 gap-3 max-w-[1920px] mx-auto w-full min-h-[calc(100vh-100px)]">
      {/* Top Banner & Quick Controls */}
      <div className="flex items-center justify-between bg-[#151E28] border border-[#263341] px-3.5 py-2 rounded-[8px]">
        <div className="flex items-center gap-3">
          {onBackToCommand && (
            <button
              onClick={onBackToCommand}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-[6px] bg-[#111821] border border-[#263341] text-xs font-sans font-medium text-[#98A6B5] hover:text-[#E8EDF3] hover:bg-[#263341] transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>Back to Command Centre</span>
            </button>
          )}
          <div className="flex items-center gap-2">
            <MapPin className="w-4 h-4 text-[#3B9EFF]" />
            <h1 className="text-xs font-semibold font-sans tracking-wider text-[#E8EDF3] uppercase">
              GEOSPATIAL SITUATIONAL COMMAND MAP &bull; REAL-TIME TRACKING
            </h1>
          </div>
        </div>

        {/* Global Coordinates & Status Summary */}
        <div className="flex items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-1.5 text-[#98A6B5]">
            <span className="text-[#687585]">SECTOR:</span>
            <span className="text-[#E8EDF3] font-medium font-sans">Yashobhoomi Dwarka Sec-25</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[#687585]">DRONE:</span>
            {hasValidDroneGps ? (
              <span className="text-[#2BC48A] font-semibold">
                {telemetry.latitude?.toFixed(4)}, {telemetry.longitude?.toFixed(4)}
              </span>
            ) : telemetry.connected ? (
              <span className="text-[#E7A83B] font-semibold">ONLINE (GPS NO FIX)</span>
            ) : (
              <span className="text-[#687585]">OFFLINE</span>
            )}
          </div>
        </div>
      </div>

      {/* Main Layout: Large Tactical Map + Situational Sidebar */}
      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-3 min-h-[640px]">
        {/* Left / Center Map (8 cols on large screens, 9 on xl) */}
        <div className="lg:col-span-8 xl:col-span-9 flex flex-col suraksha-panel overflow-hidden border border-[#263341] rounded-[8px]">
          <CommandCentreMap height="calc(100vh - 165px)" compact={false} showControls={true} />
        </div>

        {/* Right Tactical Sidebar (4 cols on lg, 3 on xl) */}
        <div className="lg:col-span-4 xl:col-span-3 flex flex-col gap-3 overflow-y-auto max-h-[calc(100vh-165px)] pr-0.5">
          {/* 1. Active Incident Position Card */}
          <div className="suraksha-panel flex flex-col p-3 rounded-[8px] border border-[#263341]">
            <div className="flex items-center justify-between pb-2 border-b border-[#263341] mb-2">
              <div className="flex items-center gap-1.5 text-[#E05252]">
                <AlertTriangle className="w-4 h-4" />
                <span className="text-xs font-semibold font-sans uppercase tracking-wider">
                  INCIDENT TARGET
                </span>
              </div>
              <span className="text-[9px] font-mono text-[#98A6B5] bg-[#151E28] border border-[#263341] px-1.5 py-0.5 rounded-[4px]">
                SOURCE: {activeIncident?.location?.source?.toUpperCase() || 'GEOCODED DEMO'}
              </span>
            </div>

            {activeIncident ? (
              <div className="space-y-2 text-xs font-mono">
                <div>
                  <div className="text-[10px] text-[#98A6B5] font-sans">Type & Severity</div>
                  <div className="text-[#E8EDF3] font-semibold capitalize font-sans">
                    {activeIncident.incident_type.replace(/_/g, ' ')} &bull;{' '}
                    <span
                      className={
                        activeIncident.severity === 'critical'
                          ? 'text-[#E05252] font-semibold'
                          : activeIncident.severity === 'high'
                          ? 'text-[#E7A83B] font-semibold'
                          : 'text-[#3B9EFF] font-semibold'
                      }
                    >
                      {activeIncident.severity.toUpperCase()}
                    </span>
                  </div>
                </div>

                <div>
                  <div className="text-[10px] text-[#98A6B5] font-sans">Assigned Location</div>
                  <div className="text-[#E8EDF3] text-[11px] leading-snug font-sans">
                    {activeIncident.location?.name || `Zone: ${activeIncident.zone || 'Main Perimeter'}`}
                  </div>
                </div>

                <div className="bg-[#151E28] border border-[#263341] p-2 rounded-[6px]">
                  <div className="text-[10px] text-[#98A6B5] font-sans">Geodetic Coordinates</div>
                  <div className="text-[#E8EDF3] font-semibold text-xs mt-0.5 font-mono">
                    {hasIncidentLocation
                      ? `${activeIncident.location!.latitude!.toFixed(5)}° N, ${activeIncident.location!.longitude!.toFixed(5)}° E`
                      : 'Coordinates Pending (Sector Grid)'}
                  </div>
                </div>

                {droneToIncidentDistanceKm !== null && (
                  <div className="bg-[#151E28] border border-[#3B9EFF]/40 p-2 rounded-[6px]">
                    <div className="text-[10px] text-[#3B9EFF] font-semibold font-sans">
                      Drone-to-Incident Separation
                    </div>
                    <div className="text-[#E8EDF3] font-semibold text-sm mt-0.5 font-mono">
                      {droneToIncidentDistanceKm.toFixed(2)} km
                    </div>
                    <div className="text-[9px] text-[#98A6B5] mt-0.5 font-sans">
                      Straight-line Euclidean air vector (Not road navigation ETA)
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-4 text-[#687585] text-xs font-sans">
                No active incident selected. Map showing Yashobhoomi reference perimeter.
              </div>
            )}
          </div>

          {/* 2. Drone Telemetry Status Card */}
          <div className="suraksha-panel flex flex-col p-3 rounded-[8px] border border-[#263341]">
            <div className="flex items-center justify-between pb-2 border-b border-[#263341] mb-2">
              <div className="flex items-center gap-1.5 text-[#3B9EFF]">
                <Navigation className="w-4 h-4" />
                <span className="text-xs font-semibold font-sans uppercase tracking-wider">
                  DRONE TELEMETRY GPS
                </span>
              </div>
              <span
                className={`text-[9px] font-mono px-1.5 py-0.5 rounded-[4px] border ${
                  hasValidDroneGps
                    ? 'bg-[#2BC48A]/20 border-[#2BC48A]/40 text-[#2BC48A]'
                    : telemetry.connected
                    ? 'bg-[#E7A83B]/20 border-[#E7A83B]/40 text-[#E7A83B]'
                    : 'bg-[#151E28] border-[#263341] text-[#687585]'
                }`}
              >
                {hasValidDroneGps
                  ? `${telemetry.gps_fix_type} (${telemetry.satellites_visible} SATS)`
                  : telemetry.connected
                  ? 'GPS NO FIX'
                  : 'OFFLINE'}
              </span>
            </div>

            <div className="space-y-1.5 text-xs font-mono">
              <div className="flex justify-between py-0.5 border-b border-[#263341]">
                <span className="text-[#98A6B5] font-sans">Altitude AGL</span>
                <span className="text-[#E8EDF3] font-medium">
                  {telemetry.altitude_relative_m.toFixed(1)} m
                </span>
              </div>
              <div className="flex justify-between py-0.5 border-b border-[#263341]">
                <span className="text-[#98A6B5] font-sans">Groundspeed</span>
                <span className="text-[#E8EDF3] font-medium">
                  {telemetry.groundspeed_m_s.toFixed(1)} m/s
                </span>
              </div>
              <div className="flex justify-between py-0.5 border-b border-[#263341]">
                <span className="text-[#98A6B5] font-sans">Heading</span>
                <span className="text-[#E8EDF3] font-medium">
                  {telemetry.heading_deg.toFixed(0)}°
                </span>
              </div>
              <div className="flex justify-between py-0.5 border-b border-[#263341]">
                <span className="text-[#98A6B5] font-sans">Battery Status</span>
                <span className="text-[#E8EDF3] font-medium">
                  {telemetry.battery_percent}% ({telemetry.battery_voltage_v.toFixed(1)}V)
                </span>
              </div>
              <div className="flex justify-between py-0.5">
                <span className="text-[#98A6B5] font-sans">Vehicle Mode</span>
                <span className="text-[#49C6D9] font-medium">{telemetry.flight_mode}</span>
              </div>
            </div>
          </div>

          {/* 3. Nearby Emergency Facilities */}
          <div className="suraksha-panel flex flex-col p-3 rounded-[8px] border border-[#263341] flex-1">
            <div className="flex items-center gap-1.5 text-[#E8EDF3] pb-2 border-b border-[#263341] mb-2">
              <Shield className="w-4 h-4 text-[#3B9EFF]" />
              <span className="text-xs font-semibold font-sans uppercase tracking-wider">
                NEARBY EMERGENCY ASSETS
              </span>
            </div>

            <div className="space-y-2.5 text-xs font-mono flex-1 overflow-y-auto pr-1">
              {/* Police Section */}
              <div>
                <div className="text-[10px] font-semibold text-[#3B9EFF] uppercase mb-1 flex items-center gap-1 font-sans">
                  <Shield className="w-3 h-3" />
                  <span>Police Stations</span>
                </div>
                <div className="space-y-1.5">
                  {defaultNearbyPolice.map((p) => (
                    <div
                      key={p.name}
                      className="bg-[#151E28] border border-[#263341] p-2 rounded-[6px] hover:border-[#3B9EFF]/40 transition-colors"
                    >
                      <div className="flex justify-between items-start">
                        <span className="text-[#E8EDF3] font-semibold text-[11px] leading-tight font-sans">
                          {p.name}
                        </span>
                        <span className="text-[#3B9EFF] font-semibold text-[10px] bg-[#111821] px-1 py-0.5 rounded-[4px] border border-[#263341] flex-shrink-0 ml-1 font-mono">
                          {p.distance_km} km
                        </span>
                      </div>
                      <div className="text-[#98A6B5] text-[10px] mt-1 flex items-center gap-1 font-mono">
                        <PhoneCall className="w-2.5 h-2.5 text-[#687585]" />
                        <span>{p.phone}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Hospital Section */}
              <div className="pt-1">
                <div className="text-[10px] font-semibold text-[#E05252] uppercase mb-1 flex items-center gap-1 font-sans">
                  <Building2 className="w-3 h-3" />
                  <span>Hospitals & Trauma Centres</span>
                </div>
                <div className="space-y-1.5">
                  {defaultNearbyHospitals.map((h) => (
                    <div
                      key={h.name}
                      className="bg-[#151E28] border border-[#263341] p-2 rounded-[6px] hover:border-[#E05252]/40 transition-colors"
                    >
                      <div className="flex justify-between items-start">
                        <span className="text-[#E8EDF3] font-semibold text-[11px] leading-tight font-sans">
                          {h.name}
                        </span>
                        <span className="text-[#E05252] font-semibold text-[10px] bg-[#151014] px-1 py-0.5 rounded-[4px] border border-[#E05252]/40 flex-shrink-0 ml-1 font-mono">
                          {h.distance_km} km
                        </span>
                      </div>
                      <div className="text-[#98A6B5] text-[10px] mt-1 flex items-center gap-1 font-mono">
                        <PhoneCall className="w-2.5 h-2.5 text-[#687585]" />
                        <span>{h.phone}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="text-[9px] text-[#687585] pt-1 border-t border-[#263341] leading-normal italic font-sans">
                * Note: Distances shown are straight-line Euclidean distance from Yashobhoomi Sector-25. Do not represent road route travel duration.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
