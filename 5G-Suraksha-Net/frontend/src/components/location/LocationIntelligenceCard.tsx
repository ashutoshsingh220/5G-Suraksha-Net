import React from 'react';
import { MapPin, ExternalLink, Navigation, Maximize2, AlertTriangle } from 'lucide-react';
import { useIncidents } from '../../context/IncidentContext';
import { useTelemetry } from '../../context/TelemetryContext';
import { CommandCentreMap, computeHaversineDistanceKm } from '../map/CommandCentreMap';

interface LocationIntelligenceCardProps {
  onExpandMap?: () => void;
}

export const LocationIntelligenceCard: React.FC<LocationIntelligenceCardProps> = ({ onExpandMap }) => {
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
  const hasIncidentCoords = Boolean(
    activeIncident?.location?.latitude !== null &&
      activeIncident?.location?.latitude !== undefined &&
      activeIncident?.location?.longitude !== null &&
      activeIncident?.location?.longitude !== undefined &&
      Math.abs(activeIncident.location.latitude) > 0.001 &&
      Math.abs(activeIncident.location.longitude) > 0.001
  );

  const locationName =
    activeIncident?.location?.name ||
    (activeIncident ? `Camera Zone: ${activeIncident.zone || activeIncident.camera_id}` : 'IMC Yashobhoomi Sector-25 Perimeter');

  const incidentCoordsFormatted = hasIncidentCoords
    ? `${activeIncident!.location!.latitude!.toFixed(5)}° N, ${activeIncident!.location!.longitude!.toFixed(5)}° E`
    : '28.55290° N, 77.06010° E';

  const source = activeIncident?.location?.source?.toUpperCase() || 'GEOCODED DEMO';

  // Calculate straight-line distance if both drone and incident locations exist
  const straightLineDistanceKm =
    hasValidDroneGps && hasIncidentCoords
      ? computeHaversineDistanceKm(
          telemetry.latitude as number,
          telemetry.longitude as number,
          activeIncident!.location!.latitude as number,
          activeIncident!.location!.longitude as number
        )
      : null;

  return (
    <div className="suraksha-panel flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="suraksha-panel-header px-3.5 py-2 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2 text-[#E8EDF3]">
          <MapPin className="w-4 h-4 text-[#3B9EFF]" />
          <h2 className="text-xs font-semibold font-sans tracking-wider uppercase">
            LOCATION INTELLIGENCE
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-[#98A6B5] bg-[#151E28] border border-[#263341] px-2 py-0.5 rounded-[4px]">
            {source}
          </span>
          {onExpandMap && (
            <button
              onClick={onExpandMap}
              className="p-1 hover:bg-[#263341] text-[#98A6B5] hover:text-[#E8EDF3] rounded-[4px] transition-colors"
              title="Expand to Full Tactical Map"
            >
              <Maximize2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      <div className="p-3 flex-1 flex flex-col gap-2.5 overflow-hidden justify-between">
        {/* Interactive Tactical Mini Map */}
        <div className="w-full h-[155px] flex-shrink-0 rounded-[6px] border border-[#263341] overflow-hidden relative">
          <CommandCentreMap compact={true} height="100%" showControls={true} onExpand={onExpandMap} />
        </div>

        {/* Location & Sensor Information Strip */}
        <div className="space-y-2 text-xs">
          {/* 1. Incident Target Coordinates */}
          <div className="bg-[#151E28] border border-[#263341] p-2 rounded-[6px] flex items-center justify-between">
            <div className="flex items-center gap-2 truncate">
              <AlertTriangle className="w-3.5 h-3.5 text-[#E05252] flex-shrink-0" />
              <div className="truncate">
                <span className="text-[#98A6B5] font-sans font-medium">Target: </span>
                <span className="text-[#E8EDF3] font-mono font-semibold">{incidentCoordsFormatted}</span>
              </div>
            </div>
            <a
              href={`https://www.google.com/maps?q=${incidentCoordsFormatted.replace(/[^\d.,-]/g, '')}`}
              target="_blank"
              rel="noreferrer"
              className="text-[#3B9EFF] hover:text-[#49C6D9] p-1 flex-shrink-0"
              title="Open in Google Maps"
            >
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>

          {/* 2. Drone GPS Real Status */}
          <div className="bg-[#151E28] border border-[#263341] p-2 rounded-[6px] flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Navigation className="w-3.5 h-3.5 text-[#49C6D9] flex-shrink-0" />
              <div>
                <span className="text-[#98A6B5] font-sans font-medium">Drone: </span>
                {hasValidDroneGps ? (
                  <span className="text-[#2BC48A] font-mono font-semibold">
                    {telemetry.latitude?.toFixed(4)}, {telemetry.longitude?.toFixed(4)} ({telemetry.altitude_relative_m.toFixed(1)}m)
                  </span>
                ) : telemetry.connected ? (
                  <span className="text-[#E7A83B] font-mono font-semibold">ONLINE &bull; GPS NO FIX</span>
                ) : (
                  <span className="text-[#687585] font-mono font-medium">OFFLINE</span>
                )}
              </div>
            </div>
            {straightLineDistanceKm !== null ? (
              <span className="text-[#49C6D9] font-mono font-semibold text-xs bg-[#111821] border border-[#263341] px-2 py-0.5 rounded-[4px]" title="Straight-line air distance">
                {straightLineDistanceKm.toFixed(2)} km
              </span>
            ) : null}
          </div>

          {/* Name & Sector */}
          <div className="text-xs text-[#98A6B5] font-sans flex items-center justify-between pt-0.5">
            <span className="truncate font-medium text-[#E8EDF3]" title={locationName}>
              {locationName}
            </span>
            <span className="text-xs text-[#687585] font-mono flex-shrink-0 ml-2">
              {hasIncidentCoords ? 'Direct Geospatial Fix' : 'Sec-25 Yashobhoomi'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
