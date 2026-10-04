import React, { useEffect, useRef, useState } from 'react';
import {
  Plus,
  Minus,
  Navigation,
  AlertTriangle,
  Maximize2,
} from 'lucide-react';
import { useIncidents } from '../../context/IncidentContext';
import { useTelemetry } from '../../context/TelemetryContext';
import { useMapFocus } from '../../context/MapFocusContext';
import {
  defaultNearbyPolice,
  defaultNearbyHospitals,
  DEFAULT_COMMAND_CENTER_LOCATION,
} from '../../data/emergencyResources';
import { loadGoogleMaps } from '../../utils/googleMapsLoader';

interface CommandCentreMapProps {
  height?: string;
  compact?: boolean;
  showControls?: boolean;
  onExpand?: () => void;
}

// Deterministic Haversine distance calculation (km)
export function computeHaversineDistanceKm(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const R = 6371;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lon2 - lon1) * Math.PI) / 180;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return Math.round(R * c * 100) / 100;
}

// Clean bright Google Maps style (clear streets, vibrant landmarks, excellent legibility)
const brightMapStyles: any[] = [];

export const CommandCentreMap: React.FC<CommandCentreMapProps> = ({
  height = '100%',
  compact = false,
  showControls = true,
  onExpand,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const gmapRef = useRef<any>(null);
  const googleApiRef = useRef<any>(null);

  const { activeIncident } = useIncidents();
  const { telemetry } = useTelemetry();
  const { focusTarget } = useMapFocus();

  const [mapType, setMapType] = useState<'roadmap' | 'hybrid'>('roadmap');
  const [googleMapsLoaded, setGoogleMapsLoaded] = useState<boolean>(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [layers, setLayers] = useState({
    drone: true,
    incident: true,
    police: true,
    hospital: true,
  });

  // Marker references
  const incidentMarkerRef = useRef<any>(null);
  const droneMarkerRef = useRef<any>(null);
  const policeMarkersRef = useRef<any[]>([]);
  const hospitalMarkersRef = useRef<any[]>([]);
  const polylineRef = useRef<any>(null);
  const infoWindowRef = useRef<any>(null);

  // Validate drone GPS
  const hasValidDroneGps = Boolean(
    telemetry.connected &&
      telemetry.latitude !== null &&
      telemetry.longitude !== null &&
      Math.abs(telemetry.latitude) > 0.001 &&
      Math.abs(telemetry.longitude) > 0.001 &&
      telemetry.gps_fix_type !== 'No GPS' &&
      telemetry.gps_fix_type !== 'No Fix'
  );

  // Validate incident location
  const hasIncidentLocation = Boolean(
    activeIncident?.location?.latitude !== null &&
      activeIncident?.location?.latitude !== undefined &&
      activeIncident?.location?.longitude !== null &&
      activeIncident?.location?.longitude !== undefined &&
      Math.abs(activeIncident.location.latitude) > 0.001 &&
      Math.abs(activeIncident.location.longitude) > 0.001
  );

  // Initialize Google Maps with bright, standard styling
  useEffect(() => {
    let isMounted = true;

    loadGoogleMaps()
      .then((google) => {
        if (!isMounted || !containerRef.current || gmapRef.current) return;
        googleApiRef.current = google;

        const initLat = hasIncidentLocation
          ? (activeIncident!.location!.latitude as number)
          : hasValidDroneGps
          ? (telemetry.latitude as number)
          : DEFAULT_COMMAND_CENTER_LOCATION.latitude;

        const initLng = hasIncidentLocation
          ? (activeIncident!.location!.longitude as number)
          : hasValidDroneGps
          ? (telemetry.longitude as number)
          : DEFAULT_COMMAND_CENTER_LOCATION.longitude;

        const map = new google.maps.Map(containerRef.current, {
          center: { lat: initLat, lng: initLng },
          zoom: compact ? 15 : 15,
          mapTypeId: mapType === 'hybrid' ? google.maps.MapTypeId.HYBRID : google.maps.MapTypeId.ROADMAP,
          styles: brightMapStyles, // Bright, authentic Google Maps road styling
          disableDefaultUI: true,
          zoomControl: false,
          mapTypeControl: false,
          streetViewControl: false,
          fullscreenControl: false,
          gestureHandling: 'greedy',
        });

        infoWindowRef.current = new google.maps.InfoWindow();
        gmapRef.current = map;
        setGoogleMapsLoaded(true);
        setLoadError(null);
      })
      .catch((err) => {
        if (isMounted) {
          console.warn('Google Maps JS API load failed; falling back:', err);
          setLoadError('Google Maps API connecting...');
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Update Map Type (Roadmap vs Hybrid Satellite)
  useEffect(() => {
    if (!gmapRef.current || !googleApiRef.current) return;
    const google = googleApiRef.current;
    if (mapType === 'hybrid') {
      gmapRef.current.setMapTypeId(google.maps.MapTypeId.HYBRID);
      gmapRef.current.setOptions({ styles: null });
    } else {
      gmapRef.current.setMapTypeId(google.maps.MapTypeId.ROADMAP);
      gmapRef.current.setOptions({ styles: null });
    }
  }, [mapType]);

  // 1. Stable Incident Target Marker (NO re-drop jitter on telemetry ticks)
  useEffect(() => {
    if (!gmapRef.current || !googleApiRef.current) return;
    const google = googleApiRef.current;
    const map = gmapRef.current;

    if (incidentMarkerRef.current) {
      incidentMarkerRef.current.setMap(null);
      incidentMarkerRef.current = null;
    }

    if (layers.incident) {
      const incLat = hasIncidentLocation
        ? (activeIncident!.location!.latitude as number)
        : DEFAULT_COMMAND_CENTER_LOCATION.latitude;
      const incLng = hasIncidentLocation
        ? (activeIncident!.location!.longitude as number)
        : DEFAULT_COMMAND_CENTER_LOCATION.longitude;

      const incTitle = activeIncident
        ? `${activeIncident.incident_type.toUpperCase()} #${activeIncident.incident_id.slice(-4)}`
        : 'Yashobhoomi Command Base';

      // Solid bright red pin (stable, no animation: DROP)
      const incidentSvg = {
        path: 'M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 9.5c-1.38 0-2.5-1.12-2.5-2.5s1.12-2.5 2.5-2.5 2.5 1.12 2.5 2.5-1.12 2.5-2.5 2.5z',
        fillColor: '#EF4444',
        fillOpacity: 1.0,
        strokeColor: '#FFFFFF',
        strokeWeight: 2,
        scale: 1.8,
        anchor: new google.maps.Point(12, 22),
      };

      const marker = new google.maps.Marker({
        position: { lat: incLat, lng: incLng },
        map,
        title: incTitle,
        icon: incidentSvg,
        zIndex: 999,
      });

      marker.addListener('click', () => {
        const content = `
          <div style="font-family: sans-serif; font-size: 12px; color: #0F172A; padding: 6px; line-height: 1.4;">
            <div style="color: #DC2626; font-size: 13px; font-weight: bold; border-bottom: 1px solid #CBD5E1; padding-bottom: 3px; margin-bottom: 4px;">INCIDENT TARGET</div>
            <div><strong>Type:</strong> ${activeIncident?.incident_type?.toUpperCase() || 'MONITORED BASE'}</div>
            <div><strong>ID:</strong> #${activeIncident?.incident_id || 'DEMO'}</div>
            <div><strong>Location:</strong> ${activeIncident?.location?.name || 'Yashobhoomi Sector 25'}</div>
            <div style="font-family: monospace; color: #475569; font-size: 11px; margin-top: 2px;">${incLat.toFixed(5)}°N, ${incLng.toFixed(5)}°E</div>
          </div>
        `;
        infoWindowRef.current.setContent(content);
        infoWindowRef.current.open(map, marker);
      });

      incidentMarkerRef.current = marker;
    }
  }, [googleMapsLoaded, activeIncident?.incident_id, activeIncident?.location?.latitude, activeIncident?.location?.longitude, layers.incident]);

  // 2. Drone UAV Marker (Updates position smoothly without destroying and re-dropping)
  useEffect(() => {
    if (!gmapRef.current || !googleApiRef.current) return;
    const google = googleApiRef.current;
    const map = gmapRef.current;

    if (!layers.drone || !hasValidDroneGps) {
      if (droneMarkerRef.current) {
        droneMarkerRef.current.setMap(null);
        droneMarkerRef.current = null;
      }
      return;
    }

    const pos = { lat: telemetry.latitude as number, lng: telemetry.longitude as number };

    const droneSvg = {
      path: 'M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z',
      fillColor: '#0284C7',
      fillOpacity: 1.0,
      strokeColor: '#FFFFFF',
      strokeWeight: 2,
      scale: 1.6,
      rotation: telemetry.heading_deg || 0,
      anchor: new google.maps.Point(12, 12),
    };

    if (droneMarkerRef.current) {
      droneMarkerRef.current.setPosition(pos);
      droneMarkerRef.current.setIcon(droneSvg);
    } else {
      const marker = new google.maps.Marker({
        position: pos,
        map,
        title: `Drone UAV (Alt: ${(telemetry.altitude_relative_m ?? 0).toFixed(1)}m)`,
        icon: droneSvg,
        zIndex: 1000,
      });

      marker.addListener('click', () => {
        const content = `
          <div style="font-family: sans-serif; font-size: 12px; color: #0F172A; padding: 6px; line-height: 1.4;">
            <div style="color: #0284C7; font-size: 13px; font-weight: bold; border-bottom: 1px solid #CBD5E1; padding-bottom: 3px; margin-bottom: 4px;">DRONE TELEMETRY</div>
            <div><strong>Fix:</strong> ${telemetry.gps_fix_type} (${telemetry.satellites_visible} sats)</div>
            <div><strong>Altitude:</strong> ${(telemetry.altitude_relative_m ?? 0).toFixed(1)} m</div>
            <div><strong>Speed:</strong> ${(telemetry.groundspeed_m_s ?? 0).toFixed(1)} m/s</div>
            <div style="font-family: monospace; color: #475569; font-size: 11px; margin-top: 2px;">${(telemetry.latitude as number).toFixed(5)}°N, ${(telemetry.longitude as number).toFixed(5)}°E</div>
          </div>
        `;
        infoWindowRef.current.setContent(content);
        infoWindowRef.current.open(map, marker);
      });

      droneMarkerRef.current = marker;
    }
  }, [googleMapsLoaded, telemetry.latitude, telemetry.longitude, telemetry.heading_deg, telemetry.gps_fix_type, layers.drone, hasValidDroneGps]);

  // 3. Police & Hospital Pinpointed Markers (< 3.0 km)
  useEffect(() => {
    if (!gmapRef.current || !googleApiRef.current) return;
    const google = googleApiRef.current;
    const map = gmapRef.current;

    // Clear old police markers
    policeMarkersRef.current.forEach((m) => m.setMap(null));
    policeMarkersRef.current = [];

    if (layers.police) {
      // Vibrant blue shield pin for police
      const policeSvg = {
        path: 'M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-2 16l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z',
        fillColor: '#2563EB',
        fillOpacity: 1.0,
        strokeColor: '#FFFFFF',
        strokeWeight: 1.8,
        scale: 1.4,
        anchor: new google.maps.Point(12, 12),
      };

      defaultNearbyPolice.forEach((p) => {
        const marker = new google.maps.Marker({
          position: { lat: p.latitude, lng: p.longitude },
          map,
          title: `Police: ${p.name} (${p.distance_km} km)`,
          icon: policeSvg,
          zIndex: 500,
        });

        marker.addListener('click', () => {
          const content = `
            <div style="font-family: sans-serif; font-size: 12px; color: #0F172A; padding: 6px; line-height: 1.4;">
              <div style="color: #2563EB; font-size: 13px; font-weight: bold; border-bottom: 1px solid #CBD5E1; padding-bottom: 3px; margin-bottom: 4px;">POLICE FACILITY (< 3 KM)</div>
              <div><strong>${p.name}</strong></div>
              <div><strong>Distance:</strong> ${p.distance_km} km</div>
              <div><strong>Phone:</strong> ${p.phone}</div>
              <div style="color: #64748B; font-size: 11px;">${p.address}</div>
            </div>
          `;
          infoWindowRef.current.setContent(content);
          infoWindowRef.current.open(map, marker);
        });

        policeMarkersRef.current.push(marker);
      });
    }

    // Clear old hospital markers
    hospitalMarkersRef.current.forEach((m) => m.setMap(null));
    hospitalMarkersRef.current = [];

    if (layers.hospital) {
      // Vibrant medical red cross pin for hospitals
      const hospitalSvg = {
        path: 'M19 10.5h-4.5V6h-5v4.5H5v5h4.5V20h5v-4.5H19z',
        fillColor: '#DC2626',
        fillOpacity: 1.0,
        strokeColor: '#FFFFFF',
        strokeWeight: 1.8,
        scale: 1.4,
        anchor: new google.maps.Point(12, 12),
      };

      defaultNearbyHospitals.forEach((h) => {
        const marker = new google.maps.Marker({
          position: { lat: h.latitude, lng: h.longitude },
          map,
          title: `Hospital: ${h.name} (${h.distance_km} km)`,
          icon: hospitalSvg,
          zIndex: 500,
        });

        marker.addListener('click', () => {
          const content = `
            <div style="font-family: sans-serif; font-size: 12px; color: #0F172A; padding: 6px; line-height: 1.4;">
              <div style="color: #DC2626; font-size: 13px; font-weight: bold; border-bottom: 1px solid #CBD5E1; padding-bottom: 3px; margin-bottom: 4px;">HOSPITAL & TRAUMA (< 3 KM)</div>
              <div><strong>${h.name}</strong></div>
              <div><strong>Distance:</strong> ${h.distance_km} km</div>
              <div><strong>Phone:</strong> ${h.phone}</div>
              <div style="color: #64748B; font-size: 11px;">${h.address}</div>
            </div>
          `;
          infoWindowRef.current.setContent(content);
          infoWindowRef.current.open(map, marker);
        });

        hospitalMarkersRef.current.push(marker);
      });
    }
  }, [googleMapsLoaded, layers.police, layers.hospital]);

  // 4. Distance Vector Polyline
  useEffect(() => {
    if (!gmapRef.current || !googleApiRef.current) return;
    const google = googleApiRef.current;
    const map = gmapRef.current;

    if (polylineRef.current) {
      polylineRef.current.setMap(null);
      polylineRef.current = null;
    }

    if (hasValidDroneGps && hasIncidentLocation && layers.drone && layers.incident) {
      const poly = new google.maps.Polyline({
        path: [
          { lat: telemetry.latitude as number, lng: telemetry.longitude as number },
          { lat: activeIncident!.location!.latitude as number, lng: activeIncident!.location!.longitude as number },
        ],
        geodesic: true,
        strokeColor: '#0284C7',
        strokeOpacity: 0.9,
        strokeWeight: 2.5,
        map,
      });
      polylineRef.current = poly;
    }
  }, [googleMapsLoaded, telemetry.latitude, telemetry.longitude, activeIncident?.location?.latitude, activeIncident?.location?.longitude, layers.drone, layers.incident, hasValidDroneGps, hasIncidentLocation]);

  // Handle Focus Target from MapFocusContext
  useEffect(() => {
    if (!gmapRef.current || !focusTarget) return;
    gmapRef.current.panTo({ lat: focusTarget.latitude, lng: focusTarget.longitude });
    gmapRef.current.setZoom(16);
  }, [focusTarget]);

  // Map Controls
  const handleZoomIn = () => {
    if (gmapRef.current) {
      gmapRef.current.setZoom(gmapRef.current.getZoom() + 1);
    }
  };

  const handleZoomOut = () => {
    if (gmapRef.current) {
      gmapRef.current.setZoom(gmapRef.current.getZoom() - 1);
    }
  };

  const handleCenterIncident = () => {
    if (!gmapRef.current) return;
    const lat = hasIncidentLocation
      ? (activeIncident!.location!.latitude as number)
      : DEFAULT_COMMAND_CENTER_LOCATION.latitude;
    const lng = hasIncidentLocation
      ? (activeIncident!.location!.longitude as number)
      : DEFAULT_COMMAND_CENTER_LOCATION.longitude;
    gmapRef.current.panTo({ lat, lng });
    gmapRef.current.setZoom(16);
  };

  const handleCenterDrone = () => {
    if (!gmapRef.current || !hasValidDroneGps) return;
    gmapRef.current.panTo({ lat: telemetry.latitude as number, lng: telemetry.longitude as number });
    gmapRef.current.setZoom(16);
  };

  const handleFitAll = () => {
    if (!gmapRef.current || !googleApiRef.current) return;
    const google = googleApiRef.current;
    const bounds = new google.maps.LatLngBounds();

    if (hasIncidentLocation) {
      bounds.extend({
        lat: activeIncident!.location!.latitude as number,
        lng: activeIncident!.location!.longitude as number,
      });
    }
    if (hasValidDroneGps) {
      bounds.extend({
        lat: telemetry.latitude as number,
        lng: telemetry.longitude as number,
      });
    }
    defaultNearbyPolice.forEach((p) => bounds.extend({ lat: p.latitude, lng: p.longitude }));
    defaultNearbyHospitals.forEach((h) => bounds.extend({ lat: h.latitude, lng: h.longitude }));

    gmapRef.current.fitBounds(bounds);
  };

  return (
    <div
      style={{ height, width: '100%' }}
      className="relative flex flex-col overflow-hidden bg-[#060a12] select-none"
    >
      {/* Map Container */}
      <div ref={containerRef} className="w-full h-full" />

      {/* Loading Overlay */}
      {!googleMapsLoaded && (
        <div className="absolute inset-0 bg-[#060a12] flex flex-col items-center justify-center gap-2 z-20">
          <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
          <span className="text-[11px] font-mono text-slate-300">
            {loadError || 'Initializing Google Maps Infrastructure...'}
          </span>
        </div>
      )}

      {/* Top Floating Quick Layer & Type Badges */}
      <div className="absolute top-2 left-2 z-10 flex flex-wrap items-center gap-1.5 pointer-events-auto">
        {/* Google Maps Realistic Type Toggle */}
        <div className="flex items-center bg-[#070e1b]/90 border border-[#16253c] rounded p-0.5 backdrop-blur-sm shadow-md">
          <button
            onClick={() => setMapType('roadmap')}
            className={`px-2 py-0.5 rounded text-[9.5px] font-mono font-bold transition-colors ${
              mapType === 'roadmap' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Tactical Road
          </button>
          <button
            onClick={() => setMapType('hybrid')}
            className={`px-2 py-0.5 rounded text-[9.5px] font-mono font-bold transition-colors ${
              mapType === 'hybrid' ? 'bg-blue-600 text-white' : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Satellite Hybrid
          </button>
        </div>

        {/* Drone Telemetry Status Badge */}
        <div
          className={`flex items-center gap-1 text-[9px] font-mono px-2 py-0.5 rounded border shadow-sm backdrop-blur-sm ${
            hasValidDroneGps
              ? 'bg-[#06241a]/90 border-emerald-600/50 text-emerald-300'
              : 'bg-[#0f172a]/90 border-slate-700/60 text-slate-400'
          }`}
        >
          <span
            className={`w-1.5 h-1.5 rounded-full ${
              hasValidDroneGps ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'
            }`}
          />
          <span>{hasValidDroneGps ? 'DRONE LIVE GPS' : 'DRONE TELEMETRY OFFLINE'}</span>
        </div>
      </div>

      {/* Layer Toggles (Top Right) */}
      <div className="absolute top-2 right-2 z-10 flex items-center gap-1 pointer-events-auto">
        <button
          onClick={() => setLayers((prev) => ({ ...prev, incident: !prev.incident }))}
          className={`px-2 py-0.5 rounded-[4px] text-[9px] font-mono font-semibold border transition-colors ${
            layers.incident
              ? 'bg-[#151014] border-[#E05252]/60 text-[#E05252]'
              : 'bg-[#111821]/90 border-[#263341] text-[#687585]'
          }`}
        >
          INCIDENT
        </button>
        <button
          onClick={() => setLayers((prev) => ({ ...prev, police: !prev.police }))}
          className={`px-2 py-0.5 rounded-[4px] text-[9px] font-mono font-semibold border transition-colors ${
            layers.police
              ? 'bg-[#111821] border-[#3B9EFF]/60 text-[#3B9EFF]'
              : 'bg-[#111821]/90 border-[#263341] text-[#687585]'
          }`}
        >
          POLICE
        </button>
        <button
          onClick={() => setLayers((prev) => ({ ...prev, hospital: !prev.hospital }))}
          className={`px-2 py-0.5 rounded-[4px] text-[9px] font-mono font-semibold border transition-colors ${
            layers.hospital
              ? 'bg-[#151014] border-[#E05252]/60 text-[#E05252]'
              : 'bg-[#111821]/90 border-[#263341] text-[#687585]'
          }`}
        >
          HOSPITAL
        </button>
      </div>

      {/* Bottom Tactical Navigation Controls Bar */}
      {showControls && (
        <div className="absolute bottom-2 left-2 right-2 z-10 flex items-center justify-between pointer-events-none">
          <div className="flex items-center gap-1 bg-[#151E28]/95 border border-[#263341] p-1 rounded-[6px] backdrop-blur-sm pointer-events-auto shadow-md text-[10px] font-mono">
            <button
              onClick={handleZoomIn}
              className="p-1 hover:bg-[#263341] text-[#98A6B5] hover:text-[#E8EDF3] rounded-[4px] transition-colors"
              title="Zoom In"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-1 hover:bg-[#263341] text-[#98A6B5] hover:text-[#E8EDF3] rounded-[4px] transition-colors"
              title="Zoom Out"
            >
              <Minus className="w-3.5 h-3.5" />
            </button>
            <div className="w-[1px] h-3 bg-[#263341] mx-0.5" />
            <button
              onClick={handleCenterDrone}
              disabled={!hasValidDroneGps}
              className="px-2 py-0.5 hover:bg-[#263341] disabled:opacity-30 text-[#49C6D9] rounded-[4px] flex items-center gap-1 transition-colors font-sans font-medium"
              title="Center Drone"
            >
              <Navigation className="w-3 h-3" />
              <span>Drone</span>
            </button>
            <button
              onClick={handleCenterIncident}
              className="px-2 py-0.5 hover:bg-[#263341] text-[#E05252] rounded-[4px] flex items-center gap-1 transition-colors font-sans font-medium"
              title="Center Incident Target"
            >
              <AlertTriangle className="w-3 h-3" />
              <span>Incident</span>
            </button>
            <button
              onClick={handleFitAll}
              className="px-2 py-0.5 hover:bg-[#263341] text-[#E8EDF3] rounded-[4px] font-sans font-medium transition-colors"
              title="Fit All Resources in View"
            >
              Fit All
            </button>
            {onExpand && (
              <>
                <div className="w-[1px] h-3 bg-[#263341] mx-0.5" />
                <button
                  onClick={onExpand}
                  className="p-1 hover:bg-[#263341] text-[#98A6B5] hover:text-[#E8EDF3] rounded-[4px] transition-colors"
                  title="Expand Map"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
