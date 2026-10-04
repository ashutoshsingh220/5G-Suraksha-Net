/**
 * Strongly typed incident contracts matching the authoritative backend schema
 * (src/suraksha/incidents/schemas.py and src/suraksha/location/schemas.py).
 */

export type IncidentType =
  | 'fight'
  | 'crowd_density_high'
  | 'crowd_density_critical'
  | 'crowd_rapid_growth'
  | 'crowd_panic'
  | 'weapon'
  | 'armed_fight';

export type IncidentSeverity =
  | 'low'
  | 'medium'
  | 'moderate'
  | 'high'
  | 'critical';

export type IncidentStatus =
  | 'candidate'
  | 'verified'
  | 'recording_post_event'
  | 'finalized';

export interface IncidentBBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface IncidentEvidence {
  snapshot_path?: string | null;
  clip_path?: string | null;
  frame_idx?: number | null;
}

export interface IncidentLocation {
  name: string;
  latitude?: number | null;
  longitude?: number | null;
  source?: string;
  timestamp?: string;
  accuracy?: number | null;
  camera_id?: string | null;
  source_id?: string | null;
  has_coordinates?: boolean;
}

export interface IncidentReport {
  schema_version: string;
  source_module: string;
  incident_id: string;
  status: IncidentStatus;
  camera_id: string;
  incident_type: IncidentType;
  severity: IncidentSeverity;
  confidence: number;
  start_time: string;
  end_time?: string | null;
  zone?: string | null;
  track_ids: number[];
  bbox?: IncidentBBox | null;
  person_count?: number | null;
  details: Record<string, any>;
  evidence: IncidentEvidence;
  location?: IncidentLocation | null;
  source_mode?: 'REAL' | 'DEMO';
  finalized_at?: string | null;
}

export type IncidentStreamConnectionStatus =
  | 'CONNECTING'
  | 'CONNECTED'
  | 'RECONNECTING'
  | 'DISCONNECTED'
  | 'ERROR';

export interface EmergencyFacility {
  facility_id: string;
  name: string;
  facility_type: string;
  address: string;
  phone: string;
  distance_km: number;
  eta_minutes: number;
  latitude: number;
  longitude: number;
}

export interface IncidentResponsePlan {
  schema_version: string;
  incident_id: string;
  incident_type: string;
  severity: string;
  location: IncidentLocation;
  primary_responder: string;
  police: EmergencyFacility;
  hospital: EmergencyFacility;
  actions: string[];
  recommended_actions: string[];
  requires_human_approval: boolean;
  approval_status: string;
  generated_at: string;
}
