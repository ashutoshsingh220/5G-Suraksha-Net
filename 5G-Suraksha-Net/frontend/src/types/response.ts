/**
 * Data contracts for Phase 3C & Part 6 Response Intelligence.
 * Strictly mirrors authoritative backend schemas in src/suraksha/response/schemas.py.
 */

import type { IncidentType, IncidentSeverity, IncidentLocation, IncidentEvidence } from './incidents';

export type ActionType = 'POLICE_SECURITY' | 'MEDICAL_ASSISTANCE';

export type ActionPriority = 'low' | 'medium' | 'moderate' | 'high' | 'critical';

export type ResponseStatus =
  | 'RECOMMENDED'
  | 'AWAITING_HUMAN_APPROVAL'
  | 'NO_ACTION'
  | 'RESOURCE_UNAVAILABLE';

export interface EmergencyResource {
  name: string;
  category?: string | null;
  place_type?: string | null;
  distance_km?: number | null;
  distance_m?: number | null;
  formatted_address?: string | null;
  latitude?: number | null;
  longitude?: number | null;
  phone_number?: string | null;
  rating?: number | null;
  user_ratings_count?: number | null;
  website?: string | null;
  place_id?: string | null;
  distance_type?: string; // "straight_line"
}

export interface ResponseAction {
  action_type: ActionType;
  target: string; // e.g. "POLICE_SECURITY_DISPATCH" | "MEDICAL_TRAUMA_ASSISTANCE"
  priority: string;
  reason: string;
  human_approval_required: boolean;
  status: ResponseStatus;
  recommended_resource?: EmergencyResource | null;
}

export interface IncidentResponse {
  schema_version: string;
  incident_id: string;
  incident_type: IncidentType | string;
  severity: IncidentSeverity | string;
  timestamp: string;
  location?: IncidentLocation | null;
  response_actions: ResponseAction[];
  police_resources: EmergencyResource[];
  medical_resources: EmergencyResource[];
  evidence?: IncidentEvidence;
  human_approval_required: boolean;
  status: ResponseStatus;
  warning?: string | null;
  email_status?: string | null;
  disclaimer: string;
}
