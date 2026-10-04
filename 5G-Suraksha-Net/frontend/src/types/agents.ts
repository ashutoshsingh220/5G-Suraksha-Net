/**
 * Data contracts for Phase 3E Agentic Emergency Orchestration.
 * All assessments are advisory, traceable, and require human approval.
 */

export type ActionStepStatus =
  | 'COMPLETED'
  | 'REQUIRED'
  | 'ADVISORY'
  | 'AWAITING_APPROVAL';

export interface ActionStep {
  step_number: number;
  title: string;
  description: string;
  status: ActionStepStatus;
  advisory_only: boolean;
}

export interface SeverityAssessment {
  severity: string;
  explanation: string;
  is_deterministic: boolean;
  incident_type: string;
}

export interface LocationAssessment {
  location_name: string | null;
  latitude: number | null;
  longitude: number | null;
  nearby_police_count: number;
  nearby_hospital_count: number;
  closest_police_facility: string | null;
  closest_police_distance_km: number | null;
  closest_hospital_facility: string | null;
  closest_hospital_distance_km: number | null;
  gps_fix_available: boolean;
  summary: string;
}

export interface EvidenceAssessment {
  snapshot_available: boolean;
  snapshot_path: string | null;
  video_available: boolean;
  video_path: string | null;
  summary: string;
}

export interface ResponseAssessment {
  status: string;
  police_recommended: boolean;
  medical_recommended: boolean;
  medical_conditional: boolean;
  actions_summary: string;
  human_approval_required: boolean;
}

export interface NetworkAssessment {
  policy: string;
  priority: string;
  actual_network_control: boolean;
  control_plane_connected: boolean;
  summary: string;
}

export interface DataSourceTraceability {
  incident_report: boolean;
  response_planner: boolean;
  location_intelligence: boolean;
  evidence_metadata: boolean;
  network_policy: boolean;
}

export interface AgentAssessment {
  incident_id: string;
  situation_summary: string;
  severity_assessment: SeverityAssessment;
  location_assessment: LocationAssessment;
  evidence_assessment: EvidenceAssessment;
  response_assessment: ResponseAssessment;
  network_assessment: NetworkAssessment;
  recommended_sequence: ActionStep[];
  missing_information: string[];
  data_sources: DataSourceTraceability;
  confidence: number;
  human_approval_required: boolean;
  generated_at: string;
  source: string;
  disclaimer: string;
}
