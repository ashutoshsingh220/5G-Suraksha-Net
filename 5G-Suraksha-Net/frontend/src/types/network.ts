/**
 * Network and 5G policy data contracts for 5G Suraksha-Net.
 * Adheres strictly to zero-fabrication rules: flags actual_network_control as false.
 */

export type NetworkPolicy = 'NORMAL' | 'EVENT' | 'CRITICAL';

export type ApplicationPriority = 'ROUTINE' | 'HIGH' | 'CRITICAL';

export type PolicyEventType =
  | 'POLICY_INITIALIZED'
  | 'POLICY_ESCALATED'
  | 'POLICY_DEESCALATED'
  | 'INCIDENT_PRIORITY_ASSIGNED'
  | 'EVIDENCE_READY';

export interface NetworkPolicyState {
  policy: NetworkPolicy;
  priority: ApplicationPriority;
  active_incident_id?: string | null;
  active_incident_type?: string | null;
  active_severity?: string | null;
  reason: string;
  actual_network_control: boolean;
  control_plane_connected: boolean;
  policy_scope: string;
  slice_allocated: string;
  measurement_source: string;
  timestamp: string;
}

export interface NetworkEventMetric {
  event_id: string;
  event_type: PolicyEventType;
  policy: NetworkPolicy;
  priority: ApplicationPriority;
  incident_id?: string | null;
  details: string;
  timestamp: string;
  actual_network_control: boolean;
}

export interface NetworkEventsResponse {
  count: number;
  events: NetworkEventMetric[];
}
