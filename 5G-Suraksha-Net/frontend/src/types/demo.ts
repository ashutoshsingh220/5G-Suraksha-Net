import type { IncidentReport } from './incidents';

export type DemoScenarioId = 'WEAPON' | 'CROWD_PANIC' | 'ARMED_FIGHT' | 'NORMAL';

export interface DemoVideoChoice {
  id: string;
  label: string;
}

export interface DemoScenarioMeta {
  id: DemoScenarioId;
  name: string;
  description: string;
  incident_type: string | null;
  severity: string | null;
  expected_network_policy: string;
  expected_network_priority: string;
  location_name: string;
  video_choices?: DemoVideoChoice[];
}

export interface DemoSimulationStep {
  step_id: string;
  timestamp_label: string;
  title: string;
  details: string;
  status: string;
}

export interface DemoStatusResponse {
  is_demo_active: boolean;
  active_scenario: DemoScenarioId | null;
  source_mode: 'REAL' | 'DEMO';
  incident: IncidentReport | null;
  simulation_timeline: DemoSimulationStep[];
  video_file?: string | null;
  video_choice?: string | null;
}
