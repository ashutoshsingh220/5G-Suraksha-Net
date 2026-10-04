export interface DroneTelemetryData {
  connected: boolean;
  status: 'ONLINE' | 'STALE' | 'OFFLINE';
  telemetry_age_s: number | null;
  last_heartbeat_timestamp: number | null;
  drone_id: string;
  platform: string;
  system_type: string;
  armed: boolean;
  flight_mode: string;
  altitude_relative_m: number;
  altitude_amsl_m: number;
  groundspeed_m_s: number;
  airspeed_m_s: number;
  climb_rate_m_s: number;
  heading_deg: number;
  latitude: number | null;
  longitude: number | null;
  battery_percent: number;
  battery_voltage_v: number;
  battery_current_a: number;
  gps_fix_type: string;
  satellites_visible: number;
  hdop: number;
  roll_deg: number;
  pitch_deg: number;
  yaw_deg: number;
  edge_compute: string;
  camera_status: string;
}

export const initialDroneTelemetry: DroneTelemetryData = {
  connected: false,
  status: 'OFFLINE',
  telemetry_age_s: null,
  last_heartbeat_timestamp: null,
  drone_id: 'SURAKSHA-DRONE-01',
  platform: 'Edge AI (Raspberry Pi + ArduPilot)',
  system_type: 'QUADROTOR',
  armed: false,
  flight_mode: 'UNKNOWN',
  altitude_relative_m: 0.0,
  altitude_amsl_m: 0.0,
  groundspeed_m_s: 0.0,
  airspeed_m_s: 0.0,
  climb_rate_m_s: 0.0,
  heading_deg: 0.0,
  latitude: null,
  longitude: null,
  battery_percent: 0,
  battery_voltage_v: 0.0,
  battery_current_a: 0.0,
  gps_fix_type: 'No GPS',
  satellites_visible: 0,
  hdop: 0.0,
  roll_deg: 0.0,
  pitch_deg: 0.0,
  yaw_deg: 0.0,
  edge_compute: 'Active',
  camera_status: 'Online',
};
