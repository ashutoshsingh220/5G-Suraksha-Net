export interface VideoSourceStatus {
  status: 'LIVE' | 'STALE' | 'READY' | 'OFFLINE';
  source_kind: string;
  source_url: string;
  resolution: string | null;
  source_fps: number | null;
  effective_fps: number;
  frames_processed: number;
  last_frame_age_s: number | null;
  has_live_frame: boolean;
}

export interface LatencyBreakdown {
  detect_track?: number;
  crowd?: number;
  fight?: number;
  weapon?: number;
  total?: number;
}

export interface AIInferenceStatus {
  status: 'ACTIVE' | 'IDLE' | 'OFFLINE';
  device: string;
  effective_fps: number;
  inference_latency_ms: number | null;
  latency_breakdown: LatencyBreakdown;
}

export interface GPUStatus {
  available: boolean;
  name: string | null;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  utilization_percent: number | null;
}

export interface DroneTelemetryStatus {
  connected: boolean;
  gps_fix_type: number;
  satellites_visible: number;
  last_heartbeat_s: number | null;
}

export interface SystemStatusResponse {
  status: 'ONLINE' | 'DEGRADED' | 'OFFLINE';
  uptime_seconds: number;
  timestamp: string;
  video_source: VideoSourceStatus;
  ai_inference: AIInferenceStatus;
  gpu: GPUStatus;
  drone_telemetry: DroneTelemetryStatus | null;
  incidents_count: number;
}
