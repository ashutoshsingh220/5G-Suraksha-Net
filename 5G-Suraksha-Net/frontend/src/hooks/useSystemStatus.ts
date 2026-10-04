import { useState, useEffect, useCallback, useRef } from 'react';
import type { SystemStatusResponse } from '../types/system';

interface UseSystemStatusOptions {
  pollingIntervalMs?: number;
  apiBaseUrl?: string;
}

interface UseSystemStatusResult {
  status: SystemStatusResponse | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

const DEFAULT_BASE_URL = 'http://localhost:8100';
const DEFAULT_INTERVAL_MS = 3000;

export function useSystemStatus(options?: UseSystemStatusOptions): UseSystemStatusResult {
  const apiBaseUrl = options?.apiBaseUrl || DEFAULT_BASE_URL;
  const intervalMs = options?.pollingIntervalMs ?? DEFAULT_INTERVAL_MS;

  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const isMountedRef = useRef<boolean>(true);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${apiBaseUrl}/system/status`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data: SystemStatusResponse = await res.json();
      if (isMountedRef.current) {
        setStatus(data);
        setError(null);
        setLoading(false);
      }
    } catch (err: any) {
      if (isMountedRef.current) {
        setError(err?.message || 'Failed to fetch system status');
        // When backend is unreachable, preserve last status or synthesize truthful OFFLINE state
        setStatus((prev) =>
          prev
            ? {
                ...prev,
                status: 'OFFLINE',
                video_source: { ...prev.video_source, status: 'OFFLINE' },
                ai_inference: { ...prev.ai_inference, status: 'OFFLINE' },
              }
            : {
                status: 'OFFLINE',
                uptime_seconds: 0,
                timestamp: new Date().toISOString(),
                video_source: {
                  status: 'OFFLINE',
                  source_kind: 'none',
                  source_url: '',
                  resolution: null,
                  source_fps: null,
                  effective_fps: 0,
                  frames_processed: 0,
                  last_frame_age_s: null,
                  has_live_frame: false,
                },
                ai_inference: {
                  status: 'OFFLINE',
                  device: 'none',
                  effective_fps: 0,
                  inference_latency_ms: null,
                  latency_breakdown: {},
                },
                gpu: {
                  available: false,
                  name: null,
                  memory_used_mb: null,
                  memory_total_mb: null,
                  utilization_percent: null,
                },
                drone_telemetry: null,
                incidents_count: 0,
              }
        );
        setLoading(false);
      }
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    isMountedRef.current = true;
    fetchStatus();

    const intervalId = setInterval(fetchStatus, intervalMs);

    return () => {
      isMountedRef.current = false;
      clearInterval(intervalId);
    };
  }, [fetchStatus, intervalMs]);

  return {
    status,
    loading,
    error,
    refetch: fetchStatus,
  };
}
