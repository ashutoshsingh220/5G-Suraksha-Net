import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { SystemStatusBar } from '../SystemStatusBar';
import { IncidentProvider } from '../../../context/IncidentContext';

class MockWebSocket {
  onopen: any = null;
  onmessage: any = null;
  onerror: any = null;
  onclose: any = null;
  readyState = 1;
  send = vi.fn();
  close = vi.fn();
  addEventListener = vi.fn();
  removeEventListener = vi.fn();
}

const mockSystemStatusOnline = {
  status: 'ONLINE',
  uptime_seconds: 142.5,
  timestamp: '2026-10-03T15:30:00Z',
  video_source: {
    status: 'LIVE',
    source_kind: 'webcam',
    source_url: '0',
    resolution: '1280x720',
    source_fps: 30.0,
    effective_fps: 28.4,
    frames_processed: 4050,
    last_frame_age_s: 0.05,
    has_live_frame: true,
  },
  ai_inference: {
    status: 'ACTIVE',
    device: 'cuda',
    effective_fps: 28.4,
    inference_latency_ms: 12.8,
    latency_breakdown: {
      detect_track: 8.1,
      crowd: 1.2,
      fight: 1.5,
      weapon: 2.0,
      total: 12.8,
    },
  },
  gpu: {
    available: true,
    name: 'NVIDIA GeForce RTX 4050 Laptop GPU',
    memory_used_mb: 2840,
    memory_total_mb: 6144,
    utilization_percent: null, // Strictly unmeasured without continuous profiler
  },
  drone_telemetry: {
    connected: true,
    gps_fix_type: 3,
    satellites_visible: 14,
    last_heartbeat_s: 0.2,
  },
  incidents_count: 0,
};

describe('SystemStatusBar Component', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket);
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('1. Renders truthful online system, AI, and network status without fake 5G label', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/system/status')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockSystemStatusOnline,
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({ count: 0, incidents: [] }),
      });
    });

    render(
      <IncidentProvider>
        <SystemStatusBar />
      </IncidentProvider>
    );

    // Verify System status
    await waitFor(() => {
      expect(screen.getByText(/SYSTEM:/i)).toBeDefined();
      expect(screen.getAllByText('ONLINE').length).toBeGreaterThanOrEqual(1);
    });

    // Verify NETWORK LINK label exists and is NOT labeled "5G LINK" or "5G STABLE"
    expect(screen.getByText(/NETWORK LINK:/i)).toBeDefined();
    expect(screen.queryByText(/5G LINK/i)).toBeNull();
    expect(screen.queryByText(/5G STABLE/i)).toBeNull();

    // Verify AI Engine
    expect(screen.getByText(/AI ENGINE:/i)).toBeDefined();
    expect(screen.getByText('CUDA (ACTIVE)')).toBeDefined();

    // Verify Source and Resolution
    expect(screen.getByText(/SOURCE:/i)).toBeDefined();
    expect(screen.getByText(/WEBCAM \[1280x720\] \(LIVE\)/i)).toBeDefined();

    // Verify FPS and Latency
    expect(screen.getByText('28.4')).toBeDefined();
    expect(screen.getByText('12.8 ms')).toBeDefined();

    // Verify Truthful Accelerator VRAM and NO fabricated utilization percentage
    expect(screen.getByText(/ACCELERATOR:/i)).toBeDefined();
    expect(screen.getByText(/2840\/6144 MB/i)).toBeDefined();
    expect(screen.queryByText(/%/i)).toBeNull();

    // Verify Drone MAVLink status
    expect(screen.getByText(/DRONE LINK:/i)).toBeDefined();
    expect(screen.getByText('MAVLINK (14 SATS)')).toBeDefined();
  });

  it('2. Gracefully renders truthful OFFLINE state when backend fails or is unreachable', async () => {
    (fetch as any).mockRejectedValue(new Error('Network error'));

    render(
      <IncidentProvider>
        <SystemStatusBar />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/SYSTEM:/i)).toBeDefined();
    });

    expect(screen.getAllByText('OFFLINE').length).toBeGreaterThan(0);
    expect(screen.getByText('CPU (N/A)')).toBeDefined();
    expect(screen.getByText('STANDBY')).toBeDefined();
  });
});
