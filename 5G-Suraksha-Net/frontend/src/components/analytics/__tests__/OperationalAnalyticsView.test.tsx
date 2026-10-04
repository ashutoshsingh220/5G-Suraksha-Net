import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { OperationalAnalyticsView } from '../OperationalAnalyticsView';
import { IncidentProvider } from '../../../context/IncidentContext';
import type { IncidentReport } from '../../../types/incidents';

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

const mockAnalyticsIncidents: IncidentReport[] = [
  {
    schema_version: '1.0',
    source_module: 'crowd_fight',
    incident_id: 'inc_an_01',
    status: 'finalized',
    camera_id: 'cam_default',
    incident_type: 'weapon',
    severity: 'critical',
    confidence: 0.9,
    start_time: '2026-10-03T10:00:00Z',
    track_ids: [1],
    details: {},
    evidence: {
      snapshot_path: 'outputs/snapshots/inc_an_01.jpg',
    },
  },
  {
    schema_version: '1.0',
    source_module: 'crowd_fight',
    incident_id: 'inc_an_02',
    status: 'verified',
    camera_id: 'cam_default',
    incident_type: 'fight',
    severity: 'high',
    confidence: 0.8,
    start_time: '2026-10-03T11:00:00Z',
    track_ids: [2, 3],
    details: {},
    evidence: {
      clip_path: 'outputs/clips/inc_an_02.mp4',
    },
  },
  {
    schema_version: '1.0',
    source_module: 'crowd_fight',
    incident_id: 'inc_an_03',
    status: 'candidate',
    camera_id: 'cam_pi_drone',
    incident_type: 'crowd_density_high',
    severity: 'moderate',
    confidence: 0.7,
    start_time: '2026-10-03T12:00:00Z',
    track_ids: [],
    details: {},
    evidence: {},
  },
];

describe('OperationalAnalyticsView Component', () => {
  const onBack = vi.fn();

  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket);
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('1. Renders honest empty state when 0 incidents exist', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ count: 0, incidents: [] }),
    });

    render(
      <IncidentProvider>
        <OperationalAnalyticsView onBackToCommand={onBack} />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/NO INCIDENTS RECORDED IN SESSION/i)).toBeDefined();
    });
  });

  it('2. Dynamically calculates accurate metrics strictly from real incident records', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({
        count: 3,
        incidents: mockAnalyticsIncidents,
      }),
    });

    render(
      <IncidentProvider>
        <OperationalAnalyticsView onBackToCommand={onBack} />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('3 incidents')).toBeDefined();
    });

    // Total = 3
    expect(screen.getAllByText('3').length).toBeGreaterThanOrEqual(1);

    // Verified = 2 (finalized + verified)
    expect(screen.getAllByText('2').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/67% verified rate/i)).toBeDefined();

    // High / Critical = 2 (1 critical + 1 high)
    expect(screen.getByText(/Urgent Response/i)).toBeDefined();

    // Evidence rate = 2 / 3 = 67%
    expect(screen.getAllByText('67%').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/2 with artifact/i)).toBeDefined();

    // Average confidence = (0.9 + 0.8 + 0.7) / 3 = 80%
    expect(screen.getByText('80%')).toBeDefined();

    // Breakdown labels
    expect(screen.getByText(/Severity Distribution/i)).toBeDefined();
    expect(screen.getByText(/Incident Type Breakdown/i)).toBeDefined();
    expect(screen.getByText(/Surveillance Sources/i)).toBeDefined();
  });
});
