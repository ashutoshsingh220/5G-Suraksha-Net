import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { IncidentHistoryView } from '../IncidentHistoryView';
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

const mockIncidentA: IncidentReport = {
  schema_version: '1.0',
  source_module: 'crowd_fight',
  incident_id: 'inc_weapon_01',
  status: 'finalized',
  camera_id: 'cam_default',
  incident_type: 'weapon',
  severity: 'high',
  confidence: 0.89,
  start_time: '2026-10-03T14:30:00Z',
  finalized_at: '2026-10-03T14:30:10Z',
  track_ids: [4],
  details: { weapon_class: 'knife' },
  evidence: {
    snapshot_path: 'outputs/snapshots/inc_weapon_01.jpg',
    clip_path: 'outputs/clips/inc_weapon_01.mp4',
    frame_idx: 120,
  },
  location: {
    name: 'Sector 25 Yashobhoomi Main Gate',
    latitude: 28.5529,
    longitude: 77.0601,
    source: 'fixed',
  },
};

const mockIncidentB: IncidentReport = {
  schema_version: '1.0',
  source_module: 'crowd_fight',
  incident_id: 'inc_fight_02',
  status: 'verified',
  camera_id: 'cam_pi_drone',
  incident_type: 'fight',
  severity: 'moderate',
  confidence: 0.76,
  start_time: '2026-10-03T14:35:00Z',
  track_ids: [10, 11],
  details: {},
  evidence: {
    snapshot_path: null,
    clip_path: null,
  },
  location: {
    name: 'Convention Center Plaza',
    latitude: 28.5535,
    longitude: 77.0612,
  },
};

const mockIncidentC: IncidentReport = {
  schema_version: '1.0',
  source_module: 'crowd_fight',
  incident_id: 'inc_crowd_03',
  status: 'candidate',
  camera_id: 'cam_default',
  incident_type: 'crowd_density_critical',
  severity: 'critical',
  confidence: 0.95,
  start_time: '2026-10-03T14:40:00Z',
  track_ids: [],
  person_count: 85,
  details: {},
  evidence: {},
  location: null, // missing location test
};

describe('IncidentHistoryView Component', () => {
  const onBack = vi.fn();

  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket);
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('1. Renders professional empty state when no incidents exist', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ count: 0, incidents: [] }),
    });

    render(
      <IncidentProvider>
        <IncidentHistoryView onBackToCommand={onBack} />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/NO INCIDENTS MATCHING FILTER/i)).toBeDefined();
      expect(screen.getByText(/SYSTEM MONITORING/i)).toBeDefined();
    });
  });

  it('2. Renders real incident rows with accurate IDs, severities, and locations', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({
        count: 3,
        incidents: [mockIncidentA, mockIncidentB, mockIncidentC],
      }),
    });

    render(
      <IncidentProvider>
        <IncidentHistoryView onBackToCommand={onBack} />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getAllByText(/#pon_01/i).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/#ght_02/i).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/#owd_03/i).length).toBeGreaterThanOrEqual(1);
    });

    // Check types and severities
    expect(screen.getByText('WEAPON')).toBeDefined();
    expect(screen.getByText('FIGHT')).toBeDefined();
    expect(screen.getByText('DENSITY CRIT')).toBeDefined();
    expect(screen.getAllByText('CRITICAL').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('HIGH').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('MODERATE').length).toBeGreaterThanOrEqual(1);

    // Check location
    expect(screen.getByText('Sector 25 Yashobhoomi Main Gate')).toBeDefined();
    expect(screen.getByText('Convention Center Plaza')).toBeDefined();
  });

  it('3. Filters incidents by search query text', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({
        count: 3,
        incidents: [mockIncidentA, mockIncidentB, mockIncidentC],
      }),
    });

    render(
      <IncidentProvider>
        <IncidentHistoryView onBackToCommand={onBack} />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getAllByText(/#pon_01/i).length).toBeGreaterThanOrEqual(1);
    });

    const searchInput = screen.getByPlaceholderText(/Search by ID, location, or camera/i);
    fireEvent.change(searchInput, { target: { value: 'drone' } });

    // Table should only show drone incident
    const table = document.querySelector('table');
    expect(table).toBeDefined();
    expect(table?.textContent).toContain('ght_02');
    expect(table?.textContent).not.toContain('pon_01');
    expect(table?.textContent).not.toContain('owd_03');
  });

  it('4. Filters incidents by severity dropdown', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({
        count: 3,
        incidents: [mockIncidentA, mockIncidentB, mockIncidentC],
      }),
    });

    render(
      <IncidentProvider>
        <IncidentHistoryView onBackToCommand={onBack} />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getAllByText(/#pon_01/i).length).toBeGreaterThanOrEqual(1);
    });

    // Select Critical
    const selects = screen.getAllByRole('combobox');
    const severitySelect = selects[2]; // [0]=Time, [1]=Type, [2]=Severity, [3]=Status
    fireEvent.change(severitySelect, { target: { value: 'critical' } });

    expect(screen.queryByText(/#pon_01/i)).toBeNull();
    expect(screen.queryByText(/#ght_02/i)).toBeNull();
    expect(screen.getAllByText(/#owd_03/i).length).toBeGreaterThanOrEqual(1);
  });

  it('5. Selecting an incident displays its chronological lifecycle timeline', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            count: 2,
            incidents: [mockIncidentA, mockIncidentB],
          }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(
      <IncidentProvider>
        <IncidentHistoryView onBackToCommand={onBack} />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getAllByText(/#pon_01/i).length).toBeGreaterThanOrEqual(1);
    });

    // Select row
    const rowA = screen.getAllByText(/#pon_01/i)[0].closest('tr');
    fireEvent.click(rowA!);

    // Check timeline lifecycle node headers
    await waitFor(() => {
      expect(screen.getByText(/Incident Lifecycle Timeline/i)).toBeDefined();
      expect(screen.getByText('DETECTED')).toBeDefined();
      expect(screen.getByText('VERIFIED')).toBeDefined();
      expect(screen.getByText('EVIDENCE RECORDING')).toBeDefined();
      expect(screen.getByText('FINALIZED')).toBeDefined();
      expect(screen.getByText('RESPONSE DISPATCH')).toBeDefined();
    });
  });

  it('6. Handles missing optional fields gracefully without crash', async () => {
    (fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({
        count: 1,
        incidents: [mockIncidentC], // has no location, no tracks, no bbox, no evidence
      }),
    });

    render(
      <IncidentProvider>
        <IncidentHistoryView onBackToCommand={onBack} />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getAllByText(/#owd_03/i).length).toBeGreaterThanOrEqual(1);
    });

    // Fallback to camera_id or default when location is null
    expect(screen.getAllByText('cam_default').length).toBeGreaterThanOrEqual(1);
  });
});
