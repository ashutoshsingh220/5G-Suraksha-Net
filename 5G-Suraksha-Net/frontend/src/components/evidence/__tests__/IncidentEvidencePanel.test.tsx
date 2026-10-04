import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { IncidentEvidencePanel } from '../IncidentEvidencePanel';
import { IncidentProvider } from '../../../context/IncidentContext';
import type { IncidentReport } from '../../../types/incidents';

// Mock WebSocket
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

const mockIncidentFinalized: IncidentReport = {
  schema_version: '1.0',
  source_module: 'crowd_fight',
  incident_id: '56dc59e0f761',
  status: 'finalized',
  camera_id: 'cam_dwarka_sec25_01',
  incident_type: 'armed_fight',
  severity: 'critical',
  confidence: 0.96,
  start_time: '2026-10-03T14:30:00Z',
  track_ids: [10, 11],
  details: { weapon_class: 'knife' },
  evidence: {
    snapshot_path: 'outputs/snapshots/56dc59e0f761.jpg',
    clip_path: 'outputs/clips/56dc59e0f761.mp4',
    frame_idx: 1420,
  },
  location: {
    name: 'Sector 25 Yashobhoomi Main Gate',
    latitude: 28.5529,
    longitude: 77.0601,
  },
};

const mockIncidentRecording: IncidentReport = {
  ...mockIncidentFinalized,
  incident_id: 'rec_inc_999',
  status: 'recording_post_event',
};

describe('IncidentEvidencePanel Component', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket);
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('1. Displays standby state when no incident is selected', async () => {
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ count: 0, incidents: [] }),
    });

    render(
      <IncidentProvider>
        <IncidentEvidencePanel />
      </IncidentProvider>
    );

    expect(screen.getByText(/NO INCIDENT SELECTED/i)).toBeDefined();
    expect(screen.getByText(/SYSTEM MONITORING • ZERO SIMULATED EVIDENCE/i)).toBeDefined();
  });

  it('2. Renders real snapshot and metadata when finalized incident is active', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncidentFinalized] }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(
      <IncidentProvider>
        <IncidentEvidencePanel />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/56dc59e0f761/i)).toBeDefined();
    });

    // Check metadata
    expect(screen.getByText(/cam_dwarka_sec25_01/i)).toBeDefined();
    expect(screen.getByText(/Frame #1420/i)).toBeDefined();
    expect(screen.getByText(/armed fight/i)).toBeDefined();

    // Check image src
    const img = screen.getByRole('img');
    expect(img.getAttribute('src')).toBe('http://localhost:8100/incidents/56dc59e0f761/snapshot');
  });

  it('3. Handles snapshot error gracefully', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncidentFinalized] }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(
      <IncidentProvider>
        <IncidentEvidencePanel />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByRole('img')).toBeDefined();
    });

    // Simulate image error
    const img = screen.getByRole('img');
    fireEvent.error(img);

    await waitFor(() => {
      expect(screen.getByText(/SNAPSHOT UNAVAILABLE/i)).toBeDefined();
    });
  });

  it('4. Switches to video clip view and renders video element for finalized incident', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncidentFinalized] }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(
      <IncidentProvider>
        <IncidentEvidencePanel />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/Evidence Clip/i)).toBeDefined();
    });

    const clipTabBtn = screen.getByText(/Evidence Clip/i);
    fireEvent.click(clipTabBtn);

    // Verify video tag with correct source URL
    const video = document.querySelector('video');
    expect(video).toBeDefined();
    const source = video?.querySelector('source');
    expect(source?.getAttribute('src')).toBe('http://localhost:8100/incidents/56dc59e0f761/clip');
  });

  it('5. Displays EVIDENCE RECORDING when incident status is recording_post_event', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncidentRecording] }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(
      <IncidentProvider>
        <IncidentEvidencePanel />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/EVIDENCE RECORDING/i)).toBeDefined();
    });

    // Switch to clip tab to verify temporal window recording message
    const clipTabBtn = screen.getByText(/Evidence Clip/i);
    fireEvent.click(clipTabBtn);

    expect(screen.getByText(/RECORDING POST-EVENT EVIDENCE CLIP/i)).toBeDefined();
  });

  it('6. Opens and closes fullscreen forensic investigation modal', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncidentFinalized] }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(
      <IncidentProvider>
        <IncidentEvidencePanel />
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/View Fullscreen/i)).toBeDefined();
    });

    // Open modal
    const fullscreenBtn = screen.getByText(/View Fullscreen/i);
    fireEvent.click(fullscreenBtn);

    expect(screen.getByText(/FORENSIC INVESTIGATION VIEWER/i)).toBeDefined();
    expect(screen.getByText(/Sector 25 Yashobhoomi Main Gate/i)).toBeDefined();

    // Close via ESC key
    fireEvent.keyDown(window, { key: 'Escape', code: 'Escape' });

    await waitFor(() => {
      expect(screen.queryByText(/FORENSIC INVESTIGATION VIEWER/i)).toBeNull();
    });
  });
});
