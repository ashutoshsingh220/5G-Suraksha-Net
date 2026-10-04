import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { RecommendedResponseCard } from '../RecommendedResponseCard';
import { IncidentProvider } from '../../../context/IncidentContext';
import { MapFocusProvider } from '../../../context/MapFocusContext';
import type { IncidentReport } from '../../../types/incidents';
import type { IncidentResponse } from '../../../types/response';

// Mock WebSocket class to prevent network socket errors in jsdom
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

// Mock test incident report
const mockIncident: IncidentReport = {
  schema_version: '1.0',
  source_module: 'crowd_fight',
  incident_id: 'inc_test_001',
  status: 'verified',
  camera_id: 'cam_yashobhoomi_01',
  incident_type: 'armed_fight',
  severity: 'critical',
  confidence: 0.94,
  start_time: '2026-10-03T14:00:00Z',
  track_ids: [1, 2],
  details: { weapon_class: 'pistol' },
  evidence: { snapshot_path: 'snapshots/inc_test_001.jpg' },
  location: {
    name: 'Yashobhoomi Main Convention Hall A',
    latitude: 28.55255,
    longitude: 77.04489,
    source: 'geocoded_demo',
  },
};

// Mock Joint Response Plan (Police + Medical)
const mockJointPlan: IncidentResponse = {
  schema_version: '1.0',
  incident_id: 'inc_test_001',
  incident_type: 'armed_fight',
  severity: 'critical',
  timestamp: '2026-10-03T14:00:00Z',
  location: mockIncident.location,
  human_approval_required: true,
  status: 'AWAITING_HUMAN_APPROVAL',
  disclaimer:
    'Phase 3C provides decision support only. Straight-line geographic distance; not road distance or ETA. No external emergency service is contacted automatically.',
  response_actions: [
    {
      action_type: 'POLICE_SECURITY',
      target: 'POLICE_SECURITY_DISPATCH',
      priority: 'critical',
      reason:
        'Armed fight combines a confirmed weapon with a physical fight; police/security response is required.',
      human_approval_required: true,
      status: 'AWAITING_HUMAN_APPROVAL',
      recommended_resource: {
        name: 'Police Station Dwarka Sector 23',
        distance_km: 1.2,
        distance_type: 'straight_line',
        formatted_address: 'Sector 23 Dwarka, New Delhi',
        phone_number: '011 2805 1585',
        latitude: 28.5582,
        longitude: 77.0521,
      },
    },
    {
      action_type: 'MEDICAL_ASSISTANCE',
      target: 'MEDICAL_TRAUMA_ASSISTANCE',
      priority: 'critical',
      reason:
        'Armed fight indicates high risk of critical trauma; medical assistance readiness is required.',
      human_approval_required: true,
      status: 'AWAITING_HUMAN_APPROVAL',
      recommended_resource: {
        name: 'Max Super Speciality Hospital Dwarka',
        distance_km: 1.6,
        distance_type: 'straight_line',
        formatted_address: 'Sector 10 Dwarka, New Delhi',
        phone_number: '088604 44888',
        latitude: 28.5812,
        longitude: 77.0592,
      },
    },
  ],
  police_resources: [],
  medical_resources: [],
};

// Mock Police-Only Plan
const mockPoliceOnlyPlan: IncidentResponse = {
  ...mockJointPlan,
  incident_type: 'fight',
  severity: 'high',
  response_actions: [mockJointPlan.response_actions[0]],
};

// Mock Medical-Only Plan
const mockMedicalOnlyPlan: IncidentResponse = {
  ...mockJointPlan,
  response_actions: [mockJointPlan.response_actions[1]],
};

// Mock No-Action Plan
const mockNoActionPlan: IncidentResponse = {
  ...mockJointPlan,
  incident_type: 'crowd_density_high',
  severity: 'moderate',
  status: 'NO_ACTION',
  warning:
    'Crowd density/growth is under routine monitoring; no emergency intervention policy configured.',
  response_actions: [],
};

describe('RecommendedResponseCard Operational Component', () => {
  beforeEach(() => {
    vi.stubGlobal('WebSocket', MockWebSocket);
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('1. Renders standby state when no active incident exists', async () => {
    (fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ count: 0, incidents: [] }),
    });

    render(
      <IncidentProvider>
        <MapFocusProvider>
          <RecommendedResponseCard />
        </MapFocusProvider>
      </IncidentProvider>
    );

    expect(screen.getByText(/RESPONSE STANDBY/i)).toBeDefined();
    expect(screen.getByText(/No active incident requires emergency response/i)).toBeDefined();
    expect(screen.getAllByText(/DECISION SUPPORT ONLY/i).length).toBeGreaterThan(0);
  });

  it('2. Renders joint Police + Medical response plan with human approval gate', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncident] }),
        });
      }
      if (url.includes('/response')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockJointPlan,
        });
      }
      return Promise.reject(new Error('Unknown url'));
    });

    render(
      <IncidentProvider>
        <MapFocusProvider>
          <RecommendedResponseCard />
        </MapFocusProvider>
      </IncidentProvider>
    );

    // Wait for response plan to load
    await waitFor(() => {
      expect(screen.getByText(/POLICE SUPPORT/i)).toBeDefined();
      expect(screen.getByText(/MEDICAL SUPPORT/i)).toBeDefined();
    });

    // Check facility names
    expect(screen.getByText(/Police Station Dwarka Sector 23/i)).toBeDefined();
    expect(screen.getByText(/Max Super Speciality Hospital Dwarka/i)).toBeDefined();

    // Check human approval gate representation
    expect(screen.getAllByText(/AWAITING HUMAN APPROVAL/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/DECISION SUPPORT ONLY/i).length).toBeGreaterThan(0);
  });

  it('3. Renders Police-only response plan', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncident] }),
        });
      }
      if (url.includes('/response')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockPoliceOnlyPlan,
        });
      }
      return Promise.reject(new Error('Unknown url'));
    });

    render(
      <IncidentProvider>
        <MapFocusProvider>
          <RecommendedResponseCard />
        </MapFocusProvider>
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/POLICE SUPPORT/i)).toBeDefined();
    });

    // Should NOT have medical support
    expect(screen.queryByText(/MEDICAL SUPPORT/i)).toBeNull();
  });

  it('4. Renders Medical-only response plan', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncident] }),
        });
      }
      if (url.includes('/response')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockMedicalOnlyPlan,
        });
      }
      return Promise.reject(new Error('Unknown url'));
    });

    render(
      <IncidentProvider>
        <MapFocusProvider>
          <RecommendedResponseCard />
        </MapFocusProvider>
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/MEDICAL SUPPORT/i)).toBeDefined();
    });

    // Should NOT have police support
    expect(screen.queryByText(/POLICE SUPPORT/i)).toBeNull();
  });

  it('5. Renders No-Action plan with deterministic policy explanation', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncident] }),
        });
      }
      if (url.includes('/response')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockNoActionPlan,
        });
      }
      return Promise.reject(new Error('Unknown url'));
    });

    render(
      <IncidentProvider>
        <MapFocusProvider>
          <RecommendedResponseCard />
        </MapFocusProvider>
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/NO IMMEDIATE RESPONSE RECOMMENDED/i)).toBeDefined();
    });

    expect(screen.getByText(/System continues routine monitoring/i)).toBeDefined();
    expect(
      screen.getByText(/Crowd density\/growth is under routine monitoring/i)
    ).toBeDefined();
  });

  it('6. Correctly renders distance as straight-line and does NOT fabricate ETA', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncident] }),
        });
      }
      if (url.includes('/response')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockJointPlan,
        });
      }
      return Promise.reject(new Error('Unknown url'));
    });

    render(
      <IncidentProvider>
        <MapFocusProvider>
          <RecommendedResponseCard />
        </MapFocusProvider>
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/1.2 km/i)).toBeDefined();
      expect(screen.getByText(/1.6 km/i)).toBeDefined();
    });

    // Ensure no fake arrival minutes or fake ETA are fabricated in the output
    const containerText = document.body.textContent || '';
    expect(containerText).not.toContain('min away');
    expect(containerText).not.toContain('ETA:');
    expect(containerText).not.toContain('Fastest Route');
  });

  it('7. Handles 404 / plan unavailable gracefully while preserving incident context', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncident] }),
        });
      }
      if (url.includes('/response')) {
        return Promise.resolve({
          ok: false,
          status: 404,
        });
      }
      return Promise.reject(new Error('Unknown url'));
    });

    render(
      <IncidentProvider>
        <MapFocusProvider>
          <RecommendedResponseCard />
        </MapFocusProvider>
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/RESPONSE PLAN UNAVAILABLE/i)).toBeDefined();
    });

    // Verify retained incident context
    expect(screen.getByText(/Retained Incident Context:/i)).toBeDefined();
  });

  it('8. Supports map focus trigger when operator clicks Focus on Map', async () => {
    (fetch as any).mockImplementation((url: string) => {
      if (url.includes('/incidents?limit=')) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ count: 1, incidents: [mockIncident] }),
        });
      }
      if (url.includes('/response')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockJointPlan,
        });
      }
      return Promise.reject(new Error('Unknown url'));
    });

    const onExpandMock = vi.fn();

    render(
      <IncidentProvider>
        <MapFocusProvider>
          <RecommendedResponseCard onExpandMap={onExpandMock} />
        </MapFocusProvider>
      </IncidentProvider>
    );

    await waitFor(() => {
      expect(screen.getAllByText(/Focus on Map/i).length).toBeGreaterThan(0);
    });

    const focusButtons = screen.getAllByText(/Focus on Map/i);
    fireEvent.click(focusButtons[0]);

    expect(onExpandMock).toHaveBeenCalled();
  });
});
