import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { NetworkPolicyCard } from '../NetworkPolicyCard';
import type { NetworkPolicyState, NetworkEventsResponse } from '../../../types/network';

describe('NetworkPolicyCard Component', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  it('1. Renders truthful NORMAL policy, ROUTINE priority, and NOT CONNECTED 5G plane', async () => {
    const mockPolicy: NetworkPolicyState = {
      policy: 'NORMAL',
      priority: 'ROUTINE',
      active_incident_id: null,
      active_incident_type: null,
      active_severity: null,
      reason: 'Routine monitoring active — no verified critical or elevated incidents.',
      actual_network_control: false,
      control_plane_connected: false,
      policy_scope: 'APPLICATION_LAYER',
      slice_allocated: 'DEFAULT_BE',
      measurement_source: 'APPLICATION_POLICY',
      timestamp: '2026-10-03T16:00:00Z',
    };

    const mockEvents: NetworkEventsResponse = {
      count: 1,
      events: [
        {
          event_id: 'pe-001',
          event_type: 'POLICY_INITIALIZED',
          policy: 'NORMAL',
          priority: 'ROUTINE',
          incident_id: null,
          details: 'Network policy engine initialized in NORMAL standby mode.',
          timestamp: '2026-10-03T16:00:00Z',
          actual_network_control: false,
        },
      ],
    };

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/network/policy')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockPolicy,
        });
      }
      if (url.includes('/network/events')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockEvents,
        });
      }
      return Promise.reject(new Error('Unknown url'));
    });

    render(<NetworkPolicyCard />);

    await waitFor(() => {
      expect(screen.getByText(/5G & NETWORK INTELLIGENCE/i)).toBeDefined();
    });

    expect(screen.getByText(/APPLICATION-LEVEL POLICY/i)).toBeDefined();
    expect(screen.getAllByText(/NORMAL/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/ROUTINE/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/NOT CONNECTED/i)).toBeDefined();
    expect(screen.getByText(/No physical slice controller attached/i)).toBeDefined();
  });

  it('2. Correctly renders escalated CRITICAL policy state with active trigger', async () => {
    const mockCriticalPolicy: NetworkPolicyState = {
      policy: 'CRITICAL',
      priority: 'CRITICAL',
      active_incident_id: 'inc-crit-99',
      active_incident_type: 'armed_fight',
      active_severity: 'CRITICAL',
      reason: 'Active CRITICAL incident [armed_fight] (#inc-crit-99) requires maximum transmission priority.',
      actual_network_control: false,
      control_plane_connected: false,
      policy_scope: 'APPLICATION_LAYER',
      slice_allocated: 'DEFAULT_BE',
      measurement_source: 'APPLICATION_POLICY',
      timestamp: '2026-10-03T16:05:00Z',
    };

    const mockEvents: NetworkEventsResponse = {
      count: 2,
      events: [
        {
          event_id: 'pe-002',
          event_type: 'POLICY_ESCALATED',
          policy: 'CRITICAL',
          priority: 'CRITICAL',
          incident_id: 'inc-crit-99',
          details: 'Policy transitioned from NORMAL to CRITICAL (priority: CRITICAL).',
          timestamp: '2026-10-03T16:05:00Z',
          actual_network_control: false,
        },
      ],
    };

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/network/policy')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockCriticalPolicy,
        });
      }
      if (url.includes('/network/events')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockEvents,
        });
      }
      return Promise.reject(new Error('Unknown url'));
    });

    render(<NetworkPolicyCard />);

    await waitFor(() => {
      expect(screen.getAllByText(/CRITICAL/i).length).toBeGreaterThan(0);
    });

    expect(screen.getByText(/#inc-crit-99 \(armed_fight\)/i)).toBeDefined();
    expect(screen.getByText(/requires maximum transmission priority/i)).toBeDefined();

    // Toggle audit events
    const toggleButton = screen.getByRole('button', { name: /Policy Audit Events/i });
    fireEvent.click(toggleButton);

    expect(screen.getByText(/ESCALATED/i)).toBeDefined();
    expect(screen.getByText(/Policy transitioned from NORMAL to CRITICAL/i)).toBeDefined();
  });
});
