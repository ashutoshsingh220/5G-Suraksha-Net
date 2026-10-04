import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { AgentAssessmentCard } from '../AgentAssessmentCard';
import type { AgentAssessment } from '../../../types/agents';

describe('AgentAssessmentCard Component', () => {
  const originalFetch = globalThis.fetch;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    globalThis.fetch = originalFetch;
  });

  const mockCriticalAssessment: AgentAssessment = {
    incident_id: 'inc-wpn-101',
    situation_summary:
      'Verified CRITICAL-severity weapon incident detected at Yashobhoomi Convention Centre (AI confidence: 94.0%). Forensic snapshot is secured and video clip is stored. The deterministic response planner recommends immediate police/security coordination; medical assistance remains conditional on verified injury. Application network priority is CRITICAL. Human operator approval is strictly mandatory before emergency dispatch.',
    severity_assessment: {
      severity: 'CRITICAL',
      explanation: "Classified as CRITICAL due to verified high-risk event 'weapon' with 94.0% AI confidence.",
      is_deterministic: true,
      incident_type: 'weapon',
    },
    location_assessment: {
      location_name: 'Yashobhoomi Convention Centre, Sector 25, Dwarka, New Delhi',
      latitude: 28.5528,
      longitude: 77.0601,
      nearby_police_count: 3,
      nearby_hospital_count: 3,
      closest_police_facility: 'Dwarka Sector 23 Police Station',
      closest_police_distance_km: 1.25,
      closest_hospital_facility: 'Indira Gandhi Hospital',
      closest_hospital_distance_km: 2.1,
      gps_fix_available: true,
      summary: 'Incident localized to Yashobhoomi Convention Centre.',
    },
    evidence_assessment: {
      snapshot_available: true,
      snapshot_path: 'outputs/snapshots/inc-wpn-101.jpg',
      video_available: true,
      video_path: 'outputs/clips/inc-wpn-101.mp4',
      summary: 'Forensic artifacts: Snapshot AVAILABLE, 10-second video clip AVAILABLE.',
    },
    response_assessment: {
      status: 'AWAITING_HUMAN_APPROVAL',
      police_recommended: true,
      medical_recommended: false,
      medical_conditional: true,
      actions_summary: 'Deterministic Response Planner recommends: Dispatch Police/Security unit.',
      human_approval_required: true,
    },
    network_assessment: {
      policy: 'CRITICAL',
      priority: 'CRITICAL',
      actual_network_control: false,
      control_plane_connected: false,
      summary: 'Application network policy: CRITICAL (Data Priority Tier: CRITICAL).',
    },
    recommended_sequence: [
      {
        step_number: 1,
        title: 'VERIFY INCIDENT',
        description: 'Operator inspection of live video and target bounding box.',
        status: 'COMPLETED',
        advisory_only: true,
      },
      {
        step_number: 2,
        title: 'PRESERVE EVIDENCE',
        description: 'Secure forensic snapshot and locked 10-second MP4 clip.',
        status: 'COMPLETED',
        advisory_only: true,
      },
      {
        step_number: 3,
        title: 'COORDINATE POLICE RESPONSE',
        description: 'Notify sector security with incident coordinates.',
        status: 'AWAITING_APPROVAL',
        advisory_only: true,
      },
    ],
    missing_information: [
      'Medical injury status not confirmed from optical CCTV feed; triage conditional upon ground verification.',
    ],
    data_sources: {
      incident_report: true,
      response_planner: true,
      location_intelligence: true,
      evidence_metadata: true,
      network_policy: true,
    },
    confidence: 0.94,
    human_approval_required: true,
    generated_at: '2026-10-03T16:00:00Z',
    source: 'DETERMINISTIC_SUPERVISOR_AGENT',
    disclaimer:
      'Agentic Intelligence provides situational briefing and decision support only. Deterministic safety logic is authoritative. Human verification and approval is strictly mandatory.',
  };

  it('1. Renders standby state when no incidentId is passed', () => {
    render(<AgentAssessmentCard incidentId={null} />);
    expect(screen.getByText(/Agentic Orchestration Standby/i)).toBeDefined();
  });

  it('2. Successfully fetches and renders complete structured assessment', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockCriticalAssessment,
    });

    render(<AgentAssessmentCard incidentId="inc-wpn-101" />);

    await waitFor(() => {
      expect(screen.getByText(/AGENTIC INTELLIGENCE BRIEFING/i)).toBeDefined();
    });

    // Check Executive Briefing
    expect(screen.getByText(/Verified CRITICAL-severity weapon incident/i)).toBeDefined();

    // Check Authoritative Severity
    expect(screen.getAllByText(/CRITICAL/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/AUTHORITATIVE DETERMINISTIC/i)).toBeDefined();

    // Check Mandatory Human Approval Gate
    expect(screen.getByText(/OPERATOR GATE: HUMAN APPROVAL REQUIRED/i)).toBeDefined();
    expect(screen.getByText(/Autonomous emergency dispatch is strictly prohibited/i)).toBeDefined();

    // Check Recommended Action Sequence
    expect(screen.getByText(/VERIFY INCIDENT/i)).toBeDefined();
    expect(screen.getByText(/PRESERVE EVIDENCE/i)).toBeDefined();
    expect(screen.getByText(/COORDINATE POLICE RESPONSE/i)).toBeDefined();

    // Check Information Gaps
    expect(screen.getByText(/Medical injury status not confirmed/i)).toBeDefined();

    // Check Data Sources Traceability
    expect(screen.getByText(/Incident Report: VERIFIED/i)).toBeDefined();
    expect(screen.getByText(/Response Planner: VERIFIED/i)).toBeDefined();
    expect(screen.getByText(/Network Policy: VERIFIED/i)).toBeDefined();

    // Check Network Priority & 5G Control Plane state
    expect(screen.getByText(/5G CONTROL PLANE: NOT CONNECTED/i)).toBeDefined();
  });

  it('3. Renders error message and retry button on fetch failure', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
    });

    render(<AgentAssessmentCard incidentId="inc-fail-001" />);

    await waitFor(() => {
      expect(screen.getByText(/Assessment Unavailable/i)).toBeDefined();
    });

    expect(screen.getByRole('button', { name: /Retry/i })).toBeDefined();
  });
});
