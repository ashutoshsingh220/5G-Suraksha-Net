import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DemoModeBanner } from '../DemoModeBanner';
import { DemoControlPanel } from '../DemoControlPanel';
import type { DemoScenarioMeta, DemoStatusResponse } from '../../../types/demo';

const mockScenarios: DemoScenarioMeta[] = [
  {
    id: 'WEAPON',
    name: 'Weapon Detection (High)',
    description: 'Simulated weapon detection; triggers elevated network priority and police response.',
    incident_type: 'weapon',
    severity: 'high',
    expected_network_policy: 'EVENT',
    expected_network_priority: 'HIGH',
    location_name: 'Yashobhoomi, Dwarka Sector 25, New Delhi',
  },
  {
    id: 'CROWD_PANIC',
    name: 'Crowd Panic (High)',
    description: 'Simulated crowd surge and panic movement.',
    incident_type: 'crowd_panic',
    severity: 'high',
    expected_network_policy: 'EVENT',
    expected_network_priority: 'HIGH',
    location_name: 'Yashobhoomi, Dwarka Sector 25, New Delhi',
  },
  {
    id: 'ARMED_FIGHT',
    name: 'Armed Fight (Critical)',
    description: 'Simulated violent altercation; triggers critical priority.',
    incident_type: 'armed_fight',
    severity: 'critical',
    expected_network_policy: 'CRITICAL',
    expected_network_priority: 'CRITICAL',
    location_name: 'Yashobhoomi, Dwarka Sector 25, New Delhi',
  },
  {
    id: 'NORMAL',
    name: 'Normal Monitoring (Standby)',
    description: 'Routine baseline monitoring.',
    incident_type: null,
    severity: null,
    expected_network_policy: 'NORMAL',
    expected_network_priority: 'ROUTINE',
    location_name: 'Command Base',
  },
];

const mockActiveStatus: DemoStatusResponse = {
  is_demo_active: true,
  active_scenario: 'WEAPON',
  source_mode: 'DEMO',
  incident: {
    schema_version: '1.0',
    source_module: 'crowd_fight',
    incident_id: 'demo-weapon-001',
    status: 'verified',
    camera_id: 'cam_imc_demo',
    incident_type: 'weapon',
    severity: 'high',
    confidence: 0.93,
    start_time: '2026-10-03T16:00:00Z',
    track_ids: [1, 2],
    details: { is_simulation: true },
    evidence: { snapshot_path: 'outputs/demo/snapshots/demo_evidence.jpg' },
    source_mode: 'DEMO',
  },
  simulation_timeline: [
    {
      step_id: 's1',
      timestamp_label: 'T+00.0s',
      title: 'SCENARIO ACTIVATED',
      details: 'Weapon detection scenario initiated.',
      status: 'COMPLETED',
    },
    {
      step_id: 's2',
      timestamp_label: 'T+01.0s',
      title: 'INCIDENT VERIFIED',
      details: 'Confirmed weapon object at 93.0% confidence.',
      status: 'COMPLETED',
    },
  ],
};

const mockInactiveStatus: DemoStatusResponse = {
  is_demo_active: false,
  active_scenario: null,
  source_mode: 'REAL',
  incident: null,
  simulation_timeline: [],
};

describe('DemoModeBanner Component', () => {
  it('1. Does not render when demo mode is inactive', () => {
    const { container } = render(
      <DemoModeBanner
        demoStatus={mockInactiveStatus}
        onOpenControlPanel={vi.fn()}
        onReset={vi.fn()}
      />
    );
    expect(container.firstChild).toBeNull();
  });

  it('2. Renders prominent amber banner and scenario title when active', () => {
    const handleOpen = vi.fn();
    const handleReset = vi.fn();

    render(
      <DemoModeBanner
        demoStatus={mockActiveStatus}
        onOpenControlPanel={handleOpen}
        onReset={handleReset}
      />
    );

    expect(screen.getByText(/DEMO MODE ACTIVE/i)).toBeDefined();
    expect(screen.getByText(/WEAPON/i)).toBeDefined();
    expect(
      screen.getByText(/SIMULATED INCIDENT DATA \(NOT A LIVE CAMERA DETECTION\)/i)
    ).toBeDefined();

    // Click scenarios button
    fireEvent.click(screen.getByRole('button', { name: /scenarios/i }));
    expect(handleOpen).toHaveBeenCalledTimes(1);

    // Click reset demo button
    fireEvent.click(screen.getByRole('button', { name: /reset demo/i }));
    expect(handleReset).toHaveBeenCalledTimes(1);
  });
});

describe('DemoControlPanel Component', () => {
  it('3. Renders scenarios, safety disclaimers, and triggers start scenario', () => {
    const handleStart = vi.fn().mockResolvedValue(true);
    const handleReset = vi.fn().mockResolvedValue(true);
    const handleClose = vi.fn();

    render(
      <DemoControlPanel
        isOpen={true}
        onClose={handleClose}
        demoStatus={mockInactiveStatus}
        scenarios={mockScenarios}
        actionLoading={false}
        onStartScenario={handleStart}
        onResetDemo={handleReset}
      />
    );

    expect(
      screen.getByText(/IMC LIVE DEMONSTRATION & CONTROLLED SCENARIOS/i)
    ).toBeDefined();
    expect(screen.getByText(/SAFETY & INTEGRITY POLICY:/i)).toBeDefined();
    expect(screen.getByText(/Weapon Detection \(High\)/i)).toBeDefined();
    expect(screen.getByText(/Crowd Panic \(High\)/i)).toBeDefined();
    expect(screen.getByText(/Armed Fight \(Critical\)/i)).toBeDefined();

    // Trigger weapon scenario
    const buttons = screen.getAllByRole('button', { name: /activate scenario/i });
    expect(buttons.length).toBeGreaterThan(0);
    fireEvent.click(buttons[0]);
    expect(handleStart).toHaveBeenCalledWith('WEAPON');

    // Close panel
    fireEvent.click(screen.getByRole('button', { name: /close panel/i }));
    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it('4. Renders simulation timeline and active badge when scenario is active', () => {
    const handleReset = vi.fn().mockResolvedValue(true);

    render(
      <DemoControlPanel
        isOpen={true}
        onClose={vi.fn()}
        demoStatus={mockActiveStatus}
        scenarios={mockScenarios}
        actionLoading={false}
        onStartScenario={vi.fn()}
        onResetDemo={handleReset}
      />
    );

    expect(screen.getByText(/DEMONSTRATION \(SIMULATED\)/i)).toBeDefined();
    expect(screen.getByText(/demo-weapon-001/i)).toBeDefined();
    expect(screen.getByText(/SCENARIO ACTIVATED/i)).toBeDefined();
    expect(screen.getByText(/INCIDENT VERIFIED/i)).toBeDefined();

    // Click reset demo & clear simulation button
    fireEvent.click(
      screen.getByRole('button', { name: /reset demo & clear simulation/i })
    );
    expect(handleReset).toHaveBeenCalledTimes(1);
  });
});
