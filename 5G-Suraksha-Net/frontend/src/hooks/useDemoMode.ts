import { useState, useEffect, useCallback } from 'react';
import type { DemoScenarioId, DemoScenarioMeta, DemoStatusResponse } from '../types/demo';

interface UseDemoModeReturn {
  demoStatus: DemoStatusResponse | null;
  scenarios: DemoScenarioMeta[];
  loading: boolean;
  actionLoading: boolean;
  error: string | null;
  startScenario: (scenarioId: DemoScenarioId, videoChoice?: string) => Promise<boolean>;
  resetDemo: () => Promise<boolean>;
  refresh: () => Promise<void>;
}

export const useDemoMode = (pollIntervalMs: number = 3000): UseDemoModeReturn => {
  const [demoStatus, setDemoStatus] = useState<DemoStatusResponse | null>(null);
  const [scenarios, setScenarios] = useState<DemoScenarioMeta[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [actionLoading, setActionLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8100/demo/status');
      if (res.ok) {
        const data: DemoStatusResponse = await res.json();
        setDemoStatus(data);
        setError(null);
      }
    } catch (err: any) {
      setError(err?.message || 'Demo status unavailable');
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchScenarios = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8100/demo/scenarios');
      if (res.ok) {
        const data: DemoScenarioMeta[] = await res.json();
        setScenarios(data);
      }
    } catch (err) {
      console.warn('Failed loading demo scenarios', err);
    }
  }, []);

  useEffect(() => {
    fetchScenarios();
    fetchStatus();
    const interval = setInterval(fetchStatus, pollIntervalMs);
    return () => clearInterval(interval);
  }, [fetchScenarios, fetchStatus, pollIntervalMs]);

  const startScenario = useCallback(
    async (scenarioId: DemoScenarioId, videoChoice?: string): Promise<boolean> => {
      setActionLoading(true);
      setError(null);
      try {
        const payload: { scenario_id: string; video_choice?: string } = { scenario_id: scenarioId };
        if (videoChoice) {
          payload.video_choice = videoChoice;
        }
        const res = await fetch('http://localhost:8100/demo/scenario', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          throw new Error(`Failed to activate scenario (${res.status})`);
        }
        const updated: DemoStatusResponse = await res.json();
        setDemoStatus(updated);
        return true;
      } catch (err: any) {
        setError(err?.message || 'Failed to start demo scenario');
        return false;
      } finally {
        setActionLoading(false);
      }
    },
    []
  );

  const resetDemo = useCallback(async (): Promise<boolean> => {
    setActionLoading(true);
    setError(null);
    try {
      const res = await fetch('http://localhost:8100/demo/reset', {
        method: 'POST',
      });
      if (!res.ok) {
        throw new Error(`Failed to reset demo (${res.status})`);
      }
      const updated: DemoStatusResponse = await res.json();
      setDemoStatus(updated);
      try {
        await fetch('http://localhost:8100/incidents/clear', { method: 'POST' });
      } catch {}
      return true;
    } catch (err: any) {
      setError(err?.message || 'Failed to reset demo');
      return false;
    } finally {
      setActionLoading(false);
    }
  }, []);

  return {
    demoStatus,
    scenarios,
    loading,
    actionLoading,
    error,
    startScenario,
    resetDemo,
    refresh: fetchStatus,
  };
};
