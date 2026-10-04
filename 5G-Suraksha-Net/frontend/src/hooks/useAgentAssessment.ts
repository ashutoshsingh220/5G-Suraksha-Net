import { useState, useEffect, useCallback } from 'react';
import type { AgentAssessment } from '../types/agents';

interface UseAgentAssessmentReturn {
  assessment: AgentAssessment | null;
  loading: boolean;
  error: string | null;
  refresh: (force?: boolean) => Promise<void>;
}

export const useAgentAssessment = (incidentId: string | null): UseAgentAssessmentReturn => {
  const [assessment, setAssessment] = useState<AgentAssessment | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAssessment = useCallback(
    async (force: boolean = false) => {
      if (!incidentId) {
        setAssessment(null);
        setError(null);
        setLoading(false);
        return;
      }

      setLoading(true);
      setError(null);
      try {
        const url = `http://localhost:8100/incidents/${encodeURIComponent(incidentId)}/assessment${
          force ? '?force_refresh=true' : ''
        }`;
        const res = await fetch(url);
        if (!res.ok) {
          if (res.status === 404) {
            throw new Error(`Incident #${incidentId} assessment not found`);
          }
          throw new Error(`Failed to fetch assessment (${res.status})`);
        }
        const data: AgentAssessment = await res.json();
        setAssessment(data);
        setError(null);
      } catch (err: any) {
        setError(err?.message || 'Failed to load agent assessment');
        setAssessment(null);
      } finally {
        setLoading(false);
      }
    },
    [incidentId]
  );

  useEffect(() => {
    fetchAssessment(false);
  }, [fetchAssessment]);

  return {
    assessment,
    loading,
    error,
    refresh: fetchAssessment,
  };
};
