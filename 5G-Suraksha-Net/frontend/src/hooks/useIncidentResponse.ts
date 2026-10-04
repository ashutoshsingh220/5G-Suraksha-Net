import { useState, useEffect } from 'react';
import type { IncidentResponse } from '../types/response';

interface UseIncidentResponseResult {
  responsePlan: IncidentResponse | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useIncidentResponse(
  incidentId: string | null | undefined,
  apiBaseUrl: string = 'http://localhost:8100'
): UseIncidentResponseResult {
  const [responsePlan, setResponsePlan] = useState<IncidentResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshTrigger, setRefreshTrigger] = useState<number>(0);

  const refetch = () => {
    setRefreshTrigger((prev) => prev + 1);
  };

  useEffect(() => {
    if (!incidentId) {
      setResponsePlan(null);
      setLoading(false);
      setError(null);
      return;
    }

    let isMounted = true;
    setLoading(true);
    setError(null);

    const controller = new AbortController();

    fetch(`${apiBaseUrl}/incidents/${incidentId}/response`, {
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) {
          if (res.status === 404) {
            throw new Error('Response plan not found for incident');
          }
          throw new Error(`HTTP error ${res.status}`);
        }
        return res.json();
      })
      .then((data: IncidentResponse) => {
        if (isMounted) {
          setResponsePlan(data);
          setLoading(false);
          setError(null);
        }
      })
      .catch((err) => {
        if (isMounted && err.name !== 'AbortError') {
          setResponsePlan(null);
          setLoading(false);
          setError(err.message || 'Failed to fetch response plan');
        }
      });

    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [incidentId, apiBaseUrl, refreshTrigger]);

  return { responsePlan, loading, error, refetch };
}
