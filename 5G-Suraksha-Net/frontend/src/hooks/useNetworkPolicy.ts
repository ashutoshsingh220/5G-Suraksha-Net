import { useState, useEffect, useCallback } from 'react';
import type {
  NetworkPolicyState,
  NetworkEventMetric,
  NetworkEventsResponse,
} from '../types/network';

interface UseNetworkPolicyReturn {
  policyState: NetworkPolicyState | null;
  events: NetworkEventMetric[];
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export const useNetworkPolicy = (pollIntervalMs: number = 3000): UseNetworkPolicyReturn => {
  const [policyState, setPolicyState] = useState<NetworkPolicyState | null>(null);
  const [events, setEvents] = useState<NetworkEventMetric[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPolicy = useCallback(async () => {
    try {
      const [resPolicy, resEvents] = await Promise.all([
        fetch('http://localhost:8100/network/policy'),
        fetch('http://localhost:8100/network/events?limit=20'),
      ]);

      if (!resPolicy.ok) {
        throw new Error(`Failed to fetch policy (${resPolicy.status})`);
      }

      const policyData: NetworkPolicyState = await resPolicy.json();
      setPolicyState(policyData);

      if (resEvents.ok) {
        const eventsData: NetworkEventsResponse = await resEvents.json();
        setEvents(eventsData.events || []);
      }

      setError(null);
    } catch (err: any) {
      setError(err?.message || 'Network policy unreachable');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPolicy();
    const interval = setInterval(fetchPolicy, pollIntervalMs);
    return () => clearInterval(interval);
  }, [fetchPolicy, pollIntervalMs]);

  return {
    policyState,
    events,
    loading,
    error,
    refresh: fetchPolicy,
  };
};
