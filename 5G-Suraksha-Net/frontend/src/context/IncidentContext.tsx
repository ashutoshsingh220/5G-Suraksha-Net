import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
} from 'react';
import type {
  IncidentReport,
  IncidentStreamConnectionStatus,
} from '../types/incidents';

interface IncidentContextValue {
  activeIncident: IncidentReport | null;
  incidentHistory: IncidentReport[];
  connectionStatus: IncidentStreamConnectionStatus;
  newIncidentHighlight: boolean;
  totalIncidentCount: number;
  selectIncident: (id: string) => void;
  clearActiveIncident: () => void;
  clearAllIncidents: () => Promise<void>;
  reconnect: () => void;
  refreshHistory: () => Promise<void>;
}

const IncidentContext = createContext<IncidentContextValue | undefined>(undefined);

const MAX_HISTORY = 100;
const INITIAL_RETRY_MS = 1500;
const MAX_RETRY_MS = 10000;

export const IncidentProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [activeIncident, setActiveIncident] = useState<IncidentReport | null>(null);
  const [incidentHistory, setIncidentHistory] = useState<IncidentReport[]>([]);
  const [connectionStatus, setConnectionStatus] =
    useState<IncidentStreamConnectionStatus>('CONNECTING');
  const [newIncidentHighlight, setNewIncidentHighlight] = useState<boolean>(false);

  const wsRef = useRef<WebSocket | null>(null);
  const retryCountRef = useRef<number>(0);
  const reconnectTimeoutRef = useRef<any>(null);
  const highlightTimeoutRef = useRef<any>(null);
  const isMountedRef = useRef<boolean>(true);

  // Trigger brief highlight on new incident
  const triggerHighlight = useCallback(() => {
    setNewIncidentHighlight(true);
    if (highlightTimeoutRef.current) {
      clearTimeout(highlightTimeoutRef.current);
    }
    highlightTimeoutRef.current = setTimeout(() => {
      if (isMountedRef.current) {
        setNewIncidentHighlight(false);
      }
    }, 4000);
  }, []);

  // Ingest an incident report (handles deduplication and updates)
  const handleIncomingIncident = useCallback(
    (report: IncidentReport, isInitialLoad = false) => {
      if (!report || !report.incident_id) return;

      setIncidentHistory((prev) => {
        const existingIdx = prev.findIndex((i) => i.incident_id === report.incident_id);
        if (existingIdx !== -1) {
          // Update existing incident in-place without duplicating
          const updated = [...prev];
          updated[existingIdx] = { ...updated[existingIdx], ...report };
          return updated;
        } else {
          // Prepend new incident, keeping max bounded history
          const next = [report, ...prev];
          if (next.length > MAX_HISTORY) {
            return next.slice(0, MAX_HISTORY);
          }
          return next;
        }
      });

      // Update active incident immediately to the newly detected or updated incident
      setActiveIncident(report);

      if (!isInitialLoad) {
        triggerHighlight();
      }
    },
    [triggerHighlight]
  );

  // Fetch initial history from REST endpoint
  const fetchInitialHistory = useCallback(async () => {
    try {
      const res = await fetch('http://localhost:8100/incidents?limit=100');
      if (res.ok) {
        const data = await res.json();
        const items: IncidentReport[] = data.incidents || [];
        if (Array.isArray(items) && items.length > 0) {
          // Sort newest first
          const sorted = [...items].sort((a, b) => {
            const tA = new Date(a.start_time).getTime();
            const tB = new Date(b.start_time).getTime();
            return tB - tA;
          });

          setIncidentHistory(sorted.slice(0, MAX_HISTORY));
          setActiveIncident((curr) => {
            if (!curr) return sorted[0];
            const currTime = new Date(curr.start_time).getTime();
            const newestTime = new Date(sorted[0].start_time).getTime();
            if (newestTime > currTime) {
              return sorted[0];
            }
            return curr;
          });
        } else {
          setIncidentHistory([]);
          setActiveIncident(null);
        }
      }
    } catch (err) {
      // Backend may be starting up; will retry or receive via WS
      console.warn('Initial incidents fetch pending backend availability');
    }
  }, []);

  const clearAllIncidents = useCallback(async () => {
    try {
      await fetch('http://localhost:8100/incidents/clear', { method: 'POST' });
    } catch (err) {
      console.warn('Failed clearing backend incidents', err);
    }
    setIncidentHistory([]);
    setActiveIncident(null);
  }, []);

  // Connect to /ws/incidents with bounded exponential backoff
  const connectWebSocket = useCallback(() => {
    if (!isMountedRef.current) return;

    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {}
      wsRef.current = null;
    }

    setConnectionStatus((prev) =>
      prev === 'CONNECTED' ? 'RECONNECTING' : 'CONNECTING'
    );

    try {
      const ws = new WebSocket('ws://localhost:8100/ws/incidents');
      wsRef.current = ws;

      ws.onopen = () => {
        if (!isMountedRef.current) return;
        setConnectionStatus('CONNECTED');
        retryCountRef.current = 0;
        // Fetch current list on connect to guarantee sync
        fetchInitialHistory();
      };

      ws.onmessage = (event) => {
        if (!isMountedRef.current) return;
        try {
          const parsed = JSON.parse(event.data);
          if (parsed && (parsed.event === 'incidents_cleared' || parsed.count === 0)) {
            setIncidentHistory([]);
            setActiveIncident(null);
            return;
          }
          const report: IncidentReport = parsed;
          handleIncomingIncident(report, false);
        } catch (e) {
          console.error('Malformed incident WebSocket payload:', e);
        }
      };

      ws.onerror = () => {
        if (!isMountedRef.current) return;
        setConnectionStatus('ERROR');
      };

      ws.onclose = () => {
        if (!isMountedRef.current) return;
        setConnectionStatus('RECONNECTING');

        // Calculate bounded backoff: 1.5s, 2.25s, 3.375s ... up to 10s
        const delay = Math.min(
          MAX_RETRY_MS,
          INITIAL_RETRY_MS * Math.pow(1.5, retryCountRef.current)
        );
        retryCountRef.current += 1;

        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, delay);
      };
    } catch {
      setConnectionStatus('DISCONNECTED');
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, INITIAL_RETRY_MS);
    }
  }, [fetchInitialHistory, handleIncomingIncident]);

  useEffect(() => {
    isMountedRef.current = true;
    fetchInitialHistory();
    connectWebSocket();

    const pollInterval = setInterval(() => {
      if (isMountedRef.current) {
        fetchInitialHistory();
      }
    }, 2500);

    return () => {
      isMountedRef.current = false;
      clearInterval(pollInterval);
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (highlightTimeoutRef.current) clearTimeout(highlightTimeoutRef.current);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket, fetchInitialHistory]);

  const selectIncident = useCallback((id: string) => {
    setIncidentHistory((prev) => {
      const found = prev.find((i) => i.incident_id === id);
      if (found) {
        setActiveIncident(found);
      } else {
        fetch(`http://localhost:8100/incidents/${id}`)
          .then((res) => (res.ok ? res.json() : null))
          .then((data) => {
            if (data && isMountedRef.current) {
              setActiveIncident(data);
            }
          })
          .catch(() => {});
      }
      return prev;
    });
  }, []);

  const clearActiveIncident = useCallback(() => {
    setActiveIncident(null);
  }, []);

  const value: IncidentContextValue = {
    activeIncident,
    incidentHistory,
    connectionStatus,
    newIncidentHighlight,
    totalIncidentCount: incidentHistory.length,
    selectIncident,
    clearActiveIncident,
    clearAllIncidents,
    reconnect: connectWebSocket,
    refreshHistory: fetchInitialHistory,
  };

  return (
    <IncidentContext.Provider value={value}>
      {children}
    </IncidentContext.Provider>
  );
};

export const useIncidents = (): IncidentContextValue => {
  const ctx = useContext(IncidentContext);
  if (!ctx) {
    throw new Error('useIncidents must be used within an IncidentProvider');
  }
  return ctx;
};
