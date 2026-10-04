import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useRef,
} from 'react';
import type { DroneTelemetryData } from '../types/telemetry';
import { initialDroneTelemetry } from '../types/telemetry';

interface TelemetryContextValue {
  telemetry: DroneTelemetryData;
  connected: boolean;
}

const TelemetryContext = createContext<TelemetryContextValue | undefined>(undefined);

export const TelemetryProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [telemetry, setTelemetry] = useState<DroneTelemetryData>(initialDroneTelemetry);
  const [connected, setConnected] = useState<boolean>(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let isMounted = true;
    let reconnectTimer: any = null;
    let pollInterval: any = null;

    const connectWs = () => {
      if (!isMounted) return;
      try {
        const ws = new WebSocket('ws://localhost:8100/ws/telemetry');
        wsRef.current = ws;

        ws.onopen = () => {
          if (!isMounted) return;
          setConnected(true);
          if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
          }
        };

        ws.onmessage = (event) => {
          if (!isMounted) return;
          try {
            const data: DroneTelemetryData = JSON.parse(event.data);
            setTelemetry(data);
          } catch (e) {
            console.error('Failed to parse telemetry message', e);
          }
        };

        ws.onerror = () => {
          if (!isMounted) return;
          setConnected(false);
        };

        ws.onclose = () => {
          if (!isMounted) return;
          setConnected(false);
          startPolling();
          reconnectTimer = setTimeout(connectWs, 3000);
        };
      } catch {
        setConnected(false);
        startPolling();
        reconnectTimer = setTimeout(connectWs, 3000);
      }
    };

    const startPolling = () => {
      if (pollInterval) return;
      pollInterval = setInterval(async () => {
        if (!isMounted) return;
        try {
          const res = await fetch('http://localhost:8100/drone/telemetry');
          if (res.ok) {
            const data: DroneTelemetryData = await res.json();
            setTelemetry(data);
          }
        } catch {
          // Keep current state if server unreachable
        }
      }, 1500);
    };

    connectWs();

    return () => {
      isMounted = false;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (pollInterval) clearInterval(pollInterval);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <TelemetryContext.Provider value={{ telemetry, connected }}>
      {children}
    </TelemetryContext.Provider>
  );
};

export const useTelemetry = (): TelemetryContextValue => {
  const ctx = useContext(TelemetryContext);
  if (!ctx) {
    throw new Error('useTelemetry must be used within a TelemetryProvider');
  }
  return ctx;
};
