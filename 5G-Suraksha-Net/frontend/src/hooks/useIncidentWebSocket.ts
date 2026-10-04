/**
 * Dedicated hook wrapping real-time incident WebSocket state.
 */
import { useIncidents } from '../context/IncidentContext';

export const useIncidentWebSocket = () => {
  return useIncidents();
};
