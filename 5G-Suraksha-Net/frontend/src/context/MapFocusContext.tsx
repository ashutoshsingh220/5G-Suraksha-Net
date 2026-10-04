import React, { createContext, useContext, useState, useCallback } from 'react';

export interface MapFocusTarget {
  latitude: number;
  longitude: number;
  label?: string;
  zoom?: number;
  facilityType?: 'police' | 'hospital' | 'incident' | 'drone';
  timestamp: number;
}

interface MapFocusContextValue {
  focusTarget: MapFocusTarget | null;
  focusLocation: (target: {
    latitude: number;
    longitude: number;
    label?: string;
    zoom?: number;
    facilityType?: 'police' | 'hospital' | 'incident' | 'drone';
  }) => void;
  clearFocus: () => void;
}

const MapFocusContext = createContext<MapFocusContextValue | undefined>(undefined);

export const MapFocusProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [focusTarget, setFocusTarget] = useState<MapFocusTarget | null>(null);

  const focusLocation = useCallback(
    (target: {
      latitude: number;
      longitude: number;
      label?: string;
      zoom?: number;
      facilityType?: 'police' | 'hospital' | 'incident' | 'drone';
    }) => {
      setFocusTarget({
        ...target,
        timestamp: Date.now(),
      });
    },
    []
  );

  const clearFocus = useCallback(() => {
    setFocusTarget(null);
  }, []);

  return (
    <MapFocusContext.Provider value={{ focusTarget, focusLocation, clearFocus }}>
      {children}
    </MapFocusContext.Provider>
  );
};

export const useMapFocus = (): MapFocusContextValue => {
  const ctx = useContext(MapFocusContext);
  if (!ctx) {
    throw new Error('useMapFocus must be used within a MapFocusProvider');
  }
  return ctx;
};
