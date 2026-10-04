import { AppShell } from './components/layout/AppShell';
import { IncidentProvider } from './context/IncidentContext';
import { TelemetryProvider } from './context/TelemetryContext';
import { MapFocusProvider } from './context/MapFocusContext';

export function App() {
  return (
    <IncidentProvider>
      <TelemetryProvider>
        <MapFocusProvider>
          <AppShell />
        </MapFocusProvider>
      </TelemetryProvider>
    </IncidentProvider>
  );
}

export default App;
