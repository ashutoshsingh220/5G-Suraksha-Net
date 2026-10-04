import React, { useState } from 'react';
import { TopNavigation, type NavTabId } from './TopNavigation';
import { SystemStatusBar } from './SystemStatusBar';
import { AerialIntelligence } from '../aerial/AerialIntelligence';
import { VideoSources } from '../sources/VideoSources';
import { LiveVideoPanel } from '../video/LiveVideoPanel';
import { IncidentDetectedCard } from '../incidents/IncidentDetectedCard';
import { LocationIntelligenceCard } from '../location/LocationIntelligenceCard';
import { RecommendedResponseCard } from '../response/RecommendedResponseCard';
import { IncidentFeedTable } from '../feed/IncidentFeedTable';
import { IncidentEvidencePanel } from '../evidence/IncidentEvidencePanel';
import { LiveMapView } from '../map/LiveMapView';
import { IncidentHistoryView } from '../incidents/IncidentHistoryView';
import { DemoModeBanner } from '../demo/DemoModeBanner';
import { DemoControlPanel } from '../demo/DemoControlPanel';
import { useDemoMode } from '../../hooks/useDemoMode';

export const AppShell: React.FC = () => {
  const [activeTab, setActiveTab] = useState<NavTabId>('command');
  const [selectedSourceId, setSelectedSourceId] = useState<string>('drone-cam');
  const [isDemoModalOpen, setIsDemoModalOpen] = useState(false);
  const { demoStatus, scenarios, actionLoading, startScenario, resetDemo } = useDemoMode();

  return (
    <div className="min-h-screen bg-[#0F172A] text-[#E8EDF3] flex flex-col select-none overflow-x-hidden bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(59,130,246,0.06),rgba(255,255,255,0))]">
      {/* 1. Top Tactical Navigation Bar (48px fixed) */}
      <TopNavigation
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        isDemoActive={demoStatus?.is_demo_active}
        onOpenDemoPanel={() => setIsDemoModalOpen(true)}
      />

      {/* 2. System Status & Metrics Strip (30px fixed) */}
      <SystemStatusBar />

      {/* 2b. High-Visibility Demonstration Mode Banner (When Active) */}
      <DemoModeBanner
        demoStatus={demoStatus}
        onOpenControlPanel={() => setIsDemoModalOpen(true)}
        onReset={resetDemo}
        actionLoading={actionLoading}
      />

      {/* 3. Main Workspace: Command Centre, Live Map, or Incident Archive */}
      {activeTab === 'map' ? (
        <LiveMapView onBackToCommand={() => setActiveTab('command')} />
      ) : activeTab === 'incidents' ? (
        <IncidentHistoryView onBackToCommand={() => setActiveTab('command')} />
      ) : (
        <main className="flex-1 p-3 flex flex-col gap-3.5 max-w-[1920px] mx-auto w-full">
          {/* Upper Hero Operational Area: Dominant Live Center Video + Tactical Feeds */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 min-h-[580px] xl:min-h-[640px]">
            {/* Left Column: Aerial Intelligence + Video Sources (3 cols) */}
            <div className="lg:col-span-3 flex flex-col gap-3 min-h-0">
              <div className="flex-1 min-h-[280px]">
                <AerialIntelligence />
              </div>
              <div className="flex-1 min-h-[280px]">
                <VideoSources
                  selectedSourceId={selectedSourceId}
                  onSelectSource={setSelectedSourceId}
                />
              </div>
            </div>

            {/* Center Column: Live AI Video Centerpiece (Dominant 6 cols, Generous Height & Width) */}
            <div className="lg:col-span-6 flex flex-col min-h-[560px] xl:min-h-[620px]">
              <LiveVideoPanel
                selectedSourceId={selectedSourceId}
                onSelectSource={setSelectedSourceId}
              />
            </div>

            {/* Right Column: Incident Detected + Location + Response (3 cols) */}
            <div className="lg:col-span-3 flex flex-col gap-3 min-h-0">
              <div className="flex-shrink-0">
                <IncidentDetectedCard />
              </div>
              <div className="flex-1 min-h-[250px]">
                <LocationIntelligenceCard onExpandMap={() => setActiveTab('map')} />
              </div>
              <div className="flex-1 min-h-[240px]">
                <RecommendedResponseCard onExpandMap={() => setActiveTab('map')} />
              </div>
            </div>
          </div>

          {/* Section Divider for Supporting Operational Modules */}
          <div className="flex items-center gap-2.5 px-1 pt-2 pb-0.5">
            <div className="w-1.5 h-1.5 rounded-full bg-[#3B9EFF]" />
            <span className="text-[11px] font-mono font-semibold tracking-wider text-[#98A6B5] uppercase">
              SUPPORTING OPERATIONAL AUDIT LOGS & FORENSIC EVIDENCE
            </span>
            <div className="flex-1 h-[1px] bg-[#263341]" />
            <span className="text-[10px] font-mono text-[#687585]">
              SCROLLABLE SUPPORTING MODULES
            </span>
          </div>

          {/* Supporting Operational Logs & Forensic Evidence Area (7 + 5 cols) */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 h-[380px] pb-4">
            {/* Live Incident Feed & Timeline (7 cols) */}
            <div className="lg:col-span-7 flex flex-col h-full overflow-hidden">
              <IncidentFeedTable />
            </div>

            {/* Incident Evidence Snapshots / Clips (5 cols) */}
            <div className="lg:col-span-5 flex flex-col h-full overflow-hidden">
              <IncidentEvidencePanel />
            </div>
          </div>
        </main>
      )}

      {/* 4. Restrained Tactical Footer Strip (20px fixed) */}
      <footer className="h-[20px] bg-[#0B0F14] border-t border-[#263341] px-3 flex items-center justify-between text-[9px] font-mono text-[#687585] flex-shrink-0">
        <span>5G SURAKSHA-NET &bull; CROWD & FIGHT SURVEILLANCE MODULE</span>
        <span>DECISION SUPPORT ONLY &bull; HUMAN VERIFICATION MANDATORY</span>
        <span>YASHOBHOOMI SECTOR-25, NEW DELHI</span>
      </footer>

      {/* 5. Controlled Scenario Launcher & Demo Control Center Modal */}
      <DemoControlPanel
        isOpen={isDemoModalOpen}
        onClose={() => setIsDemoModalOpen(false)}
        demoStatus={demoStatus}
        scenarios={scenarios}
        actionLoading={actionLoading}
        onStartScenario={startScenario}
        onResetDemo={resetDemo}
      />
    </div>
  );
};
