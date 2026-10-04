import React from 'react';
import { AlertCircle, RotateCcw, Sliders } from 'lucide-react';
import type { DemoStatusResponse } from '../../types/demo';

interface DemoModeBannerProps {
  demoStatus: DemoStatusResponse | null;
  onOpenControlPanel: () => void;
  onReset: () => void;
  actionLoading?: boolean;
}

export const DemoModeBanner: React.FC<DemoModeBannerProps> = ({
  demoStatus,
  onOpenControlPanel,
  onReset,
  actionLoading = false,
}) => {
  if (!demoStatus?.is_demo_active) {
    return null;
  }

  const scenarioName = demoStatus.active_scenario?.replace(/_/g, ' ') || 'ACTIVE SIMULATION';

  return (
    <div
      role="banner"
      aria-label="Demonstration Mode Banner"
      className="bg-[#151E28] border-b border-[#E7A83B]/50 px-3.5 py-1.5 flex items-center justify-between z-30 select-none shadow-sm backdrop-blur-sm"
    >
      <div className="flex items-center gap-2.5">
        <div className="flex items-center gap-1.5 bg-[#E7A83B] text-[#0B0F14] font-semibold text-[10px] font-sans px-2 py-0.5 rounded-[4px] tracking-wider uppercase animate-pulse">
          <AlertCircle className="w-3 h-3 stroke-[3]" />
          <span>DEMO MODE ACTIVE</span>
        </div>

        <div className="hidden sm:flex items-center gap-2 text-xs font-mono">
          <span className="text-[#E7A83B] font-semibold uppercase tracking-wide">
            {scenarioName}
          </span>
          <span className="text-[#687585]">•</span>
          <span className="text-[#98A6B5] text-[10.5px] font-sans">
            SIMULATED INCIDENT DATA (NOT A LIVE CAMERA DETECTION) • HUMAN AUTHORIZATION MANDATORY
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={onOpenControlPanel}
          className="flex items-center gap-1 bg-[#111821] hover:bg-[#263341] text-[#E7A83B] border border-[#E7A83B]/40 text-[10.5px] font-sans font-medium px-2 py-0.5 rounded-[4px] transition-colors"
        >
          <Sliders className="w-3 h-3 text-[#E7A83B]" />
          <span>Scenarios</span>
        </button>

        <button
          onClick={onReset}
          disabled={actionLoading}
          className="flex items-center gap-1 bg-[#E7A83B] hover:bg-[#E7A83B]/90 disabled:opacity-50 text-[#0B0F14] font-semibold text-[10.5px] font-sans px-2.5 py-0.5 rounded-[4px] transition-colors"
        >
          <RotateCcw className={`w-3 h-3 ${actionLoading ? 'animate-spin' : ''}`} />
          <span>Reset Demo</span>
        </button>
      </div>
    </div>
  );
};
