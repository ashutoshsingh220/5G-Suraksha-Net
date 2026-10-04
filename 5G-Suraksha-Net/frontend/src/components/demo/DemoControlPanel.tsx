import React from 'react';
import {
  X,
  Play,
  RotateCcw,
  ShieldAlert,
  Clock,
  CheckCircle,
  Info,
} from 'lucide-react';
import type { DemoScenarioId, DemoScenarioMeta, DemoStatusResponse } from '../../types/demo';

interface DemoControlPanelProps {
  isOpen: boolean;
  onClose: () => void;
  demoStatus: DemoStatusResponse | null;
  scenarios: DemoScenarioMeta[];
  actionLoading: boolean;
  onStartScenario: (scenarioId: DemoScenarioId, videoChoice?: string) => Promise<boolean>;
  onResetDemo: () => Promise<boolean>;
}

export const DemoControlPanel: React.FC<DemoControlPanelProps> = ({
  isOpen,
  onClose,
  demoStatus,
  scenarios,
  actionLoading,
  onStartScenario,
  onResetDemo,
}) => {
  if (!isOpen) return null;

  const isActive = Boolean(demoStatus?.is_demo_active);
  const activeScenId = demoStatus?.active_scenario;

  const [selectedClips, setSelectedClips] = React.useState<Record<string, string>>({
    ARMED_FIGHT: 'armedfight.mp4',
    WEAPON: 'weapondetection.mp4',
    CROWD_PANIC: 'crowd_panic.mp4',
  });

  const handleClipSelect = (scenarioId: string, clipId: string) => {
    setSelectedClips((prev) => ({ ...prev, [scenarioId]: clipId }));
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label="Demonstration Control Center"
      className="fixed inset-0 z-50 flex items-center justify-center p-3 bg-black/80 backdrop-blur-sm"
    >
      <div className="bg-[#111821] border border-[#263341] rounded-[8px] shadow-2xl max-w-3xl w-full max-h-[90vh] flex flex-col overflow-hidden text-[#E8EDF3] animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="px-4 py-3 border-b border-[#263341] bg-[#151E28] flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-[6px] bg-[#111821] border border-[#E7A83B]/40 flex items-center justify-center text-[#E7A83B]">
              <ShieldAlert className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-xs font-semibold font-sans tracking-wider text-[#E8EDF3] uppercase">
                IMC LIVE DEMONSTRATION & CONTROLLED SCENARIOS
              </h2>
              <p className="text-[10px] font-sans text-[#98A6B5]">
                Execute repeatable demonstration sequences without fabricating live camera detections
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-[4px] text-[#98A6B5] hover:text-[#E8EDF3] hover:bg-[#263341] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3.5 text-xs font-mono">
          {/* Safety & Operational Notice */}
          <div className="bg-[#151E28] border border-[#E7A83B]/40 rounded-[6px] p-2.5 flex items-start gap-2.5">
            <Info className="w-4 h-4 text-[#E7A83B] flex-shrink-0 mt-0.5" />
            <div className="text-[11px] leading-relaxed text-[#98A6B5] font-sans">
              <strong className="text-[#E7A83B] font-semibold">SAFETY & INTEGRITY POLICY: </strong>
              Simulated demonstration events are isolated in <code className="text-[#E8EDF3] bg-[#111821] border border-[#263341] px-1 py-0.5 rounded-[4px] font-mono text-[10px]">outputs/demo/</code>. Activating a scenario streams the selected video footage through live AI detection (YOLO11s, ByteTrack, crowd analytics, fight recognition) with real-time role-specific email evidence dispatch.
            </div>
          </div>

          {/* Current Status Strip */}
          <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-3 grid grid-cols-1 sm:grid-cols-3 gap-2 text-center">
            <div>
              <div className="text-[9.5px] text-[#98A6B5] uppercase font-sans font-medium">System Mode</div>
              <div className="font-semibold text-xs mt-0.5">
                {isActive ? (
                  <span className="text-[#E7A83B] flex items-center justify-center gap-1.5 font-sans">
                    <span className="w-2 h-2 rounded-full bg-[#E7A83B] animate-pulse" />
                    DEMONSTRATION (SIMULATED)
                  </span>
                ) : (
                  <span className="text-[#2BC48A] flex items-center justify-center gap-1.5 font-sans">
                    <span className="w-2 h-2 rounded-full bg-[#2BC48A]" />
                    LIVE OPERATIONAL
                  </span>
                )}
              </div>
            </div>
            <div>
              <div className="text-[9.5px] text-[#98A6B5] uppercase font-sans font-medium">Active Scenario</div>
              <div className="font-semibold text-xs mt-0.5 text-[#E8EDF3] font-sans">
                {activeScenId ? activeScenId.replace(/_/g, ' ') : 'NORMAL (STANDBY)'}
              </div>
            </div>
            <div>
              <div className="text-[9.5px] text-[#98A6B5] uppercase font-sans font-medium">Incident ID</div>
              <div className="font-mono text-xs mt-0.5 text-[#49C6D9]">
                {demoStatus?.incident?.incident_id || 'NONE'}
              </div>
            </div>
          </div>

          {/* Available Scenarios Grid */}
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wider text-[#98A6B5] mb-2 flex items-center justify-between font-sans">
              <span>Available Scenarios</span>
              <span className="text-[9px] text-[#687585] font-normal">Select a scenario to trigger</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
              {scenarios.map((scen) => {
                const isNormal = scen.id === 'NORMAL';
                const isCurrent = activeScenId === scen.id || (!isActive && isNormal);

                return (
                  <div
                    key={scen.id}
                    className={`rounded-[6px] border p-3 flex flex-col justify-between transition-all ${
                      isCurrent
                        ? 'bg-[#151E28] border-2 border-[#E7A83B]'
                        : 'bg-[#151E28] border-[#263341] hover:border-[#3B9EFF]/40'
                    }`}
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2 mb-1.5">
                        <span className="font-semibold text-[#E8EDF3] text-xs font-sans">{scen.name}</span>
                        {scen.severity && (
                          <span
                            className={`px-1.5 py-0.5 rounded-[4px] text-[8.5px] font-semibold uppercase font-mono ${
                              scen.severity === 'critical'
                                ? 'bg-[#E05252]/20 text-[#E05252] border border-[#E05252]/40'
                                : 'bg-[#E7A83B]/20 text-[#E7A83B] border border-[#E7A83B]/40'
                            }`}
                          >
                            {scen.severity}
                          </span>
                        )}
                        {isNormal && (
                          <span className="px-1.5 py-0.5 rounded-[4px] text-[8.5px] font-semibold uppercase font-mono bg-[#2BC48A]/20 text-[#2BC48A] border border-[#2BC48A]/40">
                            BASELINE
                          </span>
                        )}
                      </div>

                      <p className="text-[10px] text-[#98A6B5] mb-2 leading-relaxed font-sans">
                        {scen.description}
                      </p>

                      <div className="space-y-1 text-[9.5px] text-[#E8EDF3] mb-3 bg-[#111821] p-2 rounded-[4px] border border-[#263341] font-mono">
                        <div className="flex items-center justify-between">
                          <span className="text-[#98A6B5]">Site Location:</span>
                          <span className="text-[#E8EDF3]">{scen.location_name}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-[#98A6B5]">5G Policy / Priority:</span>
                          <span className="text-[#49C6D9] font-semibold">
                            {scen.expected_network_policy} / {scen.expected_network_priority}
                          </span>
                        </div>
                      </div>

                      {/* Video Footage Source & Clip Selector */}
                      {scen.video_choices && scen.video_choices.length > 0 && (
                        <div className="mb-3 bg-[#111821] p-2 rounded-[4px] border border-[#263341]">
                          <div className="text-[9.5px] text-[#98A6B5] font-sans font-medium mb-1.5 flex items-center justify-between">
                            <span>Demonstration Video Footage:</span>
                            {scen.video_choices.length > 1 && (
                              <span className="text-[8.5px] text-[#E7A83B] font-mono font-medium">Select Clip</span>
                            )}
                          </div>
                          {scen.video_choices.length > 1 ? (
                            <div className="grid grid-cols-2 gap-1.5">
                              {scen.video_choices.map((choice) => {
                                const isSelected = (selectedClips[scen.id] || scen.video_choices?.[0]?.id) === choice.id;
                                return (
                                  <button
                                    key={choice.id}
                                    type="button"
                                    disabled={actionLoading || isCurrent}
                                    onClick={() => handleClipSelect(scen.id, choice.id)}
                                    className={`px-2 py-1.5 text-[9.5px] rounded-[4px] font-mono border text-left transition-all ${
                                      isSelected
                                        ? 'bg-[#3B9EFF]/20 border-[#3B9EFF] text-[#E8EDF3] font-semibold'
                                        : 'bg-[#151E28] border-[#263341] text-[#98A6B5] hover:text-[#E8EDF3] hover:border-[#3B9EFF]/40'
                                    }`}
                                  >
                                    <div className="flex items-center gap-1.5">
                                      <span className={isSelected ? 'text-[#3B9EFF]' : 'text-[#687585]'}>●</span>
                                      <span className="truncate">{choice.label}</span>
                                    </div>
                                  </button>
                                );
                              })}
                            </div>
                          ) : (
                            <div className="text-[10px] text-[#49C6D9] font-mono flex items-center gap-1.5">
                              <span>📹</span>
                              <span>{scen.video_choices[0].label}</span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    <button
                      onClick={() => {
                        if (isNormal) {
                          onResetDemo();
                        } else if (scen.video_choices && scen.video_choices.length > 0) {
                          onStartScenario(scen.id, selectedClips[scen.id] || scen.video_choices[0].id);
                        } else {
                          onStartScenario(scen.id);
                        }
                      }}
                      disabled={actionLoading || isCurrent}
                      className={`w-full py-1.5 px-3 rounded-[6px] flex items-center justify-center gap-1.5 text-xs font-semibold font-sans transition-all ${
                        isCurrent
                          ? 'bg-[#111821] text-[#E7A83B] border border-[#E7A83B]/50 cursor-default'
                          : isNormal
                          ? 'bg-[#111821] hover:bg-[#263341] text-[#E8EDF3] border border-[#263341]'
                          : 'bg-[#3B9EFF] hover:bg-[#3B9EFF]/90 text-[#0B0F14]'
                      } disabled:opacity-50`}
                    >
                      {isCurrent ? (
                        <>
                          <CheckCircle className="w-3.5 h-3.5 text-[#E7A83B]" />
                          <span>Currently Active</span>
                        </>
                      ) : (
                        <>
                          <Play className="w-3.5 h-3.5" />
                          <span>{isNormal ? 'Reset to Baseline' : 'Activate Scenario'}</span>
                        </>
                      )}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Simulation Timeline (When Active) */}
          {Boolean(demoStatus?.simulation_timeline?.length) && (
            <div className="bg-[#151E28] border border-[#263341] rounded-[6px] p-3">
              <div className="flex items-center gap-1.5 text-[#E8EDF3] font-semibold text-xs uppercase mb-2.5 font-sans">
                <Clock className="w-3.5 h-3.5 text-[#3B9EFF]" />
                <span>Deterministic Simulation Timeline</span>
              </div>

              <div className="space-y-1.5">
                {demoStatus?.simulation_timeline.map((step) => (
                  <div
                    key={step.step_id}
                    className="flex items-start gap-2 bg-[#111821] p-1.5 rounded-[4px] border border-[#263341] text-[10px]"
                  >
                    <span className="text-[#3B9EFF] font-mono font-semibold flex-shrink-0 w-14">
                      {step.timestamp_label}
                    </span>
                    <span className="font-semibold text-[#E8EDF3] flex-shrink-0 w-36 font-sans">
                      {step.title}
                    </span>
                    <span className="text-[#98A6B5] flex-1 font-sans">{step.details}</span>
                    {step.status && (
                      <span className="text-[8.5px] px-1 py-0.5 rounded-[4px] bg-[#E7A83B]/20 text-[#E7A83B] border border-[#E7A83B]/40 font-semibold uppercase font-mono flex-shrink-0">
                        {step.status}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-2.5 border-t border-[#263341] bg-[#151E28] flex items-center justify-between flex-shrink-0">
          <button
            onClick={onResetDemo}
            disabled={actionLoading || !isActive}
            className="flex items-center gap-1.5 bg-[#151014] hover:bg-[#E05252]/20 disabled:opacity-40 text-[#E05252] border border-[#E05252]/50 text-xs font-semibold px-3 py-1.5 rounded-[6px] transition-colors font-sans"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${actionLoading ? 'animate-spin' : ''}`} />
            <span>Reset Demo & Clear Simulation</span>
          </button>

          <button
            onClick={onClose}
            className="bg-[#111821] hover:bg-[#263341] text-[#E8EDF3] border border-[#263341] text-xs font-semibold px-4 py-1.5 rounded-[6px] transition-colors font-sans"
          >
            Close Panel
          </button>
        </div>
      </div>
    </div>
  );
};
