import React, { useMemo } from 'react';
import {
  ArrowLeft,
  BarChart3,
  Calendar,
} from 'lucide-react';
import { useIncidents } from '../../context/IncidentContext';

interface OperationalAnalyticsViewProps {
  onBackToCommand: () => void;
}

export const OperationalAnalyticsView: React.FC<OperationalAnalyticsViewProps> = ({
  onBackToCommand,
}) => {
  const { incidentHistory, totalIncidentCount } = useIncidents();

  // Compute operational metrics from real incident records
  const metrics = useMemo(() => {
    const total = incidentHistory.length;
    if (total === 0) {
      return {
        total: 0,
        verified: 0,
        active: 0,
        criticalHigh: 0,
        evidenceCount: 0,
        evidenceRate: 0,
        avgConfidence: 0,
        byType: {} as Record<string, number>,
        bySeverity: { critical: 0, high: 0, moderate: 0, low: 0 },
        byCamera: {} as Record<string, number>,
        timelineBuckets: [] as { label: string; count: number }[],
      };
    }

    let verified = 0;
    let active = 0;
    let criticalHigh = 0;
    let evidenceCount = 0;
    let confSum = 0;

    const byType: Record<string, number> = {};
    const bySeverity = { critical: 0, high: 0, moderate: 0, low: 0 };
    const byCamera: Record<string, number> = {};
    const bucketsMap: Record<string, number> = {};

    incidentHistory.forEach((inc) => {
      // Status
      if (
        inc.status === 'verified' ||
        inc.status === 'finalized' ||
        inc.status === 'recording_post_event'
      ) {
        verified++;
      }
      if (
        inc.status === 'candidate' ||
        inc.status === 'recording_post_event' ||
        inc.status === 'verified'
      ) {
        active++;
      }

      // Severity
      const sev = String(inc.severity).toLowerCase();
      if (sev === 'critical') {
        bySeverity.critical++;
        criticalHigh++;
      } else if (sev === 'high') {
        bySeverity.high++;
        criticalHigh++;
      } else if (sev === 'medium' || sev === 'moderate') {
        bySeverity.moderate++;
      } else {
        bySeverity.low++;
      }

      // Evidence
      if (inc.evidence?.snapshot_path || inc.evidence?.clip_path) {
        evidenceCount++;
      }

      // Confidence
      confSum += inc.confidence || 0;

      // Type
      byType[inc.incident_type] = (byType[inc.incident_type] || 0) + 1;

      // Camera
      const cam = inc.camera_id || 'unknown';
      byCamera[cam] = (byCamera[cam] || 0) + 1;

      // Timeline bucket by date or hour
      try {
        const d = new Date(inc.start_time);
        const key = d.toISOString().slice(0, 10); // YYYY-MM-DD
        bucketsMap[key] = (bucketsMap[key] || 0) + 1;
      } catch {}
    });

    const timelineBuckets = Object.entries(bucketsMap)
      .map(([label, count]) => ({ label, count }))
      .sort((a, b) => a.label.localeCompare(b.label));

    return {
      total,
      verified,
      active,
      criticalHigh,
      evidenceCount,
      evidenceRate: Math.round((evidenceCount / total) * 100),
      avgConfidence: Math.round((confSum / total) * 100),
      byType,
      bySeverity,
      byCamera,
      timelineBuckets,
    };
  }, [incidentHistory]);

  const maxTimelineCount = Math.max(
    ...metrics.timelineBuckets.map((b) => b.count),
    1
  );

  return (
    <div className="flex-1 p-3 flex flex-col gap-3 max-w-[1920px] mx-auto w-full">
      {/* Top Header */}
      <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-2.5 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToCommand}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-[#111821] hover:bg-[#263341] border border-[#263341] rounded-[6px] text-xs font-sans font-medium text-[#98A6B5] hover:text-[#E8EDF3] transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Command Centre</span>
          </button>

          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xs font-semibold font-sans text-[#E8EDF3] uppercase tracking-wider">
                Operational Incident Analytics
              </h1>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-[4px] bg-[#2BC48A]/20 text-[#2BC48A] border border-[#2BC48A]/40 font-semibold">
                Live Data Driven
              </span>
            </div>
            <div className="text-[10px] font-mono text-[#98A6B5]">
              DERIVED EXCLUSIVELY FROM AUTHORITATIVE INCIDENT RECORDS &bull; ZERO FABRICATED DATA
            </div>
          </div>
        </div>

        <div className="text-right text-xs font-mono text-[#98A6B5]">
          <span className="text-[#687585]">Session Sample Size:</span>{' '}
          <span className="text-[#E8EDF3] font-semibold">{totalIncidentCount} incidents</span>
        </div>
      </div>

      {metrics.total === 0 ? (
        /* Empty State */
        <div className="flex-1 bg-[#111821] border border-[#263341] rounded-[8px] p-12 flex flex-col items-center justify-center text-center font-sans">
          <div className="w-12 h-12 rounded-[8px] bg-[#151E28] border border-[#263341] flex items-center justify-center text-[#3B9EFF] mb-3">
            <BarChart3 className="w-6 h-6" />
          </div>
          <div className="text-[#E8EDF3] text-sm font-semibold uppercase tracking-wider mb-1">
            NO INCIDENTS RECORDED IN SESSION
          </div>
          <p className="text-xs text-[#98A6B5] max-w-md leading-relaxed">
            The AI surveillance pipeline is actively monitoring video feeds on CUDA. Operational charts and analytics will populate automatically as soon as verified incidents are generated.
          </p>
        </div>
      ) : (
        /* Populated Analytics Dashboard */
        <div className="flex flex-col gap-3">
          {/* Key Stat Cards Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2.5 font-mono">
            <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-3">
              <div className="text-[#98A6B5] text-[10px] uppercase tracking-wider font-sans font-medium">
                Total Incidents
              </div>
              <div className="text-2xl font-bold text-[#E8EDF3] mt-1">
                {metrics.total}
              </div>
              <div className="text-[9.5px] text-[#3B9EFF] mt-0.5 font-sans">Authoritative Records</div>
            </div>

            <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-3">
              <div className="text-[#98A6B5] text-[10px] uppercase tracking-wider font-sans font-medium">
                Verified
              </div>
              <div className="text-2xl font-bold text-[#2BC48A] mt-1">
                {metrics.verified}
              </div>
              <div className="text-[9.5px] text-[#98A6B5] mt-0.5 font-sans">
                {Math.round((metrics.verified / metrics.total) * 100)}% verified rate
              </div>
            </div>

            <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-3">
              <div className="text-[#98A6B5] text-[10px] uppercase tracking-wider font-sans font-medium">
                High / Critical
              </div>
              <div className="text-2xl font-bold text-[#E05252] mt-1">
                {metrics.criticalHigh}
              </div>
              <div className="text-[9.5px] text-[#98A6B5] mt-0.5 font-sans">Urgent Response</div>
            </div>

            <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-3">
              <div className="text-[#98A6B5] text-[10px] uppercase tracking-wider font-sans font-medium">
                Evidence Rate
              </div>
              <div className="text-2xl font-bold text-[#49C6D9] mt-1">
                {metrics.evidenceRate}%
              </div>
              <div className="text-[9.5px] text-[#98A6B5] mt-0.5 font-sans">
                {metrics.evidenceCount} with artifact
              </div>
            </div>

            <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-3">
              <div className="text-[#98A6B5] text-[10px] uppercase tracking-wider font-sans font-medium">
                Avg Confidence
              </div>
              <div className="text-2xl font-bold text-[#3B9EFF] mt-1">
                {metrics.avgConfidence}%
              </div>
              <div className="text-[9.5px] text-[#98A6B5] mt-0.5 font-sans">Temporal Classifier</div>
            </div>

            <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-3">
              <div className="text-[#98A6B5] text-[10px] uppercase tracking-wider font-sans font-medium">
                Active / In-Flight
              </div>
              <div className="text-2xl font-bold text-[#E7A83B] mt-1">
                {metrics.active}
              </div>
              <div className="text-[9.5px] text-[#98A6B5] mt-0.5 font-sans">Live tracking</div>
            </div>
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
            {/* Chart 1: Severity Distribution */}
            <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-3.5 flex flex-col font-mono text-xs">
              <div className="flex items-center justify-between border-b border-[#263341] pb-2 mb-3">
                <span className="font-semibold text-[#E8EDF3] uppercase text-[11px] font-sans">
                  Severity Distribution
                </span>
                <span className="text-[10px] text-[#98A6B5] font-sans">Semantic Triage</span>
              </div>

              <div className="space-y-3 flex-1 justify-center flex flex-col">
                <div>
                  <div className="flex justify-between text-[11px] mb-1 font-sans">
                    <span className="text-[#E05252] font-semibold">CRITICAL</span>
                    <span className="text-[#E8EDF3] font-mono font-semibold">{metrics.bySeverity.critical}</span>
                  </div>
                  <div className="h-2 bg-[#111821] border border-[#263341] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#E05252] rounded-full"
                      style={{
                        width: `${(metrics.bySeverity.critical / metrics.total) * 100}%`,
                      }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] mb-1 font-sans">
                    <span className="text-[#E7A83B] font-semibold">HIGH</span>
                    <span className="text-[#E8EDF3] font-mono font-semibold">{metrics.bySeverity.high}</span>
                  </div>
                  <div className="h-2 bg-[#111821] border border-[#263341] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#E7A83B] rounded-full"
                      style={{
                        width: `${(metrics.bySeverity.high / metrics.total) * 100}%`,
                      }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] mb-1 font-sans">
                    <span className="text-[#E7A83B] font-semibold">MODERATE / MEDIUM</span>
                    <span className="text-[#E8EDF3] font-mono font-semibold">{metrics.bySeverity.moderate}</span>
                  </div>
                  <div className="h-2 bg-[#111821] border border-[#263341] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#E7A83B]/80 rounded-full"
                      style={{
                        width: `${(metrics.bySeverity.moderate / metrics.total) * 100}%`,
                      }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-[11px] mb-1 font-sans">
                    <span className="text-[#3B9EFF] font-semibold">LOW</span>
                    <span className="text-[#E8EDF3] font-mono font-semibold">{metrics.bySeverity.low}</span>
                  </div>
                  <div className="h-2 bg-[#111821] border border-[#263341] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#3B9EFF] rounded-full"
                      style={{
                        width: `${(metrics.bySeverity.low / metrics.total) * 100}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Chart 2: Incident Type Distribution */}
            <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-3.5 flex flex-col font-mono text-xs">
              <div className="flex items-center justify-between border-b border-[#263341] pb-2 mb-3">
                <span className="font-semibold text-[#E8EDF3] uppercase text-[11px] font-sans">
                  Incident Type Breakdown
                </span>
                <span className="text-[10px] text-[#98A6B5] font-sans">Class Distribution</span>
              </div>

              <div className="space-y-2.5 flex-1 justify-center flex flex-col overflow-y-auto">
                {Object.entries(metrics.byType).map(([type, count]) => {
                  const pct = Math.round((count / metrics.total) * 100);
                  return (
                    <div key={type}>
                      <div className="flex justify-between text-[10.5px] mb-1">
                        <span className="text-[#E8EDF3] font-medium uppercase font-sans">
                          {type.replace(/_/g, ' ')}
                        </span>
                        <span className="text-[#98A6B5] font-mono">
                          {count} ({pct}%)
                        </span>
                      </div>
                      <div className="h-2 bg-[#111821] border border-[#263341] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#3B9EFF] rounded-full"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Chart 3: Camera / Source Breakdown */}
            <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-3.5 flex flex-col font-mono text-xs">
              <div className="flex items-center justify-between border-b border-[#263341] pb-2 mb-3">
                <span className="font-semibold text-[#E8EDF3] uppercase text-[11px] font-sans">
                  Surveillance Sources
                </span>
                <span className="text-[10px] text-[#98A6B5] font-sans">Input Origins</span>
              </div>

              <div className="space-y-2.5 flex-1 justify-center flex flex-col">
                {Object.entries(metrics.byCamera).map(([cam, count]) => {
                  const pct = Math.round((count / metrics.total) * 100);
                  return (
                    <div key={cam}>
                      <div className="flex justify-between text-[10.5px] mb-1">
                        <span className="text-[#E8EDF3] font-medium font-sans">{cam}</span>
                        <span className="text-[#98A6B5] font-mono">
                          {count} ({pct}%)
                        </span>
                      </div>
                      <div className="h-2 bg-[#111821] border border-[#263341] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-[#49C6D9] rounded-full"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Volume Over Time Bar Chart */}
          <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-3.5 font-mono text-xs">
            <div className="flex items-center justify-between border-b border-[#263341] pb-2 mb-3">
              <div className="flex items-center gap-2">
                <Calendar className="w-3.5 h-3.5 text-[#3B9EFF]" />
                <span className="font-semibold text-[#E8EDF3] uppercase text-[11px] font-sans">
                  Incident Volume Timeline
                </span>
              </div>
              <span className="text-[10px] text-[#98A6B5] font-sans">
                Temporal Distribution
              </span>
            </div>

            {metrics.timelineBuckets.length === 0 ? (
              <div className="p-4 text-center text-[#687585] text-xs font-sans">
                No temporal distribution data available.
              </div>
            ) : (
              <div className="flex items-end gap-3 h-28 pt-4 pb-2 px-2 overflow-x-auto">
                {metrics.timelineBuckets.map((b) => {
                  const heightPercent = Math.max(
                    Math.round((b.count / maxTimelineCount) * 100),
                    8
                  );
                  return (
                    <div
                      key={b.label}
                      className="flex flex-col items-center gap-1 flex-1 min-w-[50px] max-w-[90px]"
                    >
                      <span className="text-[9.5px] text-[#E8EDF3] font-semibold">
                        {b.count}
                      </span>
                      <div className="w-full bg-[#111821] border border-[#263341] h-18 rounded-t flex items-end">
                        <div
                          className="w-full bg-[#3B9EFF] hover:bg-[#3B9EFF]/90 rounded-t transition-all"
                          style={{ height: `${heightPercent}%` }}
                          title={`${b.label}: ${b.count} incidents`}
                        />
                      </div>
                      <span className="text-[8.5px] text-[#98A6B5] truncate w-full text-center">
                        {b.label}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
