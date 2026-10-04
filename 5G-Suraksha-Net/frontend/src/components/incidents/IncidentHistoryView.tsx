import React, { useState, useMemo } from 'react';
import {
  ArrowLeft,
  Search,
  RefreshCw,
  Trash2,
  Camera,
  Video,
  ShieldCheck,
  Eye,
  Archive,
} from 'lucide-react';
import { useIncidents } from '../../context/IncidentContext';
import type { IncidentSeverity, IncidentType } from '../../types/incidents';
import { IncidentLifecycleTimeline } from './IncidentLifecycleTimeline';
import { IncidentEvidencePanel } from '../evidence/IncidentEvidencePanel';
import { RecommendedResponseCard } from '../response/RecommendedResponseCard';
import { AgentAssessmentCard } from '../agents/AgentAssessmentCard';

interface IncidentHistoryViewProps {
  onBackToCommand: () => void;
}

export const IncidentHistoryView: React.FC<IncidentHistoryViewProps> = ({
  onBackToCommand,
}) => {
  const { incidentHistory, activeIncident, selectIncident, refreshHistory, clearAllIncidents } =
    useIncidents();

  // Filters state
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedType, setSelectedType] = useState<string>('all');
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [selectedTimeRange, setSelectedTimeRange] = useState<string>('all');
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [detailTab, setDetailTab] = useState<'timeline' | 'agent' | 'evidence' | 'response'>('timeline');

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await refreshHistory();
    } finally {
      setIsRefreshing(false);
    }
  };

  const handleClearAll = async () => {
    if (window.confirm("Are you sure you want to permanently clear all incident logs and reset the counter to 0?")) {
      setIsRefreshing(true);
      try {
        await clearAllIncidents();
      } finally {
        setIsRefreshing(false);
      }
    }
  };

  // Filter logic
  const filteredIncidents = useMemo(() => {
    const now = Date.now();

    return incidentHistory.filter((inc) => {
      // 1. Text search
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        const matchesId = inc.incident_id.toLowerCase().includes(q);
        const matchesCam = inc.camera_id.toLowerCase().includes(q);
        const matchesLoc = inc.location?.name?.toLowerCase().includes(q);
        const matchesType = inc.incident_type.toLowerCase().includes(q);
        if (!matchesId && !matchesCam && !matchesLoc && !matchesType) {
          return false;
        }
      }

      // 2. Incident Type
      if (selectedType !== 'all' && inc.incident_type !== selectedType) {
        return false;
      }

      // 3. Severity
      if (selectedSeverity !== 'all') {
        const incSev = String(inc.severity).toLowerCase();
        if (selectedSeverity === 'medium' || selectedSeverity === 'moderate') {
          if (incSev !== 'medium' && incSev !== 'moderate') return false;
        } else if (incSev !== selectedSeverity) {
          return false;
        }
      }

      // 4. Status
      if (selectedStatus !== 'all' && inc.status !== selectedStatus) {
        return false;
      }

      // 5. Time range
      if (selectedTimeRange !== 'all') {
        try {
          const incTime = new Date(inc.start_time).getTime();
          const diffMs = now - incTime;

          if (selectedTimeRange === '15m' && diffMs > 15 * 60 * 1000) return false;
          if (selectedTimeRange === '1h' && diffMs > 60 * 60 * 1000) return false;
          if (selectedTimeRange === '24h' && diffMs > 24 * 60 * 60 * 1000) return false;
          if (selectedTimeRange === 'today') {
            const incDate = new Date(inc.start_time).toDateString();
            const todayDate = new Date().toDateString();
            if (incDate !== todayDate) return false;
          }
        } catch {
          // If parse fails, keep in list
        }
      }

      return true;
    });
  }, [
    incidentHistory,
    searchQuery,
    selectedType,
    selectedSeverity,
    selectedStatus,
    selectedTimeRange,
  ]);

  const getSeverityBadge = (sev: IncidentSeverity) => {
    const s = String(sev).toLowerCase();
    switch (s) {
      case 'critical':
        return (
          <span className="bg-[#E05252]/20 text-[#E05252] border border-[#E05252]/40 font-mono text-[9px] font-semibold px-1.5 py-0.5 rounded-[4px] tracking-wider uppercase">
            CRITICAL
          </span>
        );
      case 'high':
        return (
          <span className="bg-[#E7A83B]/20 text-[#E7A83B] border border-[#E7A83B]/40 font-mono text-[9px] font-semibold px-1.5 py-0.5 rounded-[4px] tracking-wider uppercase">
            HIGH
          </span>
        );
      case 'medium':
      case 'moderate':
        return (
          <span className="bg-[#E7A83B]/20 text-[#E7A83B] border border-[#E7A83B]/40 font-mono text-[9px] font-semibold px-1.5 py-0.5 rounded-[4px] tracking-wider uppercase">
            {s.toUpperCase()}
          </span>
        );
      case 'low':
      default:
        return (
          <span className="bg-[#151E28] text-[#98A6B5] border border-[#263341] font-mono text-[9px] font-semibold px-1.5 py-0.5 rounded-[4px] tracking-wider uppercase">
            LOW
          </span>
        );
    }
  };

  const formatIncidentType = (type: IncidentType): string => {
    switch (type) {
      case 'armed_fight':
        return 'ARMED FIGHT';
      case 'weapon':
        return 'WEAPON';
      case 'fight':
        return 'FIGHT';
      case 'crowd_panic':
        return 'CROWD PANIC';
      case 'crowd_density_critical':
        return 'DENSITY CRIT';
      case 'crowd_density_high':
        return 'DENSITY HIGH';
      case 'crowd_rapid_growth':
        return 'CROWD SURGE';
      default:
        return String(type).toUpperCase();
    }
  };

  const formatTime = (isoString: string): string => {
    try {
      const d = new Date(isoString);
      return (
        d.toLocaleTimeString('en-IN', {
          hour12: false,
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }) + ' IST'
      );
    } catch {
      return isoString;
    }
  };

  const formatDate = (isoString: string): string => {
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString('en-IN', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });
    } catch {
      return '';
    }
  };

  return (
    <div className="flex-1 p-3 flex flex-col gap-3 max-w-[1920px] mx-auto w-full">
      {/* Top Action Bar */}
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
                Incident History & Forensic Archive
              </h1>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-[4px] bg-[#111821] text-[#3B9EFF] border border-[#263341] font-semibold">
                {incidentHistory.length} Total Records
              </span>
            </div>
            <div className="text-[10px] font-mono text-[#98A6B5]">
              RUNTIME PERSISTED AUDIT LOG &bull; YASHOBHOOMI COMMAND SECTOR
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleClearAll}
            disabled={isRefreshing}
            className="flex items-center gap-1.5 px-2.5 py-1.5 bg-[#1A1118] hover:bg-[#E05252]/20 border border-[#E05252]/40 rounded-[6px] text-xs font-sans font-medium text-[#E05252] transition-colors disabled:opacity-50"
            title="Permanently remove all incident logs and restart counter at 0"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Logs (Reset to 0)</span>
          </button>

          <button
            onClick={handleRefresh}
            disabled={isRefreshing}
            className="flex items-center gap-1.5 px-2.5 py-1.5 bg-[#111821] hover:bg-[#263341] border border-[#263341] rounded-[6px] text-xs font-sans font-medium text-[#98A6B5] hover:text-[#E8EDF3] transition-colors disabled:opacity-50"
            title="Refresh from backend storage"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Filter Toolbar */}
      <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-2.5 flex flex-wrap items-center gap-2.5 text-xs font-mono">
        <div className="flex items-center gap-1.5 flex-1 min-w-[200px] bg-[#111821] border border-[#263341] rounded-[6px] px-2.5 py-1">
          <Search className="w-3.5 h-3.5 text-[#687585]" />
          <input
            type="text"
            placeholder="Search by ID, location, or camera..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="bg-transparent border-none text-[#E8EDF3] placeholder-[#687585] focus:outline-none text-xs w-full font-mono"
          />
        </div>

        {/* Time Range Filter */}
        <div className="flex items-center gap-1.5">
          <span className="text-[#98A6B5] text-[10px] uppercase font-sans font-medium">Time:</span>
          <select
            value={selectedTimeRange}
            onChange={(e) => setSelectedTimeRange(e.target.value)}
            className="bg-[#111821] border border-[#263341] text-[#E8EDF3] rounded-[6px] px-2 py-1 text-xs focus:outline-none focus:border-[#3B9EFF]"
          >
            <option value="all">All Records</option>
            <option value="15m">Last 15 Minutes</option>
            <option value="1h">Last 1 Hour</option>
            <option value="24h">Last 24 Hours</option>
            <option value="today">Today</option>
          </select>
        </div>

        {/* Type Filter */}
        <div className="flex items-center gap-1.5">
          <span className="text-[#98A6B5] text-[10px] uppercase font-sans font-medium">Type:</span>
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="bg-[#111821] border border-[#263341] text-[#E8EDF3] rounded-[6px] px-2 py-1 text-xs focus:outline-none focus:border-[#3B9EFF]"
          >
            <option value="all">All Types</option>
            <option value="fight">Fight</option>
            <option value="armed_fight">Armed Fight</option>
            <option value="weapon">Weapon</option>
            <option value="crowd_density_high">Crowd Density High</option>
            <option value="crowd_density_critical">Crowd Density Critical</option>
            <option value="crowd_rapid_growth">Crowd Rapid Surge</option>
            <option value="crowd_panic">Crowd Panic</option>
          </select>
        </div>

        {/* Severity Filter */}
        <div className="flex items-center gap-1.5">
          <span className="text-[#98A6B5] text-[10px] uppercase font-sans font-medium">Severity:</span>
          <select
            value={selectedSeverity}
            onChange={(e) => setSelectedSeverity(e.target.value)}
            className="bg-[#111821] border border-[#263341] text-[#E8EDF3] rounded-[6px] px-2 py-1 text-xs focus:outline-none focus:border-[#3B9EFF]"
          >
            <option value="all">All Severities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="moderate">Moderate / Medium</option>
            <option value="low">Low</option>
          </select>
        </div>

        {/* Status Filter */}
        <div className="flex items-center gap-1.5">
          <span className="text-[#98A6B5] text-[10px] uppercase font-sans font-medium">Status:</span>
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="bg-[#111821] border border-[#263341] text-[#E8EDF3] rounded-[6px] px-2 py-1 text-xs focus:outline-none focus:border-[#3B9EFF]"
          >
            <option value="all">All Statuses</option>
            <option value="verified">Verified</option>
            <option value="finalized">Finalized</option>
            <option value="recording_post_event">Recording Post-Event</option>
            <option value="candidate">Candidate</option>
          </select>
        </div>

        {(searchQuery ||
          selectedType !== 'all' ||
          selectedSeverity !== 'all' ||
          selectedStatus !== 'all' ||
          selectedTimeRange !== 'all') && (
          <button
            onClick={() => {
              setSearchQuery('');
              setSelectedType('all');
              setSelectedSeverity('all');
              setSelectedStatus('all');
              setSelectedTimeRange('all');
            }}
            className="text-[10px] text-[#3B9EFF] hover:underline font-mono"
          >
            Clear Filters
          </button>
        )}
      </div>

      {/* Main Content Grid: Left Table (7 cols) + Right Detail (5 cols) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 flex-1 min-h-[550px]">
        {/* Left Column: Historical Records Table */}
        <div className="lg:col-span-7 bg-[#111821] border border-[#263341] rounded-[8px] flex flex-col overflow-hidden">
          <div className="p-2.5 border-b border-[#263341] bg-[#151E28] flex items-center justify-between flex-shrink-0 text-xs font-mono">
            <div className="flex items-center gap-2">
              <Archive className="w-3.5 h-3.5 text-[#3B9EFF]" />
              <span className="font-semibold text-[#E8EDF3] uppercase text-[11px] font-sans">
                Incident Records ({filteredIncidents.length})
              </span>
            </div>
            <span className="text-[10px] text-[#98A6B5]">
              Showing {filteredIncidents.length} of {incidentHistory.length}
            </span>
          </div>

          <div className="flex-1 overflow-y-auto overflow-x-auto min-h-0">
            {filteredIncidents.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center p-8 text-center text-[#98A6B5] font-sans text-xs">
                <ShieldCheck className="w-8 h-8 text-[#687585] mb-2" />
                <div className="text-[#E8EDF3] font-semibold mb-1">
                  NO INCIDENTS MATCHING FILTER
                </div>
                <div className="text-[10.5px] max-w-sm text-[#98A6B5]">
                  SYSTEM MONITORING &bull; No verified incidents available for the selected criteria. Adjust filters or view live command centre.
                </div>
              </div>
            ) : (
              <table className="w-full text-left border-collapse text-xs font-mono">
                <thead className="sticky top-0 z-10 bg-[#111821] border-b border-[#263341]">
                  <tr className="text-[9.5px] uppercase tracking-wider text-[#98A6B5] font-sans">
                    <th className="py-2 px-3 w-16 font-semibold">ID</th>
                    <th className="py-2 px-3 w-28 font-semibold">DATE / TIME</th>
                    <th className="py-2 px-3 w-24 font-semibold">TYPE</th>
                    <th className="py-2 px-3 w-32 font-semibold">LOCATION</th>
                    <th className="py-2 px-3 w-16 text-center font-semibold">SEVERITY</th>
                    <th className="py-2 px-3 w-20 text-center font-semibold">STATUS</th>
                    <th className="py-2 px-3 w-16 text-center font-semibold">CONF</th>
                    <th className="py-2 px-3 w-16 text-center font-semibold">EVIDENCE</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#263341]">
                  {filteredIncidents.map((row) => {
                    const isSelected = activeIncident?.incident_id === row.incident_id;
                    const hasSnap = Boolean(row.evidence?.snapshot_path);
                    const hasClip = Boolean(row.evidence?.clip_path);

                    return (
                      <tr
                        key={row.incident_id}
                        onClick={() => selectIncident(row.incident_id)}
                        className={`cursor-pointer transition-colors ${
                          isSelected
                            ? 'bg-[#151E28] border-l-2 border-l-[#3B9EFF]'
                            : 'hover:bg-[#151E28]'
                        }`}
                      >
                        <td className="py-2 px-3 text-[#E8EDF3] text-[10px] font-semibold">
                          #{row.incident_id.slice(-6)}
                        </td>
                        <td className="py-2 px-3 text-[#E8EDF3] text-[10px]">
                          <div>{formatTime(row.start_time)}</div>
                          <div className="text-[9px] text-[#687585]">
                            {formatDate(row.start_time)}
                          </div>
                        </td>
                        <td className="py-2 px-3">
                          <span className="text-[#E8EDF3] text-[10px] font-medium font-sans">
                            {formatIncidentType(row.incident_type)}
                          </span>
                        </td>
                        <td className="py-2 px-3 text-[#98A6B5] text-[10px] truncate max-w-[140px] font-sans">
                          {row.location?.name || row.zone || row.camera_id}
                        </td>
                        <td className="py-2 px-3 text-center">
                          {getSeverityBadge(row.severity)}
                        </td>
                        <td className="py-2 px-3 text-center">
                          <span className="text-[9px] font-mono text-[#98A6B5] uppercase">
                            {row.status}
                          </span>
                        </td>
                        <td className="py-2 px-3 text-center text-[#E8EDF3] text-[10px] font-semibold">
                          {(row.confidence * 100).toFixed(0)}%
                        </td>
                        <td className="py-2 px-3 text-center">
                          <div className="flex items-center justify-center gap-1 text-[#98A6B5]">
                            {hasSnap ? (
                              <span title="Snapshot available">
                                <Camera className="w-3 h-3 text-[#2BC48A]" />
                              </span>
                            ) : (
                              <span className="text-[#687585] text-[9px]">—</span>
                            )}
                            {hasClip ? (
                              <span title="Clip available">
                                <Video className="w-3 h-3 text-[#3B9EFF]" />
                              </span>
                            ) : null}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Right Column: Detailed Investigation, Lifecycle Timeline & Evidence */}
        <div className="lg:col-span-5 flex flex-col gap-3 min-h-0">
          {activeIncident ? (
            <>
              {/* Detail Navigation Tabs */}
              <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-1 flex items-center justify-between text-xs font-mono">
                <div className="flex items-center gap-1">
                  <button
                    onClick={() => setDetailTab('agent')}
                    className={`px-3 py-1 rounded-[6px] text-[10.5px] font-semibold transition-colors flex items-center gap-1.5 font-sans ${
                      detailTab === 'agent'
                        ? 'bg-[#3B9EFF] text-[#0B0F14]'
                        : 'text-[#98A6B5] hover:text-[#E8EDF3]'
                    }`}
                  >
                    <span>Agentic Briefing</span>
                  </button>
                  <button
                    onClick={() => setDetailTab('timeline')}
                    className={`px-3 py-1 rounded-[6px] text-[10.5px] font-semibold transition-colors font-sans ${
                      detailTab === 'timeline'
                        ? 'bg-[#3B9EFF] text-[#0B0F14]'
                        : 'text-[#98A6B5] hover:text-[#E8EDF3]'
                    }`}
                  >
                    Timeline Lifecycle
                  </button>
                  <button
                    onClick={() => setDetailTab('evidence')}
                    className={`px-3 py-1 rounded-[6px] text-[10.5px] font-semibold transition-colors font-sans ${
                      detailTab === 'evidence'
                        ? 'bg-[#3B9EFF] text-[#0B0F14]'
                        : 'text-[#98A6B5] hover:text-[#E8EDF3]'
                    }`}
                  >
                    Forensic Evidence
                  </button>
                  <button
                    onClick={() => setDetailTab('response')}
                    className={`px-3 py-1 rounded-[6px] text-[10.5px] font-semibold transition-colors font-sans ${
                      detailTab === 'response'
                        ? 'bg-[#3B9EFF] text-[#0B0F14]'
                        : 'text-[#98A6B5] hover:text-[#E8EDF3]'
                    }`}
                  >
                    Response Plan
                  </button>
                </div>

                <div className="text-[9.5px] font-mono text-[#98A6B5] pr-2">
                  Active: #{activeIncident.incident_id.slice(-6)}
                </div>
              </div>

              {/* Detail Content based on Tab */}
              <div className="flex-1 min-h-[460px] flex flex-col">
                {detailTab === 'agent' && (
                  <div className="h-full">
                    <AgentAssessmentCard incidentId={activeIncident.incident_id} />
                  </div>
                )}

                {detailTab === 'timeline' && (
                  <div className="flex flex-col gap-3 h-full">
                    <IncidentLifecycleTimeline incident={activeIncident} />

                    {/* Metadata Card */}
                    <div className="bg-[#151E28] border border-[#263341] rounded-[8px] p-3 text-xs font-mono flex-1">
                      <div className="font-semibold text-[#E8EDF3] border-b border-[#263341] pb-1.5 mb-2 text-[11px] uppercase font-sans">
                        Forensic Metadata
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-[10.5px]">
                        <div>
                          <span className="text-[#98A6B5] font-sans">Camera / Source:</span>
                          <div className="text-[#E8EDF3] font-semibold">{activeIncident.camera_id}</div>
                        </div>
                        <div>
                          <span className="text-[#98A6B5] font-sans">Severity:</span>
                          <div>{getSeverityBadge(activeIncident.severity)}</div>
                        </div>
                        <div>
                          <span className="text-[#98A6B5] font-sans">Confidence:</span>
                          <div className="text-[#E8EDF3] font-semibold">
                            {(activeIncident.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div>
                          <span className="text-[#98A6B5] font-sans">Status:</span>
                          <div className="text-[#E8EDF3] uppercase font-semibold">{activeIncident.status}</div>
                        </div>
                        <div className="col-span-2">
                          <span className="text-[#98A6B5] font-sans">Location:</span>
                          <div className="text-[#E8EDF3] font-semibold font-sans">
                            {activeIncident.location?.name || activeIncident.zone || 'Yashobhoomi Sector 25'}
                          </div>
                          {activeIncident.location?.latitude && (
                            <div className="text-[#98A6B5] text-[9.5px]">
                              {activeIncident.location.latitude.toFixed(6)}° N, {activeIncident.location.longitude?.toFixed(6)}° E ({activeIncident.location.source || 'Fixed GPS'})
                            </div>
                          )}
                        </div>
                        {activeIncident.track_ids.length > 0 && (
                          <div className="col-span-2">
                            <span className="text-[#98A6B5] font-sans">Tracked ByteTrack IDs:</span>
                            <div className="text-[#49C6D9]">
                              [{activeIncident.track_ids.join(', ')}]
                            </div>
                          </div>
                        )}
                        {activeIncident.details && Object.keys(activeIncident.details).length > 0 && (
                          <div className="col-span-2 bg-[#111821] p-2 rounded-[6px] border border-[#263341] mt-1">
                            <span className="text-[#98A6B5] text-[10px] font-sans">Algorithm Details:</span>
                            <pre className="text-[10px] text-[#E8EDF3] overflow-x-auto whitespace-pre-wrap mt-0.5 font-mono">
                              {JSON.stringify(activeIncident.details, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {detailTab === 'evidence' && (
                  <div className="h-full">
                    <IncidentEvidencePanel />
                  </div>
                )}

                {detailTab === 'response' && (
                  <div className="h-full">
                    <RecommendedResponseCard onExpandMap={onBackToCommand} />
                  </div>
                )}
              </div>
            </>
          ) : (
            <div className="h-full bg-[#111821] border border-[#263341] rounded-[8px] p-6 flex flex-col items-center justify-center text-center font-sans text-xs text-[#98A6B5]">
              <Eye className="w-8 h-8 text-[#687585] mb-2" />
              <div className="text-[#E8EDF3] font-semibold mb-1">NO INCIDENT SELECTED</div>
              <div className="text-[10.5px] max-w-xs text-[#98A6B5]">
                Select an incident from the archive list on the left to inspect its complete chronological timeline, forensic evidence, and emergency response plan.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
