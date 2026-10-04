import React, { useState, useEffect } from 'react';
import {
  Shield,
  LayoutGrid,
  MapPin,
  AlertTriangle,
  Sliders,
  Maximize2,
  Minimize2,
} from 'lucide-react';
import { useSystemStatus } from '../../hooks/useSystemStatus';

export type NavTabId = 'command' | 'map' | 'incidents';

interface TopNavigationProps {
  activeTab?: NavTabId;
  onSelectTab?: (tab: NavTabId) => void;
  isDemoActive?: boolean;
  onOpenDemoPanel?: () => void;
}

export const TopNavigation: React.FC<TopNavigationProps> = ({
  activeTab = 'command',
  onSelectTab,
  isDemoActive = false,
  onOpenDemoPanel,
}) => {
  const [internalTab, setInternalTab] = useState<NavTabId>('command');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [currentTime, setCurrentTime] = useState<Date>(new Date());
  const { status: sysStatus, loading } = useSystemStatus({ pollingIntervalMs: 5000 });

  const currentTab = onSelectTab ? activeTab : internalTab;

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const handleTabClick = (tabId: NavTabId) => {
    if (onSelectTab) {
      onSelectTab(tabId);
    } else {
      setInternalTab(tabId);
    }
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen().catch(() => {});
        setIsFullscreen(false);
      }
    }
  };

  const navItems = [
    { id: 'command' as const, label: 'Command Centre', icon: <LayoutGrid className="w-3.5 h-3.5" /> },
    { id: 'map' as const, label: 'Live Map', icon: <MapPin className="w-3.5 h-3.5" /> },
    { id: 'incidents' as const, label: 'Incidents', icon: <AlertTriangle className="w-3.5 h-3.5" /> },
  ];

  const getStatusBadge = () => {
    if (loading && !sysStatus) {
      return {
        dotClass: 'bg-[#98A6B5]',
        borderClass: 'border-[#263341] bg-[#151E28]',
        textClass: 'text-[#98A6B5]',
        label: 'Connecting...',
      };
    }

    switch (sysStatus?.status) {
      case 'ONLINE':
        return {
          dotClass: 'bg-[#2BC48A]',
          borderClass: 'border-[#2BC48A]/40 bg-[#151E28]',
          textClass: 'text-[#2BC48A]',
          label: 'System Online',
        };
      case 'DEGRADED':
        return {
          dotClass: 'bg-[#E7A83B]',
          borderClass: 'border-[#E7A83B]/40 bg-[#151E28]',
          textClass: 'text-[#E7A83B]',
          label: 'System Degraded',
        };
      case 'OFFLINE':
      default:
        return {
          dotClass: 'bg-[#E05252]',
          borderClass: 'border-[#E05252]/40 bg-[#151E28]',
          textClass: 'text-[#E05252]',
          label: 'System Offline',
        };
    }
  };

  const statusBadge = getStatusBadge();

  return (
    <header className="h-[48px] bg-[#111821] border-b border-[#263341] px-3.5 flex items-center justify-between flex-shrink-0 z-40 select-none">
      <div className="flex items-center gap-3">
        {/* Shield Icon */}
        <div className="w-7 h-7 rounded-[6px] bg-[#151E28] border border-[#263341] flex items-center justify-center text-[#3B9EFF]">
          <Shield className="w-3.5 h-3.5" />
        </div>
        <div>
          <div className="text-sm font-bold tracking-wide text-[#E8EDF3] font-sans uppercase leading-none">
            5G SURAKSHA-NET
          </div>
          <div className="text-[10.5px] font-mono tracking-widest text-[#98A6B5] uppercase leading-tight mt-0.5">
            AI POWERED AERIAL INTELLIGENCE
          </div>
        </div>
      </div>

      {/* Center Navigation Tabs */}
      <nav className="hidden md:flex items-center gap-1 bg-[#0B0F14] border border-[#263341] rounded-[8px] p-0.5">
        {navItems.map((item) => {
          const isActive = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => handleTabClick(item.id)}
              className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-[6px] text-xs font-sans font-medium transition-colors ${
                isActive
                  ? 'bg-[#3B9EFF] text-white font-semibold'
                  : 'text-[#98A6B5] hover:text-[#E8EDF3] hover:bg-[#151E28]'
              }`}
            >
              {item.icon}
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Right Controls */}
      <div className="flex items-center gap-2.5">
        {/* Controlled Demo Mode Launcher */}
        <button
          onClick={onOpenDemoPanel}
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-[6px] text-xs font-mono font-bold transition-all ${
            isDemoActive
              ? 'bg-[#E7A83B] text-black shadow-md'
              : 'bg-[#151E28] hover:bg-[#1c2736] text-[#E7A83B] border border-[#E7A83B]/40'
          }`}
          title="IMC Live Demonstration & Controlled Scenario Launcher"
        >
          <Sliders className="w-3.5 h-3.5" />
          <span>{isDemoActive ? 'DEMO ACTIVE' : 'DEMO MODE'}</span>
        </button>

        {/* Live System Health Badge */}
        <div className={`flex items-center gap-1.5 border rounded-full px-2.5 py-0.5 ${statusBadge.borderClass}`}>
          <span className={`w-1.5 h-1.5 rounded-full ${statusBadge.dotClass}`} />
          <span className={`text-xs font-sans font-medium ${statusBadge.textClass}`}>
            {statusBadge.label}
          </span>
        </div>

        {/* Live Clock */}
        <div className="text-right pl-2 border-l border-[#263341] text-xs font-mono">
          <span className="font-semibold text-[#E8EDF3]">
            {currentTime.toTimeString().slice(0, 8)}
          </span>
          <span className="text-[#98A6B5] text-xs ml-1.5">
            {currentTime.toISOString().slice(0, 10)}
          </span>
        </div>

        {/* Fullscreen Button */}
        <button
          onClick={toggleFullscreen}
          title={isFullscreen ? 'Exit Fullscreen' : 'Enter Fullscreen'}
          className="p-1 text-[#98A6B5] hover:text-[#E8EDF3] hover:bg-[#151E28] border border-[#263341] rounded-[6px] transition-colors"
        >
          {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
        </button>
      </div>
    </header>
  );
};
