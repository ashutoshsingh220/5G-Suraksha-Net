import React from 'react';

interface StatusBadgeProps {
  status: 'ONLINE' | 'ACTIVE' | 'STANDBY' | 'WARNING' | 'ALERT' | 'HIGH' | 'CRITICAL' | 'MEDIUM' | 'INFO';
  pulse?: boolean;
  className?: string;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  pulse = false,
  className = '',
  size = 'md',
}) => {
  const getStyles = () => {
    switch (status) {
      case 'ONLINE':
      case 'ACTIVE':
        return {
          bg: 'bg-emerald-950/70 border-emerald-500/50 text-emerald-400',
          dot: 'bg-emerald-400',
        };
      case 'WARNING':
      case 'MEDIUM':
        return {
          bg: 'bg-amber-950/70 border-amber-500/50 text-amber-400',
          dot: 'bg-amber-400',
        };
      case 'ALERT':
      case 'HIGH':
      case 'CRITICAL':
        return {
          bg: 'bg-red-950/70 border-red-500/60 text-red-400',
          dot: 'bg-red-400',
        };
      case 'INFO':
        return {
          bg: 'bg-blue-950/70 border-blue-500/50 text-blue-400',
          dot: 'bg-blue-400',
        };
      case 'STANDBY':
      default:
        return {
          bg: 'bg-slate-800/80 border-slate-600/50 text-slate-400',
          dot: 'bg-slate-400',
        };
    }
  };

  const { bg, dot } = getStyles();
  const sizeClasses = size === 'sm' ? 'px-1.5 py-0.5 text-[10px]' : 'px-2 py-0.5 text-xs';

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-mono font-medium tracking-wider uppercase ${bg} ${sizeClasses} ${className}`}
    >
      <span className="relative flex h-2 w-2">
        {pulse && (
          <span
            className={`absolute inline-flex h-full w-full rounded-full opacity-75 animate-ping ${dot}`}
          />
        )}
        <span className={`relative inline-flex h-2 w-2 rounded-full ${dot}`} />
      </span>
      {status}
    </span>
  );
};
