import React, { type ReactNode } from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  icon?: ReactNode;
  variant?: 'default' | 'cyan' | 'amber' | 'red' | 'purple' | 'green';
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subValue,
  icon,
  variant = 'default',
  className = '',
}) => {
  const getVariantStyles = () => {
    switch (variant) {
      case 'cyan':
        return {
          icon: 'text-cyan-400',
          val: 'text-cyan-300',
          border: 'border-cyan-500/20 hover:border-cyan-500/40',
        };
      case 'amber':
        return {
          icon: 'text-amber-400',
          val: 'text-amber-300',
          border: 'border-amber-500/20 hover:border-amber-500/40',
        };
      case 'red':
        return {
          icon: 'text-red-400',
          val: 'text-red-300',
          border: 'border-red-500/20 hover:border-red-500/40',
        };
      case 'purple':
        return {
          icon: 'text-purple-400',
          val: 'text-purple-300',
          border: 'border-purple-500/20 hover:border-purple-500/40',
        };
      case 'green':
        return {
          icon: 'text-emerald-400',
          val: 'text-emerald-300',
          border: 'border-emerald-500/20 hover:border-emerald-500/40',
        };
      default:
        return {
          icon: 'text-slate-400',
          val: 'text-slate-100',
          border: 'border-slate-800 hover:border-slate-700',
        };
    }
  };

  const styles = getVariantStyles();

  return (
    <div
      className={`bg-slate-900/60 border rounded-lg p-2.5 flex items-center justify-between transition-colors ${styles.border} ${className}`}
    >
      <div className="flex items-center gap-2.5">
        {icon && <div className={`p-1.5 rounded bg-slate-800/80 ${styles.icon}`}>{icon}</div>}
        <div>
          <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400 leading-none mb-1">
            {label}
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className={`text-base font-bold font-mono tracking-tight ${styles.val}`}>
              {value}
            </span>
            {subValue && (
              <span className="text-[11px] font-mono text-slate-500">{subValue}</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
