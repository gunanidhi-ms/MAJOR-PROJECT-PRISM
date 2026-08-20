import React from 'react';

interface BadgeProps {
  variant?: 'cyan' | 'green' | 'amber' | 'red' | 'violet' | 'muted';
  children: React.ReactNode;
  className?: string;
}

const variantStyles: Record<string, React.CSSProperties> = {
  cyan:   { background: 'rgba(0,212,255,0.12)',   color: '#00d4ff',  border: '1px solid rgba(0,212,255,0.25)' },
  green:  { background: 'rgba(34,197,94,0.12)',   color: '#22c55e',  border: '1px solid rgba(34,197,94,0.25)' },
  amber:  { background: 'rgba(245,158,11,0.12)',  color: '#f59e0b',  border: '1px solid rgba(245,158,11,0.25)' },
  red:    { background: 'rgba(239,68,68,0.12)',   color: '#ef4444',  border: '1px solid rgba(239,68,68,0.25)' },
  violet: { background: 'rgba(168,85,247,0.12)',  color: '#a855f7',  border: '1px solid rgba(168,85,247,0.25)' },
  muted:  { background: 'rgba(255,255,255,0.06)', color: '#94a3b8',  border: '1px solid rgba(255,255,255,0.1)' },
};

export function Badge({ variant = 'muted', children, className = '' }: BadgeProps) {
  return (
    <span
      className={className}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '4px',
        padding: '2px 10px',
        borderRadius: '9999px',
        fontSize: '11px',
        fontWeight: 600,
        letterSpacing: '0.02em',
        ...variantStyles[variant],
      }}
    >
      {children}
    </span>
  );
}

export function statusToBadgeVariant(status: string): BadgeProps['variant'] {
  if (status === 'signed') return 'green';
  if (status === 'draft') return 'amber';
  return 'muted';
}

export function sourceToBadgeVariant(source: string): BadgeProps['variant'] {
  if (source === 'llm') return 'violet';
  if (source === 'api') return 'cyan';
  return 'muted';
}
