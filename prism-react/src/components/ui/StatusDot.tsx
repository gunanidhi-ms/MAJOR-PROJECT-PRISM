interface StatusDotProps {
  online: boolean;
  label?: string;
  size?: 'sm' | 'md';
}

export function StatusDot({ online, label, size = 'md' }: StatusDotProps) {
  const dotSize = size === 'sm' ? 7 : 9;
  const color = online ? '#22c55e' : '#ef4444';

  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 7 }}>
      <span style={{ position: 'relative', display: 'inline-flex', width: dotSize, height: dotSize }}>
        <span
          style={{
            position: 'absolute', inset: 0,
            borderRadius: '50%',
            background: color,
            animation: online ? 'pulse-ring 1.5s ease-out infinite' : 'none',
            opacity: 0.5,
          }}
        />
        <span
          style={{
            position: 'relative',
            display: 'inline-flex',
            width: dotSize,
            height: dotSize,
            borderRadius: '50%',
            background: color,
            boxShadow: `0 0 8px ${color}`,
          }}
        />
      </span>
      {label && (
        <span style={{ fontSize: 12, fontWeight: 600, color: online ? '#22c55e' : '#ef4444', letterSpacing: '0.02em' }}>
          {label}
        </span>
      )}
    </div>
  );
}
