import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  children: React.ReactNode;
}

const baseStyle: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '8px',
  fontWeight: 600,
  cursor: 'pointer',
  border: 'none',
  transition: 'all 150ms cubic-bezier(0.4, 0, 0.2, 1)',
  fontFamily: 'inherit',
  letterSpacing: '0.01em',
  textDecoration: 'none',
  whiteSpace: 'nowrap',
};

const variants: Record<string, React.CSSProperties> = {
  primary: {
    background: 'linear-gradient(135deg, #0ea5e9, #0284c7)',
    color: '#fff',
    boxShadow: '0 0 20px rgba(14,165,233,0.3), inset 0 1px 0 rgba(255,255,255,0.1)',
  },
  secondary: {
    background: 'rgba(255,255,255,0.06)',
    color: '#e2e8f0',
    border: '1px solid rgba(255,255,255,0.1)',
  },
  success: {
    background: 'linear-gradient(135deg, #16a34a, #15803d)',
    color: '#fff',
    boxShadow: '0 0 20px rgba(22,163,74,0.3)',
  },
  danger: {
    background: 'linear-gradient(135deg, #dc2626, #b91c1c)',
    color: '#fff',
    boxShadow: '0 0 20px rgba(220,38,38,0.3)',
  },
  ghost: {
    background: 'transparent',
    color: '#94a3b8',
    border: '1px solid rgba(255,255,255,0.08)',
  },
};

const sizes: Record<string, React.CSSProperties> = {
  sm: { padding: '6px 14px', fontSize: '12px', borderRadius: '8px' },
  md: { padding: '10px 20px', fontSize: '14px', borderRadius: '10px' },
  lg: { padding: '13px 28px', fontSize: '15px', borderRadius: '12px' },
};

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled,
  children,
  style,
  ...props
}: ButtonProps) {
  const isDisabled = disabled || loading;

  return (
    <button
      {...props}
      disabled={isDisabled}
      style={{
        ...baseStyle,
        ...variants[variant],
        ...sizes[size],
        opacity: isDisabled ? 0.5 : 1,
        cursor: isDisabled ? 'not-allowed' : 'pointer',
        ...style,
      }}
    >
      {loading && (
        <svg
          style={{ width: 14, height: 14 }}
          className="animate-spin"
          viewBox="0 0 24 24"
          fill="none"
        >
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
        </svg>
      )}
      {children}
    </button>
  );
}
