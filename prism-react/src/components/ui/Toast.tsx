import { useEffect, useState } from 'react';

interface ToastProps {
  message: string;
  type: 'success' | 'error' | 'info';
  visible: boolean;
  onClose: () => void;
}

export function Toast({ message, type, visible, onClose }: ToastProps) {
  useEffect(() => {
    if (visible) {
      const t = setTimeout(onClose, 4500);
      return () => clearTimeout(t);
    }
  }, [visible, onClose]);

  const colors = {
    success: { bg: 'rgba(22,163,74,0.15)',  border: 'rgba(34,197,94,0.3)',  accent: '#22c55e' },
    error:   { bg: 'rgba(220,38,38,0.15)',  border: 'rgba(239,68,68,0.3)',  accent: '#ef4444' },
    info:    { bg: 'rgba(14,165,233,0.15)', border: 'rgba(0,212,255,0.3)',  accent: '#00d4ff' },
  }[type];

  const icons = {
    success: (
      <svg style={{ width: 16, height: 16, color: colors.accent }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    ),
    error: (
      <svg style={{ width: 16, height: 16, color: colors.accent }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
      </svg>
    ),
    info: (
      <svg style={{ width: 16, height: 16, color: colors.accent }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
  }[type];

  return (
    <div
      style={{
        position: 'fixed',
        bottom: 24,
        right: 24,
        zIndex: 9999,
        maxWidth: 380,
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '12px 16px',
        borderRadius: 12,
        border: `1px solid ${colors.border}`,
        background: colors.bg,
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
        transform: visible ? 'translateY(0)' : 'translateY(20px)',
        opacity: visible ? 1 : 0,
        transition: 'all 300ms cubic-bezier(0.4, 0, 0.2, 1)',
        pointerEvents: visible ? 'auto' : 'none',
      }}
    >
      {icons}
      <span style={{ fontSize: 13, fontWeight: 500, color: '#e2e8f0', flex: 1 }}>{message}</span>
      <button
        onClick={onClose}
        style={{
          background: 'none',
          border: 'none',
          cursor: 'pointer',
          color: '#64748b',
          padding: 2,
          lineHeight: 1,
        }}
      >
        <svg style={{ width: 14, height: 14 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}

export function useToast() {
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' | 'info'; visible: boolean }>({
    message: '',
    type: 'success',
    visible: false,
  });

  const show = (message: string, type: 'success' | 'error' | 'info' = 'success') => {
    setToast({ message, type, visible: true });
  };

  const hide = () => setToast((t) => ({ ...t, visible: false }));

  return { toast, show, hide };
}
