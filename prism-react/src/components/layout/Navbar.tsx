import { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useHealth } from '../../hooks/useHealth';
import { StatusDot } from '../ui/StatusDot';

/* ── Live Clock ─────────────────────────────────────────── */
function LiveClock() {
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const pad = (n: number) => String(n).padStart(2, '0');
  const hh = pad(time.getHours());
  const mm = pad(time.getMinutes());
  const ss = pad(time.getSeconds());
  const dateStr = time.toLocaleDateString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric', timeZone: 'Asia/Kolkata',
  });

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'flex-end',
      padding: '4px 12px',
      borderRadius: 9,
      background: 'rgba(255,255,255,0.04)',
      border: '1px solid rgba(255,255,255,0.07)',
    }}>
      <span style={{
        fontFamily: "'JetBrains Mono', monospace",
        fontSize: 16,
        fontWeight: 600,
        color: '#00d4ff',
        letterSpacing: '0.05em',
        lineHeight: 1.1,
      }}>
        {hh}
        <span style={{ opacity: 0.5, animation: 'blink-colon 1s step-start infinite' }}>:</span>
        {mm}
        <span style={{ opacity: 0.5, animation: 'blink-colon 1s step-start infinite' }}>:</span>
        <span style={{ color: '#475569' }}>{ss}</span>
      </span>
      <span style={{ fontSize: 9, color: '#334155', letterSpacing: '0.04em', marginTop: 1 }}>
        {dateStr} IST
      </span>
    </div>
  );
}

/* ── Page title map ─────────────────────────────────────── */
function usePageLabel() {
  const { pathname } = useLocation();
  if (pathname === '/') return { label: 'Dashboard', sub: 'Overview & Recent Reports' };
  if (pathname.startsWith('/generate')) return { label: 'AI Report Workstation', sub: 'Generate Draft Reports' };
  if (pathname.startsWith('/reports')) return { label: 'All Reports', sub: 'Browse & Filter' };
  if (pathname.startsWith('/report/')) {
    const id = pathname.split('/report/')[1]?.replace(/\/$/, '') || '';
    return { label: id, sub: 'Report Editor' };
  }
  return { label: 'PRISM', sub: '' };
}

/* ── Nav items ──────────────────────────────────────────── */
const NAV_ITEMS = [
  {
    to: '/',
    label: 'Dashboard',
    shortcut: 'D',
    icon: (
      <svg style={{ width: 15, height: 15 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
      </svg>
    ),
  },
  {
    to: '/generate',
    label: 'Generate',
    shortcut: 'G',
    icon: (
      <svg style={{ width: 15, height: 15 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
      </svg>
    ),
  },
  {
    to: '/reports',
    label: 'Reports',
    shortcut: 'R',
    icon: (
      <svg style={{ width: 15, height: 15 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
  },
];

/* ── Main Navbar ────────────────────────────────────────── */
export function Navbar() {
  const { health } = useHealth();
  const page = usePageLabel();
  const isOnline = !!health?.ollama_reachable;

  return (
    <>
      {/* Inject blink keyframe via style tag */}
      <style>{`
        @keyframes blink-colon {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.15; }
        }
        @keyframes logo-glow {
          0%, 100% { box-shadow: 0 0 16px rgba(14,165,233,0.4); }
          50%       { box-shadow: 0 0 32px rgba(14,165,233,0.7); }
        }
        @keyframes top-bar-shimmer {
          0%   { background-position: -400% 0; }
          100% { background-position: 400% 0; }
        }
      `}</style>

      {/* ── Top accent bar ── */}
      <div style={{
        height: 2,
        background: 'linear-gradient(90deg, #6366f1, #0ea5e9, #22c55e, #0ea5e9, #6366f1)',
        backgroundSize: '400% 100%',
        animation: 'top-bar-shimmer 6s linear infinite',
      }} />

      {/* ── Main header ── */}
      <header style={{
        position: 'sticky', top: 2, zIndex: 100,
        background: 'rgba(7,11,18,0.92)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
      }}>
        <div style={{
          maxWidth: 1280, margin: '0 auto',
          padding: '0 24px',
          height: 60,
          display: 'flex',
          alignItems: 'center',
          gap: 16,
        }}>

          {/* Logo */}
          <NavLink to="/" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none', flexShrink: 0 }}>
            <div style={{
              width: 36, height: 36, borderRadius: 10,
              background: 'linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              animation: 'logo-glow 3s ease-in-out infinite',
              flexShrink: 0,
            }}>
              <svg style={{ width: 19, height: 19, color: '#fff' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M5 14.5l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.611L5 14.5" />
              </svg>
            </div>
            <div>
              <div style={{
                fontSize: 16, fontWeight: 800, color: '#e2e8f0',
                letterSpacing: '-0.02em', lineHeight: 1,
                background: 'linear-gradient(90deg, #e2e8f0, #00d4ff)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>
                PRISM
              </div>
              <div style={{ fontSize: 9, color: '#334155', fontWeight: 500, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                Phase 3 · Radiology AI
              </div>
            </div>
          </NavLink>

          {/* Divider */}
          <div style={{ width: 1, height: 28, background: 'rgba(255,255,255,0.07)', flexShrink: 0 }} />

          {/* Page breadcrumb */}
          <div style={{ flex: '0 0 auto' }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: '#94a3b8', lineHeight: 1.1 }}>{page.label}</div>
            {page.sub && <div style={{ fontSize: 10, color: '#334155', marginTop: 1 }}>{page.sub}</div>}
          </div>

          {/* Spacer */}
          <div style={{ flex: 1 }} />

          {/* Nav links */}
          <nav style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {NAV_ITEMS.map(({ to, label, icon, shortcut }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                title={`${label} (${shortcut})`}
                style={({ isActive }) => ({
                  display: 'flex', alignItems: 'center', gap: 6,
                  padding: '6px 12px',
                  borderRadius: 8,
                  fontSize: 12, fontWeight: 500,
                  textDecoration: 'none',
                  transition: 'all 150ms ease',
                  color: isActive ? '#00d4ff' : '#475569',
                  background: isActive ? 'rgba(0,212,255,0.08)' : 'transparent',
                  border: isActive ? '1px solid rgba(0,212,255,0.15)' : '1px solid transparent',
                  position: 'relative',
                })}
              >
                {icon}
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>

          {/* Divider */}
          <div style={{ width: 1, height: 28, background: 'rgba(255,255,255,0.07)', flexShrink: 0 }} />

          {/* AI status */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 7,
            padding: '5px 12px', borderRadius: 20,
            background: isOnline ? 'rgba(34,197,94,0.07)' : 'rgba(239,68,68,0.07)',
            border: `1px solid ${isOnline ? 'rgba(34,197,94,0.18)' : 'rgba(239,68,68,0.18)'}`,
            flexShrink: 0,
          }}>
            <StatusDot online={isOnline} size="sm" />
            <span style={{ fontSize: 11, fontWeight: 600, color: isOnline ? '#22c55e' : '#ef4444', letterSpacing: '0.03em' }}>
              {isOnline ? 'AI Online' : 'Offline'}
            </span>
            {health?.ollama_model && (
              <span style={{ fontSize: 9, color: '#334155', fontFamily: "'JetBrains Mono', monospace", marginLeft: 2 }}>
                · {health.ollama_model}
              </span>
            )}
          </div>

          {/* Live clock */}
          <LiveClock />
        </div>
      </header>
    </>
  );
}
