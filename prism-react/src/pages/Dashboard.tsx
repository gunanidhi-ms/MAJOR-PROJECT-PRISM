import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useReports } from '../hooks/useReports';
import { useHealth } from '../hooks/useHealth';
import { Badge, statusToBadgeVariant, sourceToBadgeVariant } from '../components/ui/Badge';
import { StatusDot } from '../components/ui/StatusDot';
import type { Report } from '../api/prismApi';

const SAMPLE_CASES = [
  {
    id: 'normal_case',
    title: 'Normal CT KUB',
    protocol: 'CT KUB — All organs within normal limits',
    badge: 'Normal',
    badgeVariant: 'green' as const,
    image: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=400&h=200&fit=crop',
  },
  {
    id: 'single_lesion',
    title: 'Single Renal Lesion',
    protocol: 'CT KUB — Hypodense lesion, right kidney',
    badge: 'Anomaly',
    badgeVariant: 'amber' as const,
    image: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=400&h=200&fit=crop',
  },
  {
    id: 'multi_organ',
    title: 'Multi-Organ Findings',
    protocol: 'CT Abdomen — Hepatic & renal findings',
    badge: 'Complex',
    badgeVariant: 'red' as const,
    image: 'https://images.unsplash.com/photo-1530497610245-94d3c16cda28?w=400&h=200&fit=crop',
  },
];

function AnimatedStat({ value, label, color }: { value: number; label: string; color: string }) {
  const [displayed, setDisplayed] = useState(0);

  useEffect(() => {
    if (value === 0) return;
    let start = 0;
    const increment = Math.ceil(value / 20);
    const timer = setInterval(() => {
      start += increment;
      if (start >= value) { setDisplayed(value); clearInterval(timer); }
      else setDisplayed(start);
    }, 40);
    return () => clearInterval(timer);
  }, [value]);

  return (
    <div
      className="glass-card animate-fade-up"
      style={{ padding: '20px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}
    >
      <div>
        <p style={{ fontSize: 12, fontWeight: 500, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</p>
        <p style={{ fontSize: 36, fontWeight: 700, color, lineHeight: 1.1, marginTop: 4, fontFamily: "'JetBrains Mono', monospace" }}>
          {displayed}
        </p>
      </div>
      <div style={{ width: 48, height: 48, borderRadius: 12, background: `${color}15`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <div style={{ width: 20, height: 20, borderRadius: '50%', background: color, opacity: 0.8 }} />
      </div>
    </div>
  );
}

function RecentReportRow({ report, index }: { report: Report; index: number }) {
  return (
    <tr
      className={`stagger-${Math.min(index + 1, 6)}`}
      style={{
        borderBottom: '1px solid rgba(255,255,255,0.04)',
        transition: 'background 150ms ease',
        animation: 'fade-up 0.3s ease-out both',
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(255,255,255,0.03)')}
      onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
    >
      <td style={{ padding: '12px 16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 32, height: 32, borderRadius: 8, background: 'rgba(14,165,233,0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
            <svg style={{ width: 14, height: 14, color: '#0ea5e9' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <span style={{ fontSize: 13, fontWeight: 500, color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace" }}>
            {report.study_id}
          </span>
        </div>
      </td>
      <td style={{ padding: '12px 16px' }}>
        <Badge variant={statusToBadgeVariant(report.status)}>{report.status.charAt(0).toUpperCase() + report.status.slice(1)}</Badge>
      </td>
      <td style={{ padding: '12px 16px' }}>
        <Badge variant={sourceToBadgeVariant(report.source)}>{report.source?.toUpperCase() || '—'}</Badge>
      </td>
      <td style={{ padding: '12px 16px', fontSize: 12, color: '#475569', fontFamily: "'JetBrains Mono', monospace" }}>
        {report.updated_at?.slice(0, 16) || '—'}
      </td>
      <td style={{ padding: '12px 16px', textAlign: 'right' }}>
        <Link
          to={`/report/${encodeURIComponent(report.study_id)}`}
          style={{ fontSize: 12, fontWeight: 600, color: '#00d4ff', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 4 }}
        >
          Open
          <svg style={{ width: 12, height: 12 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
          </svg>
        </Link>
      </td>
    </tr>
  );
}

export default function Dashboard() {
  const { data, loading, error: apiError } = useReports();
  const { health } = useHealth();
  const reports = data.reports;
  const draftCount = reports.filter((r) => r.status === 'draft').length;
  const signedCount = reports.filter((r) => r.status === 'signed').length;

  return (
    <div>
      {/* Hero */}
      <section
        className="animate-fade-up"
        style={{
          position: 'relative',
          borderRadius: 20,
          overflow: 'hidden',
          marginBottom: 32,
          background: 'linear-gradient(135deg, #0a1628 0%, #0d1f3e 40%, #0a2555 100%)',
          border: '1px solid rgba(0,212,255,0.12)',
          boxShadow: '0 0 60px rgba(0,212,255,0.08)',
        }}
      >
        {/* Grid pattern */}
        <div
          className="grid-bg"
          style={{ position: 'absolute', inset: 0, opacity: 0.4, pointerEvents: 'none' }}
        />
        {/* Glow orb */}
        <div style={{
          position: 'absolute', top: -60, right: -60, width: 280, height: 280,
          borderRadius: '50%', background: 'radial-gradient(circle, rgba(0,212,255,0.12) 0%, transparent 70%)',
          pointerEvents: 'none',
        }} />

        <div style={{ position: 'relative', padding: '48px 48px', maxWidth: 640 }}>
          <p style={{ fontSize: 11, fontWeight: 600, color: '#00d4ff', textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 12 }}>
            Clinical Intelligence Platform
          </p>
          <h1 style={{ fontSize: 38, fontWeight: 700, color: '#f1f5f9', lineHeight: 1.15, letterSpacing: '-0.02em', marginBottom: 16 }}>
            AI-Powered<br />
            <span style={{ color: '#00d4ff' }}>Radiology Reports</span>
          </h1>
          <p style={{ fontSize: 15, color: '#64748b', lineHeight: 1.7, marginBottom: 28, maxWidth: 480 }}>
            Transform structured CT findings into validated clinical draft reports. Review, edit, and sign — all in one AI-assisted workflow.
          </p>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <Link
              to="/generate"
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 8,
                padding: '11px 22px', borderRadius: 10,
                background: 'linear-gradient(135deg, #0ea5e9, #0284c7)',
                color: '#fff', fontSize: 14, fontWeight: 600,
                textDecoration: 'none',
                boxShadow: '0 0 24px rgba(14,165,233,0.35)',
                transition: 'all 200ms ease',
              }}
            >
              <svg style={{ width: 16, height: 16 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
              New Report
            </Link>
            <Link
              to="/reports"
              style={{
                display: 'inline-flex', alignItems: 'center', gap: 8,
                padding: '11px 22px', borderRadius: 10,
                background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
                color: '#94a3b8', fontSize: 14, fontWeight: 600,
                textDecoration: 'none',
                transition: 'all 200ms ease',
              }}
            >
              View All Reports
            </Link>
          </div>
        </div>
      </section>

      {/* API Error banner */}
      {(apiError || data.error) && (
        <div style={{
          marginBottom: 24, padding: '12px 16px', borderRadius: 10,
          background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)',
          fontSize: 13, color: '#f59e0b',
        }}>
          <strong>Backend notice:</strong> {apiError || data.error} — Start the FastAPI server:{' '}
          <code style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 11, background: 'rgba(245,158,11,0.12)', padding: '2px 6px', borderRadius: 4 }}>
            uvicorn app.main:app --reload
          </code>
        </div>
      )}

      {/* Stats */}
      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 32 }}>
        <AnimatedStat value={loading ? 0 : data.total} label="Total Reports" color="#00d4ff" />
        <AnimatedStat value={loading ? 0 : draftCount} label="Draft Reports" color="#f59e0b" />
        <AnimatedStat value={loading ? 0 : signedCount} label="Signed Reports" color="#22c55e" />
        <div
          className="glass-card animate-fade-up stagger-4"
          style={{ padding: '20px 24px', display: 'flex', alignItems: 'center', gap: 12 }}
        >
          <StatusDot online={!!health?.ollama_reachable} />
          <div>
            <p style={{ fontSize: 12, fontWeight: 500, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em' }}>AI Engine</p>
            <p style={{ fontSize: 14, fontWeight: 600, color: health?.ollama_reachable ? '#22c55e' : '#475569', marginTop: 2 }}>
              {health?.ollama_reachable ? 'Online' : 'Template Mode'}
            </p>
            {health?.ollama_model && (
              <p style={{ fontSize: 11, color: '#334155', fontFamily: "'JetBrains Mono', monospace", marginTop: 2 }}>
                {health.ollama_model}
              </p>
            )}
          </div>
        </div>
      </section>

      {/* Main content grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 320px', gap: 24, alignItems: 'start' }}>
        {/* Reports table */}
        <div className="surface-card animate-fade-up stagger-3" style={{ overflow: 'hidden' }}>
          <div style={{ padding: '20px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
            <h2 style={{ fontSize: 15, fontWeight: 700, color: '#e2e8f0' }}>Recent Reports</h2>
            <Link to="/reports" style={{ fontSize: 12, fontWeight: 600, color: '#00d4ff', textDecoration: 'none' }}>
              View all →
            </Link>
          </div>

          {loading ? (
            <div style={{ padding: 24 }}>
              {[1, 2, 3].map((i) => (
                <div key={i} className="skeleton" style={{ height: 48, marginBottom: 8 }} />
              ))}
            </div>
          ) : reports.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    {['Study ID', 'Status', 'Source', 'Updated', ''].map((h) => (
                      <th key={h} style={{ padding: '10px 16px', textAlign: 'left', fontSize: 10, fontWeight: 600, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {reports.slice(0, 8).map((r, i) => (
                    <RecentReportRow key={r.study_id} report={r} index={i} />
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ padding: '64px 24px', textAlign: 'center' }}>
              <div style={{ width: 64, height: 64, borderRadius: '50%', background: 'rgba(14,165,233,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
                <svg style={{ width: 28, height: 28, color: '#0ea5e9' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <p style={{ fontSize: 14, fontWeight: 500, color: '#475569', marginBottom: 6 }}>No reports yet</p>
              <p style={{ fontSize: 12, color: '#334155', marginBottom: 20 }}>Generate your first report from structured findings.</p>
              <Link
                to="/generate"
                style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '9px 18px', borderRadius: 9, background: 'rgba(14,165,233,0.12)', border: '1px solid rgba(14,165,233,0.2)', color: '#0ea5e9', fontSize: 13, fontWeight: 600, textDecoration: 'none' }}
              >
                Generate Report
              </Link>
            </div>
          )}
        </div>

        {/* Quick-start cases */}
        <div>
          <h2 style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0', marginBottom: 14, letterSpacing: '-0.01em' }}>Quick-Start Cases</h2>
          <p style={{ fontSize: 12, color: '#475569', marginBottom: 16 }}>Load a sample study and generate a draft instantly.</p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {SAMPLE_CASES.map((c, i) => (
              <Link
                key={c.id}
                to={`/generate?case=${c.id}`}
                className={`glass-card animate-fade-up stagger-${i + 2}`}
                style={{ textDecoration: 'none', overflow: 'hidden', display: 'block' }}
              >
                <div style={{ position: 'relative', height: 90, overflow: 'hidden' }}>
                  <img src={c.image} alt={c.title} style={{ width: '100%', height: '100%', objectFit: 'cover', transition: 'transform 300ms ease' }}
                    onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.05)')}
                    onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1)')}
                  />
                  <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(0,0,0,0.7) 0%, transparent 100%)' }} />
                  <div style={{ position: 'absolute', bottom: 8, left: 10 }}>
                    <Badge variant={c.badgeVariant}>{c.badge}</Badge>
                  </div>
                </div>
                <div style={{ padding: '10px 12px' }}>
                  <p style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0', marginBottom: 2 }}>{c.title}</p>
                  <p style={{ fontSize: 11, color: '#475569' }}>{c.protocol}</p>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
