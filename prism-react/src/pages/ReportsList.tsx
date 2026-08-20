import { useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useReports } from '../hooks/useReports';
import { Badge, statusToBadgeVariant, sourceToBadgeVariant } from '../components/ui/Badge';

type StatusFilter = 'all' | 'draft' | 'signed';

export default function ReportsList() {
  const { data, loading } = useReports();
  const [filter, setFilter] = useState<StatusFilter>('all');
  const [search, setSearch] = useState('');

  const filtered = useMemo(() => {
    return data.reports.filter((r) => {
      const matchStatus = filter === 'all' || r.status === filter;
      const matchSearch = !search || r.study_id.toLowerCase().includes(search.toLowerCase());
      return matchStatus && matchSearch;
    });
  }, [data.reports, filter, search]);

  return (
    <div>
      {/* Header */}
      <div className="animate-fade-up" style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#e2e8f0', letterSpacing: '-0.02em' }}>All Reports</h1>
        <p style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>
          {data.total} total report{data.total !== 1 ? 's' : ''} stored in the system
        </p>
      </div>

      {/* Filters bar */}
      <div
        className="surface-card animate-fade-up stagger-1"
        style={{ padding: '14px 20px', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}
      >
        {/* Search */}
        <div style={{ position: 'relative', flex: 1, minWidth: 200 }}>
          <svg
            style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', width: 14, height: 14, color: '#334155' }}
            fill="none" viewBox="0 0 24 24" stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            type="text"
            placeholder="Search by Study ID…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{
              width: '100%', paddingLeft: 32, paddingRight: 12, paddingTop: 8, paddingBottom: 8,
              fontSize: 13, fontFamily: "'JetBrains Mono', monospace",
              background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)',
              borderRadius: 8, color: '#e2e8f0', outline: 'none',
            }}
          />
        </div>

        {/* Status pills */}
        <div style={{ display: 'flex', gap: 6 }}>
          {(['all', 'draft', 'signed'] as StatusFilter[]).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              style={{
                padding: '6px 14px', borderRadius: 20, fontSize: 12, fontWeight: 600, cursor: 'pointer',
                transition: 'all 150ms ease',
                background: filter === f ? (f === 'all' ? 'rgba(0,212,255,0.15)' : f === 'draft' ? 'rgba(245,158,11,0.15)' : 'rgba(34,197,94,0.15)') : 'rgba(255,255,255,0.04)',
                border: filter === f ? `1px solid ${f === 'all' ? 'rgba(0,212,255,0.3)' : f === 'draft' ? 'rgba(245,158,11,0.3)' : 'rgba(34,197,94,0.3)'}` : '1px solid rgba(255,255,255,0.06)',
                color: filter === f ? (f === 'all' ? '#00d4ff' : f === 'draft' ? '#f59e0b' : '#22c55e') : '#475569',
              }}
            >
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div className="surface-card animate-fade-up stagger-2" style={{ overflow: 'hidden' }}>
        {loading ? (
          <div style={{ padding: 24 }}>
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="skeleton" style={{ height: 52, marginBottom: 8 }} />
            ))}
          </div>
        ) : filtered.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                  {['Study ID', 'Status', 'Source', 'Validated', 'Created', 'Updated', ''].map((h) => (
                    <th key={h} style={{ padding: '12px 16px', textAlign: 'left', fontSize: 10, fontWeight: 600, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((report, i) => (
                  <tr
                    key={report.study_id}
                    className={`stagger-${Math.min(i + 1, 6)}`}
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', animation: 'fade-up 0.3s ease-out both', transition: 'background 150ms ease' }}
                    onMouseEnter={(e) => (e.currentTarget.style.background = 'rgba(255,255,255,0.03)')}
                    onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                  >
                    <td style={{ padding: '13px 16px' }}>
                      <span style={{ fontSize: 13, fontWeight: 500, color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace" }}>
                        {report.study_id}
                      </span>
                    </td>
                    <td style={{ padding: '13px 16px' }}>
                      <Badge variant={statusToBadgeVariant(report.status)}>
                        {report.status.charAt(0).toUpperCase() + report.status.slice(1)}
                      </Badge>
                    </td>
                    <td style={{ padding: '13px 16px' }}>
                      <Badge variant={sourceToBadgeVariant(report.source)}>
                        {report.source?.toUpperCase() || '—'}
                      </Badge>
                    </td>
                    <td style={{ padding: '13px 16px' }}>
                      <span style={{ fontSize: 12, color: report.validated ? '#22c55e' : '#ef4444' }}>
                        {report.validated ? '✓ Yes' : '✗ No'}
                      </span>
                    </td>
                    <td style={{ padding: '13px 16px', fontSize: 11, color: '#334155', fontFamily: "'JetBrains Mono', monospace" }}>
                      {report.created_at?.slice(0, 16) || '—'}
                    </td>
                    <td style={{ padding: '13px 16px', fontSize: 11, color: '#334155', fontFamily: "'JetBrains Mono', monospace" }}>
                      {report.updated_at?.slice(0, 16) || '—'}
                    </td>
                    <td style={{ padding: '13px 16px', textAlign: 'right' }}>
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
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ padding: '64px 24px', textAlign: 'center' }}>
            <div style={{ width: 64, height: 64, borderRadius: '50%', background: 'rgba(14,165,233,0.08)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px' }}>
              <svg style={{ width: 28, height: 28, color: '#334155' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <p style={{ fontSize: 14, color: '#475569' }}>
              {search ? `No reports matching "${search}"` : 'No reports found'}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
