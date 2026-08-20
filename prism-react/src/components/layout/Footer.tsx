import { Link } from 'react-router-dom';

const QUICK_LINKS = [
  { to: '/', label: 'Dashboard' },
  { to: '/generate', label: 'Generate Report' },
  { to: '/reports', label: 'All Reports' },
];

const CENTRE = {
  name: 'PRISM Diagnostic Centre',
  address: '4th Floor, MedTech Tower, Bengaluru — 560001',
  phone: '+91 80 4567 8900',
  email: 'reports@prism-diagnostic.in',
  regNo: 'KAR/DIA/2024/0042',
};

const TECH_STACK = ['Django 5.2', 'FastAPI', 'React 19', 'TypeScript', 'Ollama LLM'];

function currentYear() { return new Date().getFullYear(); }

export function Footer() {
  return (
    <footer style={{
      background: 'rgba(5,8,14,0.98)',
      borderTop: '1px solid rgba(255,255,255,0.06)',
      marginTop: 'auto',
    }}>
      {/* Main footer content */}
      <div style={{
        maxWidth: 1280, margin: '0 auto',
        padding: '40px 24px 24px',
        display: 'grid',
        gridTemplateColumns: '1.4fr 1fr 1fr',
        gap: 40,
      }}>

        {/* Col 1 — Brand */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8,
              background: 'linear-gradient(135deg, #0ea5e9, #6366f1)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              flexShrink: 0,
            }}>
              <svg style={{ width: 17, height: 17, color: '#fff' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3" />
              </svg>
            </div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 800, color: '#e2e8f0', letterSpacing: '-0.02em' }}>PRISM</div>
              <div style={{ fontSize: 9, color: '#334155', fontWeight: 500, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                Autonomous Radiology Intelligence
              </div>
            </div>
          </div>
          <p style={{ fontSize: 12, color: '#475569', lineHeight: 1.7, maxWidth: 260, marginBottom: 16 }}>
            AI-powered CT report generation platform for radiologists. Phase 3 — autonomous draft generation with LLM refinement and validator gate.
          </p>
          {/* Phase badge */}
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, padding: '4px 10px', borderRadius: 20, background: 'rgba(99,102,241,0.1)', border: '1px solid rgba(99,102,241,0.2)' }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#6366f1', display: 'inline-block' }} />
            <span style={{ fontSize: 10, fontWeight: 600, color: '#818cf8' }}>Phase 3 · Report Generation</span>
          </div>
        </div>

        {/* Col 2 — Quick links */}
        <div>
          <h4 style={{ fontSize: 11, fontWeight: 700, color: '#e2e8f0', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 14 }}>
            Navigation
          </h4>
          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 8 }}>
            {QUICK_LINKS.map((l) => (
              <li key={l.to}>
                <Link
                  to={l.to}
                  style={{
                    fontSize: 13, color: '#475569', textDecoration: 'none',
                    display: 'flex', alignItems: 'center', gap: 6,
                    transition: 'color 150ms ease',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = '#00d4ff')}
                  onMouseLeave={(e) => (e.currentTarget.style.color = '#475569')}
                >
                  <svg style={{ width: 12, height: 12, flexShrink: 0 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                  {l.label}
                </Link>
              </li>
            ))}
          </ul>

          <h4 style={{ fontSize: 11, fontWeight: 700, color: '#e2e8f0', textTransform: 'uppercase', letterSpacing: '0.1em', margin: '20px 0 10px' }}>
            Tech Stack
          </h4>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {TECH_STACK.map((t) => (
              <span key={t} style={{
                fontSize: 10, fontWeight: 600, color: '#334155',
                background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)',
                padding: '3px 8px', borderRadius: 5,
                fontFamily: "'JetBrains Mono', monospace",
              }}>
                {t}
              </span>
            ))}
          </div>
        </div>

        {/* Col 3 — Diagnostic centre info */}
        <div>
          <h4 style={{ fontSize: 11, fontWeight: 700, color: '#e2e8f0', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 14 }}>
            Diagnostic Centre
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
              <svg style={{ width: 13, height: 13, color: '#334155', flexShrink: 0, marginTop: 1 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
              </svg>
              <div>
                <p style={{ fontSize: 12, fontWeight: 600, color: '#94a3b8' }}>{CENTRE.name}</p>
                <p style={{ fontSize: 11, color: '#475569', marginTop: 1 }}>{CENTRE.address}</p>
              </div>
            </div>
            {[
              { icon: 'M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z', value: CENTRE.phone },
              { icon: 'M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z', value: CENTRE.email },
            ].map((item) => (
              <div key={item.value} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <svg style={{ width: 13, height: 13, color: '#334155', flexShrink: 0 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={item.icon} />
                </svg>
                <span style={{ fontSize: 11, color: '#475569', fontFamily: "'JetBrains Mono', monospace" }}>{item.value}</span>
              </div>
            ))}
            <div style={{ marginTop: 4, padding: '6px 10px', borderRadius: 6, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
              <span style={{ fontSize: 10, color: '#334155', fontFamily: "'JetBrains Mono', monospace" }}>
                Reg. No:&nbsp;
              </span>
              <span style={{ fontSize: 10, color: '#475569', fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
                {CENTRE.regNo}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom bar */}
      <div style={{
        maxWidth: 1280, margin: '0 auto',
        padding: '14px 24px',
        borderTop: '1px solid rgba(255,255,255,0.04)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 16,
        flexWrap: 'wrap',
      }}>
        <p style={{ fontSize: 11, color: '#1e293b' }}>
          © {currentYear()} {CENTRE.name} · All rights reserved.
        </p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 10, color: '#1e293b' }}>
            Disclaimer: AI-generated reports require radiologist sign-off before clinical use.
          </span>
          <span style={{
            fontSize: 10, fontWeight: 700, color: '#0ea5e9',
            background: 'rgba(14,165,233,0.08)', padding: '2px 8px', borderRadius: 4,
          }}>
            PRISM AI · Phase 3
          </span>
        </div>
      </div>
    </footer>
  );
}
