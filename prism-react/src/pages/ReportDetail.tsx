import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useReport } from '../hooks/useReports';
import { prismApi } from '../api/prismApi';
import { Badge, statusToBadgeVariant, sourceToBadgeVariant } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Modal } from '../components/ui/Modal';
import { Toast, useToast } from '../components/ui/Toast';
import { SignaturePad } from '../components/ui/SignaturePad';
import { ReportTemplate, type ReportTemplateData } from '../components/reports/ReportTemplate';
import { downloadReportPdf } from '../utils/pdfGenerator';

const WORKFLOW_STEPS = [
  { label: 'Draft Generated',    desc: 'Template + optional LLM refinement' },
  { label: 'Radiologist Review', desc: 'Edit findings & add impression' },
  { label: 'Sign & Finalise',    desc: 'Lock report for distribution' },
];

const TEMPLATE_ID = 'prism-report-template';

export default function ReportDetail() {
  const { studyId = '' } = useParams<{ studyId: string }>();
  const navigate = useNavigate();
  const { report, loading, error, setReport } = useReport(studyId);
  const { toast, show: showToast, hide: hideToast } = useToast();

  const [findings, setFindings]         = useState('');
  const [impression, setImpression]     = useState('');
  const [origFindings, setOrigFindings] = useState('');
  const [origImpression, setOrigImpression] = useState('');
  const [dirty, setDirty]               = useState(false);
  const [saving, setSaving]             = useState(false);

  const [showSignModal, setShowSignModal]   = useState(false);
  const [signing, setSigning]               = useState(false);
  const [radName, setRadName]               = useState('');
  const [radDesig, setRadDesig]             = useState('MD Radiology | FRCR');
  const [sigDataUrl, setSigDataUrl]         = useState<string | null>(null);
  const [sigApplied, setSigApplied]         = useState(false);

  const [generatingPdf, setGeneratingPdf]   = useState(false);
  const [storedSig, setStoredSig]           = useState<string | null>(null);
  const [storedName, setStoredName]         = useState('');
  const [storedDesig, setStoredDesig]       = useState('');
  const [signedAt, setSignedAt]             = useState('');

  useEffect(() => {
    if (!report) return;
    const f = report.findings || '';
    const i = report.impression || '';
    setFindings(f); setImpression(i);
    setOrigFindings(f); setOrigImpression(i);
  }, [report]);

  useEffect(() => {
    setDirty(findings !== origFindings || impression !== origImpression);
  }, [findings, impression, origFindings, origImpression]);

  const saveReport = useCallback(async () => {
    setSaving(true);
    try {
      const updated = await prismApi.updateReport(studyId, { findings, impression });
      setReport(updated);
      setOrigFindings(findings); setOrigImpression(impression);
      setDirty(false);
      showToast('Report saved successfully.', 'success');
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Failed to save', 'error');
    } finally { setSaving(false); }
  }, [studyId, findings, impression, setReport, showToast]);

  const signReport = useCallback(async () => {
    if (!sigApplied) { showToast('Draw and apply your signature first.', 'error'); return; }
    setSigning(true);
    try {
      if (dirty) await prismApi.updateReport(studyId, { findings, impression });
      await prismApi.signReport(studyId);
      const now = new Date().toISOString();
      setSignedAt(now);
      setStoredSig(sigDataUrl);
      setStoredName(radName || 'Dr. [Name]');
      setStoredDesig(radDesig);
      setShowSignModal(false);
      showToast('Report signed! Download the PDF below.', 'success');
      setTimeout(() => navigate(0), 1200);
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Failed to sign', 'error');
    } finally { setSigning(false); }
  }, [sigApplied, sigDataUrl, dirty, studyId, findings, impression, radName, radDesig, showToast, navigate]);

  const handlePdf = useCallback(async () => {
    const dateSlug = new Date().toISOString().slice(0, 10);
    await downloadReportPdf({
      elementId: TEMPLATE_ID,
      filename: `PRISM_Report_${studyId}_${dateSlug}.pdf`,
      onStart: () => setGeneratingPdf(true),
      onDone: () => { setGeneratingPdf(false); showToast('PDF downloaded!', 'success'); },
      onError: (e) => { setGeneratingPdf(false); showToast('PDF error: ' + e.message, 'error'); },
    });
  }, [studyId, showToast]);

  const isSigned = report?.status === 'signed';
  const currentStep = isSigned ? 2 : 1;

  const templateData: ReportTemplateData | null = report ? {
    studyId, status: report.status, source: report.source,
    validated: report.validated, findings, impression,
    createdAt: report.created_at, updatedAt: report.updated_at,
    signedAt: signedAt || report.updated_at,
    radiologistName: storedName || radName || undefined,
    radiologistDesignation: storedDesig || radDesig || undefined,
    signatureDataUrl: storedSig || sigDataUrl || undefined,
  } : null;

  if (loading) {
    return (
      <div style={{ padding: '40px 0' }}>
        {[220, 80, 80].map((h, i) => (
          <div key={i} className="skeleton" style={{ height: h, marginBottom: 16, borderRadius: 14 }} />
        ))}
      </div>
    );
  }

  if (error || !report) {
    return (
      <div style={{ padding: 48, textAlign: 'center', borderRadius: 16, border: '1px solid rgba(239,68,68,0.2)', background: 'rgba(239,68,68,0.04)' }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, color: '#ef4444', marginBottom: 8 }}>Report Not Found</h2>
        <p style={{ fontSize: 13, color: '#64748b', marginBottom: 20 }}>{error}</p>
        <Button variant="danger" size="sm" onClick={() => navigate('/generate')}>Generate New Report</Button>
      </div>
    );
  }

  return (
    <div>
      <Toast {...toast} onClose={hideToast} />

      {/* Hidden A4 template for PDF capture */}
      {templateData && (
        <div style={{ position: 'fixed', top: '-99999px', left: '-99999px', visibility: 'hidden', zIndex: -1, pointerEvents: 'none' }}>
          <ReportTemplate data={templateData} id={TEMPLATE_ID} />
        </div>
      )}

      {/* Sign Modal */}
      <Modal open={showSignModal} onClose={() => !signing && setShowSignModal(false)} title="Sign & Finalise Report">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 16 }}>
          <div>
            <label style={{ fontSize: 11, fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 5 }}>Radiologist Name</label>
            <input type="text" value={radName} onChange={(e) => setRadName(e.target.value)} placeholder="Dr. Firstname Lastname"
              style={{ width: '100%', padding: '8px 12px', fontSize: 13, background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#e2e8f0', outline: 'none', fontFamily: 'inherit' }} />
          </div>
          <div>
            <label style={{ fontSize: 11, fontWeight: 600, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em', display: 'block', marginBottom: 5 }}>Designation</label>
            <input type="text" value={radDesig} onChange={(e) => setRadDesig(e.target.value)} placeholder="MD Radiology | FRCR"
              style={{ width: '100%', padding: '8px 12px', fontSize: 13, background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#e2e8f0', outline: 'none', fontFamily: 'inherit' }} />
          </div>
        </div>

        <SignaturePad
          onSave={(url) => { setSigDataUrl(url); setSigApplied(true); showToast('Signature applied.', 'success'); }}
          onClear={() => { setSigDataUrl(null); setSigApplied(false); }}
        />

        {sigApplied && (
          <div style={{ marginTop: 10, padding: '6px 12px', borderRadius: 7, background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)', fontSize: 12, color: '#22c55e' }}>
            ✓ Signature applied — ready to finalise
          </div>
        )}

        <p style={{ fontSize: 12, color: '#64748b', margin: '14px 0 16px', lineHeight: 1.6 }}>
          This action is <strong style={{ color: '#e2e8f0' }}>final</strong>. Signed reports cannot be edited.
        </p>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
          <Button variant="secondary" onClick={() => setShowSignModal(false)} disabled={signing}>Cancel</Button>
          <Button variant="success" loading={signing} onClick={signReport} disabled={!sigApplied}>
            Confirm Sign & Lock
          </Button>
        </div>
      </Modal>

      {/* Page header */}
      <div className="animate-fade-up" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, marginBottom: 24, flexWrap: 'wrap' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 6 }}>
            <h1 style={{ fontSize: 20, fontWeight: 700, color: '#e2e8f0', fontFamily: "'JetBrains Mono', monospace" }}>{studyId}</h1>
            <Badge variant={statusToBadgeVariant(report.status)}>{report.status.charAt(0).toUpperCase() + report.status.slice(1)}</Badge>
            <Badge variant={sourceToBadgeVariant(report.source)}>{report.source?.toUpperCase() || '—'}</Badge>
            <Badge variant={report.validated ? 'green' : 'red'}>{report.validated ? '✓ Validated' : '✗ Validation Failed'}</Badge>
            {dirty && <Badge variant="amber">Unsaved Changes</Badge>}
          </div>
          <p style={{ fontSize: 11, color: '#334155', fontFamily: "'JetBrains Mono', monospace" }}>
            Created {report.created_at?.slice(0, 16)} · Updated {report.updated_at?.slice(0, 16)}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <Button variant="ghost" size="sm" onClick={() => navigate('/reports')}>← Reports</Button>
          {!isSigned && (
            <>
              <Button variant="secondary" size="sm" loading={saving} disabled={!dirty} onClick={saveReport}>
                {saving ? 'Saving…' : 'Save Changes'}
              </Button>
              <Button variant="success" size="sm" onClick={() => setShowSignModal(true)}>✍ Sign Report</Button>
            </>
          )}
          <Button variant="primary" size="sm" loading={generatingPdf} onClick={handlePdf} style={{ gap: 6 }}>
            <svg style={{ width: 14, height: 14 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            {generatingPdf ? 'Generating…' : 'Download PDF'}
          </Button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 24, alignItems: 'start' }}>
        {/* Editor column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          {!isSigned && (
            <div className="animate-fade-up" style={{ padding: '10px 16px', borderRadius: 10, background: 'rgba(245,158,11,0.07)', border: '1px solid rgba(245,158,11,0.2)', display: 'flex', alignItems: 'flex-start', gap: 10, fontSize: 12, color: '#f59e0b' }}>
              <span style={{ fontWeight: 700, flexShrink: 0 }}>DRAFT</span>
              <span>AI-generated draft — review and edit before signing. PDF available at any stage.</span>
            </div>
          )}

          <div className="surface-card animate-fade-up stagger-1" style={{ padding: 24 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
              <h2 style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0' }}>Clinical Findings</h2>
              {!isSigned && <span style={{ fontSize: 11, color: '#334155' }}>{findings.length} chars</span>}
            </div>
            <textarea value={findings} onChange={(e) => setFindings(e.target.value)} readOnly={isSigned}
              placeholder="AI-generated clinical findings…"
              style={{ width: '100%', minHeight: 240, resize: 'vertical', fontFamily: "'JetBrains Mono', monospace", fontSize: 12, lineHeight: 1.85, color: isSigned ? '#475569' : '#86efac', background: isSigned ? 'rgba(255,255,255,0.02)' : '#060b12', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10, padding: '14px 16px', outline: 'none', cursor: isSigned ? 'default' : 'text', transition: 'border-color 200ms ease' }}
              onFocus={(e) => !isSigned && (e.target.style.borderColor = 'rgba(0,212,255,0.35)')}
              onBlur={(e) => (e.target.style.borderColor = 'rgba(255,255,255,0.07)')} />
          </div>

          <div className="surface-card animate-fade-up stagger-2" style={{ padding: 24 }}>
            <h2 style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0', marginBottom: 14 }}>Radiologist Impression</h2>
            <textarea value={impression} onChange={(e) => setImpression(e.target.value)} readOnly={isSigned}
              placeholder="Clinical summary and recommendations…"
              style={{ width: '100%', minHeight: 130, resize: 'vertical', fontFamily: "'JetBrains Mono', monospace", fontSize: 12, lineHeight: 1.85, color: isSigned ? '#475569' : '#cbd5e1', background: isSigned ? 'rgba(255,255,255,0.02)' : '#060b12', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 10, padding: '14px 16px', outline: 'none', cursor: isSigned ? 'default' : 'text', transition: 'border-color 200ms ease' }}
              onFocus={(e) => !isSigned && (e.target.style.borderColor = 'rgba(0,212,255,0.35)')}
              onBlur={(e) => (e.target.style.borderColor = 'rgba(255,255,255,0.07)')} />
          </div>

          {/* Signature preview */}
          {isSigned && storedSig && (
            <div className="surface-card animate-fade-up stagger-3" style={{ padding: 20 }}>
              <p style={{ fontSize: 11, fontWeight: 600, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>Digital Signature on File</p>
              <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                <div style={{ padding: '8px 16px', borderRadius: 8, background: '#f8fafc', border: '1px solid rgba(255,255,255,0.1)' }}>
                  <img src={storedSig} alt="Signature" style={{ maxHeight: 56, maxWidth: 180 }} />
                </div>
                <div>
                  <p style={{ fontSize: 13, fontWeight: 700, color: '#e2e8f0' }}>{storedName}</p>
                  <p style={{ fontSize: 11, color: '#64748b' }}>{storedDesig}</p>
                  <p style={{ fontSize: 10, color: '#334155', marginTop: 4 }}>
                    Signed {signedAt ? new Date(signedAt).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' }) : ''}
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="surface-card animate-fade-up stagger-1 scan-overlay" style={{ overflow: 'hidden' }}>
            <div style={{ position: 'relative', height: 130 }}>
              <img src="https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=600&h=300&fit=crop" alt="CT scan" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(13,20,32,0.92) 0%, transparent 55%)' }} />
              <div style={{ position: 'absolute', bottom: 10, left: 14 }}>
                <p style={{ fontSize: 9, fontWeight: 600, color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Modality</p>
                <p style={{ fontSize: 12, fontWeight: 600, color: '#e2e8f0' }}>CT — Structured Analysis</p>
              </div>
            </div>
            <div style={{ padding: '12px 16px' }}>
              {(['Study ID', 'Source', 'Validated'] as const).map((lbl) => {
                const val = lbl === 'Study ID' ? studyId
                  : lbl === 'Source' ? (report.source?.charAt(0).toUpperCase() + report.source?.slice(1) || '—')
                  : (report.validated ? 'Yes' : 'No');
                const color = lbl === 'Validated' ? (report.validated ? '#22c55e' : '#ef4444') : '#e2e8f0';
                return (
                  <div key={lbl} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '5px 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                    <span style={{ fontSize: 11, color: '#334155' }}>{lbl}</span>
                    <span style={{ fontSize: 11, fontWeight: 600, color, fontFamily: "'JetBrains Mono', monospace" }}>{val}</span>
                  </div>
                );
              })}
              {report.validation_reason && <p style={{ fontSize: 10, color: '#475569', marginTop: 8, lineHeight: 1.5 }}>{report.validation_reason}</p>}
            </div>
          </div>

          {/* Workflow stepper */}
          <div className="surface-card animate-fade-up stagger-2" style={{ padding: 18 }}>
            <h3 style={{ fontSize: 10, fontWeight: 700, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.12em', marginBottom: 16 }}>Workflow</h3>
            <ol style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 14, position: 'relative' }}>
              <div style={{ position: 'absolute', left: 11, top: 24, bottom: 24, width: 1, background: 'rgba(255,255,255,0.05)' }} />
              {WORKFLOW_STEPS.map((step, i) => {
                const isDone = i <= currentStep;
                const isCurrent = i === currentStep && !isSigned;
                return (
                  <li key={step.label} style={{ display: 'flex', gap: 12, position: 'relative' }}>
                    <span style={{
                      width: 23, height: 23, borderRadius: '50%', flexShrink: 0, zIndex: 1,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 10, fontWeight: 700,
                      background: isDone ? '#22c55e' : isCurrent ? 'rgba(14,165,233,0.15)' : 'rgba(255,255,255,0.04)',
                      color: isDone ? '#fff' : isCurrent ? '#0ea5e9' : '#334155',
                      border: isCurrent ? '1px solid rgba(14,165,233,0.3)' : 'none',
                      boxShadow: isDone ? '0 0 10px rgba(34,197,94,0.3)' : 'none',
                    }}>
                      {isDone ? '✓' : i + 1}
                    </span>
                    <div>
                      <p style={{ fontSize: 12, fontWeight: 600, color: isDone || isCurrent ? '#e2e8f0' : '#475569' }}>{step.label}</p>
                      <p style={{ fontSize: 10, color: '#334155', marginTop: 1 }}>{step.desc}</p>
                    </div>
                  </li>
                );
              })}
            </ol>
          </div>

          {/* Status card */}
          {isSigned ? (
            <div className="animate-fade-up stagger-3" style={{ padding: 16, borderRadius: 12, background: 'rgba(34,197,94,0.07)', border: '1px solid rgba(34,197,94,0.18)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                <svg style={{ width: 16, height: 16, color: '#22c55e' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <p style={{ fontSize: 13, fontWeight: 700, color: '#22c55e' }}>Signed & Locked</p>
              </div>
              <p style={{ fontSize: 11, color: '#475569', lineHeight: 1.5, marginBottom: 12 }}>This report is finalised. Download the PDF to distribute.</p>
              <button onClick={handlePdf} disabled={generatingPdf}
                style={{ width: '100%', padding: '9px 0', borderRadius: 9, fontSize: 12, fontWeight: 700, cursor: 'pointer', background: generatingPdf ? 'rgba(14,165,233,0.1)' : 'linear-gradient(135deg, #0ea5e9, #0284c7)', border: 'none', color: '#fff', boxShadow: generatingPdf ? 'none' : '0 0 20px rgba(14,165,233,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
                <svg style={{ width: 14, height: 14 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                {generatingPdf ? 'Generating PDF…' : 'Download Signed PDF'}
              </button>
            </div>
          ) : (
            <div className="animate-fade-up stagger-3" style={{ padding: 16, borderRadius: 12, background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.15)' }}>
              <p style={{ fontSize: 12, fontWeight: 700, color: '#f59e0b', marginBottom: 4 }}>⏳ Awaiting Signature</p>
              <p style={{ fontSize: 11, color: '#475569', lineHeight: 1.5 }}>Review findings, then sign to lock and distribute the report.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
