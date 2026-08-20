import { useState, useEffect, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { prismApi } from '../api/prismApi';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';
import { Toast, useToast } from '../components/ui/Toast';

const SAMPLE_CASES = [
  {
    id: 'normal_case',
    title: 'Normal CT KUB',
    description: 'All organs within normal limits — baseline study.',
    modality: 'CT KUB',
    badge: 'Normal',
    badgeVariant: 'green' as const,
    image: 'https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?w=400&h=250&fit=crop',
  },
  {
    id: 'single_lesion',
    title: 'Single Renal Lesion',
    description: 'Well-defined hypodense lesion in the right kidney lower pole.',
    modality: 'CT KUB',
    badge: 'Anomaly',
    badgeVariant: 'amber' as const,
    image: 'https://images.unsplash.com/photo-1559757148-5c350d0d3c56?w=400&h=250&fit=crop',
  },
  {
    id: 'multi_lesion',
    title: 'Multiple Renal Lesions',
    description: 'Bilateral renal lesions requiring detailed characterisation.',
    modality: 'CT KUB',
    badge: 'Complex',
    badgeVariant: 'red' as const,
    image: 'https://images.unsplash.com/photo-1582719478250-c89cae4dc85b?w=400&h=250&fit=crop',
  },
  {
    id: 'multi_organ',
    title: 'Multi-Organ Findings',
    description: 'Hepatic and renal abnormalities on contrast-enhanced CT.',
    modality: 'CT Abdomen',
    badge: 'Multi-organ',
    badgeVariant: 'violet' as const,
    image: 'https://images.unsplash.com/photo-1530497610245-94d3c16cda28?w=400&h=250&fit=crop',
  },
];

const PIPELINE_STAGES = [
  { num: 1, label: 'Template Engine', desc: 'Deterministic report from structured data', color: '#0ea5e9' },
  { num: 2, label: 'LLM Refinement', desc: 'Ollama grammar polish (optional)', color: '#a855f7' },
  { num: 3, label: 'Validator Gate', desc: 'Anti-hallucination check before save', color: '#22c55e' },
];

export default function Generate() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { toast, show: showToast, hide: hideToast } = useToast();

  const [selectedCase, setSelectedCase] = useState(searchParams.get('case') || 'normal_case');
  const [jsonText, setJsonText] = useState('');
  const [jsonError, setJsonError] = useState('');
  const [jsonValid, setJsonValid] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingCase, setLoadingCase] = useState(false);
  const [pipelineActive, setPipelineActive] = useState(-1);

  const loadCase = useCallback(async (caseId: string) => {
    setSelectedCase(caseId);
    setLoadingCase(true);
    try {
      const data = await prismApi.loadSampleCase(caseId);
      setJsonText(JSON.stringify(data, null, 2));
      setJsonError('');
      setJsonValid(true);
    } catch (err) {
      showToast(err instanceof Error ? err.message : 'Failed to load case', 'error');
    } finally {
      setLoadingCase(false);
    }
  }, [showToast]);

  useEffect(() => { loadCase(selectedCase); }, []); // eslint-disable-line

  function formatJson() {
    try {
      const parsed = JSON.parse(jsonText);
      setJsonText(JSON.stringify(parsed, null, 2));
      setJsonError(''); setJsonValid(true);
    } catch (err) {
      setJsonError('Invalid JSON: ' + (err instanceof Error ? err.message : 'parse error'));
      setJsonValid(false);
    }
  }

  function validateJson() {
    try {
      const parsed = JSON.parse(jsonText) as Record<string, unknown>;
      if (!parsed.study_id) throw new Error('Missing study_id');
      if (!Array.isArray(parsed.organs) || !parsed.organs.length) throw new Error('Missing organs array');
      setJsonError(''); setJsonValid(true);
      showToast('JSON is valid and ready to submit.', 'success');
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Validation error';
      setJsonError(msg); setJsonValid(false);
    }
  }

  async function generateReport() {
    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(jsonText);
    } catch {
      setJsonError('Invalid JSON — fix syntax errors first.'); return;
    }

    setLoading(true);
    setPipelineActive(0);

    // Animate stages
    const stageInterval = setInterval(() => {
      setPipelineActive((prev) => {
        if (prev >= 2) { clearInterval(stageInterval); return prev; }
        return prev + 1;
      });
    }, 700);

    try {
      const result = await prismApi.generateReport(payload);
      clearInterval(stageInterval);
      setPipelineActive(2);
      showToast('Report generated! Redirecting…', 'success');
      setTimeout(() => navigate(`/report/${encodeURIComponent(result.study_id)}`), 800);
    } catch (err) {
      clearInterval(stageInterval);
      setPipelineActive(-1);
      showToast(err instanceof Error ? err.message : 'Generation failed', 'error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <Toast {...toast} onClose={hideToast} />

      {/* Page header */}
      <div className="animate-fade-up" style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#e2e8f0', letterSpacing: '-0.02em' }}>
          AI Report Workstation
        </h1>
        <p style={{ fontSize: 13, color: '#475569', marginTop: 4 }}>
          Generate clinical radiology reports from structured CT findings JSON
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 24, alignItems: 'start' }}>
        {/* Case selector */}
        <div className="animate-fade-up stagger-1">
          <p style={{ fontSize: 10, fontWeight: 600, color: '#334155', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 12 }}>
            Sample Cases
          </p>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {SAMPLE_CASES.map((c) => {
              const isSelected = selectedCase === c.id;
              return (
                <button
                  key={c.id}
                  onClick={() => loadCase(c.id)}
                  style={{
                    width: '100%', textAlign: 'left', background: 'none', cursor: 'pointer',
                    border: `1px solid ${isSelected ? 'rgba(0,212,255,0.4)' : 'rgba(255,255,255,0.06)'}`,
                    borderRadius: 12, overflow: 'hidden',
                    transition: 'all 200ms ease',
                    boxShadow: isSelected ? '0 0 20px rgba(0,212,255,0.15)' : 'none',
                    outline: 'none',
                  }}
                >
                  <div style={{ position: 'relative', height: 100, overflow: 'hidden' }}>
                    <img src={c.image} alt={c.title} style={{ width: '100%', height: '100%', objectFit: 'cover', transition: 'transform 300ms ease',
                      transform: isSelected ? 'scale(1.03)' : 'scale(1)' }} />
                    <div style={{ position: 'absolute', inset: 0, background: 'linear-gradient(to top, rgba(0,0,0,0.75) 0%, transparent 60%)' }} />
                    <div style={{ position: 'absolute', top: 8, right: 8 }}>
                      <Badge variant={c.badgeVariant}>{c.badge}</Badge>
                    </div>
                    <div style={{ position: 'absolute', bottom: 8, left: 10 }}>
                      <p style={{ fontSize: 11, fontWeight: 600, color: '#e2e8f0' }}>{c.modality}</p>
                    </div>
                  </div>
                  <div style={{ padding: '10px 12px', background: isSelected ? 'rgba(0,212,255,0.06)' : 'rgba(255,255,255,0.02)' }}>
                    <p style={{ fontSize: 12, fontWeight: 600, color: isSelected ? '#00d4ff' : '#e2e8f0', marginBottom: 2 }}>{c.title}</p>
                    <p style={{ fontSize: 11, color: '#475569' }}>{c.description}</p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Right panel */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* AI Pipeline */}
          <div className="surface-card animate-fade-up stagger-2" style={{ padding: 24 }}>
            <h2 style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0', marginBottom: 4 }}>AI Analysis Pipeline</h2>
            <p style={{ fontSize: 12, color: '#475569', marginBottom: 18 }}>Three-stage quality assurance for clinical accuracy</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              {PIPELINE_STAGES.map((stage, i) => {
                const isActive = pipelineActive === i;
                const isDone = pipelineActive > i;
                return (
                  <div
                    key={stage.num}
                    style={{
                      padding: '14px 16px', borderRadius: 12,
                      border: `1px solid ${isActive || isDone ? `${stage.color}30` : 'rgba(255,255,255,0.06)'}`,
                      background: isActive || isDone ? `${stage.color}08` : 'rgba(255,255,255,0.02)',
                      transition: 'all 400ms ease',
                    }}
                  >
                    <div style={{
                      width: 32, height: 32, borderRadius: 9, background: isDone ? stage.color : isActive ? `${stage.color}30` : 'rgba(255,255,255,0.06)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 10,
                      fontSize: 12, fontWeight: 700, color: isDone ? '#fff' : stage.color,
                      transition: 'all 400ms ease',
                    }}>
                      {isDone ? '✓' : stage.num}
                    </div>
                    <p style={{ fontSize: 12, fontWeight: 600, color: isActive || isDone ? '#e2e8f0' : '#475569', marginBottom: 4 }}>{stage.label}</p>
                    <p style={{ fontSize: 11, color: '#334155' }}>{stage.desc}</p>
                    {(isActive || isDone) && (
                      <p style={{ fontSize: 10, color: stage.color, marginTop: 6, fontWeight: 600 }}>
                        {isDone ? 'Done ✓' : 'Processing…'}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* JSON Editor */}
          <div className="surface-card animate-fade-up stagger-3" style={{ padding: 24 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
              <div>
                <h2 style={{ fontSize: 14, fontWeight: 700, color: '#e2e8f0', marginBottom: 2 }}>Structured Findings JSON</h2>
                <p style={{ fontSize: 12, color: '#475569' }}>Phase 2 output — organs, anomalies, measurements</p>
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button onClick={formatJson} style={{ padding: '6px 12px', fontSize: 11, fontWeight: 600, color: '#94a3b8', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 7, cursor: 'pointer' }}>
                  Format
                </button>
                <button onClick={validateJson} style={{ padding: '6px 12px', fontSize: 11, fontWeight: 600, color: '#94a3b8', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 7, cursor: 'pointer' }}>
                  Validate
                </button>
              </div>
            </div>

            <textarea
              value={loadingCase ? 'Loading…' : jsonText}
              onChange={(e) => { setJsonText(e.target.value); setJsonError(''); setJsonValid(false); }}
              spellCheck={false}
              placeholder='{"study_id": "...", "modality": "CT", "organs": [...]}'
              style={{
                width: '100%', minHeight: 260, resize: 'vertical',
                fontFamily: "'JetBrains Mono', monospace", fontSize: 12, lineHeight: 1.7,
                color: '#86efac', background: '#060b12',
                border: `1px solid ${jsonError ? 'rgba(239,68,68,0.4)' : jsonValid ? 'rgba(34,197,94,0.3)' : 'rgba(255,255,255,0.07)'}`,
                borderRadius: 10, padding: '14px 16px',
                outline: 'none', transition: 'border-color 200ms ease',
              }}
              onFocus={(e) => (e.target.style.borderColor = 'rgba(0,212,255,0.4)')}
              onBlur={(e) => (e.target.style.borderColor = jsonError ? 'rgba(239,68,68,0.4)' : jsonValid ? 'rgba(34,197,94,0.3)' : 'rgba(255,255,255,0.07)')}
            />

            {jsonError && (
              <p style={{ marginTop: 8, fontSize: 12, color: '#ef4444', display: 'flex', alignItems: 'center', gap: 6 }}>
                <svg style={{ width: 14, height: 14 }} fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                {jsonError}
              </p>
            )}
            {jsonValid && !jsonError && (
              <p style={{ marginTop: 8, fontSize: 12, color: '#22c55e', display: 'flex', alignItems: 'center', gap: 6 }}>
                <svg style={{ width: 14, height: 14 }} fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
                Valid JSON structure
              </p>
            )}

            <div style={{ marginTop: 20, paddingTop: 20, borderTop: '1px solid rgba(255,255,255,0.05)', display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
              <Button
                onClick={generateReport}
                loading={loading}
                disabled={loading || !!jsonError}
                size="lg"
              >
                <svg style={{ width: 16, height: 16 }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
                </svg>
                {loading ? 'Generating…' : 'Generate Draft Report'}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
