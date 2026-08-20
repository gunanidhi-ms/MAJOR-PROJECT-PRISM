import { useRef, useEffect, useCallback } from 'react';

interface SignaturePadProps {
  onSave: (dataUrl: string) => void;
  onClear?: () => void;
}

export function SignaturePad({ onSave, onClear }: SignaturePadProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);
  const hasStrokes = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // High-DPI support
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // Placeholder text
    drawPlaceholder(ctx, rect.width, rect.height);
  }, []);

  function drawPlaceholder(ctx: CanvasRenderingContext2D, w: number, h: number) {
    ctx.save();
    ctx.font = '13px Inter, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'center';
    ctx.fillText('Sign here ↓', w / 2, h / 2 - 4);
    ctx.restore();
  }

  function getPos(e: React.MouseEvent | React.TouchEvent) {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    if ('touches' in e) {
      return {
        x: e.touches[0].clientX - rect.left,
        y: e.touches[0].clientY - rect.top,
      };
    }
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function startDraw(e: React.MouseEvent | React.TouchEvent) {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;

    if (!hasStrokes.current) {
      // Clear placeholder text on first stroke
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width * dpr, rect.height * dpr);
      hasStrokes.current = true;
    }

    isDrawing.current = true;
    const { x, y } = getPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
  }

  function draw(e: React.MouseEvent | React.TouchEvent) {
    e.preventDefault();
    if (!isDrawing.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  }

  function endDraw() {
    isDrawing.current = false;
  }

  const clearPad = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width * dpr, rect.height * dpr);
    hasStrokes.current = false;
    drawPlaceholder(ctx, rect.width, rect.height);
    onClear?.();
  }, [onClear]);

  function saveSignature() {
    const canvas = canvasRef.current;
    if (!canvas || !hasStrokes.current) return;
    // Export on white background
    const offscreen = document.createElement('canvas');
    offscreen.width = canvas.width;
    offscreen.height = canvas.height;
    const ctx = offscreen.getContext('2d')!;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, offscreen.width, offscreen.height);
    ctx.drawImage(canvas, 0, 0);
    onSave(offscreen.toDataURL('image/png'));
  }

  return (
    <div>
      <p style={{ fontSize: 11, color: '#64748b', marginBottom: 8 }}>
        Draw your signature below using mouse or touch:
      </p>
      <div style={{ position: 'relative', borderRadius: 10, overflow: 'hidden', border: '1px solid rgba(255,255,255,0.12)', background: '#f8fafc' }}>
        <canvas
          ref={canvasRef}
          style={{ width: '100%', height: 140, display: 'block', cursor: 'crosshair', touchAction: 'none' }}
          onMouseDown={startDraw}
          onMouseMove={draw}
          onMouseUp={endDraw}
          onMouseLeave={endDraw}
          onTouchStart={startDraw}
          onTouchMove={draw}
          onTouchEnd={endDraw}
        />
      </div>
      <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
        <button
          onClick={clearPad}
          style={{
            flex: 1, padding: '7px 0', fontSize: 12, fontWeight: 600,
            background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 8, color: '#94a3b8', cursor: 'pointer',
          }}
        >
          Clear
        </button>
        <button
          onClick={saveSignature}
          style={{
            flex: 2, padding: '7px 0', fontSize: 12, fontWeight: 600,
            background: 'linear-gradient(135deg, #16a34a, #15803d)',
            border: 'none', borderRadius: 8, color: '#fff', cursor: 'pointer',
            boxShadow: '0 0 16px rgba(22,163,74,0.3)',
          }}
        >
          ✓ Apply Signature
        </button>
      </div>
    </div>
  );
}
