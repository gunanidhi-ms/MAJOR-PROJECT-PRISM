// jsPDF and html2canvas are dynamically imported — they only load
// when the user clicks "Download PDF", not on initial page load.
// This keeps the main bundle small.

interface PdfOptions {
  elementId: string;
  filename: string;
  onStart?: () => void;
  onDone?: () => void;
  onError?: (err: Error) => void;
}

export async function downloadReportPdf({
  elementId,
  filename,
  onStart,
  onDone,
  onError,
}: PdfOptions): Promise<void> {
  const el = document.getElementById(elementId);
  if (!el) {
    onError?.(new Error(`Element #${elementId} not found`));
    return;
  }

  onStart?.();

  try {
    // Dynamically import heavy PDF libs only when needed
    const [{ default: jsPDF }, { default: html2canvas }] = await Promise.all([
      import('jspdf'),
      import('html2canvas'),
    ]);

    // Temporarily reveal element off-screen for capture (must be visible for html2canvas)
    const prev = {
      visibility: el.style.visibility,
      position:   el.style.position,
      top:        el.style.top,
      left:       el.style.left,
      zIndex:     el.style.zIndex,
    };

    el.style.visibility = 'visible';
    el.style.position   = 'fixed';
    el.style.top        = '-9999px';
    el.style.left       = '-9999px';
    el.style.zIndex     = '-1';

    // Capture the FULL element (no height clipping) at 2× resolution
    const canvas = await html2canvas(el, {
      scale: 2,
      useCORS: true,
      backgroundColor: '#ffffff',
      logging: false,
      width: 794,
      windowWidth: 794,
    });

    // Restore element
    Object.assign(el.style, prev);

    // A4: 210 × 297 mm  →  at 96 dpi = 794 × 1123 px
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

    const pageW_mm = pdf.internal.pageSize.getWidth();   // 210
    const pageH_mm = pdf.internal.pageSize.getHeight();  // 297

    // How tall is one A4 page in canvas pixels?
    // canvas.width = 794 * 2 (scale=2) = 1588 px → maps to 210 mm
    // So 1 mm = canvas.width / 210 canvas-px
    const pxPerMm   = canvas.width / pageW_mm;
    const pageH_px  = Math.floor(pageH_mm * pxPerMm); // height of one page in canvas px
    const totalPages = Math.ceil(canvas.height / pageH_px);

    for (let pageIdx = 0; pageIdx < totalPages; pageIdx++) {
      if (pageIdx > 0) pdf.addPage();

      // Create a temporary canvas for just this page's slice
      const pageCanvas = document.createElement('canvas');
      pageCanvas.width  = canvas.width;
      pageCanvas.height = pageH_px;

      const ctx = pageCanvas.getContext('2d')!;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, pageCanvas.width, pageCanvas.height);

      // Draw the slice of the full canvas that belongs to this page
      const srcY = pageIdx * pageH_px;
      ctx.drawImage(
        canvas,
        0, srcY,                           // source x, y
        canvas.width, pageH_px,           // source width, height
        0, 0,                              // dest x, y
        pageCanvas.width, pageH_px,        // dest width, height
      );

      const pageImgData = pageCanvas.toDataURL('image/jpeg', 0.95);
      pdf.addImage(pageImgData, 'JPEG', 0, 0, pageW_mm, pageH_mm);
    }

    pdf.save(filename);
    onDone?.();
  } catch (err) {
    onError?.(err instanceof Error ? err : new Error(String(err)));
  }
}
