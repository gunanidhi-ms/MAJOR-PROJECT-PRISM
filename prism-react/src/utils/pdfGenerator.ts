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

    // Temporarily reveal element off-screen for capture
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

    const canvas = await html2canvas(el, {
      scale: 2,              // 2× for crisp text
      useCORS: true,         // allow cross-origin images (Unsplash)
      backgroundColor: '#ffffff',
      logging: false,
      width: 794,
      windowWidth: 794,
    });

    // Restore element
    Object.assign(el.style, prev);

    const imgData = canvas.toDataURL('image/jpeg', 0.95);

    // A4: 210 × 297 mm
    const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });

    const pageW = pdf.internal.pageSize.getWidth();
    const pageH = pdf.internal.pageSize.getHeight();
    const imgW  = pageW;
    const imgH  = (canvas.height * pageW) / canvas.width;

    // Multi-page support
    let remaining = imgH;
    let pageIdx   = 0;

    while (remaining > 0) {
      if (pageIdx > 0) pdf.addPage();
      pdf.addImage(imgData, 'JPEG', 0, -(pageIdx * pageH), imgW, imgH);
      remaining -= pageH;
      pageIdx++;
    }

    pdf.save(filename);
    onDone?.();
  } catch (err) {
    onError?.(err instanceof Error ? err : new Error(String(err)));
  }
}
