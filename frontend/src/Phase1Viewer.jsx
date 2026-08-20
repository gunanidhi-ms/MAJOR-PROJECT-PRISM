import { useState, useEffect, useRef, useCallback, useMemo } from 'react'

/**
 * Phase1Viewer — Live CT Slice Viewer with DICOM-style Controls
 *
 * Features:
 *   - Real-time WebSocket slice streaming
 *   - Full-resolution CT image display
 *   - Window/Level controls with clinical presets
 *   - Slice scrolling (mouse wheel / slider)
 *   - Finding overlays with bounding boxes
 *   - Scanner control panel
 */

const WS_URL = 'ws://localhost:8001/ws/alerts'
const API_URL = 'http://localhost:8001'
const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000]

// Clinical windowing presets
const WINDOW_PRESETS = {
  'DICOM Default': { wc: 'auto', ww: 'auto' },
  'Soft Tissue': { wc: 50, ww: 350 },
  'Lung':        { wc: -600, ww: 1500 },
  'Bone':        { wc: 400, ww: 1800 },
  'Mediastinum': { wc: 50, ww: 400 },
  'Brain':       { wc: 40, ww: 80 },
  'Liver':       { wc: 60, ww: 150 },
}

export default function Phase1Viewer() {
  const [slices, setSlices] = useState([])
  const [selectedIdx, setSelectedIdx] = useState(0)
  const [connectionStatus, setConnectionStatus] = useState('disconnected')
  // Stats derived from current slices array — always reflects current patient only
  const stats = useMemo(() => {
    const total = slices.length
    const flagged = slices.filter(s => s.flagged).length
    const totalTime = slices.reduce((sum, s) => sum + (s.processing_time_ms || 0), 0)
    return {
      total,
      flagged,
      avgTime: total > 0 ? totalTime / total : 0
    }
  }, [slices])

  // Scanner Control State
  const [scannerRunning, setScannerRunning] = useState(false)
  const [datasetDir, setDatasetDir] = useState("C:\\Users\\gunan\\Downloads\\TCIA files")
  const [delayMs, setDelayMs] = useState(30)
  const [patientId, setPatientId] = useState("")

  // Windowing state
  const [windowPreset, setWindowPreset] = useState('DICOM Default')
  const [windowCenter, setWindowCenter] = useState(50)
  const [windowWidth, setWindowWidth] = useState(350)
  const [showFindings, setShowFindings] = useState(true)
  const [showPhase2, setShowPhase2] = useState(false)
  const [isAutoPlay, setIsAutoPlay] = useState(false)


  // Image display state
  const [viewerImageUrl, setViewerImageUrl] = useState(null)
  const [imageNaturalSize, setImageNaturalSize] = useState({ w: 512, h: 512 })
  const [viewerSize, setViewerSize] = useState({ w: 800, h: 600 })

  const wsRef = useRef(null)
  const reconnectAttempt = useRef(0)
  const reconnectTimer = useRef(null)
  const imgRef = useRef(null)
  const overlayRef = useRef(null)
  const sliceListRef = useRef(null)
  const viewerRef = useRef(null)
  const wrapperRef = useRef(null)
  const autoPlayRef = useRef(null)

  const displayIdx = Math.floor(selectedIdx)
  const selectedSlice = slices[displayIdx] || null

  // ── WebSocket Connection ──────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setConnectionStatus('connecting')

    try {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        setConnectionStatus('connected')
        reconnectAttempt.current = 0
      }

      ws.onmessage = (event) => {
        try {
          const alert = JSON.parse(event.data)
          if (alert.type === 'pong') return

          setSlices(prev => {
            if (prev.some(s => s.uid === alert.uid)) return prev
            const updated = [...prev, { ...alert, _isNew: true }]
            updated.sort((a, b) => a.slice_id - b.slice_id)
            return updated
          })

          

        } catch (e) {
          console.error('[PRISM] Parse error:', e)
        }
      }

      ws.onclose = () => {
        setConnectionStatus('disconnected')
        wsRef.current = null
        scheduleReconnect()
      }

      ws.onerror = (err) => {
        console.error('[PRISM] WebSocket error:', err)
        ws.close()
      }
    } catch (err) {
      setConnectionStatus('disconnected')
      scheduleReconnect()
    }
    
    // Cleanup on unmount (essential for React StrictMode)
    return () => {
      if (wsRef.current) {
        // Prevent onclose from triggering a reconnect attempt during unmount
        wsRef.current.onclose = null;
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [])

  const scheduleReconnect = useCallback(() => {
    const delay = RECONNECT_DELAYS[Math.min(reconnectAttempt.current, RECONNECT_DELAYS.length - 1)]
    reconnectAttempt.current++
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    reconnectTimer.current = setTimeout(connect, delay)
  }, [connect])

  useEffect(() => {
    connect()
    return () => {
      if (wsRef.current) wsRef.current.close()
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    }
  }, [connect])

  // ── Auto-follow latest slice ──────────────────────────────────────────────
  useEffect(() => {
    if (isAutoPlay && slices.length > 0) {
      setSelectedIdx(slices.length - 1)
    }
  }, [slices.length, isAutoPlay])

  // ── DICOM Default Windowing ───────────────────────────────────────────────
  useEffect(() => {
    if (windowPreset === 'DICOM Default' && selectedSlice) {
      const wc = selectedSlice.window_center_default ?? 50
      const ww = selectedSlice.window_width_default ?? 350
      setWindowCenter(wc)
      setWindowWidth(ww)
    }
  }, [selectedSlice, windowPreset])

  // ── Clear "new" animation ─────────────────────────────────────────────────
  useEffect(() => {
    const timer = setTimeout(() => {
      setSlices(prev => prev.map(s => ({ ...s, _isNew: false })))
    }, 3000)
    return () => clearTimeout(timer)
  }, [slices.length])

  // ── Fetch windowed image when slice or window changes ─────────────────────
  useEffect(() => {
    if (!selectedSlice) {
      setViewerImageUrl(null)
      return
    }
    const url = `${API_URL}/api/slices/${selectedSlice.slice_id}/windowed?wc=${windowCenter}&ww=${windowWidth}&t=${Date.now()}`
    setViewerImageUrl(url)
  }, [selectedSlice?.slice_id, windowCenter, windowWidth])

  // ── Resize observer to track viewer area ──────────────────────────────────
  useEffect(() => {
    const wrapper = wrapperRef.current
    if (!wrapper) return

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width > 0 && height > 0) {
          setViewerSize({ w: width, h: height })
        }
      }
    })
    ro.observe(wrapper)
    return () => ro.disconnect()
  }, [])

  // Compute the fitted image dimensions (fit image inside viewer, maintaining aspect ratio)
  const fittedSize = (() => {
    const { w: vw, h: vh } = viewerSize
    const { w: iw, h: ih } = imageNaturalSize
    if (iw === 0 || ih === 0) return { w: 0, h: 0 }
    const scaleX = vw / iw
    const scaleY = vh / ih
    const scale = Math.min(scaleX, scaleY)
    return { w: Math.floor(iw * scale), h: Math.floor(ih * scale) }
  })()

  // ── Draw findings overlay ─────────────────────────────────────────────────
  useEffect(() => {
    const overlay = overlayRef.current
    if (!overlay) return

    const ctx = overlay.getContext('2d')
    const { w: fw, h: fh } = fittedSize
    const { w: nw, h: nh } = imageNaturalSize

    // Set canvas buffer to match CSS display size
    overlay.width = fw
    overlay.height = fh
    ctx.clearRect(0, 0, fw, fh)

    if (!showFindings || !selectedSlice?.findings?.length) return

    const scaleX = fw / nw
    const scaleY = fh / nh

    for (const finding of selectedSlice.findings) {
      const [bx, by, bw, bh] = finding.bbox
      
      // Color and label based on finding type
      const FINDING_STYLES = {
        'ground_glass':        { color: '#f97316', label: 'GGO' },
        'solid_nodule':        { color: '#ef4444', label: 'NOD' },
        'hypodense_lesion':    { color: '#3b82f6', label: 'HYPO' },
        'hyperdense_lesion':   { color: '#ef4444', label: 'HYPER' },
        'hypodense_region':    { color: '#8b5cf6', label: 'HYPO' },
        'hemorrhage':          { color: '#dc2626', label: 'HEM' },
        'lytic_lesion':        { color: '#f59e0b', label: 'LYTIC' },
        'low_density_anomaly': { color: '#06b6d4', label: 'LOW' },
        'high_density_anomaly':{ color: '#ef4444', label: 'HIGH' },
      }
      const style = FINDING_STYLES[finding.finding_type] || { color: '#f97316', label: finding.finding_type?.substring(0, 4)?.toUpperCase() || '?' }
      const color = style.color

      const sx = bx * scaleX, sy = by * scaleY
      const sw = bw * scaleX, sh = bh * scaleY

      // Bounding box
      ctx.strokeStyle = color
      ctx.lineWidth = Math.max(1.5, 1.5 * Math.min(scaleX, scaleY))
      ctx.setLineDash([4 * Math.min(scaleX, scaleY), 4 * Math.min(scaleX, scaleY)])
      ctx.strokeRect(sx, sy, sw, sh)

      // Label
      const labelText = `${style.label} ${finding.hu_mean?.toFixed(0)} HU`
      ctx.setLineDash([])
      const fontSize = Math.max(10, Math.round(11 * Math.min(scaleX, scaleY)))
      ctx.font = `600 ${fontSize}px Inter, sans-serif`
      const textWidth = ctx.measureText(labelText).width + 8
      const labelHeight = fontSize + 4

      ctx.fillStyle = color + 'cc'
      ctx.fillRect(sx, sy - labelHeight, textWidth, labelHeight)

      ctx.fillStyle = '#fff'
      ctx.fillText(labelText, sx + 4, sy - 4)
    }
  }, [fittedSize.w, fittedSize.h, selectedSlice, showFindings, imageNaturalSize])

  // ── Keyboard & wheel navigation ───────────────────────────────────────────
  useEffect(() => {
    const handleWheel = (e) => {
      if (!viewerRef.current?.contains(e.target)) return
      e.preventDefault()
      setSelectedIdx(prev => {
        const next = Math.floor(prev) + (e.deltaY > 0 ? 1 : -1)
        return Math.max(0, Math.min(slices.length - 1, next))
      })
      setIsAutoPlay(false)
    }

    const handleKeyDown = (e) => {
      if (e.key === 'ArrowUp' || e.key === 'ArrowLeft') {
        e.preventDefault()
        setSelectedIdx(prev => Math.max(0, Math.floor(prev) - 1))
        setIsAutoPlay(false)
      } else if (e.key === 'ArrowDown' || e.key === 'ArrowRight') {
        e.preventDefault()
        setSelectedIdx(prev => Math.min(slices.length - 1, Math.floor(prev) + 1))
        setIsAutoPlay(false)
      }
    }

    window.addEventListener('wheel', handleWheel, { passive: false })
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('wheel', handleWheel)
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [slices.length])

  // ── Autoplay timer ────────────────────────────────────────────────────────
  useEffect(() => {
    if (isAutoPlay && slices.length > 1) {
      autoPlayRef.current = setInterval(() => {
        setSelectedIdx(prev => {
          const next = Math.floor(prev)
          if (next >= slices.length - 1) return 0
          return next + 1
        })
      }, 150)
    }
    return () => {
      if (autoPlayRef.current) clearInterval(autoPlayRef.current)
    }
  }, [isAutoPlay, slices.length])

  // ── Window preset handler ─────────────────────────────────────────────────
  const applyPreset = (name) => {
    const preset = WINDOW_PRESETS[name]
    if (preset) {
      setWindowPreset(name)
      if (preset.wc !== 'auto') setWindowCenter(preset.wc)
      if (preset.ww !== 'auto') setWindowWidth(preset.ww)
      else if (selectedSlice) {
        setWindowCenter(selectedSlice.window_center_default ?? 50)
        setWindowWidth(selectedSlice.window_width_default ?? 350)
      }
    }
  }

  // ── Scanner Control ───────────────────────────────────────────────────────
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch(`${API_URL}/api/scanner/status`)
        if (res.ok) {
          const data = await res.json()
          setScannerRunning(data.running)
        }
      } catch (e) { /* ignored */ }
    }
    checkStatus()
    const interval = setInterval(checkStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  const startScanner = async () => {
    try {
      const res = await fetch(`${API_URL}/api/scanner/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_dir: datasetDir, delay_ms: parseInt(delayMs), patient_id: patientId })
      })
      const data = await res.json()
      if (data.status === 'success') {
        setScannerRunning(true)
        setSlices([])
        setSelectedIdx(0)
      } else {
        alert(`Scanner failed to start: ${data.message}`)
      }
    } catch (e) {
      alert('Failed to connect to backend server. Is the pipeline running?')
    }
  }

  const stopScanner = async () => {
    try {
      await fetch(`${API_URL}/api/scanner/stop`, { method: 'POST' })
      setScannerRunning(false)
    } catch (e) {
      console.error('Failed to stop scanner:', e)
    }
  }

  const handleBrowse = async () => {
    try {
      const res = await fetch(`${API_URL}/api/scanner/browse`)
      const data = await res.json()
      if (data.status === 'success') {
        setDatasetDir(data.path)
      } else if (data.status === 'error') {
        alert('Failed to open native browse dialog.')
      }
    } catch (e) {
      alert('Failed to connect to backend server.')
    }
  }

  // ── Scroll the slice list to keep selection visible ───────────────────────
  useEffect(() => {
    if (sliceListRef.current && selectedSlice) {
      const el = document.getElementById(`slice-card-${selectedSlice.slice_id}`)
      if (el) {
        el.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
      }
    }
  }, [selectedIdx])

  // Resolve the image source — prefer API, fallback to base64 thumbnail
  const resolvedImageSrc = (() => {
    if (viewerImageUrl) return viewerImageUrl
    if (selectedSlice?.thumbnail) return `data:image/png;base64,${selectedSlice.thumbnail}`
    return null
  })()

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">
      {/* ── Top Bar ── */}
      <header className="topbar">
        <div className="topbar__left">
          <span className="topbar__logo">◆ PRISM</span>
          <span className="topbar__subtitle">Phase 1 — CT Ingestion & Screening</span>
        </div>
        <div className="topbar__right">
          <span className={`topbar__status topbar__status--${connectionStatus}`}>
            {connectionStatus === 'connected' ? '● Connected' :
             connectionStatus === 'connecting' ? '◌ Connecting...' : '○ Disconnected'}
          </span>
          <span className="topbar__stat">
            <span className="mono">{stats.total}</span> slices
          </span>
          <span className="topbar__stat topbar__stat--flagged">
            <span className="mono">{stats.flagged}</span> flagged
          </span>
        </div>
      </header>

      <div className="main-layout">
        {/* ── Left Panel: Scanner Controls + Slice List ── */}
        <aside className="slice-panel">
          {/* Scanner Controls */}
          <div className="scanner-controls">
            <div className="scanner-controls__title">Scanner Control</div>
            <div className="scanner-controls__body">
              <div>
                <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Dataset Directory</label>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <input 
                    type="text" 
                    value={datasetDir}
                    onChange={(e) => setDatasetDir(e.target.value)}
                    placeholder="Enter absolute path here (e.g., C:\Users\gunan\...)"
                    style={{ flex: 1, padding: '6px', fontSize: '12px', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: '#fff', borderRadius: '4px' }}
                  />
                  <button
                    onClick={handleBrowse}
                    style={{ padding: '0 10px', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', borderRadius: '4px', cursor: 'pointer', fontSize: '12px', fontWeight: '600' }}
                  >
                    📂 Browse
                  </button>
                </div>
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Patient ID (Optional)</label>
                <input 
                  type="text" 
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  placeholder="e.g. MSB-00142"
                  style={{ width: '100%', padding: '6px', fontSize: '12px', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: '#fff', borderRadius: '4px' }}
                />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>Speed (Delay ms)</label>
                <input 
                  type="number" 
                  value={delayMs}
                  onChange={(e) => setDelayMs(e.target.value)}
                  style={{ width: '100%', padding: '6px', fontSize: '12px', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: '#fff', borderRadius: '4px' }}
                />
              </div>
              <div style={{ display: 'flex', gap: '8px', marginTop: '8px' }}>
                <button 
                  onClick={scannerRunning ? stopScanner : startScanner}
                  style={{ 
                    flex: 1, 
                    padding: '8px', 
                    background: scannerRunning ? '#ef4444' : '#10b981', 
                    color: '#fff', 
                    border: 'none', 
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: '600'
                  }}
                >
                  {scannerRunning ? '⏹ Stop Scanner' : '▶ Start Scan'}
                </button>
              </div>
            </div>
          </div>

          <div className="slice-panel__header">
            <span className="slice-panel__title">Received Slices</span>
            <span className="slice-panel__count">{slices.length}</span>
          </div>

          <div className="slice-list" ref={sliceListRef}>
            {slices.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state__icon">📡</div>
                <div className="empty-state__text">Waiting for slices...</div>
                <div className="empty-state__hint">
                  Start the pipeline and scanner to see live CT slices appear here.
                </div>
              </div>
            ) : (
              slices.map((slice, idx) => (
                <div
                  key={slice.slice_id}
                  id={`slice-card-${slice.slice_id}`}
                  className={[
                    'slice-card',
                    slice.flagged ? 'slice-card--flagged' : '',
                    slice._isNew && slice.flagged ? 'slice-card--new' : '',
                    idx === displayIdx ? 'slice-card--selected' : '',
                  ].filter(Boolean).join(' ')}
                  onClick={() => { setSelectedIdx(idx); setIsAutoPlay(false) }}
                >
                  <div className="slice-card__thumbnail">
                    {slice.thumbnail ? (
                      <img
                        src={`data:image/png;base64,${slice.thumbnail}`}
                        alt={`Slice ${slice.slice_id}`}
                      />
                    ) : (
                      <div className="slice-card__thumbnail--placeholder">🫁</div>
                    )}
                  </div>

                  <div className="slice-card__info">
                    <div className="slice-card__title">Slice #{slice.slice_id}</div>
                    <div className="slice-card__meta">
                      {slice.processing_time_ms?.toFixed(1)}ms
                      {slice.findings?.length > 0 && ` · ${slice.findings.length} finding${slice.findings.length > 1 ? 's' : ''}`}
                    </div>
                  </div>

                  <div className="slice-card__status">
                    <span className={`slice-card__badge slice-card__badge--${slice.emergency_action?.toLowerCase().replace(' ', '-') || 'continue'}`}>
                      {slice.emergency_action || 'CONTINUE'}
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </aside>

        {/* ── Center: DICOM-style Viewer ── */}
        <main className="detail-panel" ref={viewerRef}>
          {/* Viewer toolbar */}
          <div className="viewer-toolbar">
            <div className="viewer-toolbar__left">
              <span className="viewer-toolbar__title">
                {selectedSlice ? `Slice #${selectedSlice.slice_id}` : 'No Slice'}
              </span>
              {selectedSlice && (
                <span className="viewer-toolbar__meta mono">
                  I:{selectedSlice.slice_id} ({displayIdx + 1}/{slices.length})
                </span>
              )}
            </div>
            <div className="viewer-toolbar__center">
              {/* Window presets */}
              {Object.keys(WINDOW_PRESETS).map(name => (
                <button
                  key={name}
                  className={`viewer-toolbar__preset ${windowPreset === name ? 'viewer-toolbar__preset--active' : ''}`}
                  onClick={() => applyPreset(name)}
                >
                  {name}
                </button>
              ))}
            </div>
            <div className="viewer-toolbar__right">
              <button
                className={`viewer-toolbar__toggle ${showFindings ? 'viewer-toolbar__toggle--active' : ''}`}
                onClick={() => setShowFindings(v => !v)}
                title="Toggle findings overlay"
              >
                Findings {showFindings ? 'ON' : 'OFF'}
              </button>
              <button
                className={`viewer-toolbar__toggle ${isAutoPlay ? 'viewer-toolbar__toggle--active' : ''}`}
                onClick={() => setIsAutoPlay(v => !v)}
                title="Auto-scroll through slices"
              >
                {isAutoPlay ? '⏸ Pause' : '▶ Play'}
              </button>
            </div>
          </div>

          <div className="detail-panel__body">
            {!selectedSlice ? (
              <div className="canvas-viewer">
                <div className="empty-state">
                  <div className="empty-state__icon">🔬</div>
                  <div className="empty-state__text">No slice selected</div>
                  <div className="empty-state__hint">
                    Start a scan or click on a slice in the left panel.
                  </div>
                </div>
              </div>
            ) : (
              <>
                <div className="canvas-viewer">
                  <div className="canvas-viewer__wrapper" ref={wrapperRef}>
                    {/* DICOM-style corner labels */}
                    <div className="dicom-label dicom-label--tl">
                      <div className="mono">W:{windowWidth} L:{windowCenter}</div>
                    </div>
                    <div className="dicom-label dicom-label--tr">
                      <div className="mono">{selectedSlice.detected_region?.toUpperCase() || 'GENERIC'}</div>
                      <div className="mono">{selectedSlice.flagged ? 'FLAGGED' : 'CLEAN'}</div>
                    </div>
                    <div className="dicom-label dicom-label--bl">
                      <div className="mono">HU μ={selectedSlice.slice_hu_mean?.toFixed(1)} σ={selectedSlice.slice_hu_std?.toFixed(1)}</div>
                    </div>
                    <div className="dicom-label dicom-label--br">
                      <div className="mono">I:{selectedSlice.slice_id} ({displayIdx + 1}/{slices.length})</div>
                    </div>

                    {/* CT Image — rendered as an <img> for reliability */}
                    <div className="canvas-viewer__image-container" style={{
                      width: fittedSize.w,
                      height: fittedSize.h,
                      position: 'relative',
                      flexShrink: 0,
                    }}>
                      <img
                        ref={imgRef}
                        src={resolvedImageSrc}
                        alt={`CT Slice ${selectedSlice.slice_id}`}
                        style={{
                          width: '100%',
                          height: '100%',
                          display: 'block',
                          objectFit: 'fill',
                          imageRendering: 'auto',
                        }}
                        onLoad={(e) => {
                          const img = e.target
                          if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                            setImageNaturalSize({ w: img.naturalWidth, h: img.naturalHeight })
                          }
                        }}
                        onError={() => {
                          // Fallback to base64 thumbnail if API fails
                          if (viewerImageUrl && selectedSlice?.thumbnail) {
                            setViewerImageUrl(null)
                          }
                        }}
                      />
                      {/* Overlay canvas for findings bounding boxes */}
                      <canvas
                        ref={overlayRef}
                        style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '100%',
                          pointerEvents: 'none',
                          backgroundColor: 'transparent',
                        }}
                      />
                    </div>

                    {/* Slice slider — overlaid at bottom of viewer */}
                    {slices.length > 1 && (
                      <div className="slice-slider">
                        <input
                          type="range"
                          min={0}
                          max={slices.length - 1}
                          step="0.01"
                          value={selectedIdx}
                          onChange={(e) => { setSelectedIdx(parseFloat(e.target.value)); setIsAutoPlay(false) }}
                          className="slice-slider__input"
                        />
                        <span className="slice-slider__label mono">
                          {displayIdx + 1} / {slices.length}
                        </span>
                      </div>
                    )}
                  </div>
                </div>

                {/* ── Findings Sidebar ── */}
                <aside className="detail-sidebar detail-sidebar--right">
                  <div style={{
                    padding: '16px',
                    marginBottom: '16px',
                    borderRadius: '8px',
                    background: 'var(--bg-elevated)',
                    border: '1px solid var(--border-subtle)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                  }}>
                    <span style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '4px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Emergency Score</span>
                    <span style={{ 
                      fontSize: '32px', 
                      fontWeight: '800', 
                      color: selectedSlice.emergency_score >= 80 ? 'var(--status-danger)' : 
                             selectedSlice.emergency_score >= 50 ? '#ff4500' : 
                             selectedSlice.emergency_score >= 20 ? 'var(--status-warning)' : 'var(--status-success)'
                    }}>
                      {selectedSlice.emergency_score || 0}
                    </span>
                  </div>

                  <div className="detail-sidebar__section">
                    <h3 className="detail-sidebar__title">Score Details</h3>
                    {selectedSlice.score_details?.length > 0 ? (
                      <div className="detail-findings-list">
                        {selectedSlice.score_details.map((detail, idx) => (
                          <div key={idx} className="finding-card">
                            <div className="finding-card__header">
                              <span className="finding-card__type">+{detail.points} pts</span>
                            </div>
                            <div className="finding-card__body">
                              <p style={{ margin: 0, fontSize: '11px', color: 'var(--text-primary)' }}>{detail.description}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="detail-findings-empty">
                        No structural anomalies detected.
                      </div>
                    )}
                  </div>

                  <div className="detail-sidebar__section" style={{ marginTop: '16px' }}>
                    <h3 className="detail-sidebar__title">Raw Findings</h3>
                  </div>

                  {(!selectedSlice.findings || selectedSlice.findings.length === 0) ? (
                    <div style={{ textAlign: 'center', padding: '16px', color: 'var(--text-muted)' }}>
                      <div style={{ fontSize: '13px' }}>No raw findings</div>
                    </div>
                  ) : (
                    selectedSlice.findings.map((finding, idx) => (
                      <div
                        key={idx}
                        className={`finding-card`}
                      >
                        <div className="finding-card__type" style={{ color: 'var(--status-warning)', textTransform: 'uppercase' }}>
                          ⚠️ {finding.finding_type}
                        </div>

                        <div className="finding-card__row">
                          <span className="finding-card__label">Bounding Box</span>
                          <span className="finding-card__value">
                            [{finding.bbox.join(', ')}]
                          </span>
                        </div>
                        <div className="finding-card__row">
                          <span className="finding-card__label">Area</span>
                          <span className="finding-card__value">{finding.area_px} px</span>
                        </div>
                        <div className="finding-card__row">
                          <span className="finding-card__label">HU Mean</span>
                          <span className="finding-card__value">{finding.hu_mean?.toFixed(1)}</span>
                        </div>
                        <div className="finding-card__row">
                          <span className="finding-card__label">HU Max</span>
                          <span className="finding-card__value">{finding.hu_max?.toFixed(1)}</span>
                        </div>
                        <div className="finding-card__row">
                          <span className="finding-card__label">HU Min</span>
                          <span className="finding-card__value">{finding.hu_min?.toFixed(1)}</span>
                        </div>
                        <div className="finding-card__row">
                          <span className="finding-card__label">Centroid</span>
                          <span className="finding-card__value">
                            ({finding.centroid?.[0]?.toFixed(0)}, {finding.centroid?.[1]?.toFixed(0)})
                          </span>
                        </div>
                      </div>
                    ))
                  )}

                  {/* Timing info */}
                  <div style={{
                    marginTop: '16px',
                    padding: '12px',
                    borderRadius: '8px',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-subtle)',
                    fontSize: '11px',
                  }}>
                    <div className="finding-card__row">
                      <span className="finding-card__label">Pipeline</span>
                      <span className="finding-card__value">
                        {selectedSlice.processing_time_ms?.toFixed(1)}ms
                      </span>
                    </div>
                    <div className="finding-card__row">
                      <span className="finding-card__label">Triage</span>
                      <span className="finding-card__value">
                        {selectedSlice.triage_time_ms?.toFixed(1)}ms
                      </span>
                    </div>
                    <div className="finding-card__row">
                      <span className="finding-card__label">Timestamp</span>
                      <span className="finding-card__value" style={{ fontSize: '10px' }}>
                        {selectedSlice.timestamp ? new Date(selectedSlice.timestamp).toLocaleTimeString() : '-'}
                      </span>
                    </div>
                  </div>

                  {/* Window/Level manual controls */}
                  <div style={{
                    marginTop: '16px',
                    padding: '12px',
                    borderRadius: '8px',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-subtle)',
                    fontSize: '11px',
                  }}>
                    <div style={{ fontWeight: '600', marginBottom: '8px', color: 'var(--text-secondary)' }}>Window / Level</div>
                    <div className="finding-card__row" style={{ marginBottom: '6px' }}>
                      <span className="finding-card__label">Center (L)</span>
                      <input
                        type="number"
                        value={windowCenter}
                        onChange={(e) => { setWindowCenter(parseInt(e.target.value) || 0); setWindowPreset('Custom') }}
                        style={{ width: '70px', padding: '3px 6px', fontSize: '11px', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: '#fff', borderRadius: '3px', textAlign: 'right' }}
                      />
                    </div>
                    <div className="finding-card__row">
                      <span className="finding-card__label">Width (W)</span>
                      <input
                        type="number"
                        value={windowWidth}
                        onChange={(e) => { setWindowWidth(parseInt(e.target.value) || 1); setWindowPreset('Custom') }}
                        style={{ width: '70px', padding: '3px 6px', fontSize: '11px', background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)', color: '#fff', borderRadius: '3px', textAlign: 'right' }}
                      />
                    </div>
                  </div>

                  {/* ── Phase 2 Segmentations Detached ── */}
                </aside>
              </>
            )}
          </div>
        </main>
      </div>


    </div>
  )
}
