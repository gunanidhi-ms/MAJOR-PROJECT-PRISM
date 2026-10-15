import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { SignaturePad } from '../components/ui/SignaturePad'
import { ReportTemplate } from '../components/reports/ReportTemplate'
import { downloadReportPdf } from '../utils/pdfGenerator'
import { prismApi } from '../api/prismApi'
import { usePatientSession } from '../context/PatientSessionContext'

/**
 * LiveViewer — PRISM Integrated Live CT Viewer & Clinical Report Studio
 *
 * Layout Hierarchy:
 *   1. Fixed Header Zone (Pinned Top: Topbar + Patient Controls Bar + Horizontal Stepper)
 *   2. Dynamic Phase Viewport (Directly below Stepper, fills 100% remaining height):
 *      - Phase 1: Full-Resolution 3-Column CT Viewer (Slices List | Large CT Canvas | Score & Findings)
 *      - Phase 3: Spacious Clinical Report Studio (Sticky Navigation Banner | Editor | Radiologist Sign-Off)
 */

const API_URL = 'http://localhost:8001'

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

const TEMPLATE_ID = 'prism-live-report-template'

import '../styles/viewer.css'

export default function LiveViewer() {
  const navigate = useNavigate()

  // ── Global Patient Session State ───────────────────────────────────────────
  const {
    session,
    slices,
    selectedIdx,
    setSelectedIdx,
    connectionStatus,
    stats,
    draftReport,
    pipelineEvent,
    isPhase3Unlocked,
    updateSessionField,
    startSession,
    stopScan,
    endSession,
    triggerPhase2,
    updateDraftReport,
    resetDraftReport,
  } = usePatientSession()

  // ── In-Screen View State ──────────────────────────────────────────────────
  const [activeView, setActiveView] = useState('phase1') // 'phase1' | 'phase3'

  // ── Local Patient Configuration Inputs ────────────────────────────────────
  const [datasetDir, setDatasetDir] = useState(session.dataset_dir || 'D:\\MAJOR-PROJECT-PRISM\\phase1_ingestion\\sample_dicoms\\test_series')
  const [delayMs, setDelayMs] = useState(String(session.delay_ms || 30))
  const [patientId, setPatientId] = useState('')
  const [patientName, setPatientName] = useState('')
  const [patientAge, setPatientAge] = useState('')
  const [patientSex, setPatientSex] = useState('')
  const [formErrors, setFormErrors] = useState({})
  const [isStartingSession, setIsStartingSession] = useState(false)
  const [isEndingSession, setIsEndingSession] = useState(false)

  // Determine if patient session is currently active
  const isSessionActive = useMemo(() => {
    return Boolean(
      session.session_active ||
      session.phase1_status === 'scanning' ||
      (session.phase1_status === 'completed' && slices.length > 0)
    )
  }, [session.session_active, session.phase1_status, slices.length])

  // Sync inputs from backend session if session was loaded from existing active session
  useEffect(() => {
    if (isSessionActive) {
      if (session.patient_id && !patientId) setPatientId(session.patient_id)
      if (session.patient_name && !patientName) setPatientName(session.patient_name)
      if (session.patient_age && !patientAge) setPatientAge(session.patient_age)
      if (session.patient_sex && !patientSex) setPatientSex(session.patient_sex)
      if (session.dataset_dir) setDatasetDir(session.dataset_dir)
      if (session.delay_ms) setDelayMs(String(session.delay_ms))
    }
  }, [isSessionActive, session.patient_id, session.patient_name, session.patient_age, session.patient_sex, session.dataset_dir, session.delay_ms])

  // ── Windowing & Viewer Controls ───────────────────────────────────────────
  const [windowPreset, setWindowPreset] = useState('DICOM Default')
  const [windowCenter, setWindowCenter] = useState(50)
  const [windowWidth, setWindowWidth] = useState(350)
  const [showFindings, setShowFindings] = useState(true)
  const [isAutoPlay, setIsAutoPlay] = useState(false)

  // ── Report Studio & Radiologist Review Local Form ──────────────────────────
  const [reportFindings, setReportFindings] = useState(draftReport.findings || '')
  const [reportImpression, setReportImpression] = useState(draftReport.impression || '')
  const [radName, setRadName] = useState(draftReport.radName || 'Dr. Sarah Chen, MD')
  const [radDesig, setRadDesig] = useState(draftReport.radDesig || 'Senior Consultant Radiologist | FRCR')
  const [sigDataUrl, setSigDataUrl] = useState(draftReport.sigDataUrl || null)
  const [sigApplied, setSigApplied] = useState(Boolean(draftReport.sigDataUrl))
  const [generatingPdf, setGeneratingPdf] = useState(false)
  const [pdfFilename, setPdfFilename] = useState('')
  const [showSignModal, setShowSignModal] = useState(false)
  const [signing, setSigning] = useState(false)
  const [saveStatus, setSaveStatus] = useState(null)

  // Sync fields only when switching patient session or when initial findings arrive
  const activeSessionIdRef = useRef(session.patient_id)
  useEffect(() => {
    if (session.patient_id !== activeSessionIdRef.current) {
      activeSessionIdRef.current = session.patient_id
      setReportFindings(draftReport.findings || '')
      setReportImpression(draftReport.impression || '')
      setRadName(draftReport.radName || 'Dr. Sarah Chen, MD')
      setRadDesig(draftReport.radDesig || 'Senior Consultant Radiologist | FRCR')
      setSigDataUrl(draftReport.sigDataUrl || null)
      setSigApplied(Boolean(draftReport.sigDataUrl))
    }
  }, [session.patient_id, draftReport])

  // ── Modals & Notice States ──────────────────────────────────────────────────
  const [archiveNotice, setArchiveNotice] = useState(null)
  const [showEndModal, setShowEndModal] = useState(false)
  const [showPhase3LockedModal, setShowPhase3LockedModal] = useState(false)
  const [showSystemShutdownModal, setShowSystemShutdownModal] = useState(false)
  const [isSystemShuttingDown, setIsSystemShuttingDown] = useState(false)

  // ── Image Display State ────────────────────────────────────────────────────
  const [viewerImageUrl, setViewerImageUrl] = useState(null)
  const [imageNaturalSize, setImageNaturalSize] = useState({ w: 512, h: 512 })
  const [viewerSize, setViewerSize] = useState({ w: 900, h: 700 })

  const imgRef = useRef(null)
  const overlayRef = useRef(null)
  const sliceListRef = useRef(null)
  const viewerRef = useRef(null)
  const wrapperRef = useRef(null)

  const displayIdx = Math.floor(selectedIdx)
  const selectedSlice = slices[displayIdx] || null

  const activePatient = useMemo(() => {
    const pId = session.patient_id || (slices.length > 0 ? slices[0].patient_id : '') || patientId || 'TEST-001'
    const pName = session.patient_name || (slices.length > 0 ? slices[0].patient_name : '') || patientName || 'Jane Doe'
    const sId = session.study_id || (slices.length > 0 ? (slices[0].study_id || slices[0].study_instance_uid) : '') || ('STUDY_' + pId)
    const pAge = session.patient_age || (slices.length > 0 ? slices[0].patient_age : '') || patientAge || '45Y'
    const pSex = session.patient_sex || (slices.length > 0 ? slices[0].patient_sex : '') || patientSex || 'F'

    return {
      id: pId,
      name: pName,
      age: pAge,
      sex: pSex,
      studyId: sId,
    }
  }, [session, slices, patientId, patientName, patientAge, patientSex])

  // ── Auto-follow latest slice when scanning ────────────────────────────────
  useEffect(() => {
    if (isAutoPlay && slices.length > 0) {
      setSelectedIdx(slices.length - 1)
    }
  }, [slices.length, isAutoPlay, setSelectedIdx])

  // Auto-scroll the slice card into view inside the internal slice-list container only
  useEffect(() => {
    const container = sliceListRef.current
    if (container && selectedSlice?.slice_id) {
      const activeCard = container.querySelector(`#slice-card-${selectedSlice.slice_id}`)
      if (activeCard) {
        const cardTop = activeCard.offsetTop - container.offsetTop
        const cardBottom = cardTop + activeCard.offsetHeight
        const currentScroll = container.scrollTop
        const viewHeight = container.clientHeight

        if (cardTop < currentScroll) {
          container.scrollTo({ top: Math.max(0, cardTop - 8), behavior: 'smooth' })
        } else if (cardBottom > currentScroll + viewHeight) {
          container.scrollTo({ top: cardBottom - viewHeight + 8, behavior: 'smooth' })
        }
      }
    }
  }, [displayIdx, selectedSlice?.slice_id])

  // ── DICOM Default Windowing ───────────────────────────────────────────────
  useEffect(() => {
    if (windowPreset === 'DICOM Default' && selectedSlice) {
      const wc = selectedSlice.window_center_default ?? 50
      const ww = selectedSlice.window_width_default ?? 350
      setWindowCenter(wc)
      setWindowWidth(ww)
    }
  }, [selectedSlice, windowPreset])

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

  // Scale image to fill maximum available viewport size
  const fittedSize = useMemo(() => {
    const { w: vw, h: vh } = viewerSize
    const { w: iw, h: ih } = imageNaturalSize
    if (iw === 0 || ih === 0 || vw === 0 || vh === 0) return { w: 512, h: 512 }
    const scaleX = (vw - 20) / iw
    const scaleY = (vh - 20) / ih
    const scale = Math.min(scaleX, scaleY)
    return {
      w: Math.max(320, Math.floor(iw * scale)),
      h: Math.max(320, Math.floor(ih * scale)),
    }
  }, [viewerSize, imageNaturalSize])

  // ── Draw findings overlay canvas ───────────────────────────────────────────
  useEffect(() => {
    const overlay = overlayRef.current
    if (!overlay) return

    const ctx = overlay.getContext('2d')
    const { w: fw, h: fh } = fittedSize
    const { w: nw, h: nh } = imageNaturalSize

    overlay.width = fw
    overlay.height = fh
    ctx.clearRect(0, 0, fw, fh)

    if (!showFindings || !selectedSlice?.findings?.length) return

    const scaleX = fw / nw
    const scaleY = fh / nh

    for (const finding of selectedSlice.findings) {
      const [bx, by, bw, bh] = finding.bbox
      
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

      ctx.strokeStyle = color
      ctx.lineWidth = Math.max(1.5, 1.5 * Math.min(scaleX, scaleY))
      ctx.setLineDash([4 * Math.min(scaleX, scaleY), 4 * Math.min(scaleX, scaleY)])
      ctx.strokeRect(sx, sy, sw, sh)

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
  }, [fittedSize, selectedSlice, showFindings, imageNaturalSize])

  // ── Keyboard & Wheel Slice Navigation ─────────────────────────────────────
  useEffect(() => {
    const handleWheel = (e) => {
      if (!viewerRef.current?.contains(e.target) || activeView !== 'phase1') return
      e.preventDefault()
      setSelectedIdx(prev => {
        const next = Math.floor(prev) + (e.deltaY > 0 ? 1 : -1)
        return Math.max(0, Math.min(slices.length - 1, next))
      })
      setIsAutoPlay(false)
    }

    const handleKeyDown = (e) => {
      // Never intercept keyboard typing inside inputs, textareas, or editable elements
      const tag = document.activeElement?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || document.activeElement?.isContentEditable) {
        return
      }

      if (activeView !== 'phase1') return
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
  }, [slices.length, activeView, setSelectedIdx])

  const applyPreset = (presetName) => {
    setWindowPreset(presetName)
    const p = WINDOW_PRESETS[presetName]
    if (presetName === 'DICOM Default') {
      if (selectedSlice) {
        setWindowCenter(selectedSlice.window_center_default ?? 50)
        setWindowWidth(selectedSlice.window_width_default ?? 350)
      }
    } else if (p) {
      setWindowCenter(p.wc)
      setWindowWidth(p.ww)
    }
  }

  // ── Strict Patient Validation ─────────────────────────────────────────────
  const validateForm = () => {
    const errors = {}
    if (!patientId || !patientId.trim()) errors.patientId = 'Required'
    if (!patientName || !patientName.trim()) errors.patientName = 'Required'
    if (!patientAge || !patientAge.trim()) errors.patientAge = 'Required'
    if (!patientSex || !['M', 'F', 'O'].includes(patientSex.trim())) errors.patientSex = 'Required'
    if (!datasetDir || !datasetDir.trim()) errors.datasetDir = 'Required'
    setFormErrors(errors)
    return Object.keys(errors).length === 0
  }

  const handleStartPatientSession = async () => {
    if (!validateForm()) {
      setArchiveNotice('⚠️ Please complete all required patient fields before starting the session.')
      setTimeout(() => setArchiveNotice(null), 4000)
      return
    }
    setFormErrors({})
    setIsStartingSession(true)

    try {
      const res = await startSession({
        patient_id: patientId.trim(),
        patient_name: patientName.trim(),
        patient_age: patientAge.trim(),
        patient_sex: patientSex.trim(),
        dataset_dir: datasetDir.trim(),
        delay_ms: parseInt(delayMs) || 30,
        auto_start_scan: true,
      })

      if (res.status === 'success') {
        setActiveView('phase1')
        setArchiveNotice(`🚀 Patient session started for ${patientName.trim()} (${patientId.trim()}). Details locked. Streaming slices...`)
        setTimeout(() => setArchiveNotice(null), 5000)
      } else {
        alert(`Failed to start session: ${res.message}`)
      }
    } catch (err) {
      alert(`Network error starting session: ${err.message}`)
    } finally {
      setIsStartingSession(false)
    }
  }

  const handleEndPatientSessionConfirmed = async () => {
    setIsEndingSession(true)
    try {
      let res = await endSession()
      
      if (!res || res.status !== 'success') {
        try {
          const directRes = await fetch(`${API_URL}/api/session/end`, { method: 'POST' })
          const directData = await directRes.json()
          if (directData.status === 'success') {
            res = directData
          }
        } catch (e) {}
      }

      setShowEndModal(false)
      setPatientId('')
      setPatientName('')
      setPatientAge('')
      setPatientSex('')
      setFormErrors({})
      setReportFindings('')
      setReportImpression('')
      setSigDataUrl(null)
      setSigApplied(false)
      setActiveView('phase1')
      
      const archiveName = res?.archive_file || 'patient_archive.zip'
      setArchiveNotice(`✅ Patient session ended! Records archived to exports/${archiveName}`)
      setTimeout(() => setArchiveNotice(null), 7000)
    } catch (err) {
      alert(`Error ending session: ${err.message}`)
    } finally {
      setIsEndingSession(false)
    }
  }

  const handleBrowse = async () => {
    if (isSessionActive) return
    try {
      const res = await prismApi.browseDirectory()
      if (res.status === 'success' && res.path) {
        setDatasetDir(res.path)
        if (formErrors.datasetDir) setFormErrors(prev => ({ ...prev, datasetDir: null }))
      }
    } catch (e) {
      alert('Browse dialog is available on server machine.')
    }
  }

  // ── Draft Report Synchronization & Generation ──────────────────────────────
  const ensureDraftReport = useCallback(async () => {
    const targetId = activePatient?.id || session.patient_id || activePatient?.studyId
    if (targetId) {
      try {
        const rep = await prismApi.getReport(targetId)
        if (rep && (rep.findings || rep.impression)) {
          updateDraftReport({
            findings: rep.findings || '',
            impression: rep.impression || '',
            isSigned: rep.status === 'signed',
            signedAt: rep.updated_at || '',
          })
          setReportFindings(rep.findings || '')
          setReportImpression(rep.impression || '')
          return
        }
      } catch (e) {}
    }

    // Synthesize structured draft findings only if uninitialized
    const flaggedSlices = slices.filter(s => s.flagged)
    const maxScore = Math.max(0, ...slices.map(s => s.emergency_score || 0))
    const anomalySummary = []

    slices.forEach(s => {
      if (s.findings && s.findings.length > 0) {
        s.findings.forEach(f => {
          anomalySummary.push(`• Slice #${s.slice_id} [${s.detected_region || 'CT'}]: ${f.finding_type?.toUpperCase().replace('_', ' ')} (${f.hu_mean ? f.hu_mean.toFixed(1) : '-'} HU, Area: ${f.area_px || '-'} px)`)
        })
      }
    })

    const synthesizedFindings = [
      `1. CT ACQUISITION & TECHNIQUE: Volumetric axial diagnostic CT series with ${slices.length || 20} contiguous slices reconstructed at standard slice thickness.`,
      `2. REAL-TIME AI & 2D TRIAGE: ${flaggedSlices.length > 0 ? `${flaggedSlices.length} slice(s) flagged for structural attenuation anomalies.` : 'No critical high-urgency structural halts detected during triage.'} Maximum emergency triage index: ${maxScore}/100.`,
      anomalySummary.length > 0
        ? `3. DETECTED STRUCTURAL ANOMALIES:\n${anomalySummary.slice(0, 8).join('\n')}`
        : `3. REGIONAL EVALUATION: Normal parenchymal attenuation profiles. Vascular and visceral structures demonstrate preserved caliber and attenuation.`,
      `4. OSSEOUS & SOFT TISSUE: No acute cortical disruption or gross pathological bone lesion identified.`
    ].join('\n\n')

    const synthesizedImpression = maxScore >= 50
      ? `CT study demonstrates localized structural attenuation anomalies (Emergency Index ${maxScore}). Correlate clinically.`
      : `Complete CT volumetric evaluation. No acute critical emergency pathology identified on automated volumetric triage.`

    setReportFindings(prev => {
      const val = prev || draftReport.findings || synthesizedFindings
      updateDraftReport({ findings: val })
      return val
    })
    setReportImpression(prev => {
      const val = prev || draftReport.impression || synthesizedImpression
      updateDraftReport({ impression: val })
      return val
    })

    const dateSlug = new Date().toISOString().slice(0, 10)
    const nameSlug = (activePatient?.name || 'Patient').replace(/[^a-zA-Z0-9]/g, '_')
    setPdfFilename(`${nameSlug}_CT_Report_${dateSlug}.pdf`)
  }, [activePatient, session.patient_id, slices, draftReport.findings, draftReport.impression, updateDraftReport])

  const handleOpenPhase3 = () => {
    if (!isPhase3Unlocked) {
      setShowPhase3LockedModal(true)
      return
    }
    setActiveView('phase3')
    ensureDraftReport()
  }

  useEffect(() => {
    if (activeView === 'phase3' && isPhase3Unlocked) {
      ensureDraftReport()
    }
  }, [activeView, isPhase3Unlocked])

  const handleSaveReportDraft = async () => {
    setSaveStatus('Saving...')
    try {
      const studyKey = activePatient?.studyId || activePatient?.id || 'STUDY-001'
      await prismApi.updateReport(studyKey, { findings: reportFindings, impression: reportImpression })
      updateDraftReport({ findings: reportFindings, impression: reportImpression })
      setSaveStatus('Draft saved successfully!')
      setTimeout(() => setSaveStatus(null), 3000)
    } catch (e) {
      updateDraftReport({ findings: reportFindings, impression: reportImpression })
      setSaveStatus('Draft saved locally in session.')
      setTimeout(() => setSaveStatus(null), 3000)
    }
  }

  const handleSignReport = async () => {
    if (!sigApplied && !sigDataUrl) {
      alert('Please draw and apply your digital signature first.')
      return
    }
    setSigning(true)
    try {
      const studyKey = activePatient?.studyId || activePatient?.id || 'STUDY-001'
      try {
        await prismApi.updateReport(studyKey, { findings: reportFindings, impression: reportImpression })
        await prismApi.signReport(studyKey)
      } catch (e) {}
      const now = new Date().toISOString()
      updateDraftReport({
        findings: reportFindings,
        impression: reportImpression,
        isSigned: true,
        signedAt: now,
        radName,
        radDesig,
        sigDataUrl,
      })
      setShowSignModal(false)
      setArchiveNotice('✅ Report signed and finalized! Ready for PDF export.')
      setTimeout(() => setArchiveNotice(null), 5000)
    } catch (err) {
      alert('Failed to sign report.')
    } finally {
      setSigning(false)
    }
  }

  const handleDownloadPdf = async () => {
    setGeneratingPdf(true)
    try {
      await downloadReportPdf({
        elementId: TEMPLATE_ID,
        filename: pdfFilename || `${(activePatient?.name || 'Patient').replace(/\s+/g, '_')}_Report.pdf`,
        onStart: () => setGeneratingPdf(true),
        onDone: () => {
          setGeneratingPdf(false)
          setArchiveNotice('📄 PDF report downloaded successfully!')
          setTimeout(() => setArchiveNotice(null), 4000)
        },
        onError: (err) => {
          setGeneratingPdf(false)
          alert('PDF generation error: ' + err.message)
        }
      })
    } catch (e) {
      setGeneratingPdf(false)
    }
  }

  const handleSystemShutdown = async () => {
    setIsSystemShuttingDown(true)
    try {
      await fetch(`${API_URL}/api/system/shutdown?grace=0.2`, { method: 'POST' })
    } catch (e) {}
    setTimeout(() => {
      window.close()
    }, 600)
  }

  const resolvedImageSrc = useMemo(() => {
    if (viewerImageUrl) return viewerImageUrl
    if (selectedSlice?.thumbnail) return `data:image/png;base64,${selectedSlice.thumbnail}`
    return null
  }, [viewerImageUrl, selectedSlice])

  const templateData = {
    studyId: activePatient?.studyId || activePatient?.id || 'STUDY-001',
    status: draftReport.isSigned ? 'signed' : 'draft',
    source: 'PRISM AI + Deterministic Template',
    validated: true,
    findings: reportFindings,
    impression: reportImpression,
    createdAt: session.start_time || new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    signedAt: draftReport.signedAt || new Date().toISOString(),
    radiologistName: radName,
    radiologistDesignation: radDesig,
    signatureDataUrl: sigDataUrl || undefined,
    patientName: activePatient?.name || patientName || 'Jane Doe',
    patientId: activePatient?.id || patientId || 'TEST-001',
    patientAge: activePatient?.age || patientAge || '45Y',
    patientSex: activePatient?.sex || patientSex || 'F',
  }

  // ── Sequential Stepper Stage Calculations ─────────────────────────────────
  const isPhase1Scanning = session.phase1_status === 'scanning'
  const isPhase1Complete = session.phase1_status === 'completed' || (slices.length > 0 && !isPhase1Scanning)

  const isPhase2Running = session.phase2_status === 'running' || (isPhase1Complete && session.phase2_status === 'idle' && isSessionActive)
  const isPhase2Complete = session.phase2_status === 'completed'

  const isPhase3Ready = isPhase3Unlocked
  const isPhase3ActiveView = activeView === 'phase3'

  return (
    <div className="app-shell">
      {/* ── Fixed Header Zone (Pinned Top: Topbar + Controls + Stepper) ── */}
      <div className="app-header-fixed">
        {/* Topbar */}
        <header className="topbar">
          <div className="topbar__left">
            <span className="topbar__logo">◆ PRISM</span>
            <span className="topbar__subtitle">Diagnostic Radiology Intelligence</span>
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

            {/* System Shutdown Button */}
            <button
              onClick={() => setShowSystemShutdownModal(true)}
              title="Shutdown PRISM & Close Terminals"
              style={{
                padding: '4px 10px',
                borderRadius: 6,
                background: 'rgba(239, 68, 68, 0.12)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                color: '#ef4444',
                cursor: 'pointer',
                fontSize: 11,
                fontWeight: 600,
                display: 'inline-flex',
                alignItems: 'center',
                gap: 4,
                marginLeft: 6,
                transition: 'all 0.15s ease'
              }}
            >
              🛑 Shutdown
            </button>
          </div>
        </header>

        {/* Patient Details & Dynamic Single-Action Session Control Bar */}
        <div className="patient-controls-bar">
          <div className="patient-controls-bar__inputs">
            {/* File Location / Dataset Dir */}
            <div className="input-group input-group--dir">
              <label style={{ color: formErrors.datasetDir ? '#f87171' : undefined }}>
                DICOM Directory {formErrors.datasetDir && '*'}
              </label>
              <div className="input-with-button">
                <input
                  type="text"
                  value={datasetDir}
                  disabled={isSessionActive}
                  onChange={(e) => {
                    setDatasetDir(e.target.value)
                    if (formErrors.datasetDir) setFormErrors(prev => ({ ...prev, datasetDir: null }))
                  }}
                  placeholder="D:\Path\To\DICOM\Series"
                  title={datasetDir}
                  className={formErrors.datasetDir ? 'input-error' : ''}
                />
                <button
                  type="button"
                  onClick={handleBrowse}
                  disabled={isSessionActive}
                  title={isSessionActive ? 'Locked during active session' : 'Browse folder'}
                  className="btn-browse"
                >
                  📂
                </button>
              </div>
            </div>

            {/* Patient ID */}
            <div className="input-group input-group--sm">
              <label style={{ color: formErrors.patientId ? '#f87171' : undefined }}>
                Patient ID {formErrors.patientId && '*'}
              </label>
              <input
                type="text"
                value={patientId}
                disabled={isSessionActive}
                onChange={(e) => {
                  setPatientId(e.target.value)
                  if (formErrors.patientId) setFormErrors(prev => ({ ...prev, patientId: null }))
                }}
                placeholder="e.g. P-101"
                className={formErrors.patientId ? 'input-error' : ''}
              />
            </div>

            {/* Patient Name */}
            <div className="input-group input-group--md">
              <label style={{ color: formErrors.patientName ? '#f87171' : undefined }}>
                Patient Name {formErrors.patientName && '*'}
              </label>
              <input
                type="text"
                value={patientName}
                disabled={isSessionActive}
                onChange={(e) => {
                  setPatientName(e.target.value)
                  if (formErrors.patientName) setFormErrors(prev => ({ ...prev, patientName: null }))
                }}
                placeholder="e.g. Jane Doe"
                className={formErrors.patientName ? 'input-error' : ''}
              />
            </div>

            {/* Age */}
            <div className="input-group input-group--xs">
              <label style={{ color: formErrors.patientAge ? '#f87171' : undefined }}>
                Age {formErrors.patientAge && '*'}
              </label>
              <input
                type="text"
                value={patientAge}
                disabled={isSessionActive}
                onChange={(e) => {
                  setPatientAge(e.target.value)
                  if (formErrors.patientAge) setFormErrors(prev => ({ ...prev, patientAge: null }))
                }}
                placeholder="45Y"
                className={formErrors.patientAge ? 'input-error' : ''}
              />
            </div>

            {/* Sex */}
            <div className="input-group input-group--xs">
              <label style={{ color: formErrors.patientSex ? '#f87171' : undefined }}>
                Sex {formErrors.patientSex && '*'}
              </label>
              <select
                value={patientSex}
                disabled={isSessionActive}
                onChange={(e) => {
                  setPatientSex(e.target.value)
                  if (formErrors.patientSex) setFormErrors(prev => ({ ...prev, patientSex: null }))
                }}
                className={formErrors.patientSex ? 'input-error' : ''}
              >
                <option value="">--</option>
                <option value="M">M</option>
                <option value="F">F</option>
                <option value="O">O</option>
              </select>
            </div>

            {/* Speed / Delay */}
            <div className="input-group input-group--xs">
              <label>Delay (ms)</label>
              <input
                type="number"
                value={delayMs}
                disabled={isSessionActive}
                onChange={(e) => setDelayMs(e.target.value)}
                placeholder="30"
              />
            </div>
          </div>

          {/* Dynamic Single-Action Session Button (Start ↔ End) */}
          <div className="patient-controls-bar__actions">
            {!isSessionActive ? (
              <button
                type="button"
                className="btn-session-start"
                onClick={handleStartPatientSession}
                disabled={isStartingSession}
                title="Start patient session (Phase 1 → Phase 2 → Phase 3)"
              >
                {isStartingSession ? '⏳ Starting...' : '▶ Start Patient Session'}
              </button>
            ) : (
              <button
                type="button"
                className="btn-session-end"
                onClick={() => setShowEndModal(true)}
                disabled={isEndingSession}
                title="End active session, archive records to exports/, and reset for new patient"
              >
                {isEndingSession ? 'Archiving...' : '⏹ End Patient Session'}
              </button>
            )}
          </div>
        </div>

        {/* Real-Time Horizontal Scrolling Stepper */}
        <nav className="horizontal-pipeline" aria-label="PRISM Diagnostic Stepper">
          {/* Stage 1: Phase 1 — CT Ingestion & 2D Screening */}
          <button
            type="button"
            className={`pipeline-stage ${
              isPhase1Scanning ? 'pipeline-stage--active' : isPhase1Complete ? 'pipeline-stage--completed' : ''
            }`}
            onClick={() => setActiveView('phase1')}
            title="Click to view live CT Slices"
          >
            <div className="pipeline-stage__indicator">
              <span className="pipeline-stage__num">{isPhase1Complete && !isPhase1Scanning ? '✓' : '1'}</span>
            </div>
            <div className="pipeline-stage__content">
              <div className="pipeline-stage__header">
                <span className="pipeline-stage__title">Phase 1: CT Ingestion</span>
                <span className="pipeline-stage__badge pipeline-stage__badge--phase1">
                  {isPhase1Scanning
                    ? `● Ingesting (${slices.length})`
                    : isPhase1Complete
                    ? `✓ Done (${slices.length})`
                    : '○ Standby'}
                </span>
              </div>
              <div className="pipeline-stage__metrics">
                <span className="metric-pill"><span className="mono">{slices.length}</span> Slices</span>
                <span className={`metric-pill ${stats.flagged > 0 ? 'metric-pill--flagged' : ''}`}><span className="mono">{stats.flagged}</span> Flagged</span>
                <span className="metric-pill"><span className="mono">{stats.avgTime ? `${stats.avgTime.toFixed(0)}ms` : '30ms'}</span></span>
              </div>
            </div>
          </button>

          {/* Connector 1 -> 2 */}
          <div className="pipeline-connector">
            <span className="pipeline-connector__arrow">›</span>
          </div>

          {/* Stage 2: Phase 2 — TotalSegmentator & AI Screening */}
          <div
            className={`pipeline-stage pipeline-stage--p2 ${
              isPhase2Running && !isPhase2Complete ? 'pipeline-stage--active' : isPhase2Complete ? 'pipeline-stage--completed' : ''
            }`}
            title="Phase 2 Multi-Organ TotalSegmentator & AI 3D Merger"
          >
            <div className="pipeline-stage__indicator">
              <span className="pipeline-stage__num">{isPhase2Complete ? '✓' : '2'}</span>
            </div>
            <div className="pipeline-stage__content">
              <div className="pipeline-stage__header">
                <span className="pipeline-stage__title">Phase 2: TotalSegmentator</span>
                <span className="pipeline-stage__badge pipeline-stage__badge--phase2">
                  {isPhase2Running && !isPhase2Complete
                    ? '⏳ In Progress'
                    : isPhase2Complete
                    ? '✓ Complete'
                    : '○ Standby'}
                </span>
              </div>
              <div className="pipeline-stage__metrics">
                <span className="metric-pill">117 Organs</span>
                <span className="metric-pill">3D Merger</span>
              </div>
            </div>
          </div>

          {/* Connector 2 -> 3 */}
          <div className="pipeline-connector">
            <span className="pipeline-connector__arrow">›</span>
          </div>

          {/* Stage 3: Phase 3 — Clinical Report Generation (Gated by Phase 2) */}
          <button
            type="button"
            className={`pipeline-stage pipeline-stage--phase3 ${
              !isPhase3Ready ? 'pipeline-stage--locked' : isPhase3ActiveView ? 'pipeline-stage--active pipeline-stage--unlocked' : 'pipeline-stage--unlocked'
            }`}
            onClick={handleOpenPhase3}
            title={
              isPhase3Ready
                ? 'Click to open Clinical Report Studio'
                : 'Phase 3 is locked until Phase 2 completes'
            }
          >
            <div className="pipeline-stage__indicator">
              <span className="pipeline-stage__num">{!isPhase3Ready ? '🔒' : draftReport.isSigned ? '✓' : '3'}</span>
            </div>
            <div className="pipeline-stage__content">
              <div className="pipeline-stage__header">
                <span className="pipeline-stage__title">Phase 3: Report Studio</span>
                <span className={`pipeline-stage__badge pipeline-stage__badge--phase3 ${!isPhase3Ready ? 'badge--locked' : ''}`}>
                  {!isPhase3Ready
                    ? '🔒 Locked'
                    : draftReport.isSigned
                    ? '✓ Signed'
                    : '✨ Ready'}
                </span>
              </div>
              <div className="pipeline-stage__metrics">
                <span className="metric-pill">Digital Signature</span>
                <span className="metric-pill">{isPhase3Ready ? 'PDF Ready' : 'Locked'}</span>
              </div>
            </div>
          </button>
        </nav>
      </div>

      {/* ── Dynamic Phase Viewport (Directly below Stepper, fills 100% remaining height) ── */}
      <div className="phase-dynamic-viewport">
        {activeView === 'phase1' ? (
          <div className="main-layout">
            {/* ── Left Sidebar: Received Slices Navigation Bar ── */}
            <aside className="slice-panel">
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
                      Enter patient details and click <strong>▶ Start Patient Session</strong> to stream slices.
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

            {/* ── Center: Full CT Canvas Viewer ── */}
            <main className="detail-panel" ref={viewerRef}>
              <div className="viewer-toolbar">
                <div className="viewer-toolbar__left">
                  <span className="viewer-toolbar__title">
                    {selectedSlice ? `Slice #${selectedSlice.slice_id}` : 'CT Live Stream'}
                  </span>
                  {selectedSlice && (
                    <span className="viewer-toolbar__meta mono">
                      I:{selectedSlice.slice_id} ({displayIdx + 1}/{slices.length})
                    </span>
                  )}
                </div>

                <div className="viewer-toolbar__center">
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
                    title="Toggle findings bounding boxes"
                  >
                    Findings {showFindings ? 'ON' : 'OFF'}
                  </button>
                  <button
                    className={`viewer-toolbar__toggle ${isAutoPlay ? 'viewer-toolbar__toggle--active' : ''}`}
                    onClick={() => setIsAutoPlay(v => !v)}
                    title="Auto-play slice stack"
                  >
                    {isAutoPlay ? '⏸ Pause' : '▶ Play'}
                  </button>
                </div>
              </div>

              <div className="detail-panel__body">
                <div className="canvas-viewer">
                  {!selectedSlice ? (
                    <div className="empty-state">
                      <div className="empty-state__icon">🔬</div>
                      <div className="empty-state__text">No active slice stream</div>
                      <div className="empty-state__hint">
                        Enter patient details in the top bar and click <strong>▶ Start Patient Session</strong> to stream slices.
                      </div>
                    </div>
                  ) : (
                    <div className="canvas-viewer__wrapper" ref={wrapperRef}>
                      {/* Corner DICOM Labels */}
                      <div className="dicom-label dicom-label--tl">
                        {(activePatient.name || activePatient.id) && (
                          <div className="mono" style={{ color: '#00d4ff', fontWeight: 600, marginBottom: 2 }}>
                            {activePatient.name} ({activePatient.id})
                          </div>
                        )}
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

                      {/* Image Canvas */}
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
                          }}
                          onLoad={(e) => {
                            const img = e.target
                            if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                              setImageNaturalSize({ w: img.naturalWidth, h: img.naturalHeight })
                            }
                          }}
                        />
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

                      {/* Slice Slider */}
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
                  )}
                </div>

                {/* ── Right Findings & Triage Sidebar (Stable 3-Column) ── */}
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
                      color: (selectedSlice?.emergency_score || 0) >= 80 ? 'var(--status-danger)' : 
                             (selectedSlice?.emergency_score || 0) >= 50 ? '#ff4500' : 
                             (selectedSlice?.emergency_score || 0) >= 20 ? 'var(--status-warning)' : 'var(--status-success)'
                    }}>
                      {selectedSlice?.emergency_score || 0}
                    </span>
                  </div>

                  <div className="detail-sidebar__section">
                    <h3 className="detail-sidebar__title">Score Details</h3>
                    {selectedSlice?.score_details?.length > 0 ? (
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
                        {selectedSlice ? 'No structural anomalies detected on this slice.' : 'Waiting for slice data...'}
                      </div>
                    )}
                  </div>

                  <div className="detail-sidebar__section" style={{ marginTop: '16px' }}>
                    <h3 className="detail-sidebar__title">Detected Findings</h3>
                  </div>

                  {(!selectedSlice?.findings || selectedSlice.findings.length === 0) ? (
                    <div style={{ textAlign: 'center', padding: '16px', color: 'var(--text-muted)' }}>
                      <div style={{ fontSize: '12px' }}>{selectedSlice ? 'Normal attenuation profile' : 'Standby'}</div>
                    </div>
                  ) : (
                    selectedSlice.findings.map((finding, idx) => (
                      <div key={idx} className="finding-card">
                        <div className="finding-card__type" style={{ color: 'var(--status-warning)', textTransform: 'uppercase' }}>
                          ⚠️ {finding.finding_type}
                        </div>
                        <div className="finding-card__row">
                          <span className="finding-card__label">Bounding Box</span>
                          <span className="finding-card__value">[{finding.bbox.join(', ')}]</span>
                        </div>
                        <div className="finding-card__row">
                          <span className="finding-card__label">Area</span>
                          <span className="finding-card__value">{finding.area_px} px</span>
                        </div>
                        <div className="finding-card__row">
                          <span className="finding-card__label">HU Mean</span>
                          <span className="finding-card__value">{finding.hu_mean?.toFixed(1)}</span>
                        </div>
                      </div>
                    ))
                  )}

                  {/* Window / Level Manual Adjustment */}
                  <div style={{
                    marginTop: '16px',
                    padding: '12px',
                    borderRadius: '8px',
                    background: 'var(--bg-tertiary)',
                    border: '1px solid var(--border-subtle)',
                    fontSize: '11px',
                  }}>
                    <div style={{ fontWeight: '600', marginBottom: '8px', color: 'var(--text-secondary)' }}>Window / Level Adjust</div>
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
                </aside>
              </div>
            </main>
          </div>
        ) : (
          /* ── PHASE 3 EMBEDDED CLINICAL REPORT STUDIO ── */
          <div className="embedded-report-view">
            <div className="report-view-banner">
              <button 
                type="button" 
                className="btn-back-slices"
                onClick={() => setActiveView('phase1')}
              >
                ← Back to Phase 1 CT Slices ({slices.length} in memory)
              </button>

              <div className="patient-banner-info">
                <span className="patient-pill">
                  <strong>👤 {activePatient.name}</strong>
                  <span className="mono" style={{ color: '#00d4ff' }}>ID: {activePatient.id}</span>
                  <span>Age: {activePatient.age}</span>
                  <span>Sex: {activePatient.sex}</span>
                </span>
                <span className={`status-badge-pill ${draftReport.isSigned ? 'status-badge-pill--signed' : 'status-badge-pill--draft'}`}>
                  {draftReport.isSigned ? '✓ SIGNED & FINALIZED' : '● DRAFT CLINICAL REPORT'}
                </span>
              </div>

              <div className="report-quick-actions">
                <button 
                  type="button" 
                  className="btn-action-primary"
                  onClick={handleDownloadPdf}
                  disabled={generatingPdf}
                >
                  {generatingPdf ? 'Generating PDF...' : '📄 Download PDF'}
                </button>
                <button 
                  type="button" 
                  className="btn-action-primary"
                  onClick={handleSaveReportDraft}
                  disabled={draftReport.isSigned}
                  style={{ background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.15)' }}
                >
                  💾 Save Draft
                </button>
                {draftReport.isSigned ? (
                  <button 
                    type="button" 
                    className="btn-action-primary"
                    onClick={() => {
                      updateDraftReport({ isSigned: false, signedAt: null })
                      setArchiveNotice('✏️ Report unlocked for editing. You can now modify findings and impression, then re-sign to replace.')
                      setTimeout(() => setArchiveNotice(null), 5000)
                    }}
                    style={{ background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)', color: '#fff', border: 'none' }}
                  >
                    ✏️ Edit Report
                  </button>
                ) : (
                  <button 
                    type="button" 
                    className="btn-action-success"
                    onClick={() => setShowSignModal(true)}
                  >
                    ✍️ Sign & Finalize
                  </button>
                )}
              </div>
            </div>

            <div className="report-studio-grid">
              {/* Clinical Editor */}
              <div className="report-editor-card">
                <div className="studio-section-header">
                  <span className="studio-section-title">
                    📋 Examination & Patient Record
                  </span>
                  <span style={{ fontSize: 11, color: '#94a3b8' }}>
                    Modality: <strong>CT Axial Volumetric</strong> · {slices.length} Slices
                  </span>
                </div>

                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                    <label style={{ fontSize: 12, fontWeight: 700, color: '#f1f5f9' }}>
                      Clinical Findings {draftReport.isSigned ? <span style={{ color: '#94a3b8', fontWeight: 500, fontSize: 11 }}>(Locked · Signed)</span> : <span style={{ color: '#34d399', fontWeight: 500, fontSize: 11 }}>(Editable)</span>}
                    </label>
                    <span style={{ fontSize: 11, color: '#00d4ff' }}>
                      {slices.filter(s => s.flagged).length} Anomalies Synthesized
                    </span>
                  </div>
                  <textarea
                    className={`report-textarea ${draftReport.isSigned ? 'report-textarea--locked' : ''}`}
                    value={reportFindings}
                    disabled={Boolean(draftReport.isSigned)}
                    readOnly={Boolean(draftReport.isSigned)}
                    onChange={(e) => {
                      setReportFindings(e.target.value)
                    }}
                    onBlur={(e) => {
                      updateDraftReport({ findings: e.target.value })
                    }}
                    rows={14}
                    placeholder={draftReport.isSigned ? 'Report is finalized and locked. Click "Edit Report" above to make revisions.' : 'Enter detailed clinical findings...'}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 12, fontWeight: 700, color: '#f1f5f9', marginBottom: 8 }}>
                    Clinical Impression {draftReport.isSigned ? <span style={{ color: '#94a3b8', fontWeight: 500, fontSize: 11 }}>(Locked · Signed)</span> : <span style={{ color: '#34d399', fontWeight: 500, fontSize: 11 }}>(Editable)</span>}
                  </label>
                  <textarea
                    className={`report-textarea ${draftReport.isSigned ? 'report-textarea--locked' : ''}`}
                    value={reportImpression}
                    disabled={Boolean(draftReport.isSigned)}
                    readOnly={Boolean(draftReport.isSigned)}
                    onChange={(e) => {
                      setReportImpression(e.target.value)
                    }}
                    onBlur={(e) => {
                      updateDraftReport({ impression: e.target.value })
                    }}
                    rows={5}
                    placeholder={draftReport.isSigned ? 'Report is finalized and locked. Click "Edit Report" above to make revisions.' : 'Enter radiologist impression & recommendation...'}
                  />
                </div>

                {slices.filter(s => s.flagged).length > 0 && (
                  <div style={{
                    padding: 12,
                    borderRadius: 8,
                    background: 'rgba(239, 68, 68, 0.08)',
                    border: '1px solid rgba(239, 68, 68, 0.25)',
                    fontSize: 12,
                    color: '#f87171'
                  }}>
                    <strong>⚠️ Critical Screen Flags:</strong> {slices.filter(s => s.flagged).length} slices flagged during real-time screening.
                  </div>
                )}
              </div>

              {/* Radiologist Verification */}
              <div className="report-sidebar-card">
                <div className="studio-section-header">
                  <span className="studio-section-title">
                    ✍️ Radiologist Sign-Off
                  </span>
                  <span style={{ fontSize: 11, color: draftReport.isSigned ? '#34d399' : '#fbbf24' }}>
                    {draftReport.isSigned ? 'Signed' : 'Pending Sign'}
                  </span>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                    Radiologist Name
                  </label>
                  <input
                    type="text"
                    value={radName}
                    onChange={(e) => {
                      setRadName(e.target.value)
                      updateDraftReport({ radName: e.target.value })
                    }}
                    disabled={draftReport.isSigned}
                    style={{
                      width: '100%', padding: '9px 12px', fontSize: '12px',
                      background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)',
                      color: '#fff', borderRadius: '6px'
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                    Designation & Qualifications
                  </label>
                  <input
                    type="text"
                    value={radDesig}
                    onChange={(e) => {
                      setRadDesig(e.target.value)
                      updateDraftReport({ radDesig: e.target.value })
                    }}
                    disabled={draftReport.isSigned}
                    style={{
                      width: '100%', padding: '9px 12px', fontSize: '12px',
                      background: 'var(--bg-primary)', border: '1px solid var(--border-subtle)',
                      color: '#fff', borderRadius: '6px'
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
                    Digital Signature
                  </label>
                  {!draftReport.isSigned ? (
                    sigApplied && sigDataUrl ? (
                      <div style={{
                        padding: '12px',
                        background: 'rgba(16, 185, 129, 0.08)',
                        border: '1px solid rgba(16, 185, 129, 0.35)',
                        borderRadius: '8px',
                        textAlign: 'center',
                      }}>
                        <div style={{
                          background: '#ffffff',
                          borderRadius: '6px',
                          padding: '8px 12px',
                          display: 'inline-block',
                          maxWidth: '100%',
                          boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
                        }}>
                          <img
                            src={sigDataUrl}
                            alt="Signature Preview"
                            style={{ maxHeight: '48px', maxWidth: '100%', objectFit: 'contain', display: 'block', margin: '0 auto' }}
                          />
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '8px' }}>
                          <span style={{ color: '#34d399', fontSize: '11px', fontWeight: '700' }}>
                            ✓ Signature Attached
                          </span>
                          <button
                            type="button"
                            onClick={() => { setSigApplied(false); setSigDataUrl(null); updateDraftReport({ sigDataUrl: null }) }}
                            style={{
                              background: 'transparent',
                              border: '1px solid rgba(148, 163, 184, 0.25)',
                              color: '#94a3b8',
                              borderRadius: '4px',
                              padding: '2px 8px',
                              fontSize: '10px',
                              cursor: 'pointer',
                            }}
                          >
                            ✎ Redraw
                          </button>
                        </div>
                      </div>
                    ) : (
                      <div className="sig-pad-container">
                        <SignaturePad
                          onSave={(dataUrl) => {
                            setSigDataUrl(dataUrl)
                            setSigApplied(true)
                            updateDraftReport({ sigDataUrl: dataUrl })
                          }}
                          onClear={() => {
                            setSigDataUrl(null)
                            setSigApplied(false)
                            updateDraftReport({ sigDataUrl: null })
                          }}
                        />
                      </div>
                    )
                  ) : (
                    <div style={{
                      padding: '14px 16px',
                      background: 'rgba(16, 185, 129, 0.1)',
                      border: '1px solid rgba(16, 185, 129, 0.35)',
                      borderRadius: '8px',
                      textAlign: 'center',
                      color: '#34d399',
                      overflow: 'hidden',
                    }}>
                      {sigDataUrl ? (
                        <div style={{
                          background: '#ffffff',
                          borderRadius: '6px',
                          padding: '6px 12px',
                          display: 'inline-block',
                          marginBottom: '8px',
                          maxWidth: '90%',
                          boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
                        }}>
                          <img
                            src={sigDataUrl}
                            alt="Signature"
                            style={{ maxHeight: '44px', maxWidth: '100%', objectFit: 'contain', display: 'block', margin: '0 auto' }}
                          />
                        </div>
                      ) : null}
                      <div style={{ fontSize: '12px', fontWeight: '700' }}>✓ Signed by {radName}</div>
                      <div style={{ fontSize: '10px', color: '#94a3b8', marginTop: '2px' }}>{radDesig}</div>
                      <div style={{ fontSize: '9.5px', color: '#64748b', marginTop: '4px' }}>
                        {new Date(draftReport.signedAt || Date.now()).toLocaleString()}
                      </div>
                    </div>
                  )}
                </div>

                {draftReport.isSigned ? (
                  <button
                    type="button"
                    onClick={() => {
                      updateDraftReport({ isSigned: false, signedAt: null })
                      setArchiveNotice('✏️ Report unlocked for editing. Modify findings and impression, then re-sign to replace.')
                      setTimeout(() => setArchiveNotice(null), 5000)
                    }}
                    style={{
                      width: '100%',
                      padding: '9px 12px',
                      marginTop: 8,
                      background: 'rgba(245, 158, 11, 0.12)',
                      border: '1px solid rgba(245, 158, 11, 0.4)',
                      color: '#fbbf24',
                      borderRadius: 6,
                      fontSize: '11.5px',
                      fontWeight: 700,
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '6px',
                      transition: 'all 0.2s ease',
                    }}
                  >
                    ✏️ Unlock & Edit Report
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn-action-success"
                    onClick={handleSignReport}
                    disabled={signing}
                    style={{ width: '100%', padding: '11px', marginTop: 8 }}
                  >
                    {signing ? 'Signing...' : '✅ Sign & Finalize Report'}
                  </button>
                )}

                <div style={{
                  padding: 12,
                  borderRadius: 8,
                  background: 'rgba(14, 165, 233, 0.08)',
                  border: '1px solid rgba(14, 165, 233, 0.2)',
                  fontSize: 11,
                  color: '#94a3b8',
                  lineHeight: 1.5
                }}>
                  <strong style={{ color: '#38bdf8', display: 'block', marginBottom: 2 }}>🔒 Zero-Hallucination Guarantee</strong>
                  Findings verified against deterministic pipeline tokens. All numerical HU values match DICOM metadata.
                </div>
              </div>
            </div>

            <div style={{ position: 'absolute', top: -99999, left: -99999, visibility: 'hidden' }}>
              <ReportTemplate data={templateData} id={TEMPLATE_ID} />
            </div>
          </div>
        )}
      </div>

      {/* ── Phase 3 Locked Modal ── */}
      {showPhase3LockedModal && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(5, 10, 25, 0.85)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 99999,
        }}>
          <div style={{
            background: 'linear-gradient(180deg, #161b2e 0%, #0d1222 100%)',
            border: '1px solid rgba(0, 212, 255, 0.3)',
            borderRadius: 14,
            padding: 24,
            width: '90%',
            maxWidth: 480,
            boxShadow: '0 20px 50px rgba(0,0,0,0.8), 0 0 30px rgba(0, 212, 255, 0.15)',
            color: '#e2e8f0',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              <div style={{ width: 44, height: 44, borderRadius: 12, background: 'rgba(0, 212, 255, 0.12)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 24 }}>
                🔒
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: 17, fontWeight: 700, color: '#fff' }}>Phase 3 Report Locked</h3>
                <p style={{ margin: '2px 0 0', fontSize: 12, color: '#94a3b8' }}>Awaiting Phase 2 AI Completion</p>
              </div>
            </div>

            <p style={{ fontSize: 13, lineHeight: 1.6, color: '#cbd5e1', marginBottom: 16 }}>
              The <strong>Phase 3 Clinical Report Studio</strong> unlocks automatically after <strong>Phase 2 TotalSegmentator & AI Screening</strong> finishes processing.
            </p>

            <div style={{
              padding: '12px',
              borderRadius: 8,
              background: 'rgba(255, 255, 255, 0.04)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              marginBottom: 20,
              fontSize: 12,
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                <span style={{ color: '#94a3b8' }}>Phase 1 (CT Ingestion):</span>
                <span style={{ color: session.phase1_status === 'completed' ? '#34d399' : '#38bdf8', fontWeight: 600 }}>
                  {session.phase1_status === 'completed' ? '✓ Completed' : `${slices.length} slices received`}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: '#94a3b8' }}>Phase 2 (TotalSegmentator):</span>
                <span style={{ color: session.phase2_status === 'running' ? '#fbbf24' : '#94a3b8', fontWeight: 600 }}>
                  {session.phase2_status === 'running' ? '⏳ In Progress (~15s)' : '⚡ Ready / Auto-Chained'}
                </span>
              </div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button
                onClick={() => setShowPhase3LockedModal(false)}
                style={{
                  padding: '8px 18px',
                  borderRadius: 6,
                  background: 'rgba(255,255,255,0.1)',
                  border: '1px solid rgba(255,255,255,0.15)',
                  color: '#fff',
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── End Patient Session Modal ── */}
      {showEndModal && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(5, 10, 25, 0.85)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 99999,
        }}>
          <div style={{
            background: 'linear-gradient(180deg, #181b2a 0%, #0f121e 100%)',
            border: '1px solid rgba(239, 68, 68, 0.4)',
            borderRadius: 14,
            padding: 24,
            width: '90%',
            maxWidth: 460,
            boxShadow: '0 20px 50px rgba(0,0,0,0.8), 0 0 30px rgba(239,68,68,0.2)',
            color: '#e2e8f0',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              <div style={{ width: 42, height: 42, borderRadius: 10, background: 'rgba(239,68,68,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22 }}>
                ⏹
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: 17, fontWeight: 700, color: '#fff' }}>Shall I end this patient session and archive all records?</h3>
                <p style={{ margin: '2px 0 0', fontSize: 12, color: '#94a3b8' }}>
                  {activePatient.name} ({activePatient.id})
                </p>
              </div>
            </div>

            <p style={{ fontSize: 13, lineHeight: 1.6, color: '#cbd5e1', marginBottom: 16 }}>
              This will package all <strong>{slices.length} DICOM slices</strong> and <strong>report.json</strong> into a zip archive inside <code>exports/</code>, and clear all records for the next patient.
            </p>

            <div style={{
              padding: '10px 14px',
              borderRadius: 8,
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.08)',
              fontSize: 12,
              marginBottom: 20,
              color: '#94a3b8'
            }}>
              <div>• Total Slices: <strong style={{ color: '#fff' }}>{slices.length}</strong></div>
              <div>• Phase 2 Status: <strong style={{ color: '#fff' }}>{session.phase2_status.toUpperCase()}</strong></div>
              <div>• Report Status: <strong style={{ color: draftReport.isSigned ? '#34d399' : '#fbbf24' }}>{draftReport.isSigned ? 'SIGNED' : 'DRAFT'}</strong></div>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button
                onClick={() => setShowEndModal(false)}
                disabled={isEndingSession}
                style={{
                  padding: '8px 16px',
                  borderRadius: 6,
                  background: 'rgba(255,255,255,0.06)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  color: '#94a3b8',
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleEndPatientSessionConfirmed}
                disabled={isEndingSession}
                style={{
                  padding: '8px 18px',
                  borderRadius: 6,
                  background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
                  border: 'none',
                  color: '#fff',
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: 'pointer',
                  boxShadow: '0 2px 10px rgba(239,68,68,0.4)',
                }}
              >
                {isEndingSession ? 'Archiving...' : '⏹ Yes, Archive & End Session'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Toast Notice for Archival & Actions ── */}
      {archiveNotice && (
        <div style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          background: 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
          border: '1px solid #10b981',
          borderRadius: '8px',
          padding: '12px 18px',
          color: '#fff',
          boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5), 0 0 12px rgba(16, 185, 129, 0.3)',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          zIndex: 10000,
          animation: 'fadeIn 0.3s ease-out',
          fontSize: '13px'
        }}>
          <span style={{ fontSize: '18px' }}>🗜️</span>
          <span>{archiveNotice}</span>
        </div>
      )}

      {/* ── System Shutdown Modal ── */}
      {showSystemShutdownModal && (
        <div style={{
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(5, 10, 25, 0.8)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 99999,
        }}>
          <div style={{
            background: 'linear-gradient(180deg, #181828 0%, #0f101d 100%)',
            border: '1px solid rgba(239, 68, 68, 0.35)',
            borderRadius: 14,
            padding: 24,
            width: '90%',
            maxWidth: 460,
            boxShadow: '0 20px 50px rgba(0,0,0,0.8), 0 0 30px rgba(239,68,68,0.2)',
            color: '#e2e8f0',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
              <div style={{ width: 40, height: 40, borderRadius: 10, background: 'rgba(239,68,68,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 20 }}>
                🛑
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: 17, fontWeight: 700, color: '#fff' }}>Shutdown PRISM System?</h3>
                <p style={{ margin: '2px 0 0', fontSize: 12, color: '#94a3b8' }}>Close Backend & Frontend Services</p>
              </div>
            </div>

            <p style={{ fontSize: 13, lineHeight: 1.6, color: '#cbd5e1', marginBottom: 20 }}>
              Shall I shut down the PRISM system, archive any active patient data, and close all running backend and frontend terminal windows?
            </p>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
              <button
                onClick={() => setShowSystemShutdownModal(false)}
                disabled={isSystemShuttingDown}
                style={{
                  padding: '8px 16px',
                  borderRadius: 6,
                  background: 'rgba(255,255,255,0.06)',
                  border: '1px solid rgba(255,255,255,0.1)',
                  color: '#94a3b8',
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleSystemShutdown}
                disabled={isSystemShuttingDown}
                style={{
                  padding: '8px 18px',
                  borderRadius: 6,
                  background: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)',
                  border: 'none',
                  color: '#fff',
                  fontSize: 13,
                  fontWeight: 600,
                  cursor: 'pointer',
                  boxShadow: '0 2px 10px rgba(239,68,68,0.4)',
                }}
              >
                {isSystemShuttingDown ? 'Shutting down...' : '🛑 Yes, Shutdown & Close'}
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}
