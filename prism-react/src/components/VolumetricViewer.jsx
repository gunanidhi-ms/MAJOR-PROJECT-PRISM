/**
 * VolumetricViewer.jsx — 3D Anatomical Reconstruction
 *
 * Renders real organ surface meshes from Phase 2 TotalSegmentator output
 * in a clean anatomical-atlas style (no hologram, no sci-fi effects).
 *
 * Visual style:
 *   • Off-white / warm grey background  (like a medical atlas page)
 *   • Organs in their natural anatomical colours with realistic lighting
 *   • Normal tissue  → natural colour, slightly translucent
 *   • Anomaly zones  → vivid red overlay only (not the whole organ)
 *   • No synthetic body shell — only anatomy reconstructed from the scan
 *   • No fog, no glow, no dark hologram effects
 *
 * Source: GET /api/phase2/organ-mesh (marching cubes on seg NIfTI)
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'

const API = 'http://localhost:8001'

// Background colour — warm clinical white
const BG_COLOR = 0xf0ede8

// Anatomical organ colours — keyed by organ label.
// These are used directly as the organ material colour so organs are
// always identifiable regardless of vertex colour data from the backend.
const ORGAN_PALETTE = {
  liver:                         '#9e4820',  // dark reddish-brown
  spleen:                        '#7a1828',  // deep burgundy
  kidney_right:                  '#b85010',  // burnt orange
  kidney_left:                   '#b85010',
  adrenal_gland_right:           '#c87830',
  adrenal_gland_left:            '#c87830',
  gallbladder:                   '#4a8822',  // olive green
  pancreas:                      '#c88030',  // golden amber
  stomach:                       '#6a6818',  // olive
  small_bowel:                   '#8a6030',  // tan
  duodenum:                      '#9a7040',
  colon:                         '#5a3820',  // dark brown
  rectum:                        '#6a4428',
  esophagus:                     '#a06040',
  urinary_bladder:               '#3878a8',  // blue
  prostate:                      '#4878b0',
  uterus:                        '#c06878',  // rose
  heart:                         '#aa1828',  // crimson
  aorta:                         '#c83010',  // bright red
  inferior_vena_cava:            '#1848a0',  // dark blue
  portal_vein_and_splenic_vein:  '#2850a8',
  lung_upper_lobe_left:          '#4870a8',  // steel blue
  lung_lower_lobe_left:          '#4870a8',
  lung_upper_lobe_right:         '#3860a0',
  lung_middle_lobe_right:        '#3860a0',
  lung_lower_lobe_right:         '#3860a0',
  trachea:                       '#6898b8',
  spinal_cord:                   '#e8d080',  // pale yellow
  vertebrae_l1:                  '#d8c898',  // bone
  vertebrae_l2:                  '#d8c898',
  vertebrae_l3:                  '#d8c898',
  vertebrae_l4:                  '#d8c898',
  vertebrae_l5:                  '#d8c898',
  sacrum:                        '#d0c090',
  hip_left:                      '#c8b888',
  hip_right:                     '#c8b888',
  femur_left:                    '#d0c090',
  femur_right:                   '#d0c090',
  iliopsoas_left:                '#c08050',  // muscle pink
  iliopsoas_right:               '#c08050',
  autochthon_left:               '#c87858',
  autochthon_right:              '#c87858',
}

// Default colour for organs not in the palette
const DEFAULT_ORGAN_COLOR = '#a07858'


const prettyName = (s = '') =>
  s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())

// ─────────────────────────────────────────────────────────────────────────────

export default function VolumetricViewer({ isVisible }) {
  const mountRef = useRef(null)
  const frameRef = useRef(null)
  const clockRef = useRef(new THREE.Clock())

  const [loading,  setLoading]  = useState(true)
  const [error,    setError]    = useState(null)
  const [meshData, setMeshData] = useState(null)
  const [selected, setSelected] = useState(null)
  const [rotating, setRotating] = useState(true)
  const [viewMode, setViewMode] = useState('solid')  // 'solid' | 'wire' | 'xray'

  // ── Fetch organ mesh data ─────────────────────────────────────────────────
  const fetchMesh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const res  = await fetch(`${API}/api/phase2/organ-mesh`)
      const data = await res.json()
      if (data.status === 'ready') {
        setMeshData(data)
      } else {
        setError(data.reason || 'Phase 2 mesh not available')
      }
    } catch (e) {
      setError('Backend unreachable: ' + e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { if (isVisible) fetchMesh() }, [isVisible, fetchMesh])

  // ── Build Three.js scene when mesh data arrives ───────────────────────────
  useEffect(() => {
    if (!meshData || !mountRef.current || !isVisible) return
    if (meshData.organs?.length === 0) return

    const W = mountRef.current.clientWidth  || 800
    const H = mountRef.current.clientHeight || 500

    // ── Renderer (clean, no tone mapping distortion) ──────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setSize(W, H)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.toneMapping = THREE.LinearToneMapping
    renderer.toneMappingExposure = 1.0
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    mountRef.current.appendChild(renderer.domElement)

    // ── Scene ─────────────────────────────────────────────────────────────
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(BG_COLOR)
    // No fog — clean anatomical view

    // ── Camera ────────────────────────────────────────────────────────────
    const camera = new THREE.PerspectiveCamera(38, W / H, 0.001, 50)
    camera.position.set(0, 0.05, 2.2)
    camera.lookAt(0, 0, 0)

    // ── Anatomical Lighting ───────────────────────────────────────────────
    // Neutral warm ambient (fills shadows, like clinical room lighting)
    scene.add(new THREE.AmbientLight(0xfff5ee, 0.72))

    // Primary key — upper-front, slightly warm (overhead surgical lamp)
    const key = new THREE.DirectionalLight(0xfffaf5, 1.1)
    key.position.set(0.5, 3.0, 2.5)
    key.castShadow = false
    scene.add(key)

    // Left fill — cooler, soft (wall bounce)
    const fillL = new THREE.DirectionalLight(0xddeeff, 0.38)
    fillL.position.set(-2.5, 1.0, 0.5)
    scene.add(fillL)

    // Right fill — slightly warm (anatomy atlas highlight from right)
    const fillR = new THREE.DirectionalLight(0xffeedd, 0.28)
    fillR.position.set(2.0, 0.5, -0.5)
    scene.add(fillR)

    // Back rim — separates organs from background
    const back = new THREE.DirectionalLight(0xffffff, 0.18)
    back.position.set(0, -1.5, -2.0)
    scene.add(back)

    // ── Subtle floor shadow plane ─────────────────────────────────────────
    const shadowGeo = new THREE.PlaneGeometry(2.5, 2.5)
    const shadowMat = new THREE.MeshBasicMaterial({
      color: 0xc8bfb0,
      transparent: true,
      opacity: 0.18,
    })
    const shadowPlane = new THREE.Mesh(shadowGeo, shadowMat)
    shadowPlane.rotation.x = -Math.PI / 2
    shadowPlane.position.y = -0.56
    scene.add(shadowPlane)

    // ── Build organ meshes ────────────────────────────────────────────────
    const organMeshes  = []
    const organGroup   = new THREE.Group()

    const organs = meshData.organs || []

    organs.forEach(organ => {
      if (!organ.vertices?.length || !organ.faces?.length) return

      const geo = new THREE.BufferGeometry()

      const vertArr = new Float32Array(organ.vertices.length * 3)
      // CT data is X (left/right), Y (anterior/posterior), Z (head/feet).
      // Three.js uses Y as up: X, Z, -Y opens as an upright atlas view.
      organ.vertices.forEach(([x, y, z], i) => {
        vertArr[i * 3]     = x
        vertArr[i * 3 + 1] = z
        vertArr[i * 3 + 2] = -y
      })
      geo.setAttribute('position', new THREE.BufferAttribute(vertArr, 3))

      const faceArr = new Uint32Array(organ.faces.length * 3)
      organ.faces.forEach(([a, b, c], i) => {
        faceArr[i * 3]     = a
        faceArr[i * 3 + 1] = b
        faceArr[i * 3 + 2] = c
      })
      geo.setIndex(new THREE.BufferAttribute(faceArr, 1))
      geo.computeVertexNormals()

      // ── Organ base colour from anatomical palette ────────────────────
      // We do NOT use vertex colours from the backend — they are unreliable
      // when thousands of Phase 1 candidates flood the colour map.
      // Instead each organ gets its true anatomical colour.
      const hexColor = ORGAN_PALETTE[organ.label] || DEFAULT_ORGAN_COLOR
      const organColor = new THREE.Color(hexColor)

      // ── Solid material ────────────────────────────────────────────────
      const solidMat = new THREE.MeshPhongMaterial({
        color: organColor,
        specular: new THREE.Color(0x331108),
        shininess: 28,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.90,
        depthWrite: true,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      })

      // ── Wireframe ─────────────────────────────────────────────────────
      const wireMat = new THREE.MeshBasicMaterial({
        color: organColor,
        wireframe: true,
        transparent: true,
        opacity: 0.50,
      })

      // ── X-Ray ─────────────────────────────────────────────────────────
      const xrayMat = new THREE.MeshPhongMaterial({
        color: organColor,
        shininess: 5,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.20,
        depthWrite: false,
      })

      const mesh = new THREE.Mesh(geo, solidMat)
      mesh.userData = { organ, solidMat, wireMat, xrayMat, isOrgan: true }
      organMeshes.push(mesh)
      organGroup.add(mesh)

      // ── Anomaly overlay: semi-transparent red on top of natural colour ─
      // Only shown when server confirms this organ has anomaly candidates.
      // Red is a thin overlay — the organ's own colour still shows through.
      if (organ.is_anomalous) {
        const anomGeo = geo.clone()
        const anomMat = new THREE.MeshPhongMaterial({
          color: 0xcc1122,
          shininess: 5,
          transparent: true,
          opacity: 0.32,
          side: THREE.DoubleSide,
          depthWrite: false,
        })
        organGroup.add(new THREE.Mesh(anomGeo, anomMat))
      }
    })

    scene.add(organGroup)

    // ── Standalone anomaly markers (unclassified candidates) ─────────────
    ;(meshData.anomalies || []).forEach(anom => {
      const [cx, cy, cz, w, h, d] = anom.bbox_norm || [0,0,0,0.05,0.05,0.05]
      const boxGeo = new THREE.BoxGeometry(
        Math.max(w, 0.015), Math.max(d, 0.015), Math.max(h, 0.015)
      )
      // Solid red marker cube — like a pathology pin
      const boxMat = new THREE.MeshPhongMaterial({
        color: 0xcc1122,
        transparent: true,
        opacity: 0.6,
        shininess: 30,
      })
      const box = new THREE.Mesh(boxGeo, boxMat)
      box.position.set(cx, cz, -cy)
      box.userData = { isAnomaly: true, candidate: anom }
      scene.add(box)
    })

    // ── Raycaster ─────────────────────────────────────────────────────────
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()

    function onCanvasClick(e) {
      const rect = renderer.domElement.getBoundingClientRect()
      mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1
      mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1
      raycaster.setFromCamera(mouse, camera)
      const hits = raycaster.intersectObjects(organMeshes, false)
      if (hits.length > 0) {
        setSelected(hits[0].object.userData.organ)
      } else {
        setSelected(null)
      }
    }
    renderer.domElement.addEventListener('click', onCanvasClick)

    // ── Simple orbit controls ─────────────────────────────────────────────
    let isDragging = false
    let prevMouse  = { x: 0, y: 0 }
    let theta = 0, phi = Math.PI / 2

    function onMouseDown(e) { isDragging = true; prevMouse = { x: e.clientX, y: e.clientY } }
    function onMouseMove(e) {
      if (!isDragging) return
      const dx = e.clientX - prevMouse.x
      const dy = e.clientY - prevMouse.y
      theta -= dx * 0.005
      phi    = Math.max(0.2, Math.min(Math.PI - 0.2, phi - dy * 0.005))
      prevMouse = { x: e.clientX, y: e.clientY }
    }
    function onMouseUp() { isDragging = false }
    function onWheel(e)  { camera.position.multiplyScalar(1 + e.deltaY * 0.001) }

    renderer.domElement.addEventListener('mousedown', onMouseDown)
    renderer.domElement.addEventListener('mousemove', onMouseMove)
    renderer.domElement.addEventListener('mouseup',   onMouseUp)
    renderer.domElement.addEventListener('wheel',     onWheel, { passive: true })

    // ── View mode switching ───────────────────────────────────────────────
    const viewModeRef = { current: viewMode }

    // ── Animate ───────────────────────────────────────────────────────────
    const rotatingRef = { current: rotating }

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate)

      if (rotatingRef.current && !isDragging) theta += 0.003

      const r = camera.position.length()
      camera.position.x = r * Math.sin(phi) * Math.sin(theta)
      camera.position.y = r * Math.cos(phi)
      camera.position.z = r * Math.sin(phi) * Math.cos(theta)
      camera.lookAt(0, 0, 0)

      // Update material based on current view mode
      const vm = viewModeRef.current
      organMeshes.forEach(m => {
        const { solidMat, wireMat, xrayMat } = m.userData
        if (vm === 'solid') m.material = solidMat
        else if (vm === 'wire') m.material = wireMat
        else m.material = xrayMat
      })

      // Sync rotating ref from React state
      rotatingRef.current = rotating

      renderer.render(scene, camera)
    }
    animate()

    // ── Resize ────────────────────────────────────────────────────────────
    const onResize = () => {
      if (!mountRef.current) return
      const w = mountRef.current.clientWidth
      const h = mountRef.current.clientHeight
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    window.addEventListener('resize', onResize)

    // ── Cleanup ───────────────────────────────────────────────────────────
    return () => {
      cancelAnimationFrame(frameRef.current)
      window.removeEventListener('resize', onResize)
      renderer.domElement.removeEventListener('click', onCanvasClick)
      renderer.domElement.removeEventListener('mousedown', onMouseDown)
      renderer.domElement.removeEventListener('mousemove', onMouseMove)
      renderer.domElement.removeEventListener('mouseup',   onMouseUp)
      renderer.domElement.removeEventListener('wheel',     onWheel)
      renderer.dispose()
      organMeshes.forEach(m => {
        m.geometry.dispose()
        if (m.userData.solidMat) m.userData.solidMat.dispose()
        if (m.userData.wireMat)  m.userData.wireMat.dispose()
        if (m.userData.xrayMat)  m.userData.xrayMat.dispose()
      })
      if (mountRef.current) mountRef.current.innerHTML = ''
    }
  }, [meshData, isVisible])

  // Sync rotating to ref without rebuilding scene
  const rotatingRef = useRef(rotating)
  useEffect(() => { rotatingRef.current = rotating }, [rotating])

  if (!isVisible) return null

  const anomalyOrgans = (meshData?.organs || []).filter(o => o.is_anomalous)
  const normalOrgans  = (meshData?.organs || []).filter(o => !o.is_anomalous)

  return (
    <div style={S.root}>

      {/* ── Top toolbar ────────────────────────────────────────────── */}
      <div style={S.toolbar}>
        <div style={S.toolbarLeft}>
          <span style={S.statusDot} />
          <span style={S.toolbarTitle}>3D Anatomical Reconstruction</span>
          <span style={S.toolbarSub}>Phase 2 · TotalSegmentator · Marching Cubes</span>
        </div>
        <div style={S.toolbarRight}>
          {(['solid', 'wire', 'xray']).map(m => (
            <button
              key={m}
              style={{ ...S.modeBtn, ...(viewMode === m ? S.modeBtnActive : {}) }}
              onClick={() => setViewMode(m)}
            >
              {m === 'solid' ? '■ Solid' : m === 'wire' ? '⬡ Wire' : '◎ X-Ray'}
            </button>
          ))}
          <button
            style={{ ...S.ctrlBtn, ...(rotating ? S.ctrlBtnOn : {}) }}
            onClick={() => setRotating(v => !v)}
          >
            {rotating ? '⏸' : '▶'}
          </button>
          <button style={S.ctrlBtn} onClick={fetchMesh} title="Refresh mesh">↻</button>
        </div>
      </div>

      {/* ── Legend ─────────────────────────────────────────────────── */}
      <div style={S.legend}>
        <span style={{ ...S.legendItem, color: '#7a5538' }}>
          <span style={{ ...S.legendSwatch, background: '#b05a2a' }} /> Normal tissue
        </span>
        <span style={{ ...S.legendItem, color: '#bb2233' }}>
          <span style={{ ...S.legendSwatch, background: '#cc1122' }} /> Anomaly zone
        </span>
        <span style={{ ...S.legendItem, color: '#6a543a' }}>
          <span style={{ ...S.legendSwatch, background: '#d8c898', border: '1px solid #bca778' }} /> Reconstructed anatomy
        </span>
        <span style={S.legendTip}>Drag to rotate · Scroll to zoom · Click organ for details</span>
      </div>

      {/* ── WebGL canvas ───────────────────────────────────────────── */}
      <div style={S.canvasWrap}>
        {loading && (
          <div style={S.overlay}>
            <div style={S.spinner} />
            <div style={S.overlayTitle}>Building 3D organ meshes…</div>
            <div style={S.overlayNote}>
              Running marching cubes on TotalSegmentator output.<br />
              This takes 5–20 seconds depending on anatomy complexity.
            </div>
          </div>
        )}
        {error && !loading && (
          <div style={S.overlay}>
            <span style={{ fontSize: 32 }}>🔬</span>
            <div style={S.overlayTitle}>Mesh not available</div>
            <div style={S.overlayNote}>{error}</div>
            <button style={S.retryBtn} onClick={fetchMesh}>↻ Retry</button>
          </div>
        )}
        {!loading && !error && meshData && meshData.organs?.length === 0 && (
          <div style={S.overlay}>
            <span style={{ fontSize: 32 }}>🫀</span>
            <div style={S.overlayTitle}>No organ meshes generated</div>
            <div style={S.overlayNote}>
              TotalSegmentator found no labelled organs in this scan region.<br />
              Check the seg NIfTI in phase2_work/.
            </div>
          </div>
        )}
        <div ref={mountRef} style={S.canvas} />
      </div>

      {/* ── Stats bar ──────────────────────────────────────────────── */}
      {meshData && !loading && !error && (
        <div style={S.statsBar}>
          <span style={S.statPill}>
            🫀 {(meshData.organs || []).length} organs reconstructed
          </span>
          <span style={{ ...S.statPill, ...(anomalyOrgans.length > 0 ? S.statPillRed : {}) }}>
            🔴 {anomalyOrgans.length} organs with anomaly zones
          </span>
          <span style={S.statPill}>
            📍 {meshData.n_anomaly_candidates || 0} candidates
          </span>
          <span style={{ ...S.statPill }}>
            🗺 {(meshData.region || 'unknown').replace(/_/g, ' ').toUpperCase()}
          </span>
        </div>
      )}

      {/* ── Organ list sidebar (right) ─────────────────────────────── */}
      {meshData && !loading && !error && (
        <div style={S.organList}>
          <div style={S.organListTitle}>Organs</div>
          {anomalyOrgans.length > 0 && (
            <>
              <div style={S.organListSectionHead}>⚠ Anomalous</div>
              {anomalyOrgans.map(o => (
                <div
                  key={o.label}
                  style={{ ...S.organRow, ...S.organRowAnomaly,
                    ...(selected?.label === o.label ? S.organRowSelected : {}) }}
                  onClick={() => setSelected(s => s?.label === o.label ? null : o)}
                >
                  <span style={{ ...S.organDot, background: '#cc1122' }} />
                  <span style={S.organName}>{prettyName(o.label)}</span>
                  <span style={S.organVol}>{o.volume_cc} cc</span>
                </div>
              ))}
            </>
          )}
          <div style={S.organListSectionHead}>Normal</div>
          {normalOrgans.slice(0, 14).map(o => (
            <div
              key={o.label}
              style={{ ...S.organRow,
                ...(selected?.label === o.label ? S.organRowSelected : {}) }}
              onClick={() => setSelected(s => s?.label === o.label ? null : o)}
            >
              <span style={{
                ...S.organDot,
                background: ORGAN_PALETTE[o.label] || '#9a6a40'
              }} />
              <span style={S.organName}>{prettyName(o.label)}</span>
              <span style={S.organVol}>{o.volume_cc} cc</span>
            </div>
          ))}
        </div>
      )}

      {/* ── Selected organ detail panel ────────────────────────────── */}
      {selected && (
        <div style={S.detail}>
          <button style={S.detailClose} onClick={() => setSelected(null)}>✕</button>
          <div style={{
            ...S.detailTitle,
            color: selected.is_anomalous ? '#cc1122' : '#6a3a18',
          }}>
            {selected.is_anomalous ? '🔴' : '🟤'} {prettyName(selected.label)}
          </div>
          <div style={S.detailGrid}>
            <div style={S.detailCell}>
              <span style={S.detailKey}>Volume</span>
              <span style={S.detailVal}>{selected.volume_cc} cc</span>
            </div>
            <div style={S.detailCell}>
              <span style={S.detailKey}>Status</span>
              <span style={{ ...S.detailVal, color: selected.is_anomalous ? '#cc1122' : '#2a7a3a' }}>
                {selected.is_anomalous ? 'Anomaly zone present' : 'Within normal range'}
              </span>
            </div>
            <div style={S.detailCell}>
              <span style={S.detailKey}>Mesh faces</span>
              <span style={S.detailVal}>{selected.faces?.length?.toLocaleString()}</span>
            </div>
            <div style={S.detailCell}>
              <span style={S.detailKey}>Mesh vertices</span>
              <span style={S.detailVal}>{selected.vertices?.length?.toLocaleString()}</span>
            </div>
          </div>
          {selected.is_anomalous && (
            <div style={S.detailNote}>
              Red zones = voxels where Phase 1 triage candidates overlap this organ's
              TotalSegmentator mask. Confirmed by intra-organ z-score deviation.<br />
              <strong style={{ color: '#b83a00' }}>Requires radiologist review.</strong>
            </div>
          )}
          <div style={S.zeroHallucinationBadge}>
            🛡 Zero Hallucination Policy · No diagnosis asserted · Descriptor only
          </div>
        </div>
      )}
    </div>
  )
}

// ── Styles — clean anatomical atlas palette ───────────────────────────────────
const S = {
  root: {
    width: '100%', height: '100%',
    background: '#f4f0eb',   // warm off-white — like an anatomy atlas page
    borderRadius: 10,
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    border: '1px solid #d8cfc4',
    boxShadow: '0 2px 16px rgba(80,50,20,0.10)',
    position: 'relative',
    fontFamily: '"Inter","Segoe UI",sans-serif',
  },
  toolbar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '7px 14px',
    borderBottom: '1px solid #ddd4c8',
    background: '#ede8e2',
    flexShrink: 0,
  },
  toolbarLeft:  { display: 'flex', alignItems: 'center', gap: 10 },
  toolbarRight: { display: 'flex', alignItems: 'center', gap: 6 },
  statusDot: {
    width: 7, height: 7, borderRadius: '50%',
    background: '#3a8a3a',   // green — "model is live"
  },
  toolbarTitle: { fontSize: 12, fontWeight: 700, color: '#3a2810', letterSpacing: '0.03em' },
  toolbarSub:   { fontSize: 10, color: '#9a7a5a', fontFamily: 'monospace' },
  modeBtn: {
    background: 'rgba(100,60,20,0.07)',
    border: '1px solid rgba(100,60,20,0.2)',
    color: 'rgba(80,50,20,0.65)',
    padding: '3px 9px', borderRadius: 5, fontSize: 10, cursor: 'pointer',
  },
  modeBtnActive: {
    background: 'rgba(120,70,20,0.14)',
    color: '#6a3810',
    border: '1px solid rgba(120,70,20,0.35)',
    fontWeight: 700,
  },
  ctrlBtn: {
    background: 'rgba(100,60,20,0.08)',
    border: '1px solid rgba(100,60,20,0.2)',
    color: '#7a5030', padding: '3px 8px',
    borderRadius: 5, fontSize: 12, cursor: 'pointer',
  },
  ctrlBtnOn: { background: 'rgba(100,60,20,0.15)' },
  legend: {
    display: 'flex', gap: 16, padding: '4px 14px', flexShrink: 0, flexWrap: 'wrap',
    borderBottom: '1px solid #ddd4c8',
    alignItems: 'center',
    background: '#ede8e2',
  },
  legendItem:   { display: 'flex', alignItems: 'center', gap: 5, fontSize: 10, fontWeight: 500 },
  legendSwatch: { width: 9, height: 9, borderRadius: 2, display: 'inline-block' },
  legendTip:    { fontSize: 9, color: '#a08060', marginLeft: 'auto' },
  canvasWrap:   { flex: 1, position: 'relative', overflow: 'hidden', minHeight: 0 },
  canvas:       { width: '100%', height: '100%' },
  overlay: {
    position: 'absolute', inset: 0, zIndex: 10,
    display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    background: 'rgba(244,240,235,0.92)', gap: 10,
  },
  spinner: {
    width: 34, height: 34,
    border: '3px solid rgba(120,70,20,0.15)',
    borderTop: '3px solid #9a5020',
    borderRadius: '50%',
    animation: 'spin 1s linear infinite',
  },
  overlayTitle: { fontSize: 14, fontWeight: 700, color: '#3a2810' },
  overlayNote:  { fontSize: 11, color: '#8a6040', textAlign: 'center', maxWidth: 300, lineHeight: 1.5 },
  retryBtn: {
    marginTop: 6, padding: '5px 14px',
    background: 'rgba(120,70,20,0.1)', border: '1px solid rgba(120,70,20,0.25)',
    color: '#7a4010', borderRadius: 6, cursor: 'pointer', fontSize: 12,
  },
  statsBar: {
    display: 'flex', gap: 8, padding: '5px 14px', flexShrink: 0, flexWrap: 'wrap',
    borderTop: '1px solid #ddd4c8', alignItems: 'center',
    background: '#ede8e2',
  },
  statPill: {
    background: 'rgba(100,60,20,0.07)',
    border: '1px solid rgba(100,60,20,0.15)',
    borderRadius: 20, padding: '2px 9px', fontSize: 10, color: '#7a5030',
  },
  statPillRed: {
    background: 'rgba(180,20,30,0.07)',
    border: '1px solid rgba(180,20,30,0.2)', color: '#aa1a22',
  },
  organList: {
    position: 'absolute', top: 76, right: 8, bottom: 36,
    width: 168, overflowY: 'auto',
    background: 'rgba(240,234,226,0.96)',
    border: '1px solid #d0c4b4',
    borderRadius: 8, padding: '6px 0',
    zIndex: 5,
    boxShadow: '0 2px 12px rgba(80,50,20,0.10)',
  },
  organListTitle: {
    fontSize: 10, fontWeight: 700, color: '#6a4020',
    textTransform: 'uppercase', letterSpacing: '0.1em',
    padding: '0 10px 4px',
    borderBottom: '1px solid #d8cfc4',
    marginBottom: 4,
  },
  organListSectionHead: {
    fontSize: 9, fontWeight: 700,
    color: '#9a7050', textTransform: 'uppercase',
    letterSpacing: '0.08em',
    padding: '4px 10px 2px',
  },
  organRow: {
    display: 'flex', alignItems: 'center', gap: 6,
    padding: '3px 10px', cursor: 'pointer',
    transition: 'background 0.1s',
  },
  organRowAnomaly:  { background: 'rgba(180,20,30,0.06)' },
  organRowSelected: { background: 'rgba(120,70,20,0.10)' },
  organDot:  { width: 7, height: 7, borderRadius: '50%', flexShrink: 0 },
  organName: { fontSize: 10, color: '#3a2010', flex: 1, lineHeight: 1.3 },
  organVol:  { fontSize: 9, color: '#9a7050', fontFamily: 'monospace' },
  detail: {
    position: 'absolute', bottom: 36, left: 14,
    width: 265,
    background: 'rgba(244,240,235,0.98)',
    border: '1px solid #ccc0b0',
    borderRadius: 10, padding: '12px 14px',
    boxShadow: '0 4px 20px rgba(80,50,20,0.12)',
    zIndex: 20,
  },
  detailClose: {
    position: 'absolute', top: 8, right: 10,
    background: 'none', border: 'none', color: '#9a8070',
    cursor: 'pointer', fontSize: 13,
  },
  detailTitle:  { fontSize: 12, fontWeight: 700, marginBottom: 8, paddingRight: 18 },
  detailGrid:   { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '5px 10px', marginBottom: 8 },
  detailCell:   { display: 'flex', flexDirection: 'column', gap: 1 },
  detailKey:    { fontSize: 9, color: '#9a7858', textTransform: 'uppercase', letterSpacing: '0.08em' },
  detailVal:    { fontSize: 11, color: '#2a1808', fontFamily: 'monospace', fontWeight: 600 },
  detailNote: {
    fontSize: 10, color: '#7a4830', lineHeight: 1.5, marginBottom: 8,
    padding: '6px 8px',
    background: 'rgba(180,20,30,0.05)',
    borderRadius: 5,
    borderLeft: '2px solid rgba(180,20,30,0.3)',
  },
  zeroHallucinationBadge: {
    fontSize: 8.5, color: '#9a7858', textAlign: 'center',
    padding: '4px 0 0', borderTop: '1px solid #d8cfc4',
  },
}
