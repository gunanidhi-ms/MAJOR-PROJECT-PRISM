import React, { createContext, useContext, useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { prismApi, type PatientSessionInfo, type StartSessionRequest } from '../api/prismApi';

export interface SliceData {
  slice_id: number;
  uid?: string;
  patient_id?: string;
  patient_name?: string;
  patient_age?: string;
  patient_sex?: string;
  study_id?: string;
  study_instance_uid?: string;
  series_instance_uid?: string;
  flagged?: boolean;
  emergency_score?: number;
  emergency_action?: string;
  score_details?: Array<{ points: number; description: string }>;
  timestamp?: string;
  findings?: Array<{
    finding_type: string;
    bbox: number[];
    area_px: number;
    hu_mean: number;
    severity?: string;
  }>;
  processing_time_ms?: number;
  triage_time_ms?: number;
  slice_hu_mean?: number;
  slice_hu_std?: number;
  detected_region?: string;
  thumbnail?: string;
  window_center_default?: number;
  window_width_default?: number;
  _isNew?: boolean;
}

export interface DraftReportState {
  findings: string;
  impression: string;
  isSigned: boolean;
  signedAt: string;
  radName: string;
  radDesig: string;
  sigDataUrl: string | null;
}

export interface PatientSessionContextType {
  session: PatientSessionInfo;
  slices: SliceData[];
  selectedIdx: number;
  setSelectedIdx: React.Dispatch<React.SetStateAction<number>>;
  connectionStatus: 'connected' | 'connecting' | 'disconnected';
  stats: { total: number; flagged: number; avgTime: number };
  draftReport: DraftReportState;
  pipelineEvent: any;
  isPhase3Unlocked: boolean;
  updateSessionField: (field: keyof PatientSessionInfo, value: any) => void;
  startSession: (config: StartSessionRequest) => Promise<{ status: string; message: string }>;
  stopScan: () => Promise<{ status: string; message: string }>;
  endSession: () => Promise<{ status: string; message: string; archive_file?: string }>;
  triggerPhase2: () => Promise<{ status: string; message: string }>;
  updateDraftReport: (updates: Partial<DraftReportState>) => void;
  resetDraftReport: () => void;
}

const defaultSession: PatientSessionInfo = {
  patient_id: '',
  patient_name: '',
  patient_age: '',
  patient_sex: '',
  dataset_dir: 'D:\\MAJOR-PROJECT-PRISM\\phase1_ingestion\\sample_dicoms\\test_series',
  delay_ms: 30,
  study_id: '',
  series_id: '',
  session_active: false,
  phase1_status: 'idle',
  phase2_status: 'idle',
  phase3_status: 'locked',
  total_slices: 0,
  flagged_slices: 0,
  start_time: '',
  scanner_running: false,
};

const defaultDraftReport: DraftReportState = {
  findings: '',
  impression: '',
  isSigned: false,
  signedAt: '',
  radName: 'Dr. Sarah Chen, MD',
  radDesig: 'Senior Consultant Radiologist | FRCR',
  sigDataUrl: null,
};

const WS_URL = 'ws://localhost:8001/ws/alerts';
const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 16000];

const PatientSessionContext = createContext<PatientSessionContextType | null>(null);

export const PatientSessionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [session, setSession] = useState<PatientSessionInfo>(() => {
    try {
      const cached = sessionStorage.getItem('prism_patient_session');
      return cached ? JSON.parse(cached) : defaultSession;
    } catch {
      return defaultSession;
    }
  });

  const [slices, setSlices] = useState<SliceData[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'connecting' | 'disconnected'>('disconnected');
  const [pipelineEvent, setPipelineEvent] = useState<any>(null);
  const [draftReport, setDraftReport] = useState<DraftReportState>(() => {
    try {
      const cached = sessionStorage.getItem('prism_draft_report');
      return cached ? JSON.parse(cached) : defaultDraftReport;
    } catch {
      return defaultDraftReport;
    }
  });

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempt = useRef(0);
  const reconnectTimer = useRef<any>(null);

  // Synchronize session info to sessionStorage
  useEffect(() => {
    try {
      sessionStorage.setItem('prism_patient_session', JSON.stringify(session));
    } catch (e) {}
  }, [session]);

  // Synchronize draft report to sessionStorage
  useEffect(() => {
    try {
      sessionStorage.setItem('prism_draft_report', JSON.stringify(draftReport));
    } catch (e) {}
  }, [draftReport]);

  const updateSessionField = useCallback((field: keyof PatientSessionInfo, value: any) => {
    setSession(prev => ({ ...prev, [field]: value }));
  }, []);

  const updateDraftReport = useCallback((updates: Partial<DraftReportState>) => {
    setDraftReport(prev => ({ ...prev, ...updates }));
  }, []);

  const resetDraftReport = useCallback(() => {
    setDraftReport(defaultDraftReport);
  }, []);

  // Compute stats
  const stats = useMemo(() => {
    const total = slices.length;
    const flagged = slices.filter(s => s.flagged).length;
    const totalTime = slices.reduce((sum, s) => sum + (s.processing_time_ms || 0), 0);
    return {
      total,
      flagged,
      avgTime: total > 0 ? totalTime / total : 0,
    };
  }, [slices]);

  // Phase 3 is unlocked ONLY when Phase 2 has completed (or report is already generated/signed)
  const isPhase3Unlocked = useMemo(() => {
    return (
      session.phase2_status === 'completed' ||
      session.phase3_status === 'unlocked' ||
      session.phase3_status === 'draft_ready' ||
      session.phase3_status === 'signed' ||
      draftReport.isSigned
    );
  }, [session.phase2_status, session.phase3_status, draftReport.isSigned]);

  // ── Global WebSocket connection & slice accumulator ─────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    setConnectionStatus('connecting');

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus('connected');
        reconnectAttempt.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const alert = JSON.parse(event.data);
          if (alert.type === 'pong') return;

          // Full active session state snapshot greeting
          if (alert.type === 'session_sync' && alert.session) {
            const s = alert.session as PatientSessionInfo;
            setSession(prev => ({
              ...prev,
              patient_id: s.patient_id || prev.patient_id,
              patient_name: s.patient_name || prev.patient_name,
              patient_age: s.patient_age || prev.patient_age,
              patient_sex: s.patient_sex || prev.patient_sex,
              dataset_dir: s.dataset_dir || prev.dataset_dir,
              delay_ms: s.delay_ms || prev.delay_ms,
              study_id: s.study_id || prev.study_id,
              series_id: s.series_id || prev.series_id,
              session_active: s.session_active,
              phase1_status: s.phase1_status || prev.phase1_status,
              phase2_status: s.phase2_status || prev.phase2_status,
              phase3_status: s.phase3_status || prev.phase3_status,
              scanner_running: s.scanner_running,
            }));
            return;
          }

          // New patient session start event
          if (alert.type === 'patient_session_start') {
            setSlices([]);
            setSelectedIdx(0);
            setPipelineEvent(null);
            setDraftReport(defaultDraftReport);
            setSession(prev => ({
              ...prev,
              patient_id: alert.patient_id || '',
              patient_name: alert.patient_name || '',
              patient_age: alert.patient_age || '',
              patient_sex: alert.patient_sex || '',
              study_id: alert.study_id || '',
              dataset_dir: alert.dataset_dir || prev.dataset_dir,
              delay_ms: alert.delay_ms || prev.delay_ms,
              session_active: true,
              phase1_status: 'idle',
              phase2_status: 'idle',
              phase3_status: 'locked',
              scanner_running: false,
            }));
            return;
          }

          // Pipeline progress events (Phase 1 / Phase 2 / Phase 3)
          if (alert.is_pipeline_event || alert.type === 'pipeline_event') {
            setPipelineEvent(alert);

            if (alert.phase === 'ingestion') {
              if (alert.status === 'completed') {
                setSession(prev => ({
                  ...prev,
                  phase1_status: 'completed',
                  scanner_running: false,
                }));
              }
            } else if (alert.phase === 'segmentation') {
              if (alert.status === 'started') {
                setSession(prev => ({
                  ...prev,
                  phase2_status: 'running',
                }));
              } else if (alert.status === 'completed') {
                setSession(prev => ({
                  ...prev,
                  phase2_status: 'completed',
                  phase3_status: 'unlocked',
                }));
              } else if (alert.status === 'failed') {
                setSession(prev => ({
                  ...prev,
                  phase2_status: 'failed',
                }));
              }
            } else if (alert.phase === 'reporting') {
              if (alert.status === 'started') {
                setSession(prev => ({
                  ...prev,
                  phase3_status: 'generating',
                }));
              } else if (alert.status === 'completed') {
                setSession(prev => ({
                  ...prev,
                  phase3_status: 'draft_ready',
                }));
              }
            }
            return;
          }

          // Incoming DICOM CT Slice
          if (alert.slice_id !== undefined) {
            setSlices(prev => {
              // Deduplicate by slice_id or uid
              const existingIdx = prev.findIndex(
                s => (s.uid && alert.uid && s.uid === alert.uid) || s.slice_id === alert.slice_id
              );
              if (existingIdx !== -1) {
                const updated = [...prev];
                updated[existingIdx] = { ...updated[existingIdx], ...alert };
                return updated;
              }
              const updated = [...prev, { ...alert, _isNew: true }];
              updated.sort((a, b) => a.slice_id - b.slice_id);
              return updated;
            });

            // Update patient metadata from slice if available
            setSession(prev => {
              const pId = alert.patient_id || prev.patient_id;
              const pName = alert.patient_name || prev.patient_name;
              const pAge = alert.patient_age || prev.patient_age;
              const pSex = alert.patient_sex || prev.patient_sex;
              const sUid = alert.study_id || alert.study_instance_uid || prev.study_id;
              const serUid = alert.series_instance_uid || prev.series_id;
              return {
                ...prev,
                patient_id: pId,
                patient_name: pName,
                patient_age: pAge,
                patient_sex: pSex,
                study_id: sUid,
                series_id: serUid,
                session_active: true,
                phase1_status: prev.phase1_status === 'completed' ? 'completed' : 'scanning',
              };
            });
          }
        } catch (e) {
          console.error('[PRISM Context] WebSocket parse error:', e);
        }
      };

      ws.onclose = () => {
        setConnectionStatus('disconnected');
        wsRef.current = null;
        scheduleReconnect();
      };

      ws.onerror = () => {
        if (wsRef.current) wsRef.current.close();
      };
    } catch (err) {
      setConnectionStatus('disconnected');
      scheduleReconnect();
    }
  }, []);

  const scheduleReconnect = useCallback(() => {
    const delay = RECONNECT_DELAYS[Math.min(reconnectAttempt.current, RECONNECT_DELAYS.length - 1)];
    reconnectAttempt.current++;
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    reconnectTimer.current = setTimeout(connect, delay);
  }, [connect]);

  useEffect(() => {
    connect();
    // Initial fetch of current session info from REST API
    prismApi.getCurrentSession().then(s => {
      if (s) {
        setSession(prev => ({
          ...prev,
          patient_id: s.patient_id || prev.patient_id,
          patient_name: s.patient_name || prev.patient_name,
          patient_age: s.patient_age || prev.patient_age,
          patient_sex: s.patient_sex || prev.patient_sex,
          dataset_dir: s.dataset_dir || prev.dataset_dir,
          delay_ms: s.delay_ms || prev.delay_ms,
          study_id: s.study_id || prev.study_id,
          series_id: s.series_id || prev.series_id,
          session_active: s.session_active,
          phase1_status: s.phase1_status || prev.phase1_status,
          phase2_status: s.phase2_status || prev.phase2_status,
          phase3_status: s.phase3_status || prev.phase3_status,
          scanner_running: s.scanner_running,
        }));
      }
    }).catch(() => {});

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [connect]);

  // Periodic polling for scanner status to sync process state
  useEffect(() => {
    const poll = async () => {
      try {
        const res = await prismApi.getScannerStatus();
        if (res) {
          setSession(prev => {
            const running = res.running;
            // Only update scanner_running flag, don't revert completed status
            return {
              ...prev,
              scanner_running: running,
              phase1_status: running ? 'scanning' : (prev.phase1_status === 'scanning' ? 'completed' : prev.phase1_status),
              phase2_status: res.phase2_status as any || prev.phase2_status,
              phase3_status: res.phase3_status as any || prev.phase3_status,
            };
          });
        }
      } catch (e) {}
    };

    poll();
    const interval = setInterval(poll, 4000);
    return () => clearInterval(interval);
  }, []);

  // ── Session Management Actions ─────────────────────────────────────────────
  const startSession = useCallback(async (config: StartSessionRequest) => {
    try {
      setSlices([]);
      setSelectedIdx(0);
      setPipelineEvent(null);
      resetDraftReport();

      const res = await prismApi.startPatientSession(config);
      if (res.status === 'success') {
        setSession({
          ...res.session,
          scanner_running: res.scan_started,
          phase1_status: res.scan_started ? 'scanning' : 'idle',
        });
        return { status: 'success', message: res.message || 'Session started.' };
      } else {
        return { status: 'error', message: res.message || 'Failed to start session.' };
      }
    } catch (err: any) {
      return { status: 'error', message: err.message || 'Network error starting session.' };
    }
  }, [resetDraftReport]);

  const stopScan = useCallback(async () => {
    try {
      const res = await prismApi.stopScanner();
      setSession(prev => ({
        ...prev,
        scanner_running: false,
        phase1_status: 'completed',
      }));
      return { status: 'success', message: res.message || 'Scan stopped.' };
    } catch (err: any) {
      return { status: 'error', message: err.message || 'Failed to stop scanner.' };
    }
  }, []);

  const endSession = useCallback(async () => {
    try {
      const res = await prismApi.endPatientSession();
      setSlices([]);
      setSelectedIdx(0);
      setPipelineEvent(null);
      resetDraftReport();
      setSession(defaultSession);
      return {
        status: 'success',
        message: res.message || 'Patient session archived and ended.',
        archive_file: res.archive_file,
      };
    } catch (err: any) {
      return { status: 'error', message: err.message || 'Failed to end session.' };
    }
  }, [resetDraftReport]);

  const triggerPhase2 = useCallback(async () => {
    try {
      const res = await prismApi.triggerPhase2();
      if (res.status === 'success') {
        setSession(prev => ({ ...prev, phase2_status: 'running' }));
        return { status: 'success', message: 'Phase 2 analysis initiated.' };
      }
      return { status: 'error', message: res.message || 'Phase 2 could not be triggered.' };
    } catch (err: any) {
      return { status: 'error', message: err.message || 'Failed to trigger Phase 2.' };
    }
  }, []);

  return (
    <PatientSessionContext.Provider
      value={{
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
      }}
    >
      {children}
    </PatientSessionContext.Provider>
  );
};

export function usePatientSession(): PatientSessionContextType {
  const ctx = useContext(PatientSessionContext);
  if (!ctx) {
    throw new Error('usePatientSession must be used within a PatientSessionProvider');
  }
  return ctx;
}
