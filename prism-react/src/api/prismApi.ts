/**
 * PRISM API Client — TypeScript port of prism-api.js
 * Proxies all requests through Django's /api/* endpoints
 */

export interface Report {
  study_id: string;
  status: 'draft' | 'signed';
  source: string;
  findings: string;
  impression: string;
  validated: boolean;
  validation_reason?: string;
  created_at: string;
  updated_at: string;
  patient_id?: string;
  patient_name?: string;
  patient_age?: string;
  patient_sex?: string;
}

export interface ReportsListResponse {
  total: number;
  reports: Report[];
  error?: string;
}

export interface HealthResponse {
  status: string;
  ollama_reachable: boolean;
  ollama_model?: string;
  error?: string;
}

export interface GenerateResponse {
  study_id: string;
  patient_id?: string;
  patient_name?: string;
  patient_age?: string;
  patient_sex?: string;
  status: string;
  findings: string;
  impression: string;
  source: string;
  validated: boolean;
}

function getCsrfToken(): string {
  const cookie = document.cookie
    .split(';')
    .map((c) => c.trim())
    .find((c) => c.startsWith('csrftoken='));
  return cookie ? decodeURIComponent(cookie.split('=')[1]) : '';
}

async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  const method = (options.method || 'GET').toUpperCase();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string> || {}),
  };

  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {
    headers['X-CSRFToken'] = getCsrfToken();
  }

  const response = await fetch(url, { ...options, headers });
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    data = {};
  }

  if (!response.ok) {
    const errData = data as Record<string, unknown>;
    const message =
      typeof errData?.detail === 'string'
        ? errData.detail
        : typeof errData?.error === 'string'
        ? errData.error
        : `Request failed (${response.status})`;
    throw new Error(message);
  }

  return data as T;
}

export interface PatientSessionInfo {
  patient_id: string;
  patient_name: string;
  patient_age: string;
  patient_sex: string;
  dataset_dir: string;
  delay_ms: number;
  study_id: string;
  series_id: string;
  session_active: boolean;
  phase1_status: 'idle' | 'scanning' | 'completed';
  phase2_status: 'idle' | 'running' | 'completed' | 'failed';
  phase3_status: 'locked' | 'unlocked' | 'generating' | 'draft_ready' | 'signed';
  total_slices: number;
  flagged_slices: number;
  start_time: string;
  phase2_info?: Record<string, unknown>;
  scanner_running: boolean;
}

export interface StartSessionRequest {
  patient_id: string;
  patient_name: string;
  patient_age: string;
  patient_sex: string;
  dataset_dir: string;
  delay_ms: number;
  auto_start_scan?: boolean;
}

export const prismApi = {
  health(): Promise<HealthResponse> {
    return request<HealthResponse>('/api/v1/health');
  },

  listReports(): Promise<ReportsListResponse> {
    return request<ReportsListResponse>('/api/v1/reports');
  },

  getReport(studyId: string): Promise<Report> {
    return request<Report>(`/api/v1/report/${encodeURIComponent(studyId)}`);
  },

  generateReport(findings: Record<string, unknown>): Promise<GenerateResponse> {
    return request<GenerateResponse>('/api/v1/generate-report', {
      method: 'POST',
      body: JSON.stringify(findings),
    });
  },

  updateReport(
    studyId: string,
    payload: { findings?: string; impression?: string }
  ): Promise<Report> {
    return request<Report>(`/api/v1/report/${encodeURIComponent(studyId)}`, {
      method: 'PUT',
      body: JSON.stringify(payload),
    });
  },

  signReport(studyId: string): Promise<{ status: string }> {
    return request<{ status: string }>('/api/v1/sign-report', {
      method: 'POST',
      body: JSON.stringify({ study_id: studyId }),
    });
  },

  loadSampleCase(caseId: string): Promise<Record<string, unknown>> {
    return request<Record<string, unknown>>(`/api/sample/${encodeURIComponent(caseId)}/`);
  },

  // ── Patient Session Endpoints ──
  getCurrentSession(): Promise<PatientSessionInfo> {
    return request<PatientSessionInfo>('/api/session/current');
  },

  startPatientSession(req: StartSessionRequest): Promise<{ status: string; session: PatientSessionInfo; scan_started: boolean; message: string }> {
    return request<{ status: string; session: PatientSessionInfo; scan_started: boolean; message: string }>('/api/session/start', {
      method: 'POST',
      body: JSON.stringify(req),
    });
  },

  endPatientSession(): Promise<{ status: string; message: string; archive_file?: string; archive_path?: string }> {
    return request<{ status: string; message: string; archive_file?: string; archive_path?: string }>('/api/session/end', {
      method: 'POST',
    });
  },

  triggerPhase2(): Promise<{ status: string; message: string }> {
    return request<{ status: string; message: string }>('/api/session/phase2-trigger', {
      method: 'POST',
    });
  },

  stopScanner(): Promise<{ status: string; message: string; phase1_status: string }> {
    return request<{ status: string; message: string; phase1_status: string }>('/api/scanner/stop', {
      method: 'POST',
    });
  },

  browseDirectory(): Promise<{ status: string; path?: string }> {
    return request<{ status: string; path?: string }>('/api/scanner/browse');
  },

  getScannerStatus(): Promise<{ running: boolean; phase1_status: string; phase2_status: string; phase3_status: string; total_slices: number }> {
    return request<{ running: boolean; phase1_status: string; phase2_status: string; phase3_status: string; total_slices: number }>('/api/scanner/status');
  },
};
