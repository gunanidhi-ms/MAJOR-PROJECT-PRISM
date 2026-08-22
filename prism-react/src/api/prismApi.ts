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
};
