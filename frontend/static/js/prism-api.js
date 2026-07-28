/**
 * PRISM Frontend API helpers (ES6)
 * Proxies requests through Django to the FastAPI backend.
 */

const PrismAPI = {
  getCsrfToken() {
    const cookie = document.cookie
      .split(';')
      .map((c) => c.trim())
      .find((c) => c.startsWith('csrftoken='));
    return cookie ? decodeURIComponent(cookie.split('=')[1]) : '';
  },

  async request(url, options = {}) {
    const headers = {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    };

    const method = (options.method || 'GET').toUpperCase();
    if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(method)) {
      headers['X-CSRFToken'] = this.getCsrfToken();
    }

    const response = await fetch(url, { ...options, headers });
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const message = data.detail || data.error || `Request failed (${response.status})`;
      throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
    }

    return data;
  },

  health() {
    return this.request('/api/health/');
  },

  listReports() {
    return this.request('/api/reports/');
  },

  getReport(studyId) {
    return this.request(`/api/report/${encodeURIComponent(studyId)}/`);
  },

  generateReport(findings) {
    return this.request('/api/generate/', {
      method: 'POST',
      body: JSON.stringify(findings),
    });
  },

  updateReport(studyId, { findings, impression } = {}) {
    return this.request(`/api/report/${encodeURIComponent(studyId)}/update/`, {
      method: 'PUT',
      body: JSON.stringify({ findings, impression }),
    });
  },

  signReport(studyId) {
    return this.request('/api/sign/', {
      method: 'POST',
      body: JSON.stringify({ study_id: studyId }),
    });
  },

  loadSampleCase(caseId) {
    return this.request(`/api/sample/${encodeURIComponent(caseId)}/`);
  },
};

window.PrismAPI = PrismAPI;

/**
 * Toast notification helper for Alpine.js components.
 */
const PrismToast = {
  show(store, message, type = 'success') {
    store.message = message;
    store.type = type;
    store.visible = true;
    clearTimeout(store._timer);
    store._timer = setTimeout(() => {
      store.visible = false;
    }, 4500);
  },
};

window.PrismToast = PrismToast;
