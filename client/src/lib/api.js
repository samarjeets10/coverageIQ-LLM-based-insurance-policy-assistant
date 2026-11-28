const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export const DEFAULT_MAX_PDF_SIZE_BYTES = Number(
  import.meta.env.VITE_MAX_PDF_SIZE_BYTES || 25 * 1024 * 1024
);

function parseApiError(statusCode, payload, headers = null) {
  const errorCode =
    Array.isArray(payload?.errors) && payload.errors[0]?.code
      ? payload.errors[0].code
      : null;

  const retryHeader =
    headers && typeof headers.get === 'function'
      ? Number(headers.get('Retry-After'))
      : Number(headers?.retryAfter);
  const retryAfterSeconds = Number.isFinite(retryHeader) && retryHeader > 0
    ? retryHeader
    : payload?.errors?.[0]?.retryAfterSeconds || null;

  const fallbackByStatus = {
    401: 'Your session has expired. Please sign in again.',
    403: 'You do not have permission to access or modify this policy.',
    409: 'This policy is archived and cannot accept new documents.',
    413: 'The uploaded PDF exceeds the maximum allowed size limit.',
    415: 'Only application/pdf files are accepted.',
    422: 'The PDF could not be processed because it is encrypted or corrupted.',
    429: retryAfterSeconds
      ? `Upload rate limit reached. Retry after ${retryAfterSeconds} seconds.`
      : 'Too many upload requests. Please wait a moment and try again.',
  };

  const message =
    payload?.message || fallbackByStatus[statusCode] || 'Request failed';

  const error = new Error(message);
  error.statusCode = statusCode;
  error.code = errorCode;
  error.retryAfterSeconds = retryAfterSeconds;
  error.data = payload || {};
  return error;
}

async function tryRefreshAccessToken() {
  const refreshed = await fetch(`${API_BASE}/users/refresh-token`, {
    method: 'POST',
    credentials: 'include',
  });
  return refreshed.ok;
}

async function request(path, options = {}, allowRefresh = true) {
  const isFormData =
    typeof FormData !== 'undefined' && options.body instanceof FormData;

  const headers = {
    ...(options.body && !isFormData ? { 'Content-Type': 'application/json' } : {}),
    ...options.headers,
  };

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: 'include',
  });

  if (
    response.status === 401 &&
    allowRefresh &&
    !path.includes('/users/refresh-token')
  ) {
    const refreshedOk = await tryRefreshAccessToken();
    if (refreshedOk) {
      return request(path, options, false);
    }
  }

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw parseApiError(response.status, payload, response.headers);
  }

  return payload;
}

function uploadFormDataWithProgress(
  path,
  formData,
  { onProgress, signal } = {},
  allowRefresh = true
) {
  if (typeof XMLHttpRequest === 'undefined') {
    if (typeof onProgress === 'function') onProgress(25);
    return request(
      path,
      { method: 'POST', body: formData, signal },
      allowRefresh
    ).then((res) => {
      if (typeof onProgress === 'function') onProgress(100);
      return res;
    });
  }

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}${path}`);
    xhr.withCredentials = true;

    const handleAbortSignal = () => {
      xhr.abort();
    };
    if (signal) {
      if (signal.aborted) {
        reject(new DOMException('Upload aborted', 'AbortError'));
        return;
      }
      signal.addEventListener('abort', handleAbortSignal, { once: true });
    }

    const cleanup = () => {
      if (signal) {
        signal.removeEventListener('abort', handleAbortSignal);
      }
    };

    xhr.upload.onprogress = (event) => {
      if (typeof onProgress === 'function' && event.lengthComputable && event.total > 0) {
        const pct = Math.min(100, Math.max(0, Math.round((event.loaded / event.total) * 100)));
        onProgress(pct);
      }
    };

    xhr.onerror = () => {
      cleanup();
      reject(new Error('Network error while uploading PDF'));
    };

    xhr.onabort = () => {
      cleanup();
      reject(new DOMException('Upload aborted', 'AbortError'));
    };

    xhr.onload = async () => {
      cleanup();
      let payload;
      try {
        payload = xhr.responseText ? JSON.parse(xhr.responseText) : {};
      } catch {
        payload = {};
      }

      if (xhr.status === 401 && allowRefresh) {
        try {
          const refreshedOk = await tryRefreshAccessToken();
          if (refreshedOk) {
            const retried = await uploadFormDataWithProgress(
              path,
              formData,
              { onProgress, signal },
              false
            );
            resolve(retried);
            return;
          }
        } catch {
          // Fall through to 401 error
        }
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        if (typeof onProgress === 'function') onProgress(100);
        resolve(payload);
        return;
      }

      const retryHeader = xhr.getResponseHeader('Retry-After');
      reject(parseApiError(xhr.status, payload, { retryAfter: retryHeader }));
    };

    // Never set Content-Type manually so browser adds multipart boundary automatically
    xhr.send(formData);
  });
}

export const authApi = {
  register: (data) =>
    request(
      '/users/register',
      {
        method: 'POST',
        body: JSON.stringify(data),
      },
      false
    ),

  login: (data) =>
    request(
      '/users/login',
      {
        method: 'POST',
        body: JSON.stringify(data),
      },
      false
    ),

  logout: () =>
    request(
      '/users/logout',
      {
        method: 'POST',
      },
      false
    ),

  getCurrentUser: () => request('/users/me'),
};

export const policiesApi = {
  list: ({ writable = false, status = '' } = {}) => {
    const params = new URLSearchParams();
    if (writable) params.set('writable', 'true');
    if (status) params.set('status', status);
    const qs = params.toString();
    return request(`/policies${qs ? `?${qs}` : ''}`);
  },

  create: (data) =>
    request('/policies', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  updateVisibility: (policyId, visibility) =>
    request(`/policies/${encodeURIComponent(policyId)}/visibility`, {
      method: 'PATCH',
      body: JSON.stringify({ visibility }),
    }),
};

export const ingestionApi = {
  uploadDocument: (policyId, { file, versionLabel = '', onProgress, signal } = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    if (versionLabel && versionLabel.trim()) {
      formData.append('version_label', versionLabel.trim());
    }
    return uploadFormDataWithProgress(
      `/policies/${encodeURIComponent(policyId)}/documents`,
      formData,
      { onProgress, signal }
    );
  },

  getJob: (jobId) => request(`/ingestion/jobs/${encodeURIComponent(jobId)}`),

  listJobs: ({ page = 1, limit = 10, status = '', policyId = '' } = {}) => {
    const params = new URLSearchParams();
    params.set('page', String(page));
    params.set('limit', String(limit));
    if (status) params.set('status', status);
    if (policyId) params.set('policyId', policyId);
    return request(`/ingestion/jobs?${params.toString()}`);
  },
};

export const qaApi = {
  ask: (policyId, { question, conversationId = '' } = {}) =>
    request(`/policies/${encodeURIComponent(policyId)}/ask`, {
      method: 'POST',
      body: JSON.stringify({
        question,
        ...(conversationId ? { conversation_id: conversationId } : {}),
      }),
    }),

  getHistory: (policyId, { page = 1, limit = 40, conversationId = '' } = {}) => {
    const params = new URLSearchParams();
    params.set('page', String(page));
    params.set('limit', String(limit));
    if (conversationId) params.set('conversationId', conversationId);
    return request(`/policies/${encodeURIComponent(policyId)}/qa-history?${params.toString()}`);
  },
};

export const recommendationsApi = {
  recommend: (payload) =>
    request('/recommendations', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
};
