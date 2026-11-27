const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api/v1';

async function request(path, options = {}, allowRefresh = true) {
  const headers = {
    ...(options.body ? { 'Content-Type': 'application/json' } : {}),
    ...options.headers,
  };

  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: 'include',
  });

  if (response.status === 401 && allowRefresh && !path.includes('/users/refresh-token')) {
    const refreshed = await fetch(`${API_BASE}/users/refresh-token`, {
      method: 'POST',
      credentials: 'include',
    });

    if (refreshed.ok) {
      return request(path, options, false);
    }
  }

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    const error = new Error(payload.message || 'Request failed');
    error.statusCode = response.status;
    error.data = payload;
    throw error;
  }

  return payload;
}

export const authApi = {
  register: (data) =>
    request('/users/register', {
      method: 'POST',
      body: JSON.stringify(data),
    }, false),

  login: (data) =>
    request('/users/login', {
      method: 'POST',
      body: JSON.stringify(data),
    }, false),

  logout: () =>
    request('/users/logout', {
      method: 'POST',
    }, false),

  getCurrentUser: () => request('/users/me'),
};
