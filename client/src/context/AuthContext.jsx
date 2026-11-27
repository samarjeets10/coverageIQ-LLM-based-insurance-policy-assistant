import { useEffect, useState } from 'react';
import { authApi } from '@/lib/api';
import { AuthContext } from '@/context/useAuth';

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    authApi
      .getCurrentUser()
      .then((res) => {
        if (active) setUser(res.data?.user || null);
      })
      .catch(() => {
        if (active) setUser(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  async function login(credentials) {
    const res = await authApi.login(credentials);
    setUser(res.data?.user || null);
    return res.data?.user;
  }

  async function register(payload) {
    const res = await authApi.register(payload);
    setUser(res.data?.user || null);
    return res.data?.user;
  }

  async function logout() {
    try {
      await authApi.logout();
    } finally {
      setUser(null);
    }
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}
