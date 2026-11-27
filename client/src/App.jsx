import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AuthProvider } from '@/context/AuthContext';
import { ProtectedRoute } from '@/components/ProtectedRoute';
import { PublicOnlyRoute } from '@/components/PublicOnlyRoute';
import { ROLES } from '@/lib/roles';
import { LandingPage } from '@/pages/LandingPage';
import { LoginPage } from '@/pages/LoginPage';
import { RegisterPage } from '@/pages/RegisterPage';
import { UnauthorizedPage } from '@/pages/UnauthorizedPage';
import { UserDashboardPage } from '@/pages/UserDashboardPage';
import { OrganisationDashboardPage } from '@/pages/OrganisationDashboardPage';
import { AdminDashboardPage } from '@/pages/AdminDashboardPage';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public */}
          <Route path="/" element={<LandingPage />} />

          {/* Public only (redirects to role dashboard if signed in) */}
          <Route element={<PublicOnlyRoute />}>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
          </Route>

          {/* Any authenticated user */}
          <Route element={<ProtectedRoute />}>
            <Route path="/unauthorized" element={<UnauthorizedPage />} />
          </Route>

          {/* RBAC-protected workspaces */}
          <Route element={<ProtectedRoute allowedRoles={[ROLES.USER]} />}>
            <Route path="/dashboard" element={<UserDashboardPage />} />
          </Route>

          <Route element={<ProtectedRoute allowedRoles={[ROLES.ORGANISATION]} />}>
            <Route path="/organisation" element={<OrganisationDashboardPage />} />
          </Route>

          <Route element={<ProtectedRoute allowedRoles={[ROLES.ADMIN]} />}>
            <Route path="/admin" element={<AdminDashboardPage />} />
          </Route>

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
