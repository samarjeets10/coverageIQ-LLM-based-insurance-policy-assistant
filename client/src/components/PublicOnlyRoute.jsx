import { Navigate, Outlet } from 'react-router-dom';
import { useAuth } from '@/context/useAuth';
import { getDashboardPath } from '@/lib/roles';

export function PublicOnlyRoute() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#FAFAFA] text-base text-slate-500">
        Checking session...
      </div>
    );
  }

  if (user) {
    return <Navigate to={getDashboardPath(user.platform_role)} replace />;
  }

  return <Outlet />;
}
