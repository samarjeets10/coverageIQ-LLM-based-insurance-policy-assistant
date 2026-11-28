import { Link } from 'react-router-dom';
import { FileUp, LogOut } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useAuth } from '@/context/useAuth';
import { getDashboardPath } from '@/lib/roles';

export function Navbar() {
  const { user, loading, logout } = useAuth();
  const dashboardPath = user ? getDashboardPath(user.platform_role) : '/dashboard';

  return (
    <header className="sticky top-0 z-40 border-b border-slate-200/80 bg-[#FAFAFA]/90 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
        <Link to="/" className="flex items-center gap-2.5 text-slate-900">
          <span className="text-xl font-bold tracking-tight">CoverageIQ</span>
        </Link>

        <nav className="hidden items-center gap-7 text-sm text-slate-600 md:flex">
          <a href="/#capabilities" className="transition-colors hover:text-slate-900">
            Features
          </a>
          <a href="/#process" className="transition-colors hover:text-slate-900">
            How It Works
          </a>
          <a href="/#roles" className="transition-colors hover:text-slate-900">
            Roles
          </a>
          <a href="/#faq" className="transition-colors hover:text-slate-900">
            FAQ
          </a>
          {!loading && user && (
            <Link
              to={dashboardPath}
              className="font-medium text-blue-600 transition-colors hover:text-blue-700"
            >
              Dashboard
            </Link>
          )}
        </nav>

        <div className="flex items-center gap-2.5">
          {!loading && user ? (
            <>
              <Link to="/upload">
                <Button variant="secondary" size="sm">
                  <FileUp className="h-4 w-4" />
                  Upload & Jobs
                </Button>
              </Link>
              <Link to={dashboardPath}>
                <Button size="sm">Dashboard</Button>
              </Link>
              <div className="hidden items-center gap-2 sm:flex">
                <span className="text-base font-medium text-slate-700">
                  {user.full_name}
                </span>
                <Badge variant="accent" className="capitalize">
                  {user.platform_role}
                </Badge>
              </div>
              <Button variant="secondary" size="sm" onClick={logout}>
                <LogOut className="h-4 w-4" />
                Sign out
              </Button>
            </>
          ) : (
            <>
              <Link to="/login">
                <Button variant="ghost" size="sm">
                  Login
                </Button>
              </Link>
              <Link to="/register">
                <Button size="sm">Sign up</Button>
              </Link>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
