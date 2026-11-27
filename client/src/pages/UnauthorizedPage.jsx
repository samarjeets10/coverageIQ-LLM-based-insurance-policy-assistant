import { Link } from 'react-router-dom';
import { ShieldAlert } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { useAuth } from '@/context/useAuth';
import { getDashboardPath } from '@/lib/roles';

export function UnauthorizedPage() {
  const { user } = useAuth();
  const homePath = user ? getDashboardPath(user.platform_role) : '/';

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#FAFAFA] bg-subtle-grid px-4">
      <Card className="w-full max-w-md p-8 text-center">
        <span className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl bg-slate-100 text-slate-900">
          <ShieldAlert className="h-5 w-5 text-blue-600" />
        </span>
        <h1 className="mt-4 text-2xl font-semibold tracking-tight text-slate-900">
          Access Restricted
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-500">
          Your account role ({user?.platform_role || 'guest'}) does not have
          permission to view this page.
        </p>
        <div className="mt-6 flex justify-center gap-3">
          <Link to={homePath}>
            <Button>Go to my workspace</Button>
          </Link>
          <Link to="/">
            <Button variant="secondary">Home</Button>
          </Link>
        </div>
      </Card>
    </div>
  );
}
