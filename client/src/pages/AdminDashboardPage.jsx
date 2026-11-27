import { Navbar } from '@/components/Navbar';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { useAuth } from '@/context/useAuth';

export function AdminDashboardPage() {
  const { user } = useAuth();

  return (
    <div className="min-h-screen bg-[#FAFAFA] text-slate-900">
      <Navbar />
      <main className="mx-auto max-w-5xl px-4 py-12 sm:px-6">
        <Card className="p-7">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
                Admin Workspace
              </h1>
              <p className="mt-1 text-sm text-slate-500">
                Signed in as {user?.full_name} ({user?.email})
              </p>
            </div>
            <Badge variant="accent" className="capitalize">
              {user?.platform_role}
            </Badge>
          </div>
          <p className="mt-6 text-sm text-slate-600">
            Protected admin route is active. Platform role management and
            system oversight will live here.
          </p>
        </Card>
      </main>
    </div>
  );
}
