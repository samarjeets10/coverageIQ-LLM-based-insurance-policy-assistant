import { ArrowRight } from 'lucide-react';
import { cn } from '@/lib/utils';

export function DashboardCard({ title, description, icon: Icon, actionLabel, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'group flex flex-col justify-between rounded-2xl border border-slate-200/90 bg-white p-6 text-left transition-all hover:border-slate-300 hover:shadow-xs cursor-pointer'
      )}
    >
      <div>
        {Icon && (
          <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-blue-100 bg-blue-50/70 text-blue-600">
            <Icon className="h-5 w-5" />
          </span>
        )}
        <h3 className="mt-4 text-base font-semibold text-slate-900">{title}</h3>
        <p className="mt-1.5 text-sm leading-relaxed text-slate-500">
          {description}
        </p>
      </div>

      {actionLabel && (
        <span className="mt-5 inline-flex items-center gap-1.5 text-sm font-medium text-slate-900 group-hover:text-blue-600">
          {actionLabel}
          <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
        </span>
      )}
    </button>
  );
}
