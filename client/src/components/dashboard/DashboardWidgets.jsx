import { Search, ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

export function DashboardToolbar({
  tabs,
  activeFilter,
  onFilterChange,
  searchQuery,
  onSearchChange,
  searchPlaceholder = 'Search...',
  sortLabel = 'Sort by',
}) {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="inline-flex w-fit items-center rounded-xl border border-slate-200/90 bg-[#FAFAFA] p-1">
        {tabs.map((tab) => {
          const isActive = activeFilter === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => onFilterChange(tab.id)}
              className={cn(
                'rounded-lg px-3.5 py-1.5 text-sm font-medium transition-all cursor-pointer',
                isActive
                  ? 'bg-white text-slate-900 shadow-xs'
                  : 'text-slate-500 hover:text-slate-900'
              )}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      <div className="flex items-center gap-2.5">
        <div className="relative flex-1 sm:w-56">
          <Search className="pointer-events-none absolute top-1/2 left-3 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder={searchPlaceholder}
            className="h-9 w-full rounded-xl border border-slate-200 bg-white pr-3 pl-9 text-sm text-slate-900 placeholder:text-slate-400 focus:border-slate-900 focus:outline-none"
          />
        </div>

        <button
          type="button"
          className="inline-flex h-9 items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 text-sm font-medium text-slate-600 hover:bg-slate-50 cursor-pointer"
        >
          <span>{sortLabel}</span>
          <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
        </button>
      </div>
    </div>
  );
}

export function DashboardCard({ title, description, badge, preview, onClick }) {
  return (
    <div
      onClick={onClick}
      className={cn(
        'group flex flex-col rounded-2xl border border-slate-200/90 bg-white p-4 transition-all hover:border-slate-300 hover:shadow-xs',
        onClick && 'cursor-pointer'
      )}
    >
      <div className="flex h-36 flex-col justify-between rounded-xl border border-slate-200/70 bg-[#FAFAFA] p-3.5">
        {preview}
      </div>

      <div className="mt-3.5 flex items-start justify-between gap-2">
        <h3 className="text-base font-semibold text-slate-900">{title}</h3>
        {badge}
      </div>
      <p className="mt-1 text-sm leading-relaxed text-slate-500">{description}</p>
    </div>
  );
}
