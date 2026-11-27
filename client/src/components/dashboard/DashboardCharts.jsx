import { cn } from '@/lib/utils';

export function KpiCard({ label, value, hint, icon: Icon }) {
  return (
    <div className="rounded-2xl border border-slate-200/90 bg-white p-5">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-slate-500">{label}</span>
        {Icon && (
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100 text-blue-600">
            <Icon className="h-4 w-4" />
          </span>
        )}
      </div>
      <p className="mt-3 text-2xl font-semibold tracking-tight text-slate-900">
        {value}
      </p>
      {hint && <p className="mt-1 text-xs text-slate-500">{hint}</p>}
    </div>
  );
}

export function HorizontalBarChart({ title, subtitle, items }) {
  const max = Math.max(...items.map((item) => item.value), 1);

  return (
    <div className="rounded-2xl border border-slate-200/90 bg-white p-6">
      <div className="mb-5">
        <h3 className="text-base font-semibold text-slate-900">{title}</h3>
        {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
      </div>

      <div className="space-y-4">
        {items.map((item) => {
          const widthPct = Math.round((item.value / max) * 100);
          return (
            <div key={item.label} className="space-y-1.5">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium text-slate-700">{item.label}</span>
                <span className="text-slate-500">{item.display ?? item.value}</span>
              </div>
              <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                <div
                  className={cn(
                    'h-full rounded-full transition-all duration-300',
                    item.color || 'bg-slate-900'
                  )}
                  style={{ width: `${widthPct}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function DonutSummary({ title, subtitle, centerValue, centerLabel, segments }) {
  const total = segments.reduce((sum, s) => sum + s.value, 0) || 1;
  const radius = 44;
  const circumference = 2 * Math.PI * radius;

  const arcs = segments.reduce((acc, seg) => {
    const prevRatio = acc.length > 0 ? acc[acc.length - 1].endRatio : 0;
    const ratio = seg.value / total;
    acc.push({
      ...seg,
      ratio,
      startRatio: prevRatio,
      endRatio: prevRatio + ratio,
    });
    return acc;
  }, []);

  return (
    <div className="rounded-2xl border border-slate-200/90 bg-white p-6">
      <div className="mb-5">
        <h3 className="text-base font-semibold text-slate-900">{title}</h3>
        {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
      </div>

      <div className="flex flex-col items-center gap-6 sm:flex-row">
        <div className="relative flex h-32 w-32 shrink-0 items-center justify-center">
          <svg className="h-32 w-32 -rotate-90" viewBox="0 0 110 110">
            <circle
              cx="55"
              cy="55"
              r={radius}
              fill="none"
              stroke="#F1F5F9"
              strokeWidth="12"
            />
            {arcs.map((seg) => {
              const dashLength = seg.ratio * circumference;
              const dashOffset = -seg.startRatio * circumference;
              return (
                <circle
                  key={seg.label}
                  cx="55"
                  cy="55"
                  r={radius}
                  fill="none"
                  stroke={seg.stroke}
                  strokeWidth="12"
                  strokeDasharray={`${dashLength} ${circumference}`}
                  strokeDashoffset={dashOffset}
                  strokeLinecap="butt"
                />
              );
            })}
          </svg>
          <div className="absolute text-center">
            <p className="text-lg font-semibold text-slate-900">{centerValue}</p>
            <p className="text-xs text-slate-500">{centerLabel}</p>
          </div>
        </div>

        <div className="w-full space-y-2.5">
          {segments.map((seg) => (
            <div
              key={seg.label}
              className="flex items-center justify-between text-sm"
            >
              <span className="flex items-center gap-2 text-slate-600">
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: seg.stroke }}
                />
                {seg.label}
              </span>
              <span className="font-medium text-slate-900">{seg.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
