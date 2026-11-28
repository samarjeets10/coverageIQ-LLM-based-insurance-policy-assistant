import React from 'react';
import { cn } from '@/lib/utils';

const VARIANTS = {
  default:
    'bg-slate-900 text-white shadow-sm hover:bg-slate-800 active:scale-[0.99]',
  secondary:
    'bg-white text-slate-900 border border-slate-200 hover:bg-slate-50',
  ghost:
    'text-slate-600 hover:text-slate-900 hover:bg-slate-100/80',
  light:
    'bg-white text-slate-900 shadow-sm hover:bg-slate-100',
};

const SIZES = {
  sm: 'h-9 px-4 text-sm',
  md: 'h-10 px-5 text-base',
  lg: 'h-11 px-6 text-base',
};

export const Button = React.forwardRef(function Button(
  { className, variant = 'default', size = 'md', type = 'button', ...props },
  ref
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-full font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-600/30 disabled:pointer-events-none disabled:opacity-50 cursor-pointer',
        VARIANTS[variant],
        SIZES[size],
        className
      )}
      {...props}
    />
  );
});
