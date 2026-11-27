import { useState } from 'react';
import {
  ChevronsLeft,
  ChevronsRight,
  LogOut,
  Menu,
  Search,
  Settings,
  HelpCircle,
  X,
} from 'lucide-react';
import { useAuth } from '@/context/useAuth';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export function DashboardLayout({
  navGroups,
  activeTab,
  onTabChange,
  pageTitle,
  primaryAction,
  children,
}) {
  const { user, logout } = useAuth();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const initials = (user?.full_name || 'U')
    .split(' ')
    .map((part) => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();

  const renderSidebarContent = () => (
    <div className="flex h-full flex-col justify-between bg-[#FAFAFA]">
      <div>
        {/* Workspace Header */}
        <div className="flex h-16 items-center justify-between border-b border-slate-200/80 px-4">
          {!collapsed && (
            <div className="flex items-center gap-2">
              <span className="text-base font-semibold tracking-tight text-slate-900">
                CoverageIQ
              </span>
              <Badge variant="accent" className="px-2 py-0.5 text-xs capitalize">
                {user?.platform_role}
              </Badge>
            </div>
          )}
          <button
            type="button"
            onClick={() => setCollapsed((prev) => !prev)}
            className="hidden h-8 w-8 items-center justify-center rounded-lg text-slate-500 hover:bg-slate-200/60 hover:text-slate-900 lg:inline-flex cursor-pointer"
            aria-label="Toggle sidebar"
          >
            {collapsed ? (
              <ChevronsRight className="h-4 w-4" />
            ) : (
              <ChevronsLeft className="h-4 w-4" />
            )}
          </button>
          <button
            type="button"
            onClick={() => setMobileOpen(false)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 hover:bg-slate-200/60 lg:hidden cursor-pointer"
            aria-label="Close menu"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Quick Search */}
        {!collapsed && (
          <div className="px-3 pt-4 pb-2">
            <div className="flex items-center justify-between rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-400">
              <span className="flex items-center gap-2">
                <Search className="h-4 w-4" />
                <span>Search...</span>
              </span>
              <kbd className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-medium text-slate-500">
                ⌘K
              </kbd>
            </div>
          </div>
        )}

        {/* Navigation Groups */}
        <nav className="space-y-5 px-3 py-3">
          {navGroups.map((group) => (
            <div key={group.label || 'main'} className="space-y-1">
              {group.label && !collapsed && (
                <p className="px-3 pb-1 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  {group.label}
                </p>
              )}
              {group.items.map((item) => {
                const Icon = item.icon;
                const isActive = activeTab === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => {
                      onTabChange(item.id);
                      setMobileOpen(false);
                    }}
                    className={cn(
                      'flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm transition-colors cursor-pointer',
                      isActive
                        ? 'bg-slate-200/75 font-medium text-slate-900'
                        : 'text-slate-600 hover:bg-slate-200/40 hover:text-slate-900',
                      collapsed && 'justify-center px-2'
                    )}
                    title={collapsed ? item.label : undefined}
                  >
                    <Icon
                      className={cn(
                        'h-4 w-4 shrink-0',
                        isActive ? 'text-blue-600' : 'text-slate-500'
                      )}
                    />
                    {!collapsed && <span className="truncate">{item.label}</span>}
                  </button>
                );
              })}
            </div>
          ))}
        </nav>
      </div>

      {/* Bottom Utility + User Profile */}
      <div className="border-t border-slate-200/80 p-3 space-y-2">
        <button
          type="button"
          onClick={() => onTabChange('settings')}
          className={cn(
            'flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm text-slate-600 hover:bg-slate-200/40 hover:text-slate-900 cursor-pointer',
            activeTab === 'settings' && 'bg-slate-200/75 font-medium text-slate-900',
            collapsed && 'justify-center px-2'
          )}
        >
          <Settings className="h-4 w-4 shrink-0 text-slate-500" />
          {!collapsed && <span>Settings</span>}
        </button>

        <button
          type="button"
          onClick={() => onTabChange('help')}
          className={cn(
            'flex w-full items-center gap-3 rounded-xl px-3 py-2 text-sm text-slate-600 hover:bg-slate-200/40 hover:text-slate-900 cursor-pointer',
            activeTab === 'help' && 'bg-slate-200/75 font-medium text-slate-900',
            collapsed && 'justify-center px-2'
          )}
        >
          <HelpCircle className="h-4 w-4 shrink-0 text-slate-500" />
          {!collapsed && <span>Help &amp; Docs</span>}
        </button>

        <div className="mt-2 flex items-center justify-between gap-2 rounded-xl border border-slate-200/80 bg-white p-2.5">
          <div className="flex min-w-0 items-center gap-2.5">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-slate-900 text-xs font-semibold text-white">
              {initials}
            </span>
            {!collapsed && (
              <div className="min-w-0">
                <p className="truncate text-sm font-medium text-slate-900">
                  {user?.full_name}
                </p>
                <p className="truncate text-xs text-slate-500">{user?.email}</p>
              </div>
            )}
          </div>
          {!collapsed && (
            <button
              type="button"
              onClick={logout}
              className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-500 hover:bg-slate-100 hover:text-slate-900 cursor-pointer"
              title="Sign out"
            >
              <LogOut className="h-4 w-4" />
            </button>
          )}
        </div>
      </div>
    </div>
  );

  return (
    <div className="flex min-h-screen bg-[#FAFAFA] text-slate-900">
      {/* Desktop Sidebar */}
      <aside
        className={cn(
          'hidden shrink-0 border-r border-slate-200/80 transition-all duration-200 lg:block',
          collapsed ? 'w-[72px]' : 'w-64'
        )}
      >
        <div className="sticky top-0 h-screen">{renderSidebarContent()}</div>
      </aside>

      {/* Mobile Sidebar Drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 flex lg:hidden">
          <div
            className="fixed inset-0 bg-slate-900/30 backdrop-blur-xs"
            onClick={() => setMobileOpen(false)}
          />
          <aside className="relative z-10 w-64 border-r border-slate-200 bg-[#FAFAFA]">
            {renderSidebarContent()}
          </aside>
        </div>
      )}

      {/* Main Content Area */}
      <div className="flex min-w-0 flex-1 flex-col bg-white">
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-slate-200/80 bg-white/95 px-4 backdrop-blur-md sm:px-8">
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => setMobileOpen(true)}
              className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 lg:hidden cursor-pointer"
              aria-label="Open menu"
            >
              <Menu className="h-4 w-4" />
            </button>
            <h1 className="text-lg font-semibold tracking-tight text-slate-900 sm:text-xl">
              {pageTitle}
            </h1>
          </div>

          {primaryAction && (
            <Button size="sm" onClick={primaryAction.onClick}>
              {primaryAction.icon && <primaryAction.icon className="h-4 w-4" />}
              {primaryAction.label}
            </Button>
          )}
        </header>

        <main className="flex-1 px-4 py-6 sm:px-8">{children}</main>
      </div>
    </div>
  );
}
