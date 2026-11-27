import { Link } from 'react-router-dom';

export function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl flex-col gap-10 px-4 py-12 sm:px-6 md:flex-row md:justify-between">
        <div className="max-w-xs space-y-3">
          <Link to="/" className="inline-block text-base font-semibold tracking-tight text-slate-900">
            CoverageIQ
          </Link>
          <p className="text-sm leading-relaxed text-slate-500">
            Clear answers from your insurance policies, backed by the exact page
            and clause.
          </p>
        </div>

        <div className="flex gap-14 text-base">
          <div className="space-y-2.5">
            <p className="text-sm font-semibold uppercase tracking-wider text-slate-900">
              Platform
            </p>
            <ul className="space-y-2 text-sm text-slate-500">
              <li>
                <a href="#capabilities" className="hover:text-slate-900">
                  Features
                </a>
              </li>
              <li>
                <a href="#process" className="hover:text-slate-900">
                  How It Works
                </a>
              </li>
              <li>
                <a href="#roles" className="hover:text-slate-900">
                  Roles
                </a>
              </li>
              <li>
                <a href="#faq" className="hover:text-slate-900">
                  FAQ
                </a>
              </li>
            </ul>
          </div>

          <div className="space-y-2.5">
            <p className="text-sm font-semibold uppercase tracking-wider text-slate-900">
              Account
            </p>
            <ul className="space-y-2 text-sm text-slate-500">
              <li>
                <Link to="/login" className="hover:text-slate-900">
                  Login
                </Link>
              </li>
              <li>
                <Link to="/register?role=user" className="hover:text-slate-900">
                  User Sign up
                </Link>
              </li>
              <li>
                <Link to="/register?role=organisation" className="hover:text-slate-900">
                  Organisation Sign up
                </Link>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </footer>
  );
}
