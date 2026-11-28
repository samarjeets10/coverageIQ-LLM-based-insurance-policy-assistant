export const ROLES = Object.freeze({
  USER: 'user',
  ORGANISATION: 'organisation',
  ADMIN: 'admin',
});

export function getDashboardPath(role) {
  switch (role) {
    case ROLES.ADMIN:
      return '/admin';
    case ROLES.ORGANISATION:
      return '/organisation';
    case ROLES.USER:
    default:
      return '/dashboard';
  }
}
