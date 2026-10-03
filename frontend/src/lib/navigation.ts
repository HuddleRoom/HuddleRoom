const normalizePath = (path: string) => {
  if (path.length > 1 && path.endsWith('/')) {
    return path.replace(/\/+$/, '')
  }
  return path || '/'
}

export const DASHBOARD_BASENAME = '/dashboard'

export function dashboardPath(path: string) {
  const normalizedPath = normalizePath(path)
  if (normalizedPath === '/') {
    return DASHBOARD_BASENAME
  }
  return `${DASHBOARD_BASENAME}${normalizedPath}`
}

export function shouldNavigate(currentPath: string, targetPath: string) {
  return normalizePath(currentPath) !== normalizePath(targetPath)
}
