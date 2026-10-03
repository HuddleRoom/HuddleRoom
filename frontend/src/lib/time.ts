/**
 * Pure time formatting functions (no React imports).
 * Used for consistent timestamp display across the dashboard.
 */

/**
 * Relative time format: "3h ago", "just now", etc.
 * Falls back to absolute format for timestamps ≥24h old.
 * Gracefully handles invalid/empty input.
 */
export function relative(iso: string): string {
  if (!iso) return ''

  let date: Date
  try {
    date = new Date(iso)
    if (isNaN(date.getTime())) return iso
  } catch {
    return iso
  }

  const diff = Date.now() - date.getTime()
  const m = Math.floor(diff / 60000)

  if (m < 1) return 'just now'
  if (m < 60) return `${m}m ago`

  const h = Math.floor(m / 60)
  if (h < 24) return `${h}h ago`

  // ≥24h old, use absolute format
  return absolute(iso)
}

/**
 * Absolute time format: "Sep 16, 8:27 AM"
 * Uses browser's default locale via Intl.DateTimeFormat.
 */
export function absolute(iso: string | number | null | undefined): string {
  if (iso == null || iso === '') return ''

  let date: Date
  try {
    date = new Date(iso)
    if (isNaN(date.getTime())) return String(iso)
  } catch {
    return String(iso)
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(date)
}

/**
 * Full ISO passthrough: returns the raw input string.
 * Used for debug/raw output.
 */
export function full(iso: string): string {
  return iso
}

/**
 * Time-of-day format: "8:27 AM"
 * Time only, no date.
 */
export function timeOfDay(iso: string): string {
  if (!iso) return ''

  let date: Date
  try {
    date = new Date(iso)
    if (isNaN(date.getTime())) return iso
  } catch {
    return iso
  }

  return new Intl.DateTimeFormat(undefined, {
    timeStyle: 'short',
  }).format(date)
}

/**
 * Date heading format: "Sep 16, 2026"
 * Date only, no time.
 */
export function dateHeading(iso: string): string {
  if (!iso) return ''

  let date: Date
  try {
    date = new Date(iso)
    if (isNaN(date.getTime())) return iso
  } catch {
    return iso
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
  }).format(date)
}
