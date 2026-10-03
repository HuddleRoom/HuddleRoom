import type { TaskPriorityLabel } from './types'
import { PRIORITY_COLORS } from './statusColors'

export const PRIORITY_INT: Record<TaskPriorityLabel, number> = {
  low: 25, medium: 50, high: 75, critical: 100,
}

export function priorityLabel(p: number): TaskPriorityLabel {
  if (p < 40) return 'low'
  if (p < 65) return 'medium'
  if (p < 90) return 'high'
  return 'critical'
}

export function priorityColor(p: number): string {
  return PRIORITY_COLORS[priorityLabel(p)]
}
