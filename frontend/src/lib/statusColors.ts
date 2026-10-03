import type { TaskPriorityLabel, TaskStatus } from './types'

export const STATUS_COLORS = {
  blue:    '#0369A1',
  amber:   '#B45309',
  green:   '#15803D',
  red:     '#B91C1C',
  neutral: '#5a6270',
}

export const PRIORITY_COLORS: Record<TaskPriorityLabel, string> = {
  low:      STATUS_COLORS.neutral,
  medium:   STATUS_COLORS.blue,
  high:     STATUS_COLORS.amber,
  critical: STATUS_COLORS.red,
}

export const TASK_COLUMN_COLOR: Record<TaskStatus, string> = {
  backlog:     STATUS_COLORS.neutral,
  ready:       STATUS_COLORS.blue,
  in_progress: STATUS_COLORS.amber,
  blocked:     STATUS_COLORS.amber,
  failed:      STATUS_COLORS.red,
  done:        STATUS_COLORS.green,
  cancelled:   STATUS_COLORS.neutral,
}
