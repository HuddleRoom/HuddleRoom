import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { DndContext, DragOverlay, useDroppable, useDraggable, pointerWithin, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core'
import type { DragStartEvent, DragEndEvent } from '@dnd-kit/core'
import { CSS } from '@dnd-kit/utilities'
import { toast } from 'sonner'
import { Dialog } from '@/components/common/Dialog'
import { Plus, X, Play, Filter, RotateCw } from 'lucide-react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useUIStore } from '@/stores/ui'
import { useAllAgents, useAllActiveAgents } from '@/api/agents'
import {
  useAllTasks,
  useCopyTask,
  useCreateTask,
  usePatchTaskStatus,
  useRunTask,
  useResumeSession,
  useTask,
  useTaskSubtasks,
  useTaskSessions,
} from '@/api/tasks'
import type { Task, TaskStatus, TaskPriorityLabel } from '@/lib/types'
import { PRIORITY_INT, priorityLabel, priorityColor } from '@/lib/priority'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'
import { TASK_COLUMN_COLOR, STATUS_COLORS } from '@/lib/statusColors'
import { Button, Input, Textarea, PageHeader, Select, SectionLabel, UI_COLORS, SkeletonTable, NoProjectSelected } from '@/components/common/uiPrimitives'
import { StatTile } from '@/components/common/StatTile'
import { absolute } from '@/lib/time'

// ─── Constants ────────────────────────────────────────────────────────────────

const VALID_TRANSITIONS: Record<TaskStatus, TaskStatus[]> = {
  backlog:     ['ready', 'cancelled'],
  ready:       ['in_progress', 'cancelled'],
  in_progress: ['blocked', 'done', 'cancelled', 'failed'],
  blocked:     ['in_progress', 'cancelled'],
  failed:      ['in_progress', 'cancelled'],
  done:        [],
  cancelled:   [],
}

const COLUMNS: { status: TaskStatus; color: string; label: string }[] = [
  { status: 'backlog',     color: TASK_COLUMN_COLOR.backlog,     label: 'Backlog'     },
  { status: 'ready',       color: TASK_COLUMN_COLOR.ready,       label: 'Ready'       },
  { status: 'in_progress', color: TASK_COLUMN_COLOR.in_progress, label: 'In progress' },
  { status: 'blocked',     color: TASK_COLUMN_COLOR.blocked,     label: 'Blocked'     },
  { status: 'failed',      color: TASK_COLUMN_COLOR.failed,      label: 'Failed'      },
  { status: 'done',        color: TASK_COLUMN_COLOR.done,        label: 'Done'        },
  { status: 'cancelled',   color: TASK_COLUMN_COLOR.cancelled,   label: 'Cancelled'   },
]


const SR_ONLY: React.CSSProperties = {
  position: 'absolute',
  width: 1,
  height: 1,
  padding: 0,
  margin: -1,
  overflow: 'hidden',
  clip: 'rect(0,0,0,0)',
  whiteSpace: 'nowrap',
  border: 0,
}

type BoardCursor = {
  columnIndex: number
  cardIndex: number
}

function fmtDate(dateStr: string): string {
  const d = new Date(dateStr)
  const now = new Date()
  const month = d.getMonth() + 1
  const day = d.getDate()
  if (d.getFullYear() === now.getFullYear()) return `${month}/${day}`
  return `${month}/${day}/${String(d.getFullYear()).slice(2)}`
}

function readableStatus(status: string): string {
  return status.replace(/_/g, ' ')
}

function getColumnLabel(status: TaskStatus): string {
  return COLUMNS.find((column) => column.status === status)?.label ?? readableStatus(status)
}

function EmptyStatePanel({
  title,
  detail,
  primaryAction,
  secondaryAction,
  prerequisites,
  alternatePaths,
}: {
  title: string
  detail: string
  primaryAction: { label: string; onClick: () => void }
  secondaryAction?: { label: string; onClick: () => void }
  prerequisites: string[]
  alternatePaths: string[]
}) {
  return (
    <div
      className="bg-huddleroom-surface border border-huddleroom-border rounded-[6px]"
      style={{
        padding: '20px',
        display: 'flex',
        flexDirection: 'column',
        gap: 16,
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxWidth: 640 }}>
        <h2 className="text-huddleroom-text-primary text-xl font-semibold" style={{ margin: 0, lineHeight: 1.2 }}>
          {title}
        </h2>
        <p className="text-huddleroom-text-secondary text-sm" style={{ margin: 0, lineHeight: 1.6 }}>
          {detail}
        </p>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
        <Button variant="primary" onClick={primaryAction.onClick}>
          {primaryAction.label}
        </Button>
        {secondaryAction && (
          <Button variant="secondary" onClick={secondaryAction.onClick}>
            {secondaryAction.label}
          </Button>
        )}
      </div>

      <details style={{ marginTop: 4 }}>
        <summary className="text-huddleroom-text-muted text-xs" style={{
          cursor: 'pointer',
          userSelect: 'none' as const,
          listStyle: 'revert',
          paddingLeft: 4,
        }}>
          Why is this empty?
        </summary>
        <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
          <div
            className="bg-huddleroom-bg-2 border border-huddleroom-border rounded-[6px]"
            style={{ padding: '14px 16px' }}
          >
            <h3 className="text-huddleroom-text-primary text-sm font-semibold" style={{ margin: '0 0 8px' }}>
              Expected prerequisites
            </h3>
            <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 8 }}>
              {prerequisites.map((item) => (
                <li key={item} className="text-huddleroom-text-secondary text-[13px]" style={{ lineHeight: 1.5 }}>
                  {item}
                </li>
              ))}
            </ul>
          </div>

          <div
            className="bg-huddleroom-bg-2 border border-huddleroom-border rounded-[6px]"
            style={{ padding: '14px 16px' }}
          >
            <h3 className="text-huddleroom-text-primary text-sm font-semibold" style={{ margin: '0 0 8px' }}>
              Alternate paths
            </h3>
            <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 8 }}>
              {alternatePaths.map((item) => (
                <li key={item} className="text-huddleroom-text-secondary text-[13px]" style={{ lineHeight: 1.5 }}>
                  {item}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </details>
    </div>
  )
}


// ─── TaskCard ─────────────────────────────────────────────────────────────────

function TaskCard({
  task,
  agentName,
  isSelected,
  isBoardFocused,
  isDraggingActive,
  onClick,
  onFocus,
  onBoardNavigate,
  onInspectShortcut,
  onQuickMove,
  registerFocusable,
}: {
  task: Task
  agentName?: string
  isSelected: boolean
  isBoardFocused: boolean
  isDraggingActive: boolean
  onClick: () => void
  onFocus: () => void
  onBoardNavigate: (key: string) => void
  onInspectShortcut: () => void
  onQuickMove: (direction: -1 | 1) => void
  registerFocusable: (node: HTMLDivElement | null) => void
}) {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id: task.id,
    data: { task },
  })

  const style: React.CSSProperties = {
    background: UI_COLORS.surface,
    border: `1px solid ${isSelected ? UI_COLORS.primary : UI_COLORS.border}`,
    borderRadius: 6,
    padding: '10px 12px',
    cursor: isDraggingActive ? 'grabbing' : 'grab',
    userSelect: 'none',
    opacity: isDragging ? 0 : 1,
    transition: 'border-color 120ms cubic-bezier(0.4, 0, 0.2, 1), background-color 120ms cubic-bezier(0.4, 0, 0.2, 1)',
    transform: transform ? CSS.Translate.toString(transform) : undefined,
    backgroundColor: isBoardFocused && !isSelected ? UI_COLORS.depth : UI_COLORS.surface,
  }

  function handleRef(node: HTMLDivElement | null) {
    setNodeRef(node)
    registerFocusable(node)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.shiftKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      e.preventDefault()
      e.stopPropagation()
      onQuickMove(e.key === 'ArrowLeft' ? -1 : 1)
      return
    }

    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'ArrowDown' || e.key === 'Home' || e.key === 'End') {
      e.preventDefault()
      e.stopPropagation()
      onBoardNavigate(e.key)
      return
    }

    if (e.key === 'Enter') {
      e.preventDefault()
      e.stopPropagation()
      onInspectShortcut()
      return
    }

    if (e.key.toLowerCase() === 'i') {
      e.preventDefault()
      e.stopPropagation()
      onInspectShortcut()
      return
    }

    const dndKeyDown = (listeners as { onKeyDown?: (e: React.KeyboardEvent<HTMLDivElement>) => void }).onKeyDown
    if (dndKeyDown) dndKeyDown(e)
  }

  return (
    <div
      ref={handleRef}
      data-testid={`task-card-${task.id}`}
      style={style}
      onClick={(e) => { e.stopPropagation(); onClick() }}
      onFocus={onFocus}
      onMouseEnter={(e) => {
        if (!isSelected) {
          ;(e.currentTarget as HTMLDivElement).style.borderColor = UI_COLORS.borderStrong
          ;(e.currentTarget as HTMLDivElement).style.backgroundColor = UI_COLORS.depth
        }
      }}
      onMouseLeave={(e) => {
        if (!isSelected) {
          ;(e.currentTarget as HTMLDivElement).style.borderColor = UI_COLORS.border
          ;(e.currentTarget as HTMLDivElement).style.backgroundColor = UI_COLORS.surface
        }
      }}
      {...attributes}
      {...listeners}
      onKeyDown={handleKeyDown}
      tabIndex={isBoardFocused ? 0 : -1}
      aria-label={`${task.title}, ${getColumnLabel(task.status)}, ${priorityLabel(task.priority)} priority`}
    >
      <div className="text-[13px] text-huddleroom-text-primary" style={{ marginBottom: 10, lineHeight: 1.45 }}>
        {task.title}
      </div>
      <div className="text-[11px]" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ color: priorityColor(task.priority) }}>{priorityLabel(task.priority)}</span>
        {agentName && (
          <span className="text-huddleroom-text-muted" style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {agentName}
          </span>
        )}
        <span className="text-huddleroom-text-muted font-mono" style={{ flexShrink: 0 }}>{fmtDate(task.created_at)}</span>
      </div>
    </div>
  )
}

// ─── KanbanColumn ─────────────────────────────────────────────────────────────

function KanbanColumn({
  status,
  label,
  color,
  columnIndex,
  tasks,
  agentMap,
  activeDragTask,
  selectedTaskId,
  onTaskClick,
  isBoardHeaderFocused,
  isBoardCardFocused,
  onHeaderFocus,
  onHeaderNavigate,
  onCardFocus,
  onCardNavigate,
  onCardInspectShortcut,
  onCardQuickMove,
  registerColumnHeader,
  registerTaskCard,
}: {
  status: TaskStatus
  label: string
  color: string
  columnIndex: number
  tasks: Task[]
  agentMap: Map<string, string>
  activeDragTask: Task | null
  selectedTaskId: string | null
  onTaskClick: (id: string) => void
  isBoardHeaderFocused: boolean
  isBoardCardFocused: (cardIndex: number) => boolean
  onHeaderFocus: () => void
  onHeaderNavigate: (key: string) => void
  onCardFocus: (cardIndex: number) => void
  onCardNavigate: (cardIndex: number, key: string) => void
  onCardInspectShortcut: (taskId: string) => void
  onCardQuickMove: (task: Task, direction: -1 | 1) => void
  registerColumnHeader: (node: HTMLDivElement | null) => void
  registerTaskCard: (taskId: string, node: HTMLDivElement | null) => void
}) {
  const { isOver, setNodeRef } = useDroppable({ id: status, data: { status } })

  const canReceive = activeDragTask
    ? VALID_TRANSITIONS[activeDragTask.status].includes(status)
    : false

  const borderColor = isOver
    ? (canReceive ? STATUS_COLORS.green : STATUS_COLORS.red)
    : UI_COLORS.border

  return (
    <div
      ref={setNodeRef}
      className="min-w-[240px] flex-1 bg-huddleroom-surface rounded-[6px]"
      style={{
        display: 'flex',
        flexDirection: 'column',
        border: `1px solid ${borderColor}`,
        maxHeight: 'calc(100vh - 200px)',
        transition: 'border-color 120ms cubic-bezier(0.4, 0, 0.2, 1)',
      }}
    >
      <div
        ref={registerColumnHeader}
        tabIndex={isBoardHeaderFocused ? 0 : -1}
        onFocus={onHeaderFocus}
        onKeyDown={(e) => {
          if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'ArrowUp' || e.key === 'ArrowDown' || e.key === 'Home' || e.key === 'End') {
            e.preventDefault()
            onHeaderNavigate(e.key)
          }
        }}
        aria-label={`${label} column ${columnIndex + 1}, ${tasks.length} ${tasks.length === 1 ? 'task' : 'tasks'}`}
        className="border-b border-huddleroom-border bg-huddleroom-bg-2"
        style={{
          padding: '10px 12px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexShrink: 0,
        }}
      >
        <span className="text-huddleroom-text-secondary text-xs font-semibold">
          {label}
        </span>
        <span className="text-huddleroom-text-muted text-[11px]">{tasks.length}</span>
        {activeDragTask && (
          <span
            aria-live="polite"
            className="text-[10px]"
            style={{ color: canReceive ? STATUS_COLORS.green : STATUS_COLORS.red, marginLeft: 4 }}
          >
            {canReceive ? '✓' : '✗'}
          </span>
        )}
      </div>

      <div className="bg-huddleroom-base" style={{ flex: 1, overflowY: 'auto', padding: 8, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {tasks.map((t) => (
          <TaskCard
            key={t.id}
            task={t}
            agentName={t.assigned_to ? agentMap.get(t.assigned_to) : undefined}
            isSelected={t.id === selectedTaskId}
            isBoardFocused={isBoardCardFocused(tasks.findIndex((candidate) => candidate.id === t.id))}
            isDraggingActive={!!activeDragTask}
            onClick={() => onTaskClick(t.id)}
            onFocus={() => onCardFocus(tasks.findIndex((candidate) => candidate.id === t.id))}
            onBoardNavigate={(key) => onCardNavigate(tasks.findIndex((candidate) => candidate.id === t.id), key)}
            onInspectShortcut={() => onCardInspectShortcut(t.id)}
            onQuickMove={(direction) => onCardQuickMove(t, direction)}
            registerFocusable={(node) => registerTaskCard(t.id, node)}
          />
        ))}
        {tasks.length === 0 && (
          <div className="text-huddleroom-border-strong text-[11px]" style={{ textAlign: 'center', padding: '24px 0' }}>—</div>
        )}
      </div>
    </div>
  )
}

// ─── Task Detail Panel ────────────────────────────────────────────────────────

function TaskDetailPanel({
  taskId,
  projectId,
  agentMap,
  focusRequest,
  onClose,
  onFocusBoard,
  isNarrow,
}: {
  taskId: string
  projectId: string
  agentMap: Map<string, string>
  focusRequest: number
  onClose: () => void
  onFocusBoard: () => void
  isNarrow?: boolean
}) {
  const { data: task, isLoading: taskLoading } = useTask(projectId, taskId)
  const { data: subtasksData } = useTaskSubtasks(projectId, taskId)
  const { data: sessionsData, isLoading: sessionsLoading, isFetching: sessionsFetching } = useTaskSessions(projectId, taskId)
  const runTask = useRunTask(projectId)
  const copyTask = useCopyTask(projectId)
  const patchStatus = usePatchTaskStatus(projectId)
  const resumeSession = useResumeSession(projectId)
  const panelRef = useRef<HTMLDivElement | null>(null)

  const subtasks = subtasksData ?? []
  const sessions = sessionsData?.items ?? []

  const hasActiveSession = sessions.some(
    (s) => s.status === 'pending' || s.status === 'running'
  )
  const canRun = task &&
    (['backlog', 'ready', 'failed', 'blocked', 'in_progress'] as TaskStatus[]).includes(task.status) &&
    !!task.assigned_to &&
    !sessionsLoading &&
    !sessionsFetching &&
    !hasActiveSession &&
    !runTask.isPending
  const quickMoves = task ? VALID_TRANSITIONS[task.status] : []

  useEffect(() => {
    if (focusRequest === 0) return
    panelRef.current?.focus()
  }, [focusRequest, taskId])

  function moveTask(direction: -1 | 1) {
    if (!task) return

    const currentIndex = COLUMNS.findIndex((column) => column.status === task.status)
    let nextIndex = currentIndex + direction
    const previousStatus = task.status

    while (nextIndex >= 0 && nextIndex < COLUMNS.length) {
      const candidate = COLUMNS[nextIndex].status
      if (VALID_TRANSITIONS[task.status].includes(candidate)) {
        patchStatus.mutate(
          { taskId: task.id, status: candidate },
          {
            onSuccess: () => toast(`Moved to ${getColumnLabel(candidate)}`, {
              action: { label: 'Undo', onClick: () => patchStatus.mutate({ taskId: task.id, status: previousStatus }, { onError: (e) => toast.error(e.message) }) },
            }),
            onError: (error) => toast.error(error.message),
          },
        )
        return
      }
      nextIndex += direction
    }
  }

  return (
    <div
      ref={panelRef}
      data-testid="task-detail-panel"
      tabIndex={-1}
      role="region"
      aria-label={task ? `Inspect task ${task.title}` : 'Inspect task'}
      onKeyDown={(e) => {
        if (e.key === 'Escape') {
          e.preventDefault()
          onClose()
          return
        }

        if (e.key.toLowerCase() === 'b') {
          e.preventDefault()
          onFocusBoard()
          return
        }

        if (e.shiftKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
          e.preventDefault()
          moveTask(e.key === 'ArrowLeft' ? -1 : 1)
        }
      }}
      className="bg-huddleroom-surface border border-huddleroom-border rounded-[6px]"
      style={{
        width: isNarrow ? '100%' : 'min(400px, 38vw)',
        minWidth: isNarrow ? undefined : 280,
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
        overflowY: 'auto',
      }}
    >
      {/* Header */}
      <div className="border-b border-huddleroom-border" style={{
        padding: '16px 18px',
        display: 'flex',
        alignItems: 'flex-start',
        gap: 8,
        flexShrink: 0,
      }}>
        <div style={{ flex: 1 }}>
          <div className="text-huddleroom-text-muted text-[13px]" style={{ marginBottom: 8 }}>
            Inspect task
          </div>
          <div className="text-sm text-huddleroom-text-primary font-bold" style={{ lineHeight: 1.4 }}>
            {task?.title ?? '…'}
          </div>
          {task && (
            <div className="text-xs" style={{ display: 'flex', gap: 8, marginTop: 6, flexWrap: 'wrap' }}>
              <span className={`status-${task.status}`} aria-label={`Task status ${readableStatus(task.status)}`}>{task.status}</span>
              <span style={{ color: priorityColor(task.priority) }}>{priorityLabel(task.priority)}</span>
              {task.assigned_to && (
                <span className="text-huddleroom-text-muted">{agentMap.get(task.assigned_to) ?? task.assigned_to}</span>
              )}
              <span className="text-huddleroom-text-muted font-mono">{fmtDate(task.created_at)}</span>
            </div>
          )}
        </div>
        <Button
          variant="ghost"
          onClick={onClose}
          style={{ padding: 2, flexShrink: 0 }}
          aria-label={task ? `Close task details for ${task.title}` : 'Close task details'}
        >
          <X size={14} />
        </Button>
      </div>

      {/* Body */}
      <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 18, flex: 1 }}>
        {taskLoading ? (
          <SkeletonTable count={4} />
        ) : (
          <>
        {task && quickMoves.length > 0 && (
          <div>
            <SectionLabel>Quick move</SectionLabel>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {quickMoves.map((status) => (
                <Button
                  key={status}
                  variant="secondary"
                  size="sm"
                  onClick={() => {
                    const previousStatus = task.status
                    patchStatus.mutate(
                      { taskId: task.id, status },
                      {
                        onSuccess: () => toast(`Moved to ${getColumnLabel(status)}`, {
                          action: { label: 'Undo', onClick: () => patchStatus.mutate({ taskId: task.id, status: previousStatus }, { onError: (e) => toast.error(e.message) }) },
                        }),
                        onError: (error) => toast.error(error.message),
                      },
                    )
                  }}
                >
                  {getColumnLabel(status)}
                </Button>
              ))}
            </div>
          </div>
        )}

        {task?.description && (
          <div>
            <SectionLabel>Description</SectionLabel>
            <div className="text-[13px] text-huddleroom-text-muted" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>
              {task.description}
            </div>
          </div>
        )}

        {task?.metadata && Object.keys(task.metadata).length > 0 && (
          <div>
            <SectionLabel>Metadata</SectionLabel>
            <pre className="text-[13px] text-huddleroom-text-muted font-mono bg-huddleroom-bg-2 rounded-[3px]" style={{
              padding: 8, overflowX: 'auto', margin: 0,
            }}>
              {JSON.stringify(task.metadata, null, 2)}
            </pre>
          </div>
        )}

        {subtasks.length > 0 && (
          <div>
            <SectionLabel>Subtasks ({subtasks.length})</SectionLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {subtasks.map((s) => (
                <div key={s.id} className="text-[13px]" style={{ display: 'flex', gap: 8 }}>
                  <span className={`status-${s.status}`} style={{ flexShrink: 0 }} aria-label={`Subtask status ${readableStatus(s.status)}`}>{s.status}</span>
                  <span className="text-huddleroom-text-primary">{s.title}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {sessions.length > 0 && (
          <div>
            <SectionLabel>Sessions ({sessions.length})</SectionLabel>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {sessions.map((s) => (
                <div key={s.id} className="border-b border-huddleroom-bg-2 text-[13px]" style={{
                  padding: '5px 0',
                }}>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <span className={`status-${s.status}`} style={{ minWidth: 60 }} aria-label={`Session status ${readableStatus(s.status)}`}>{s.status}</span>
                    <span className="text-huddleroom-text-muted font-mono" style={{ flex: 1 }}>{absolute(s.started_at)}</span>
                  </div>
                  {s.output && (
                    <pre className="text-huddleroom-text-muted font-mono text-xs" style={{
                      margin: '6px 0 0',
                      lineHeight: 1.4,
                      whiteSpace: 'pre-wrap',
                    }}>
                      {s.output}
                    </pre>
                  )}
                  {s.error && (
                    <pre className="text-huddleroom-status-red font-mono text-xs" style={{
                      margin: '6px 0 0',
                      lineHeight: 1.4,
                      whiteSpace: 'pre-wrap',
                    }}>
                      {s.error}
                    </pre>
                  )}
                  {s.status === 'failed' && s.resumable && (
                    <div style={{ marginTop: 6, display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                      <span className="text-xs font-semibold text-huddleroom-status-red">
                        Model error — can be resumed
                      </span>
                      <Button
                        variant="secondary"
                        size="sm"
                        disabled={resumeSession.isPending}
                        onClick={() => {
                          resumeSession.mutate(s.id, {
                            onSuccess: () => toast.success('Session resumed'),
                            onError: (e) => toast.error(e.message),
                          })
                        }}
                      >
                        <RotateCw size={12} />
                        {resumeSession.isPending ? 'Resuming…' : 'Resume'}
                      </Button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
          </>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-huddleroom-border" style={{ padding: '12px 16px', flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <Button
          variant="primary"
          data-testid="task-run-control"
          style={{
            width: '100%',
            justifyContent: 'center',
          }}
          disabled={!canRun}
          onClick={() => {
            if (!task) return
            runTask.mutate(task.id, {
              onSuccess: () => toast.success('Task queued'),
              onError: (e) => toast.error(e.message),
            })
          }}
        >
          <Play size={12} />
          {!task ? 'Run task'
            : hasActiveSession ? 'Session active'
            : !task.assigned_to ? 'Assign agent to run'
            : 'Run task'}
        </Button>
        <Button
          variant="secondary"
          size="sm"
          disabled={!task || copyTask.isPending}
          onClick={() => {
            if (!task) return
            copyTask.mutate(task.id, {
              onSuccess: () => toast.success('Task copied'),
              onError: (e) => toast.error(e.message),
            })
          }}
        >
          {copyTask.isPending ? 'Copying...' : 'Copy task'}
        </Button>
        <p className="text-micro text-huddleroom-text-muted m-0">Shift+Arrow: move status · B: board · Esc: close</p>
      </div>
    </div>
  )
}

function EmptyInspectPanel({ isNarrow }: { isNarrow?: boolean }) {
  return (
    <div
      className="bg-huddleroom-surface border border-huddleroom-border rounded-[6px]"
      style={{
        width: isNarrow ? '100%' : 'min(400px, 38vw)',
        minWidth: isNarrow ? undefined : 280,
        flexShrink: 0,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <div className="border-b border-huddleroom-border" style={{ padding: '16px 18px' }}>
        <div className="text-huddleroom-text-muted text-[13px]" style={{ marginBottom: 8 }}>
          Inspect task
        </div>
        <div className="text-huddleroom-text-primary text-[15px] font-bold">
          Select a task
        </div>
      </div>
      <div style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 14 }}>
        <p className="text-huddleroom-text-muted text-[13px]" style={{ margin: 0 }}>Status, assignment, sessions, and run control.</p>
      </div>
    </div>
  )
}

// ─── Create Task Modal ────────────────────────────────────────────────────────

function CreateTaskModal({
  open,
  onClose,
  projectId,
  agents,
}: {
  open: boolean
  onClose: () => void
  projectId: string
  agents: { id: string; name: string }[]
}) {
  const createTask = useCreateTask(projectId)
  const [form, setForm] = useState({
    title: '',
    description: '',
    priority: 'medium' as TaskPriorityLabel,
    assigned_to: '',
  })
  const titleInput = useRef<HTMLInputElement>(null)

  function reset() {
    setForm({ title: '', description: '', priority: 'medium' as TaskPriorityLabel, assigned_to: '' })
  }

  function handleCreate() {
    if (!form.title.trim()) return
    createTask.mutate(
      {
        title: form.title.trim(),
        description: form.description.trim() || undefined,
        priority: PRIORITY_INT[form.priority],
        assigned_to: form.assigned_to || undefined,
      },
      {
        onSuccess: () => { toast.success('Task created'); reset(); onClose() },
        onError: (e) => toast.error(e.message),
      },
    )
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => { if (!v) { reset(); onClose() } }}
      title="New task"
      description="Create or edit a task."
      size="md"
      initialFocusRef={titleInput}
      footer={{
        primaryLabel: createTask.isPending ? 'Creating...' : 'Create task',
        primaryType: 'submit',
        formId: 'create-task-form',
        isPending: createTask.isPending,
        primaryDisabled: !form.title.trim(),
      }}
    >
      <form id="create-task-form" onSubmit={(event) => { event.preventDefault(); handleCreate() }}>
        <Input
          ref={titleInput}
          label="Title *"
          type="text"
          value={form.title}
          onChange={(e) => setForm((p) => ({ ...p, title: e.target.value }))}
          placeholder="task title"
        />

        <Textarea
          label="Description"
          value={form.description}
          onChange={(e) => setForm((p) => ({ ...p, description: e.target.value }))}
          placeholder="optional description"
          rows={4}
        />

        <div style={{ display: 'flex', gap: 12 }}>
          <div style={{ flex: 1 }}>
            <Select
              label="Priority"
              value={form.priority}
              onChange={(e) => setForm((p) => ({ ...p, priority: e.target.value as TaskPriorityLabel }))}
            >
              <option value="low">low</option>
              <option value="medium">medium</option>
              <option value="high">high</option>
              <option value="critical">critical</option>
            </Select>
          </div>
          <div style={{ flex: 1 }}>
            <Select
              label="Assign to agent"
              value={form.assigned_to}
              onChange={(e) => setForm((p) => ({ ...p, assigned_to: e.target.value }))}
            >
              <option value="">unassigned</option>
              {agents.map((a) => (
                <option key={a.id} value={a.id}>{a.name}</option>
              ))}
            </Select>
          </div>
        </div>
      </form>
    </Dialog>
  )
}

// ─── TasksPage ────────────────────────────────────────────────────────────────

export function TasksPage() {
  useDocumentTitle('Tasks')
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const pid = useUIStore((s) => s.activeProjectId)
  const setCreateProjectOpen = useUIStore((s) => s.setCreateProjectOpen)
  const [activeDragTask, setActiveDragTask] = useState<Task | null>(null)
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null)
  const [detailFocusRequest, setDetailFocusRequest] = useState(0)
  const [showCreate, setShowCreate] = useState(false)
  const [isNarrow, setIsNarrow] = useState(false)
  const filterAgent = searchParams.get('agent') ?? ''
  const filterStatus = searchParams.get('status') ?? ''
  const [boardCursor, setBoardCursor] = useState<BoardCursor>({ columnIndex: 0, cardIndex: -1 })
  const columnHeaderRefs = useRef<Record<number, HTMLDivElement | null>>({})
  const taskCardRefs = useRef<Record<string, HTMLDivElement | null>>({})
  const focusReturnTaskIdRef = useRef<string | null>(null)

  const { items: tasks } = useAllTasks(pid, {
    assigned_to: filterAgent || undefined,
    status: filterStatus || undefined,
  })

  const { items: allAgents } = useAllAgents()

  const { items: activeAgents } = useAllActiveAgents()
  const hasTaskFilters = filterAgent !== '' || filterStatus !== ''

  const agentMap = useMemo(() => {
    const m = new Map<string, string>()
    allAgents.forEach((a) => m.set(a.id, a.name))
    return m
  }, [allAgents])

  const tasksByStatus = useMemo(() => {
    const map: Partial<Record<TaskStatus, Task[]>> = {}
    COLUMNS.forEach((c) => { map[c.status] = [] })
    tasks.forEach((t) => {
      if (!map[t.status]) map[t.status] = []
      map[t.status]!.push(t)
    })
    return map
  }, [tasks])
  const boardColumns = COLUMNS.map((column) => tasksByStatus[column.status] ?? [])

  const patchStatus = usePatchTaskStatus(pid)
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
    useSensor(KeyboardSensor),
  )

  function handleDragStart(event: DragStartEvent) {
    const task = event.active.data.current?.task as Task | undefined
    if (task) setActiveDragTask(task)
  }

  function handleDragEnd(event: DragEndEvent) {
    const task = activeDragTask
    setActiveDragTask(null)
    if (!task || !event.over) return

    const targetStatus = event.over.id as TaskStatus
    if (targetStatus === task.status) return

    const valid = VALID_TRANSITIONS[task.status].includes(targetStatus)
    if (!valid) {
      const from = COLUMNS.find((c) => c.status === task.status)?.label ?? task.status
      const to = COLUMNS.find((c) => c.status === targetStatus)?.label ?? targetStatus
      toast.error(`Can't move from "${from}" to "${to}"`)
      return
    }

    const previousStatus = task.status
    patchStatus.mutate(
      { taskId: task.id, status: targetStatus },
      {
        onSuccess: () => toast(`Moved to ${getColumnLabel(targetStatus)}`, {
          action: { label: 'Undo', onClick: () => patchStatus.mutate({ taskId: task.id, status: previousStatus }, { onError: (e) => toast.error(e.message) }) },
        }),
        onError: (e) => toast.error(e.message),
      },
    )
  }


  const readyCount = tasksByStatus.ready?.length ?? 0
  const inProgressCount = tasksByStatus.in_progress?.length ?? 0
  const blockedCount = tasksByStatus.blocked?.length ?? 0

  const clampBoardCursor = useCallback((columnIndex: number, cardIndex: number): BoardCursor => {
    const safeColumnIndex = Math.min(Math.max(columnIndex, 0), COLUMNS.length - 1)
    const columnTasks = boardColumns[safeColumnIndex] ?? []
    if (cardIndex < 0 || columnTasks.length === 0) {
      return { columnIndex: safeColumnIndex, cardIndex: -1 }
    }

    return {
      columnIndex: safeColumnIndex,
      cardIndex: Math.min(cardIndex, columnTasks.length - 1),
    }
  }, [boardColumns])

  function focusBoardTarget(cursor: BoardCursor) {
    requestAnimationFrame(() => {
      if (cursor.cardIndex === -1) {
        columnHeaderRefs.current[cursor.columnIndex]?.focus()
        return
      }

      const targetTask = boardColumns[cursor.columnIndex]?.[cursor.cardIndex]
      if (targetTask) {
        taskCardRefs.current[targetTask.id]?.focus()
      }
    })
  }

  function setBoardTarget(columnIndex: number, cardIndex: number, focus = false) {
    const next = clampBoardCursor(columnIndex, cardIndex)
    setBoardCursor((current) => (
      current.columnIndex === next.columnIndex && current.cardIndex === next.cardIndex
        ? current
        : next
    ))

    if (focus) {
      focusBoardTarget(next)
    }
  }

  function moveBoardCursor(from: BoardCursor, key: string) {
    const columnTasks = boardColumns[from.columnIndex] ?? []

    if (key === 'ArrowLeft') {
      setBoardTarget(from.columnIndex - 1, from.cardIndex, true)
      return
    }

    if (key === 'ArrowRight') {
      setBoardTarget(from.columnIndex + 1, from.cardIndex, true)
      return
    }

    if (key === 'ArrowUp') {
      if (from.cardIndex <= 0) {
        setBoardTarget(from.columnIndex, -1, true)
        return
      }

      setBoardTarget(from.columnIndex, from.cardIndex - 1, true)
      return
    }

    if (key === 'ArrowDown') {
      if (from.cardIndex === -1) {
        setBoardTarget(from.columnIndex, 0, true)
        return
      }

      if (from.cardIndex < columnTasks.length - 1) {
        setBoardTarget(from.columnIndex, from.cardIndex + 1, true)
      }
      return
    }

    if (key === 'Home') {
      setBoardTarget(0, from.cardIndex, true)
      return
    }

    if (key === 'End') {
      setBoardTarget(COLUMNS.length - 1, from.cardIndex, true)
    }
  }

  function openTask(taskId: string, options?: { focusDetail?: boolean }) {
    setSelectedTaskId(taskId)
    focusReturnTaskIdRef.current = taskId

    const task = tasks.find((candidate) => candidate.id === taskId)
    if (task) {
      const columnIndex = COLUMNS.findIndex((column) => column.status === task.status)
      const cardIndex = (boardColumns[columnIndex] ?? []).findIndex((candidate) => candidate.id === taskId)
      if (columnIndex >= 0 && cardIndex >= 0) {
        setBoardTarget(columnIndex, cardIndex, false)
      }
    }

    if (options?.focusDetail) {
      setDetailFocusRequest((current) => current + 1)
    }
  }

  function focusBoardSelection() {
    if (selectedTaskId) {
      const task = tasks.find((candidate) => candidate.id === selectedTaskId)
      if (task) {
        const columnIndex = COLUMNS.findIndex((column) => column.status === task.status)
        const cardIndex = (boardColumns[columnIndex] ?? []).findIndex((candidate) => candidate.id === selectedTaskId)
        if (columnIndex >= 0 && cardIndex >= 0) {
          setBoardTarget(columnIndex, cardIndex, true)
          return
        }
      }
    }

    focusBoardTarget(boardCursor)
  }

  function closeInspectPanel() {
    const returnTaskId = focusReturnTaskIdRef.current ?? selectedTaskId
    setSelectedTaskId(null)
    if (!returnTaskId) return

    requestAnimationFrame(() => {
      taskCardRefs.current[returnTaskId]?.focus()
    })
  }

  function quickMoveTask(task: Task, direction: -1 | 1) {
    const currentIndex = COLUMNS.findIndex((column) => column.status === task.status)
    let nextIndex = currentIndex + direction
    const previousStatus = task.status

    while (nextIndex >= 0 && nextIndex < COLUMNS.length) {
      const candidate = COLUMNS[nextIndex].status
      if (VALID_TRANSITIONS[task.status].includes(candidate)) {
        patchStatus.mutate(
          { taskId: task.id, status: candidate },
          {
            onSuccess: () => toast(`Moved to ${getColumnLabel(candidate)}`, {
              action: { label: 'Undo', onClick: () => patchStatus.mutate({ taskId: task.id, status: previousStatus }, { onError: (e) => toast.error(e.message) }) },
            }),
            onError: (error) => toast.error(error.message),
          },
        )
        return
      }
      nextIndex += direction
    }
  }

  useEffect(() => {
    const mq = window.matchMedia('(max-width: 900px)')
    const handler = (e: MediaQueryListEvent) => setIsNarrow(e.matches)
    setIsNarrow(mq.matches)
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [])

  useEffect(() => {
    const next = clampBoardCursor(boardCursor.columnIndex, boardCursor.cardIndex)
    if (next.columnIndex !== boardCursor.columnIndex || next.cardIndex !== boardCursor.cardIndex) {
      setBoardCursor(next)
    }
  }, [boardColumns, boardCursor, clampBoardCursor])

  if (!pid) {
    return (
      <NoProjectSelected
        pageTitle="Tasks"
        pageIntro="Tasks are scoped to the currently selected project. Choose a workspace first, then the orchestrator can load its board, detail panel, and run controls."
        title="Select a project to load the task board"
        detail="The top-bar switcher is the entry point for project-scoped work. Once you choose a project, this page will load its backlog, ready queue, active execution, and completion history."
        primaryAction={{ label: 'New project', onClick: () => setCreateProjectOpen(true) }}
        secondaryAction={{ label: 'Open dashboard', onClick: () => navigate('/') }}
        prerequisites={[
          'Your account can list at least one HuddleRoom project.',
          'The project switcher in the top bar shows the workspace you want to operate.',
          'That workspace has tasks or is ready for operators to create the first one.',
        ]}
        alternatePaths={[
          'If the switcher is empty, confirm you are in the correct environment or ask an administrator to create a project.',
          'If work has not been broken into tasks yet, start from Dashboard, Meetings, or Protocols to identify what should be queued.',
          'If you only need to inspect agent capacity first, open Agents before creating new work here.',
        ]}
      />
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 80px)', gap: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <PageHeader title="Tasks" />
        <Button variant="primary" onClick={() => setShowCreate(true)}>
          <Plus size={12} />
          New task
        </Button>
      </div>

      <div
        className="bg-huddleroom-surface border border-huddleroom-border rounded-[6px]"
        style={{
          padding: '14px 20px',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
          gap: 10,
          flexShrink: 0,
        }}
      >
        <StatTile label="Ready to run" value={readyCount} color={STATUS_COLORS.blue} />
        <StatTile label="In progress" value={inProgressCount} color={STATUS_COLORS.amber} />
        <StatTile label="Blocked" value={blockedCount} color={STATUS_COLORS.amber} />
      </div>

      <div className="bg-huddleroom-surface border border-huddleroom-border rounded-[6px]" style={{
        display: 'flex',
        alignItems: 'center',
        gap: 12,
        flexShrink: 0,
        flexWrap: 'wrap',
        padding: '12px 14px',
      }}>
        <span className="text-huddleroom-text-secondary text-xs">
          Filter board
        </span>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginLeft: 4 }}>
          <Filter size={12} className="text-huddleroom-text-muted" />
          <Select
            aria-label="Filter tasks by agent"
            value={filterAgent}
            onChange={(e) => setSearchParams((prev) => { const next = new URLSearchParams(prev); if (e.target.value) next.set('agent', e.target.value); else next.delete('agent'); return next }, { replace: true })}
            className="w-auto"
          >
            <option value="">All agents</option>
            {allAgents.map((a) => <option key={a.id} value={a.id}>{a.name}</option>)}
          </Select>

          <Select
            aria-label="Filter tasks by status"
            value={filterStatus}
            onChange={(e) => setSearchParams((prev) => { const next = new URLSearchParams(prev); if (e.target.value) next.set('status', e.target.value); else next.delete('status'); return next }, { replace: true })}
            className="w-auto"
          >
            <option value="">All statuses</option>
            {COLUMNS.map((c) => <option key={c.status} value={c.status}>{c.label}</option>)}
          </Select>
        </div>
      </div>

      <div style={{ flex: 1, display: 'flex', flexDirection: isNarrow ? 'column' : 'row', gap: 16, minHeight: 0, overflow: 'hidden' }}>
        {tasks.length === 0 ? (
          <div style={{ flex: 1, minWidth: 0, overflowY: 'auto', paddingBottom: 16 }}>
            <EmptyStatePanel
              title={hasTaskFilters ? 'No tasks match the current filters' : 'This project does not have tasks yet'}
              detail={
                hasTaskFilters
                  ? 'The board stays empty until at least one task matches the selected agent or status filters. Clear the filters to inspect the full queue, or create a task that belongs in the slice you are watching.'
                  : 'Tasks appear here after operators or upstream workflows create them for the selected project. Once tasks exist, the board becomes the place to triage state transitions, inspect execution context, and run assigned work.'
              }
              primaryAction={
                hasTaskFilters
                  ? { label: 'Clear filters', onClick: () => setSearchParams((prev) => { const next = new URLSearchParams(prev); next.delete('agent'); next.delete('status'); return next }, { replace: true }) }
                  : { label: 'Create first task', onClick: () => setShowCreate(true) }
              }
              secondaryAction={
                hasTaskFilters
                  ? { label: 'Create task anyway', onClick: () => setShowCreate(true) }
                  : { label: 'Review agents', onClick: () => navigate('/agents') }
              }
              prerequisites={
                hasTaskFilters
                  ? [
                      'The selected project already has tasks, but none match the active filters.',
                      'Agent and status filters reflect the queue slice you intend to inspect.',
                      'Newly created tasks will not appear until they match the current filter set.',
                    ]
                  : [
                      'A project is selected in the top bar.',
                      'Operators know what work item should be queued first for this workspace.',
                      'If you want runnable tasks immediately, at least one agent is available for assignment.',
                    ]
              }
              alternatePaths={
                hasTaskFilters
                  ? [
                      'Broaden the filter scope first if you are trying to understand overall workload.',
                      'Open Agents to confirm who is active before narrowing by assignee again.',
                      'Use Dashboard to see whether meetings or protocols are producing new work that has not reached this board yet.',
                    ]
                  : [
                      'Open Agents if you need to verify capacity before assigning the first task.',
                      'Open Meetings or Protocols if the next task should be derived from active coordination instead of manual entry.',
                      'If tasks should already exist, confirm you selected the correct project in the top bar.',
                    ]
              }
            />
          </div>
        ) : (
          <DndContext
            sensors={sensors}
            collisionDetection={pointerWithin}
            onDragStart={handleDragStart}
            onDragEnd={handleDragEnd}
          >
            <div
              className="bg-huddleroom-surface border border-huddleroom-border rounded-[6px]"
              style={{
                flex: 1,
                display: 'flex',
                minHeight: 0,
                minWidth: 0,
                flexDirection: 'column',
                gap: 12,
                padding: '14px',
                overflow: 'hidden',
              }}
            >
	              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'flex-start' }}>
	                <span className="text-huddleroom-text-primary text-[16px] font-bold">
	                  Task board
	                </span>
	                <span className="text-xs text-huddleroom-text-muted">Arrows move focus · Shift+Arrows move card · I inspects</span>
	              </div>

	              <div
	                aria-label="Task board columns"
	                className="overflow-x-auto"
	                style={{
	                  flex: 1,
	                  display: 'flex',
	                  gap: 12,
	                  padding: '2px 0 4px',
	                  alignItems: 'flex-start',
	                  minHeight: 0,
	                  minWidth: 0,
	                }}
	              >
	                <span style={SR_ONLY}>
	                  Use arrow keys to move between columns and cards. Press Enter or I to inspect the focused task. Press Shift and the left or right arrow key to move the focused task across valid states.
	                </span>
	                {COLUMNS.map((col, columnIndex) => (
	                  <KanbanColumn
	                    key={col.status}
	                    columnIndex={columnIndex}
	                    status={col.status}
	                    label={col.label}
	                    color={col.color}
	                    tasks={tasksByStatus[col.status] ?? []}
	                    agentMap={agentMap}
	                    activeDragTask={activeDragTask}
	                    selectedTaskId={selectedTaskId}
	                    onTaskClick={(id) => {
	                      const cardIndex = (tasksByStatus[col.status] ?? []).findIndex((task) => task.id === id)
	                      setBoardTarget(columnIndex, cardIndex, false)
	                      setSelectedTaskId((current) => current === id ? null : id)
	                      focusReturnTaskIdRef.current = id
	                    }}
	                    isBoardHeaderFocused={boardCursor.columnIndex === columnIndex && boardCursor.cardIndex === -1}
	                    isBoardCardFocused={(cardIndex) => boardCursor.columnIndex === columnIndex && boardCursor.cardIndex === cardIndex}
	                    onHeaderFocus={() => setBoardTarget(columnIndex, -1, false)}
	                    onHeaderNavigate={(key) => moveBoardCursor({ columnIndex, cardIndex: -1 }, key)}
	                    onCardFocus={(cardIndex) => setBoardTarget(columnIndex, cardIndex, false)}
	                    onCardNavigate={(cardIndex, key) => moveBoardCursor({ columnIndex, cardIndex }, key)}
	                    onCardInspectShortcut={(id) => openTask(id, { focusDetail: true })}
	                    onCardQuickMove={quickMoveTask}
	                    registerColumnHeader={(node) => {
	                      columnHeaderRefs.current[columnIndex] = node
	                    }}
	                    registerTaskCard={(taskId, node) => {
	                      taskCardRefs.current[taskId] = node
	                    }}
	                  />
	                ))}
	              </div>
            </div>

            <DragOverlay>
              {activeDragTask ? (
                <div className="bg-huddleroom-surface border border-huddleroom-primary rounded-[4px]" style={{
                  padding: '10px 12px',
                  width: 240,
                  opacity: 0.92,
                  cursor: 'grabbing',
                }}>
                  <div className="text-[13px] text-huddleroom-text-primary" style={{ marginBottom: 6, lineHeight: 1.4 }}>
                    {activeDragTask.title}
                  </div>
                  <span className="text-[11px]" style={{ color: priorityColor(activeDragTask.priority) }}>
                    {priorityLabel(activeDragTask.priority)}
                  </span>
                </div>
              ) : null}
            </DragOverlay>
          </DndContext>
        )}

	        {selectedTaskId ? (
	          <TaskDetailPanel
	            taskId={selectedTaskId}
	            projectId={pid}
	            agentMap={agentMap}
	            focusRequest={detailFocusRequest}
	            onClose={closeInspectPanel}
	            onFocusBoard={focusBoardSelection}
	            isNarrow={isNarrow}
	          />
	        ) : (
          <EmptyInspectPanel isNarrow={isNarrow} />
        )}
      </div>

      {/* Create modal */}
      <CreateTaskModal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        projectId={pid}
        agents={activeAgents}
      />
    </div>
  )
}
