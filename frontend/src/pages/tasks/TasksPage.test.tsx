import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { Session, Task } from '@/lib/types'

const navigateMock = vi.fn()
const uiState = { activeProjectId: null as string | null }
let tasksMock: Task[] = []
let taskMock: Task | undefined
let sessionsMock: Session[] = []
let selectedTaskIdMock: string | null = null
let reactStateCall = 0

vi.mock('react', async (importOriginal) => {
  const actual = await importOriginal<typeof import('react')>()
  return {
    ...actual,
    default: actual.default,
    useState: (initial: unknown) => {
      reactStateCall += 1
      if (reactStateCall === 2 && selectedTaskIdMock) {
        return actual.useState(selectedTaskIdMock)
      }
      return actual.useState(initial)
    },
  }
})

vi.mock('react-router-dom', () => ({
  useNavigate: () => navigateMock,
  useSearchParams: () => [new URLSearchParams(), vi.fn()],
}))

vi.mock('@/stores/ui', () => ({
  useUIStore: (selector: (state: { activeProjectId: string | null }) => unknown) => selector(uiState),
}))

vi.mock('@/api/agents', () => ({
  useAllAgents: () => ({ items: [{ id: 'agent-1', name: 'Agent One' }] }),
  useAllActiveAgents: () => ({ items: [] }),
}))

vi.mock('@/api/tasks', () => ({
  useAllTasks: () => ({ items: tasksMock }),
  useCopyTask: () => ({ isPending: false, mutate: vi.fn() }),
  useCreateTask: () => ({ isPending: false, mutate: vi.fn() }),
  usePatchTaskStatus: () => ({ mutate: vi.fn() }),
  useRunTask: () => ({ isPending: false, mutate: vi.fn() }),
  useResumeSession: () => ({ isPending: false, mutate: vi.fn() }),
  useTask: () => ({ data: taskMock, isLoading: false }),
  useTaskSubtasks: () => ({ data: [] }),
  useTaskSessions: () => ({ data: { items: sessionsMock }, isLoading: false, isFetching: false }),
}))

describe('TasksPage', () => {
  beforeEach(() => {
    uiState.activeProjectId = null
    tasksMock = []
    taskMock = undefined
    sessionsMock = []
    selectedTaskIdMock = null
    reactStateCall = 0
    navigateMock.mockReset()
  })

  it('shows an operational setup state when no project is selected', async () => {
    const { TasksPage } = await import('./TasksPage')

    const markup = renderToStaticMarkup(<TasksPage />)

    expect(markup).toContain('Select a project to load the task board')
    expect(markup).toContain('New project')
    expect(markup).toContain('Expected prerequisites')
  })

  it('shows next actions when a project has no tasks yet', async () => {
    uiState.activeProjectId = 'project-1'
    const { TasksPage } = await import('./TasksPage')

    const markup = renderToStaticMarkup(<TasksPage />)

    expect(markup).toContain('This project does not have tasks yet')
    expect(markup).toContain('Create first task')
    expect(markup).toContain('Review agents')
  })

  it('runs assigned backlog tasks and shows session output and errors', async () => {
    uiState.activeProjectId = 'project-1'
    const task: Task = {
      id: 'task-1',
      project_id: 'project-1',
      title: 'Backlog task',
      priority: 50,
      assigned_to: 'agent-1',
      status: 'backlog',
      created_at: '2026-07-22T10:00:00Z',
      updated_at: '2026-07-22T10:00:00Z',
    }
    tasksMock = [task]
    taskMock = task
    sessionsMock = [{
      id: 'session-1',
      project_id: 'project-1',
      task_id: 'task-1',
      agent_id: 'agent-1',
      status: 'failed',
      output: 'finished backlog work',
      error: 'but cleanup failed',
      created_at: '2026-07-22T10:01:00Z',
    }]
    selectedTaskIdMock = 'task-1'
    reactStateCall = 0
    const { TasksPage } = await import('./TasksPage')

    const markup = renderToStaticMarkup(<TasksPage />)

    expect(markup).toContain('finished backlog work')
    expect(markup).toContain('but cleanup failed')
    expect(markup).toMatch(/<button[^>]*data-testid="task-run-control"[^>]*>[\s\S]*Run task/)
    expect(markup).not.toMatch(/<button[^>]*data-testid="task-run-control"[^>]*disabled=""/)
  })

  it('renders the board with a horizontally scrollable, min-width column layout', async () => {
    uiState.activeProjectId = 'project-1'
    const task: Task = {
      id: 'task-1',
      project_id: 'project-1',
      title: 'Backlog task',
      priority: 50,
      status: 'backlog',
      created_at: '2026-07-22T10:00:00Z',
      updated_at: '2026-07-22T10:00:00Z',
    }
    tasksMock = [task]
    const { TasksPage } = await import('./TasksPage')

    const markup = renderToStaticMarkup(<TasksPage />)
    const boardMatch = markup.match(/<div aria-label="Task board columns"[^>]*>/)

    expect(boardMatch?.[0]).toMatch(/class="[^"]*overflow-x-auto[^"]*"/)
    expect(markup).toContain('min-w-[240px]')
  })
})
