import React, { act } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createMemoryRouter, MemoryRouter, RouterProvider } from 'react-router-dom'
import { descendants, mountWithTestDom, textOf } from '../../../tests/support/dom'
import type { Session, Task } from '@/lib/types'

const uiState = { activeProjectId: null as string | null }
let tasksMock: Task[] = []
let taskMock: Task | undefined
const taskByIdMock = new Map<string, Task>()
let sessionsMock: Session[] = []

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
  useTask: (projectId: string, taskId: string) => ({ data: taskByIdMock.get(`${projectId}:${taskId}`) ?? taskMock, isLoading: false }),
  useTaskSubtasks: () => ({ data: [] }),
  useTaskSessions: () => ({ data: { items: sessionsMock }, isLoading: false, isFetching: false }),
}))

describe('TasksPage', () => {
  beforeEach(() => {
    uiState.activeProjectId = null
    tasksMock = []
    taskMock = undefined
    taskByIdMock.clear()
    sessionsMock = []
  })

  it('shows an operational setup state when no project is selected', async () => {
    const { TasksPage } = await import('./TasksPage')

    const markup = renderToStaticMarkup(<MemoryRouter><TasksPage /></MemoryRouter>)

    expect(markup).toContain('Select a project to load the task board')
    expect(markup).toContain('New project')
    expect(markup).toContain('Expected prerequisites')
  })

  it('shows next actions when a project has no tasks yet', async () => {
    uiState.activeProjectId = 'project-1'
    const { TasksPage } = await import('./TasksPage')

    const markup = renderToStaticMarkup(<MemoryRouter><TasksPage /></MemoryRouter>)

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
    const { TasksPage } = await import('./TasksPage')

    const markup = renderToStaticMarkup(<MemoryRouter initialEntries={['/tasks?task=task-1']}><TasksPage /></MemoryRouter>)

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

    const markup = renderToStaticMarkup(<MemoryRouter><TasksPage /></MemoryRouter>)
    const boardMatch = markup.match(/<div aria-label="Task board columns"[^>]*>/)

    expect(boardMatch?.[0]).toMatch(/class="[^"]*overflow-x-auto[^"]*"/)
    expect(markup).toContain('min-w-[240px]')
  })

  it('SPR210 selects a task from the task query on load and browser navigation', async () => {
    uiState.activeProjectId = 'project-1'
    const boardTask: Task = {
      id: 'board-task', project_id: 'project-1', title: 'Visible board task', priority: 50,
      assigned_to: 'agent-1', status: 'blocked', created_at: '2026-07-22T10:00:00Z', updated_at: '2026-07-22T10:00:00Z',
    }
    tasksMock = [boardTask]
    taskByIdMock.set('project-1:board-task', boardTask)
    taskByIdMock.set('project-1:outside/filtered', {
      id: 'outside/filtered', project_id: 'project-1', title: 'Exact task outside board page', priority: 10,
      status: 'blocked', created_at: '2026-07-22T09:00:00Z', updated_at: '2026-07-22T09:30:00Z',
    })
    const { TasksPage } = await import('./TasksPage')
    const router = createMemoryRouter(
      [{ path: '/tasks', element: <TasksPage /> }],
      { initialEntries: ['/tasks?agent=agent-1&status=blocked&task=outside%2Ffiltered'] },
    )
    const view = await mountWithTestDom(() => {
      Object.assign(window, {
        matchMedia: () => ({
          matches: false,
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
        }),
        requestAnimationFrame: (callback: () => void) => { callback(); return 1 },
      })
      Object.assign(globalThis, {
        requestAnimationFrame: (callback: () => void) => { callback(); return 1 },
      })
      return <RouterProvider router={router} />
    }, act)
    try {
      expect(textOf(view.container)).toContain('Exact task outside board page')
      expect(textOf(view.container)).not.toContain('Inspect taskVisible board task')
      expect(router.state.location.search).toBe('?agent=agent-1&status=blocked&task=outside%2Ffiltered')

      taskByIdMock.set('project-1:outside/filtered', {
        ...taskByIdMock.get('project-1:outside/filtered')!, title: 'Refetched exact task',
      })
      await act(async () => router.navigate(
        '/tasks?agent=agent-1&status=blocked&task=outside%2Ffiltered',
        { replace: true, state: { refetched: true } },
      ))
      expect(textOf(view.container)).toContain('Refetched exact task')

      await act(async () => router.navigate('/tasks?agent=agent-1&status=blocked&task=board-task'))
      expect(textOf(view.container)).toContain('Inspect taskVisible board task')
      await act(async () => router.navigate(-1))
      expect(textOf(view.container)).toContain('Refetched exact task')
      await act(async () => router.navigate(1))
      expect(textOf(view.container)).toContain('Inspect taskVisible board task')

      const close = descendants(view.container).find((node) => node.getAttribute('aria-label') === 'Close task details for Visible board task')
      expect(close).toBeDefined()
      await act(async () => close?.click())
      expect(router.state.location.search).toBe('?agent=agent-1&status=blocked')
      expect(textOf(view.container)).toContain('Select a task')
      expect(view.activeElement?.getAttribute('data-testid')).toBe('task-card-board-task')

      const card = descendants(view.container).find((node) => node.getAttribute('data-testid') === 'task-card-board-task')
      await act(async () => card?.click())
      expect(router.state.location.search).toBe('?agent=agent-1&status=blocked&task=board-task')
      expect(textOf(view.container)).toContain('Inspect taskVisible board task')

      await act(async () => router.navigate('/tasks?agent=agent-1&status=blocked&task=missing-task'))
      expect(textOf(view.container)).toContain('Task unavailable')
      expect(textOf(view.container)).not.toContain('Inspect taskVisible board task')

      uiState.activeProjectId = 'project-2'
      await view.rerender()
      expect(textOf(view.container)).toContain('Task unavailable')
      expect(textOf(view.container)).not.toContain('Refetched exact task')
    } finally {
      view.cleanup()
      Reflect.deleteProperty(globalThis, 'requestAnimationFrame')
    }
  })
})
