import { useQuery, useQueryClient } from '@tanstack/react-query'
import { LogOut, Menu, Plus } from 'lucide-react'
import { type RefObject, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { ApiError, apiFetch, clearToken } from '@/lib/api-client'
import type { Project } from '@/lib/types'
import { STATUS_COLORS } from '@/lib/statusColors'
import { useUIStore } from '@/stores/ui'
import { useWSStore } from '@/stores/ws'
import { useCreateProject } from '@/api/projects'
import { Button, Select, Input, Textarea, UI_COLORS } from '@/components/common/uiPrimitives'
import { Dialog } from '@/components/common/Dialog'

interface TopBarProps {
  menuButtonRef?: RefObject<HTMLButtonElement | null>
}

export function TopBar({ menuButtonRef }: TopBarProps) {
  const activeProjectId = useUIStore((s) => s.activeProjectId)
  const setActiveProject = useUIStore((s) => s.setActiveProject)
  const openSidebar = useUIStore((s) => s.openSidebar)
  const sidebarOpen = useUIStore((s) => s.sidebarOpen)
  const disconnect = useWSStore((s) => s.disconnect)
  const connected = useWSStore((s) => s.connected)
  const connecting = useWSStore((s) => s.connecting)
  const retrying = useWSStore((s) => s.retrying)
  const hasConnectedOnce = useWSStore((s) => s.hasConnectedOnce)
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const createProjectOpen = useUIStore((s) => s.createProjectOpen)
  const setCreateProjectOpen = useUIStore((s) => s.setCreateProjectOpen)

  const { data: projects = [], isPending, isError, refetch } = useQuery({
    queryKey: ['projects'],
    queryFn: () => apiFetch<{ items: Project[]; next_cursor: string | null }>('/api/v1/projects')
      .then((r) => r.items),
  })

  const liveStatus = !activeProjectId
    ? { color: STATUS_COLORS.neutral, label: 'No project' }
    : connected
      ? { color: STATUS_COLORS.green, label: 'Live' }
      : connecting && !hasConnectedOnce
        ? { color: STATUS_COLORS.blue, label: 'Connecting' }
        : retrying || hasConnectedOnce
          ? { color: STATUS_COLORS.amber, label: 'Reconnecting' }
          : { color: STATUS_COLORS.neutral, label: 'Off' }

  function handleLogout() {
    queryClient.cancelQueries()
    disconnect()
    setActiveProject(null)
    clearToken()
    navigate('/login')
    queueMicrotask(() => queryClient.clear())
  }

  return (
    <header
      style={{
        height: 48,
        backgroundColor: UI_COLORS.surface,
        borderBottom: `1px solid ${UI_COLORS.border}`,
        display: 'flex',
        alignItems: 'center',
        padding: '0 16px',
        gap: 12,
        flexShrink: 0,
      }}
    >
      <Button
        ref={menuButtonRef}
        variant="ghost"
        onClick={openSidebar}
        aria-label="Toggle navigation"
        aria-expanded={sidebarOpen}
        aria-controls="huddleroom-sidebar"
        className="huddleroom-menu-button"
        style={{ width: 44, height: 44 }}
      >
        <Menu size={16} />
      </Button>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
        {isPending ? (
          <div
            aria-live="polite"
            aria-busy="true"
            style={{ display: 'flex', alignItems: 'center', gap: 8 }}
          >
            <div
              className="skeleton"
              aria-hidden="true"
              style={{ width: 176, height: 40, borderRadius: 6 }}
            />
            <span style={{ fontSize: 12, color: UI_COLORS.textMuted }}>Loading projects</span>
          </div>
        ) : isError ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{ fontSize: 12, color: STATUS_COLORS.amber }}>Projects unavailable</span>
            <button
              type="button"
              onClick={() => { void refetch() }}
              style={{
                background: UI_COLORS.surface,
                color: STATUS_COLORS.amber,
                border: `1px solid ${STATUS_COLORS.amber}`,
                borderRadius: 6,
                padding: '6px 10px',
                fontFamily: "'Inter', system-ui, sans-serif",
                fontSize: 12,
                cursor: 'pointer',
              }}
            >
              Retry
            </button>
          </div>
        ) : (
          <>
            <Select
              id="huddleroom-project-switcher"
              aria-label="Project switcher"
              value={activeProjectId ?? ''}
              onChange={(e) => setActiveProject(e.target.value || null)}
              className="py-2.5 px-3 max-w-[min(52vw,320px)]"
            >
              <option value="">Choose project</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </Select>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setCreateProjectOpen(true)}
              aria-label="Create new project"
              style={{ width: 32, height: 32, padding: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
            >
              <Plus size={16} />
            </Button>
          </>
        )}
      </div>

      <div
        aria-live="polite"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontSize: 13,
          minWidth: 0,
          padding: '3px 8px',
          borderRadius: 6,
          backgroundColor: 'transparent',
          border: '1px solid transparent',
        }}
      >
        <div
          aria-hidden="true"
          style={{
            width: 10,
            height: 10,
            borderRadius: '50%',
            backgroundColor: liveStatus.color,
            flexShrink: 0,
            transition: 'background-color 300ms',
          }}
        />
        <span style={{ color: liveStatus.color, fontWeight: 500, whiteSpace: 'nowrap' }}>{liveStatus.label}</span>
      </div>

      <div style={{ flex: 1 }} />

      <Button
        variant="ghost"
        size="lg"
        onClick={handleLogout}
        aria-label="Log out"
        style={{ fontSize: 12, gap: 6 }}
      >
        <LogOut size={12} />
        <span className="huddleroom-logout-label">Log out</span>
      </Button>

      <CreateProjectModal
        open={createProjectOpen}
        onClose={() => setCreateProjectOpen(false)}
      />
    </header>
  )
}

// ─── Create Project Modal ────────────────────────────────────────────────────────

const SANS = 'var(--huddleroom-font-sans)'

export function CreateProjectModal({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const createProject = useCreateProject()
  const setActiveProject = useUIStore((s) => s.setActiveProject)
  const [form, setForm] = useState({
    name: '',
    workspacePath: '',
    description: '',
  })
  const [workspaceError, setWorkspaceError] = useState<string | undefined>()
  const nameInput = useRef<HTMLInputElement>(null)
  const workspaceInput = useRef<HTMLInputElement>(null)

  function reset() {
    setForm({ name: '', workspacePath: '', description: '' })
    setWorkspaceError(undefined)
  }

  function handleCreate() {
    if (!form.name.trim()) {
      requestAnimationFrame(() => nameInput.current?.focus())
      return
    }
    if (!form.workspacePath.trim()) {
      setWorkspaceError('Server directory is required.')
      requestAnimationFrame(() => workspaceInput.current?.focus())
      return
    }
    createProject.mutate(
      {
        name: form.name.trim(),
        workspace_path: form.workspacePath.trim(),
        description: form.description.trim() || undefined,
      },
      {
        onSuccess: (newProject) => {
          toast.success('Project created')
          setActiveProject(newProject.id)
          reset()
          onClose()
        },
        onError: (e) => {
          const validation = e instanceof ApiError && Array.isArray(e.detail)
            ? e.detail.find((item) => typeof item === 'object' && item !== null
              && Array.isArray((item as { loc?: unknown }).loc)
              && (item as { loc: unknown[] }).loc.includes('workspace_path')) as { msg?: unknown } | undefined
            : undefined
          setWorkspaceError(typeof validation?.msg === 'string' ? validation.msg : 'Could not create the project. Try again.')
          requestAnimationFrame(() => workspaceInput.current?.focus())
        },
      },
    )
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => { if (!v) { reset(); onClose() } }}
      title="New project"
      description="Create a new project."
      size="sm"
      initialFocusRef={nameInput}
      footer={{
        primaryLabel: createProject.isPending ? 'Creating...' : 'Create project',
        primaryType: 'submit',
        formId: 'create-project-form',
        isPending: createProject.isPending,
      }}
    >
      <form id="create-project-form" onSubmit={(event) => { event.preventDefault(); handleCreate() }} style={{ fontFamily: SANS }}>
        <Input
          ref={nameInput}
          id="new-project-name"
          label="Name *"
          type="text"
          value={form.name}
          onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))}
          placeholder="project name"
        />

        <Input
          ref={workspaceInput}
          id="new-project-workspace-path"
          label="Server directory *"
          type="text"
          value={form.workspacePath}
          onChange={(e) => { setForm((p) => ({ ...p, workspacePath: e.target.value })); setWorkspaceError(undefined) }}
          placeholder="/srv/huddleroom/project"
          error={workspaceError}
          aria-invalid={!!workspaceError}
          style={{ fontFamily: 'var(--huddleroom-font-mono)' }}
        />

        <Textarea
          label="Description"
          value={form.description}
          onChange={(e) => setForm((p) => ({ ...p, description: e.target.value }))}
          placeholder="optional description"
          rows={4}
        />
      </form>
    </Dialog>
  )
}
