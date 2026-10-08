import React, { useState, useEffect, useMemo, useRef } from 'react'
import { priorityLabel, priorityColor } from '@/lib/priority'
import { useParams, useNavigate } from 'react-router-dom'
const Editor = React.lazy(() => import('@monaco-editor/react'))
import { toast } from 'sonner'
import { Plus, Pencil, Trash2, RotateCcw } from 'lucide-react'
import { useUIStore } from '@/stores/ui'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'
import { Button, Input, Textarea, Select, QueryState, UI_COLORS, PageHeader, StatusBadge, SectionLabel, ConfirmDialog } from '@/components/common/uiPrimitives'
import { Dialog } from '@/components/common/Dialog'
import { DetailHeader } from '@/components/common/DetailHeader'
import { Tabs, TabPanel } from '@/components/common/Tabs'
import { Tag } from '@/components/common/Tag'
import { EmptyState } from '@/components/common/EmptyState'
import { Banner } from '@/components/common/Banner'
import {
  useAllAgents,
  useAgent,
  useAgentSessions,
  useAgentTasks,
  useCreateAgent,
  useUpdateAgent,
  useDeleteAgent,
} from '@/api/agents'
import type { Agent, AdapterType, SessionStatus, TaskStatus } from '@/lib/types'
import { dateHeading, absolute } from '@/lib/time'

// ─── Constants / helpers ──────────────────────────────────────────────────────

const SESSION_STATUS_LABEL: Record<SessionStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

const TASK_STATUS_LABEL: Record<TaskStatus, string> = {
  backlog: 'Backlog',
  ready: 'Ready',
  in_progress: 'In progress',
  blocked: 'Blocked',
  failed: 'Failed',
  done: 'Done',
  cancelled: 'Cancelled',
}

// ─── Agent form state ─────────────────────────────────────────────────────────

interface AgentFormState {
  name: string
  role: string
  provider: string
  model: string
  adapter_type: AdapterType
  cli_runtime: string
  system_prompt: string
  description: string
  capabilities: string
  memory_enabled: boolean
  effort: string
}

const BLANK_FORM: AgentFormState = {
  name: '',
  role: '',
  provider: '',
  model: '',
  adapter_type: 'api',
  cli_runtime: '',
  system_prompt: '',
  description: '',
  capabilities: '',
  memory_enabled: false,
  effort: '',
}

const API_ADAPTER_WARNING_ID = 'agent-api-adapter-warning'
const API_ADAPTER_WARNING = 'For code development or document editing, use a CLI agent. API agents do not have a workspace toolchain.'

const CLI_RUNTIMES = [
  { value: 'claude_code', label: 'Claude Code' },
  { value: 'codex', label: 'Codex' },
  { value: 'aider', label: 'Aider' },
  { value: 'copilot', label: 'GitHub Copilot CLI' },
  { value: 'opencode', label: 'OpenCode' },
  { value: 'pi', label: 'pi (pi-agent.dev)' },
  { value: 'custom', label: 'Custom script' },
] as const

const EFFORT_RUNTIMES = ['claude_code', 'codex']
const effortLevels = (runtime: string) => (runtime === 'codex' ? ['low', 'medium', 'high', 'xhigh'] : ['low', 'medium', 'high', 'xhigh', 'max'])

function agentToForm(agent: Agent): AgentFormState {
  return {
    name:           agent.name,
    role:           agent.role,
    provider:       agent.provider,
    model:          agent.model,
    adapter_type:   agent.adapter_type,
    cli_runtime:    agent.cli_runtime ?? '',
    system_prompt:  agent.system_prompt ?? '',
    description:    agent.description ?? '',
    capabilities:   agent.capabilities.join(', '),
    memory_enabled: (agent.config?.memory_enabled as boolean | undefined) ?? false,
    effort:         (agent.config?.reasoning_effort as string | undefined) ?? '',
  }
}

function runtimeGuidance(runtime: string) {
  const guidance = {
    claude_code: ['Claude Code: unrestricted workspace access', 'Runs in the workspace with permissions bypassed. Sign in to Claude; Model is passed through; saved Claude sessions resume.', 'warning'],
    codex: ['Codex: workspace-write sandbox', 'Runs in the workspace with no approval prompts. Sign in to Codex; Model is passed through; retries start from task context.', 'info'],
    aider: ['Aider: workspace cwd', 'Runs in the workspace without an enforced sandbox. Configure provider authentication; Model is passed through; retries start from task context.', 'warning'],
    copilot: ['GitHub Copilot CLI: auto-approved workspace access', 'Runs in the workspace with tools auto-approved and without an enforced sandbox. Install and sign in to GitHub Copilot CLI; model selection and saved-session resume are passed to the installed CLI when it supports them.', 'warning'],
    opencode: ['OpenCode: auto-approved workspace access', 'Runs in the workspace with tools auto-approved and without an enforced sandbox. Install OpenCode and configure provider authentication; model selection and saved-session resume are passed to the installed CLI when it supports them.', 'warning'],
    pi: ['pi: workspace cwd', 'Runs in the workspace without an enforced sandbox. Install pi and configure provider authentication; model selection and saved-session resume are passed to the installed CLI when it supports them.', 'warning'],
    custom: ['Custom script: workspace cwd', 'Runs the configured script from the workspace without an enforced sandbox. The script controls permissions, authentication, model selection, and resume behavior.', 'warning'],
  } as const
  const [title, body, tone] = guidance[runtime as keyof typeof guidance] ?? [
    `Unknown runtime: ${runtime || 'none'}`,
    'Runs from the workspace. Verify its permissions, authentication, model handling, and resume behavior before use.',
    'warning',
  ]
  return { title, body, tone }
}

// ─── AgentFormModal ───────────────────────────────────────────────────────────

function AgentFormModal({
  open,
  onClose,
  initialData,
  mode,
}: {
  open: boolean
  onClose: () => void
  initialData?: Agent | null
  mode: 'create' | 'edit'
}) {
  const [form, setForm] = useState<AgentFormState>(BLANK_FORM)
  const createAgent = useCreateAgent()
  const updateAgent = useUpdateAgent()
  const nameRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setForm(initialData ? agentToForm(initialData) : BLANK_FORM)
    }
  }, [open, initialData])

  function set(patch: Partial<AgentFormState>) {
    setForm((p) => ({ ...p, ...patch }))
  }

  const showEffort = form.adapter_type === 'cli' && EFFORT_RUNTIMES.includes(form.cli_runtime)

  function handleSubmit() {
    if (form.adapter_type === 'cli' && !form.cli_runtime.trim()) return

    const capabilities = form.capabilities
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)

    const config: Record<string, unknown> = { ...(initialData?.config ?? {}), memory_enabled: form.memory_enabled }
    delete config.cli_runtime
    if (showEffort && effortLevels(form.cli_runtime).includes(form.effort)) config.reasoning_effort = form.effort
    else delete config.reasoning_effort

    const payload = {
      name:          form.name.trim(),
      role:          form.role.trim(),
      provider:      form.adapter_type === 'cli' ? form.cli_runtime.trim() : form.provider.trim(),
      model:         form.model.trim(),
      adapter_type:  form.adapter_type,
      cli_runtime:   form.adapter_type === 'cli' ? form.cli_runtime.trim() : null,
      system_prompt: form.system_prompt,
      description:   form.description.trim() || null,
      capabilities,
      config,
      is_active:     initialData?.is_active ?? true,
    }

    if (mode === 'create') {
      createAgent.mutate(payload as any, {
        onSuccess: () => { toast.success('Agent created'); onClose() },
        onError:   (e) => toast.error(e.message),
      })
    } else if (initialData) {
      updateAgent.mutate({ id: initialData.id, data: payload as any }, {
        onSuccess: () => { toast.success('Agent updated'); onClose() },
        onError:   (e) => toast.error(e.message),
      })
    }
  }

  const isPending = createAgent.isPending || updateAgent.isPending
  const canSubmit = Boolean(
    form.name.trim() && form.role.trim() && (form.adapter_type === 'cli' || form.provider.trim()) && form.model.trim()
    && (form.adapter_type !== 'cli' || form.cli_runtime.trim()),
  )
  const runtimeGuidanceNotice = runtimeGuidance(form.cli_runtime)
  const unknownRuntime = form.cli_runtime && !CLI_RUNTIMES.some(({ value }) => value === form.cli_runtime)

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => { if (!v) onClose() }}
      title={mode === 'create' ? 'New agent' : 'Edit agent'}
      description={
        mode === 'create'
          ? 'Create a new agent by filling in the required fields.'
          : `Edit the configuration for ${initialData?.name ?? 'this agent'}.`
      }
      size="lg"
      initialFocusRef={nameRef}
      footer={{
        primaryLabel: isPending ? 'Saving...' : mode === 'create' ? 'Create agent' : 'Save changes',
        primaryType: 'submit',
        formId: 'agent-form',
        isPending,
        primaryDisabled: !canSubmit,
      }}
    >
      <form id="agent-form" onSubmit={(event) => { event.preventDefault(); handleSubmit() }}>
            {/* Row: name + role */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 14 }}>
              <Input
                ref={nameRef}
                label="Name *"
                id="agent-name"
                value={form.name}
                onChange={(e) => set({ name: e.target.value })}
              />
              <Input
                label="Role *"
                id="agent-role"
                value={form.role}
                onChange={(e) => set({ role: e.target.value })}
              />
            </div>

            <Select
              id="agent-adapter-type"
              label="Adapter type"
              value={form.adapter_type}
              aria-describedby={form.adapter_type === 'api' ? API_ADAPTER_WARNING_ID : undefined}
              onChange={(e) => set({ adapter_type: e.target.value as AdapterType })}
            >
              <option value="api">api</option>
              <option value="cli">cli</option>
              <option value="routine">routine</option>
            </Select>

            {form.adapter_type === 'api' && (
              <div id={API_ADAPTER_WARNING_ID} style={{ marginTop: -2, marginBottom: 14 }}>
                <Banner variant="warning" title="API agent" message={API_ADAPTER_WARNING} />
              </div>
            )}

            {form.adapter_type === 'cli' && (
              <>
                <Select
                  label="CLI runtime"
                  id="agent-cli-runtime"
                  value={form.cli_runtime}
                  required
                  aria-describedby="agent-cli-runtime-guidance"
                  onChange={(e) => {
                    const newRuntime = e.target.value
                    const shouldResetEffort = form.effort && !effortLevels(newRuntime).includes(form.effort)
                    set({ cli_runtime: newRuntime, ...(shouldResetEffort ? { effort: '' } : {}) })
                  }}
                >
                  <option value="" disabled>Select a runtime</option>
                  {unknownRuntime && <option value={form.cli_runtime}>Unknown runtime: {form.cli_runtime}</option>}
                  {CLI_RUNTIMES.map(({ value, label }) => <option key={value} value={value}>{label}</option>)}
                </Select>
                <div id="agent-cli-runtime-guidance" style={{ marginTop: -2, marginBottom: 14 }}>
                  <Banner variant={runtimeGuidanceNotice.tone} title={runtimeGuidanceNotice.title} message={runtimeGuidanceNotice.body} />
                </div>
              </>
            )}

            {/* Row: provider + model */}
            {form.adapter_type !== 'cli' && (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 14 }}>
                <Input
                  label="Provider *"
                  id="agent-provider"
                  value={form.provider}
                  onChange={(e) => set({ provider: e.target.value })}
                />
                <Input
                  label="Model *"
                  id="agent-model"
                  value={form.model}
                  onChange={(e) => set({ model: e.target.value })}
                />
              </div>
            )}
            {form.adapter_type === 'cli' && (
              <div style={{ marginBottom: 14 }}>
                <Input
                  label="Model *"
                  id="agent-model"
                  value={form.model}
                  onChange={(e) => set({ model: e.target.value })}
                />
              </div>
            )}

            {showEffort && (
              <Select
                label="Effort"
                id="agent-effort"
                value={form.effort}
                onChange={(e) => set({ effort: e.target.value })}
              >
                <option value="">Default</option>
                {effortLevels(form.cli_runtime).map((l) => <option key={l} value={l}>{l}</option>)}
              </Select>
            )}

            <label className="inline-flex items-center gap-2 py-2.5 cursor-pointer" style={{ minHeight: 44 }}>
              <input
                type="checkbox"
                checked={form.memory_enabled}
                onChange={(e) => set({ memory_enabled: e.target.checked })}
                className="h-4 w-4"
                style={{ accentColor: UI_COLORS.primary }}
              />
              <span className="text-xs font-medium text-huddleroom-text-secondary" style={{ marginBottom: 0 }}>Memory enabled</span>
            </label>

            <Input
              label="Capabilities (comma-separated)"
              id="agent-capabilities"
              value={form.capabilities}
              placeholder="code, review, testing"
              onChange={(e) => set({ capabilities: e.target.value })}
            />

            <Textarea
              label="Description"
              id="agent-description"
              value={form.description}
              placeholder="optional description"
              onChange={(e) => set({ description: e.target.value })}
              style={{ height: 60, resize: 'vertical' }}
            />

            <div style={{ marginBottom: 14 }}>
              <SectionLabel className="mb-1.5">System prompt</SectionLabel>
              <div className="rounded-[3px] overflow-hidden border border-huddleroom-border">
                <React.Suspense fallback={<div className="rounded-[3px] bg-huddleroom-depth flex items-center justify-center" style={{ height: '240px' }}><span className="text-huddleroom-text-muted text-[11px]">loading editor...</span></div>}>
                  <Editor
                    height="240px"
                    defaultLanguage="markdown"
                    theme="vs-dark"
                    value={form.system_prompt}
                    onChange={(v) => set({ system_prompt: v ?? '' })}
                    options={{
                      fontSize: 12,
                      minimap: { enabled: false },
                      wordWrap: 'on',
                      lineNumbers: 'off',
                      scrollBeyondLastLine: false,
                      renderLineHighlight: 'none',
                      padding: { top: 8, bottom: 8 },
                    }}
                  />
                </React.Suspense>
              </div>
            </div>
      </form>
    </Dialog>
  )
}

// ─── AgentCard ────────────────────────────────────────────────────────────────

function AgentCard({
  agent,
  onEdit,
  onDelete,
  onReactivate,
  onClick,
}: {
  agent: Agent
  onEdit: (a: Agent) => void
  onDelete: (id: string) => void
  onReactivate: (id: string) => void
  onClick: (id: string) => void
}) {
  const caps = agent.capabilities
  const visibleCaps = caps.slice(0, 3)
  const extraCaps = caps.length > 3 ? caps.length - 3 : 0

  return (
    <div
      data-testid={`agent-card-${agent.id}`}
      className="rounded-md bg-huddleroom-surface border border-huddleroom-border"
      style={{
        transition: 'border-color 120ms cubic-bezier(0.4, 0, 0.2, 1)',
        opacity: agent.is_active ? 1 : 0.6,
      }}
      onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.borderColor = UI_COLORS.primary }}
      onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.borderColor = UI_COLORS.border }}
    >
    <div
      role="button"
      tabIndex={0}
      aria-label={`${agent.name}${agent.is_active ? '' : ' (inactive)'}`}
      style={{ padding: '14px 16px 10px', cursor: 'pointer' }}
      className="agent-card-btn rounded"
      onClick={() => onClick(agent.id)}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(agent.id) } }}
    >
      {/* Row 1: name + status + adapter */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <span className="text-sm font-bold text-huddleroom-text-primary" style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {agent.name}
        </span>
        <StatusBadge status={agent.is_active ? 'active' : 'disabled'} label={agent.is_active ? 'Active' : 'Inactive'} />
        <Tag mono className="flex-shrink-0">{agent.adapter_type}</Tag>
      </div>

      {/* Row 2: role · provider */}
      <div className="text-[13px] text-huddleroom-text-muted" style={{ marginBottom: 2 }}>
        {agent.role} · {agent.provider}
      </div>

      {/* Row 3: model */}
      <div className="font-mono text-[13px] text-huddleroom-text-muted" style={{ marginBottom: caps.length > 0 ? 8 : 0 }}>
        {agent.model}
      </div>

      {/* Capabilities */}
      {caps.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginBottom: 10 }}>
          {visibleCaps.map((c) => (
            <Tag key={c}>{c}</Tag>
          ))}
          {extraCaps > 0 && (
            <span className="text-xs text-huddleroom-text-muted" style={{ padding: '1px 5px' }}>+{extraCaps} more</span>
          )}
        </div>
      )}

    </div>

      {/* Actions — outside the navigable region to avoid nested interactive controls */}
      <div style={{ display: 'flex', gap: 6, padding: '0 16px 12px' }}>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onEdit(agent)}
          aria-label={`Edit agent ${agent.name}`}
        >
          <Pencil size={10} /> Edit
        </Button>
        {agent.is_active ? (
          <Button
            variant="danger"
            size="sm"
            onClick={() => onDelete(agent.id)}
            aria-label={`Deactivate agent ${agent.name}`}
            title="Hides this agent from new assignments. History is preserved and you can reactivate at any time."
          >
            <Trash2 size={10} /> Deactivate
          </Button>
        ) : (
          <Button
            variant="secondary"
            size="sm"
            onClick={() => onReactivate(agent.id)}
            title="Makes this agent available for new assignments again."
          >
            <RotateCcw size={10} /> Reactivate
          </Button>
        )}
      </div>
    </div>
  )
}

// ─── AgentListView ────────────────────────────────────────────────────────────

function AgentListView() {
  const navigate = useNavigate()
  const [showCreate, setShowCreate] = useState(false)
  const [editAgent, setEditAgent] = useState<Agent | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)
  const [showInactive, setShowInactive] = useState(false)

  const { items: agents, isLoading, isError, refetch, isCapped } = useAllAgents()
  const visibleAgents = useMemo(
    () => (showInactive ? agents : agents.filter((a) => a.is_active)),
    [agents, showInactive]
  )
  const deleteAgent = useDeleteAgent()
  const updateAgent = useUpdateAgent()

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center' }}>
        <PageHeader title="Agents" />
        <Button
          variant="secondary"
          onClick={() => setShowInactive((v) => !v)}
          style={{ marginLeft: 'auto' }}
        >
          {showInactive ? 'Hide inactive' : 'Show inactive'}
        </Button>
        <Button variant="primary" onClick={() => setShowCreate(true)} style={{ marginLeft: 8 }}>
          <Plus size={12} /> New agent
        </Button>
      </div>

      {/* Grid */}
      <QueryState
        query={{
          isLoading,
          isError,
          refetch,
          data: visibleAgents,
        }}
        skeleton="cards"
        skeletonCount={3}
        errorLabel="Failed to load agents"
        emptyLabel={agents.length === 0 ? 'No agents configured' : 'No active agents — toggle to show inactive'}
        emptyDetail={agents.length === 0 ? 'Create an agent to assign tasks and sessions.' : undefined}
      >
        {(displayAgents) => (
          <>
            {isCapped && (
              <div className="text-xs text-huddleroom-text-muted" style={{ textAlign: 'center', padding: '4px 0' }}>
                showing first 500 agents
              </div>
            )}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: 12,
            }}>
              {displayAgents.map((a) => (
              <AgentCard
                key={a.id}
                agent={a}
                onEdit={setEditAgent}
                onDelete={setDeleteConfirm}
                onReactivate={(id) => {
                  updateAgent.mutate({ id, data: { is_active: true } }, {
                    onSuccess: () => toast.success('Agent reactivated'),
                    onError: (e) => toast.error(e.message),
                  })
                }}
                onClick={(id) => navigate(`/agents/${id}`)}
              />
            ))}
            </div>
          </>
        )}
      </QueryState>

      {/* Create modal */}
      <AgentFormModal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        mode="create"
      />

      {/* Edit modal */}
      <AgentFormModal
        open={!!editAgent}
        onClose={() => setEditAgent(null)}
        initialData={editAgent}
        mode="edit"
      />

      {/* Delete confirmation */}
      <ConfirmDialog
        open={!!deleteConfirm}
        onOpenChange={(v) => { if (!v) setDeleteConfirm(null) }}
        title="Deactivate this agent?"
        consequence="It will no longer appear in the list. History is preserved and you can reactivate at any time."
        confirmLabel={deleteAgent.isPending ? 'Deactivating...' : 'Deactivate'}
        isPending={deleteAgent.isPending}
        onConfirm={() => {
          if (!deleteConfirm) return
          deleteAgent.mutate(deleteConfirm, {
            onSuccess: () => { toast.success('Agent deactivated'); setDeleteConfirm(null) },
            onError:   (e) => toast.error(e.message),
          })
        }}
      />
    </div>
  )
}

// ─── AgentDetailView ──────────────────────────────────────────────────────────

function AgentDetailView({ agentId }: { agentId: string }) {
  const pid = useUIStore((s) => s.activeProjectId)
  const [showEdit, setShowEdit] = useState(false)
  const [activeTab, setActiveTab] = useState<'sessions' | 'tasks'>('sessions')

  const { data: agent, isLoading, isError: agentError, refetch: agentRefetch } = useAgent(agentId)
  const { data: sessionsPage, isLoading: sessionsLoading } = useAgentSessions(agentId)
  const { data: tasksPage, isLoading: tasksLoading } = useAgentTasks(pid, agentId)

  const sessions = sessionsPage?.items ?? []
  const tasks = tasksPage?.items ?? []

  return (
    <QueryState
      query={{ isLoading, isError: agentError, refetch: agentRefetch, data: agent }}
      skeleton="rows"
      skeletonCount={4}
      errorLabel="Failed to load agent"
      emptyLabel="Agent not found"
    >
      {(agentData) => (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <DetailHeader
            backTo="/agents"
            backLabel="Agents"
            title={agentData.name}
            status={agentData.is_active ? 'active' : 'disabled'}
            statusLabel={agentData.is_active ? 'Active' : 'Inactive'}
            actions={
              <Button variant="ghost" onClick={() => setShowEdit(true)}>
                <Pencil size={10} /> Edit
              </Button>
            }
          />

          {/* Profile card */}
          <div className="rounded-[4px] bg-huddleroom-surface border border-huddleroom-border" style={{ padding: 20 }}>
            <div className="flex flex-wrap items-center gap-3 text-[13px] text-huddleroom-text-muted" style={{ marginBottom: 6 }}>
              <span>{agentData.role}</span>
              <span className="text-huddleroom-border-strong">|</span>
              <span>{agentData.provider}</span>
              <span className="text-huddleroom-border-strong">|</span>
              <span className="font-mono text-[13px]">{agentData.model}</span>
              <span className="text-huddleroom-border-strong">|</span>
              <Tag mono>{agentData.adapter_type}</Tag>
            </div>

            <div className="text-xs text-huddleroom-text-muted">
              created {dateHeading(agentData.created_at)}
            </div>

            {agentData.description && (
              <div style={{ marginTop: 12 }}>
                <SectionLabel>Description</SectionLabel>
                <div className="text-[13px] text-huddleroom-text-muted" style={{ lineHeight: 1.5 }}>{agentData.description}</div>
              </div>
            )}

            {agentData.capabilities.length > 0 && (
              <div style={{ marginTop: 12, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {agentData.capabilities.map((c) => (
                  <Tag key={c}>{c}</Tag>
                ))}
              </div>
            )}
          </div>

          {/* Tabs */}
          <div>
            <Tabs
              idPrefix="agent-detail"
              activeId={activeTab}
              onChange={(id) => setActiveTab(id as 'sessions' | 'tasks')}
              tabs={[
                { id: 'sessions', label: `Sessions${sessions.length > 0 ? ` (${sessions.length})` : ''}` },
                { id: 'tasks', label: `Tasks${tasks.length > 0 ? ` (${tasks.length})` : ''}` },
              ]}
            />

            <TabPanel tabId="sessions" activeId={activeTab} idPrefix="agent-detail">
              <div className="rounded-b-[6px] bg-huddleroom-surface border border-huddleroom-border border-t-0" style={{ minHeight: 120 }}>
                {sessionsLoading ? (
                  <div className="text-[13px] text-huddleroom-text-muted" style={{ padding: 16 }}>Loading...</div>
                ) : sessions.length === 0 ? (
                  <EmptyState
                    title="No sessions yet"
                    body="Sessions appear here when this agent starts running work."
                    className="px-4"
                  />
                ) : (
                  sessions.map((s) => (
                    <div key={s.id} className="text-[13px] border-b border-huddleroom-depth" style={{
                      display: 'flex', gap: 12, padding: '8px 12px',
                      alignItems: 'center',
                    }}>
                      <StatusBadge status={s.status} label={SESSION_STATUS_LABEL[s.status] ?? s.status} className="w-[70px] flex-shrink-0" />
                      <span className="text-huddleroom-text-muted" style={{ flex: 1 }}>{s.task_id ? `task: ${s.task_id.slice(0, 8)}…` : 'no task'}</span>
                      <span className="text-huddleroom-text-muted" style={{ flexShrink: 0 }}>{absolute(s.started_at)}</span>
                    </div>
                  ))
                )}
              </div>
            </TabPanel>

            <TabPanel tabId="tasks" activeId={activeTab} idPrefix="agent-detail">
              <div className="rounded-b-[6px] bg-huddleroom-surface border border-huddleroom-border border-t-0" style={{ minHeight: 120 }}>
                {!pid ? (
                  <div className="text-[13px] text-huddleroom-text-muted" style={{ padding: 16 }}>
                    select a project to view tasks
                  </div>
                ) : tasksLoading ? (
                  <div className="text-[13px] text-huddleroom-text-muted" style={{ padding: 16 }}>Loading...</div>
                ) : tasks.length === 0 ? (
                  <EmptyState
                    title="No tasks"
                    body="Tasks appear here when this agent is assigned work."
                    className="px-4"
                  />
                ) : (
                  tasks.map((t) => (
                    <div key={t.id} className="text-[13px] border-b border-huddleroom-depth" style={{
                      display: 'flex', gap: 12, padding: '8px 12px',
                      alignItems: 'center',
                    }}>
                      <StatusBadge status={t.status} label={TASK_STATUS_LABEL[t.status] ?? t.status} className="w-[70px] flex-shrink-0" />
                      <span className="text-huddleroom-text-primary" style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{t.title}</span>
                      <span style={{ color: priorityColor(t.priority), flexShrink: 0 }}>{priorityLabel(t.priority)}</span>
                    </div>
                  ))
                )}
              </div>
            </TabPanel>
          </div>

          {/* Edit modal */}
          <AgentFormModal
            open={showEdit}
            onClose={() => setShowEdit(false)}
            initialData={agentData}
            mode="edit"
          />
        </div>
      )}
    </QueryState>
  )
}

// ─── AgentsPage ───────────────────────────────────────────────────────────────

export function AgentsPage() {
  useDocumentTitle('Agents')
  const { agentId } = useParams<{ agentId?: string }>()
  if (agentId) return <AgentDetailView agentId={agentId} />
  return <AgentListView />
}
