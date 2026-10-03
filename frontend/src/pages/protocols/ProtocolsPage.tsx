import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
const Editor = React.lazy(() => import('@monaco-editor/react'))
const ProtocolGraph = React.lazy(() => import('./ProtocolGraph').then((m) => ({ default: m.ProtocolGraph })))
import yaml from 'js-yaml'
import { toast } from 'sonner'
import { Plus, ChevronDown, ChevronRight, Play, Pause, RotateCcw, X, Zap } from 'lucide-react'
import { Button, Input, Textarea, Select, QueryState, UI_COLORS, PageHeader, ConfirmDialog, StatusBadge, SectionLabel, StructuralLabel } from '@/components/common/uiPrimitives'
import { Dialog } from '@/components/common/Dialog'
import { DetailHeader } from '@/components/common/DetailHeader'
import { Tag } from '@/components/common/Tag'
import { EmptyState } from '@/components/common/EmptyState'
import { STATUS_COLORS } from '@/lib/statusColors'
import { useUIStore } from '@/stores/ui'
import {
  useProtocols, useProtocol, useProtocolsAll,
  useCreateProtocol, useUpdateProtocol, useDeactivateProtocol, useActivateProtocol,
  useProtocolInstances, useProtocolInstancesAll,
  useProtocolInstanceTransitions,
  usePauseInstance, useResumeInstance, useAbandonInstance,
} from '@/api/protocols'
import type { Protocol, ProtocolInstance, ProtocolTransition } from '@/lib/types'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'
import { absolute } from '@/lib/time'

// ─── Constants / helpers ──────────────────────────────────────────────────────

const MONO = 'var(--huddleroom-font-mono)'

function readableStatus(status: string): string {
  return status.replace(/_/g, ' ')
}

function instanceStatusAnnouncement(status: string): string {
  if (status === 'paused') {
    return 'Protocol instance status paused. Live execution is stopped until an operator resumes it.'
  }
  if (status === 'active') {
    return 'Protocol instance status active. Live execution is currently running.'
  }
  return `Protocol instance status ${readableStatus(status)}.`
}

// ─── Shared styles ────────────────────────────────────────────────────────────

// ─── Protocol form state ──────────────────────────────────────────────────────

interface ProtocolFormState {
  name: string
  version: string
  description: string
  definition: string
  triggers: string
}

const BLANK_FORM: ProtocolFormState = {
  name: '',
  version: '1.0.0',
  description: '',
  definition: `initial_state: pending
states:
  pending:
    type: initial
  active: {}
  completed:
    type: terminal
  failed:
    type: terminal
transitions:
  - from: pending
    to: active
    event: start
  - from: active
    to: completed
    event: done
  - from: active
    to: failed
    event: error
`,
  triggers: '',
}

function protocolToForm(protocol: Protocol): ProtocolFormState {
  return {
    name: protocol.name,
    version: protocol.version,
    description: protocol.description ?? '',
    definition: (() => {
      try {
        return typeof protocol.definition === 'object' && protocol.definition !== null
          ? yaml.dump(protocol.definition)
          : '';
      } catch {
        toast.error('Protocol definition is invalid and cannot be displayed')
        return '';
      }
    })(),
    triggers: (protocol.triggers ?? []).map((t) =>
      typeof t === 'string' ? t : String((t as Record<string, unknown>).event_type ?? JSON.stringify(t))
    ).join(', '),
  }
}

// ─── ProtocolFormModal ────────────────────────────────────────────────────────

function ProtocolFormModal({
  open,
  onClose,
  initialData,
  mode,
  pid,
}: {
  open: boolean
  onClose: () => void
  initialData?: Protocol | null
  mode: 'create' | 'edit'
  pid: string | null
}) {
  const [form, setForm] = useState<ProtocolFormState>(BLANK_FORM)
  const createProtocol = useCreateProtocol(pid)
  const updateProtocol = useUpdateProtocol(pid)

  useEffect(() => {
    if (open) {
      setForm(initialData ? protocolToForm(initialData) : BLANK_FORM)
    }
  }, [open, initialData])

  function set(patch: Partial<ProtocolFormState>) {
    setForm((p) => ({ ...p, ...patch }))
  }

  function handleSubmit() {
    try {
      const definition = yaml.load(form.definition) as Record<string, unknown>
      const triggers = form.triggers
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)

      const payload = {
        name: form.name.trim(),
        version: form.version.trim(),
        description: form.description.trim() || null,
        definition,
        triggers,
      }

      if (mode === 'create') {
        createProtocol.mutate({
          name: payload.name,
          version: payload.version,
          description: payload.description ?? undefined,
          definition: payload.definition,
          triggers: payload.triggers,
        }, {
          onSuccess: () => { toast.success('Protocol created'); onClose() },
          onError: (e) => toast.error(e.message),
        })
      } else if (initialData) {
        updateProtocol.mutate({ protocolId: initialData.id, data: {
          name: payload.name,
          version: payload.version,
          description: payload.description ?? undefined,
          definition: payload.definition,
          triggers: payload.triggers,
        } }, {
          onSuccess: () => { toast.success('Protocol updated'); onClose() },
          onError: (e) => toast.error(e.message),
        })
      }
    } catch (e) {
      toast.error(`Invalid YAML: ${e instanceof Error ? e.message : 'unknown error'}`)
    }
  }

  const isPending = createProtocol.isPending || updateProtocol.isPending
  const canSubmit = form.name.trim() && form.version.trim()

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => { if (!v) onClose() }}
      title={mode === 'create' ? 'New protocol' : 'Edit protocol'}
      description={
        mode === 'create'
          ? 'Create a new protocol with name, version, YAML definition, and triggers.'
          : `Edit the definition, version, and triggers for ${initialData?.name ?? 'this protocol'}.`
      }
      size="md"
      footer={{
        primaryLabel: isPending ? 'Saving...' : mode === 'create' ? 'Create protocol' : 'Save changes',
        primaryType: 'submit',
        formId: 'protocol-form',
        isPending,
        primaryDisabled: !canSubmit,
      }}
    >
      <form id="protocol-form" onSubmit={(e) => { e.preventDefault(); handleSubmit() }}>
            {/* Row: name + version */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 14 }}>
              <div>
                <Input id="protocol-name" label="Name *" value={form.name} placeholder="protocol name"
                  onChange={(e) => set({ name: e.target.value })}
                />
              </div>
              <div>
                <Input id="protocol-version" label="Version *" value={form.version} placeholder="1.0.0"
                  onChange={(e) => set({ version: e.target.value })}
                />
              </div>
            </div>

            <Textarea id="protocol-description" label="Description" rows={3}
              value={form.description} placeholder="optional description"
              onChange={(e) => set({ description: e.target.value })}
            />

            <div style={{ marginBottom: 14 }}>
              <label className="block text-huddleroom-text-muted text-[11px]" style={{ marginBottom: 4 }}>Definition (YAML)</label>
              <div className="border border-huddleroom-border rounded-[3px] overflow-hidden">
                <React.Suspense fallback={<div className="bg-huddleroom-depth rounded-[3px]" style={{ height: '240px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><span className="text-huddleroom-text-muted text-[11px]">loading editor...</span></div>}>
                  <Editor
                    height="240px"
                    defaultLanguage="yaml"
                    theme="vs"
                    value={form.definition}
                    onChange={(v) => set({ definition: v ?? '' })}
                    options={{
                      fontSize: 12,
                      fontFamily: MONO,
                      minimap: { enabled: false },
                      wordWrap: 'on',
                      lineNumbers: 'on',
                      scrollBeyondLastLine: false,
                      padding: { top: 8, bottom: 8 },
                    }}
                  />
                </React.Suspense>
              </div>
            </div>

            <Input id="protocol-triggers" label="Triggers (comma-separated)" value={form.triggers} placeholder="start, stop"
              onChange={(e) => set({ triggers: e.target.value })}
            />
      </form>
    </Dialog>
  )
}

// ─── Parse definition to get states and transitions ───────────────────────────

export interface ParsedStateInfo {
  id: string
  isInitial: boolean
  isTerminal: boolean
}

export interface ParsedFlow {
  states: Map<string, ParsedStateInfo>
  transitions: Array<{ from: string; to: string; event?: string }>
  initialState?: string
}

function parseProtocolDefinition(definition: Record<string, unknown>): ParsedFlow {
  const states = new Map<string, ParsedStateInfo>()
  const transitions: Array<{ from: string; to: string; event?: string }> = []
  let initialState: string | undefined

  // Parse states
  const statesObj = definition.states as Record<string, unknown> | undefined
  if (statesObj && typeof statesObj === 'object') {
    Object.entries(statesObj).forEach(([name, config]) => {
      const cfg = config as Record<string, unknown> | undefined
      const type = (cfg?.type as string | undefined) ?? ''

      states.set(name, {
        id: name,
        isInitial: type === 'initial',
        isTerminal: type === 'terminal' || type === 'end' || (cfg?.is_terminal as boolean | undefined) === true,
      })
    })
  }

  // Parse initial state
  const initialStateVal = definition.initial_state as string | undefined
  if (initialStateVal) {
    initialState = initialStateVal
  } else {
    // Look for initial state by type
    states.forEach((info, name) => {
      if (info.isInitial) initialState = name
    })
  }

  // Parse transitions
  const transitionsArr = definition.transitions as Array<Record<string, unknown>> | undefined
  if (Array.isArray(transitionsArr)) {
    transitionsArr.forEach((t) => {
      const from = (t.from as string | undefined) ?? (t.from_state as string | undefined)
      const to = (t.to as string | undefined) ?? (t.to_state as string | undefined)
      const eventRaw = t.event ?? t.event_type
      const event: string | undefined = typeof eventRaw === 'string'
        ? eventRaw
        : typeof eventRaw === 'object' && eventRaw !== null
          ? String((eventRaw as Record<string, unknown>).event_type ?? '')
          : undefined
      if (from && to) {
        // Ensure states exist
        if (!states.has(from)) {
          states.set(from, { id: from, isInitial: false, isTerminal: false })
        }
        if (!states.has(to)) {
          states.set(to, { id: to, isInitial: false, isTerminal: false })
        }
        transitions.push({ from, to, event })
      }
    })
  }

  return { states, transitions, initialState }
}

// ─── ProtocolListView ─────────────────────────────────────────────────────────

function ProtocolListView() {
  const navigate = useNavigate()
  const pid = useUIStore((s) => s.activeProjectId)
  const [showCreate, setShowCreate] = useState(false)
  const [includeInactive, setIncludeInactive] = useState(false)
  const [search, setSearch] = useState('')
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  const { items: protocols, isLoading, isError: protocolsError, refetch: protocolsRefetch } = useProtocolsAll(pid, includeInactive)
  const deactivateProtocol = useDeactivateProtocol(pid)
  const activateProtocol = useActivateProtocol(pid)
  const filteredProtocols = useMemo(() => {
    const query = search.trim().toLowerCase()
    if (!query) {
      return protocols
    }
    return protocols.filter((protocol) => {
      const haystack = `${protocol.name} ${protocol.version} ${protocol.description ?? ''}`.toLowerCase()
      return haystack.includes(query)
    })
  }, [protocols, search])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <PageHeader title="Protocols" />
        <Button
          variant="primary"
          onClick={() => setShowCreate(true)}
        >
          <Plus size={12} /> New protocol
        </Button>
      </div>

      {/* Include inactive toggle */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <label className="inline-flex items-center gap-2 py-2.5 cursor-pointer" style={{ minHeight: 44 }}>
          <input
            type="checkbox"
            className="h-4 w-4"
            checked={includeInactive}
            onChange={(e) => setIncludeInactive(e.target.checked)}
            style={{ accentColor: UI_COLORS.primary }}
          />
          Include inactive
        </label>
        <Input
          type="text"
          aria-label="Search protocols"
          placeholder="Search protocols"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ maxWidth: 240 }}
        />
      </div>

      {/* Protocol list */}
      <QueryState
        query={{
          isLoading,
          isError: protocolsError,
          refetch: protocolsRefetch,
          data: filteredProtocols,
        }}
        skeleton="rows"
        skeletonCount={3}
        errorLabel="Failed to load protocols"
        emptyLabel={search.trim() ? 'No protocols match this search' : 'No protocols'}
        emptyDetail={search.trim() ? undefined : 'Instances appear here when protocols are triggered.'}
      >
        {(displayProtocols) => (
          <div className="border border-huddleroom-border rounded-md overflow-hidden bg-huddleroom-surface">
            {displayProtocols.map((p, idx) => {
              const isGlobal = !p.project_id
              return (
                <div key={p.id}
                  role="button"
                  tabIndex={0}
                  data-testid={`protocol-card-${p.id}`}
                  className={idx < displayProtocols.length - 1 ? 'border-b border-huddleroom-depth' : ''}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 16, padding: '12px 16px', minHeight: 32,
                    transition: 'background 120ms',
                    cursor: 'pointer',
                  }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.background = UI_COLORS.depth }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.background = UI_COLORS.surface }}
                  onClick={() => navigate(`/protocols/${p.id}`)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); navigate(`/protocols/${p.id}`) } }}
                >
                  <StatusBadge
                    status={p.is_active ? 'active' : 'disabled'}
                    label={p.is_active ? 'Active' : 'Inactive'}
                    className="flex-shrink-0"
                  />

                  {/* Name & version */}
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="font-mono text-xs font-bold text-huddleroom-text-primary" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {p.name}
                    </div>
                    <Tag mono className="mt-0.5">v{p.version}</Tag>
                  </div>

                  {/* Scope badge */}
                  <Tag className="flex-shrink-0">{isGlobal ? 'global' : 'project'}</Tag>

                  {/* Actions */}
                  <div style={{ display: 'flex', gap: 4, flexShrink: 0 }} onClick={(e) => e.stopPropagation()}>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => navigate(`/protocols/${p.id}`)}
                    >
                      Edit
                    </Button>
                    {!isGlobal && (
                      p.is_active ? (
                        <Button
                          variant="danger"
                          size="sm"
                          onClick={() => setDeleteConfirm(p.id)}
                          className="text-[11px]"
                          style={{ padding: '3px 8px' }}
                        >
                          Deactivate
                        </Button>
                      ) : (
                        <Button
                          variant="secondary"
                          size="sm"
                          onClick={() => {
                            activateProtocol.mutate(p.id, {
                              onSuccess: () => toast.success('Protocol reactivated'),
                              onError: (e) => toast.error(e.message),
                            })
                          }}
                          className="text-[11px] text-huddleroom-status-green"
                          style={{ padding: '3px 8px' }}
                        >
                          Reactivate
                        </Button>
                      )
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </QueryState>

      {/* Create modal */}
      <ProtocolFormModal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        mode="create"
        pid={pid}
      />

      {/* Deactivate confirmation */}
      <ConfirmDialog
        open={!!deleteConfirm}
        onOpenChange={(v) => { if (!v) setDeleteConfirm(null) }}
        title="Deactivate this protocol?"
        consequence="It will no longer be available to agents."
        confirmLabel={deactivateProtocol.isPending ? 'Deactivating...' : 'Deactivate'}
        isPending={deactivateProtocol.isPending}
        onConfirm={() => {
          if (!deleteConfirm) return
          deactivateProtocol.mutate(deleteConfirm, {
            onSuccess: () => { toast.success('Protocol deactivated'); setDeleteConfirm(null) },
            onError: (e) => toast.error(e.message),
          })
        }}
      />
    </div>
  )
}

// ─── ProtocolDetailView ───────────────────────────────────────────────────────

function ProtocolDetailView({ protocolId }: { protocolId: string }) {
  const pid = useUIStore((s) => s.activeProjectId)
  const [showEdit, setShowEdit] = useState(false)
  const [editDefinition, setEditDefinition] = useState('')
  const [selectedInstanceId, setSelectedInstanceId] = useState<string | null>(null)
  const [searchParams, setSearchParams] = useSearchParams()
  const statusFilter = searchParams.get('instanceStatus') ?? 'all'
  const [expandedInstance, setExpandedInstance] = useState<string | null>(null)
  const [abandonConfirm, setAbandonConfirm] = useState<string | null>(null)
  const [deactivateConfirm, setDeactivateConfirm] = useState(false)

  const { data: protocol, isLoading, isError: protocolError, refetch: protocolRefetch } = useProtocol(pid, protocolId)
  const { items: instances, isLoading: instancesLoading } = useProtocolInstancesAll(pid, {
    protocol_id: protocolId,
    status: statusFilter === 'all' ? undefined : statusFilter,
  })

  const updateProtocol = useUpdateProtocol(pid)
  const deactivateProtocol = useDeactivateProtocol(pid)
  const activateProtocol = useActivateProtocol(pid)
  const pauseInstance = usePauseInstance(pid)
  const resumeInstance = useResumeInstance(pid)
  const abandonInstance = useAbandonInstance(pid)

  useEffect(() => {
    if (protocol?.definition) {
      try {
        setEditDefinition(yaml.dump(protocol.definition))
      } catch {
        toast.error('Protocol definition is invalid and cannot be displayed')
        setEditDefinition('')
      }
    }
  }, [protocol])

  const parsed = useMemo(
    () => parseProtocolDefinition(protocol?.definition ?? {}),
    [protocol?.definition]
  )
  const currentInst = selectedInstanceId ? instances.find((i) => i.id === selectedInstanceId) : null

  return (
    <QueryState
      query={{ isLoading, isError: protocolError, refetch: protocolRefetch, data: protocol }}
      skeleton="rows"
      skeletonCount={4}
      errorLabel="Failed to load protocol"
      emptyLabel="Protocol not found"
    >
      {(protocolData) => {
        const isGlobal = !protocolData.project_id

        function handleSaveDefinition() {
          try {
            const definition = yaml.load(editDefinition) as Record<string, unknown>
            updateProtocol.mutate(
              { protocolId, data: { definition } },
              {
                onSuccess: () => { toast.success('Definition saved'); setShowEdit(false) },
                onError: (e) => toast.error(e.message),
              }
            )
          } catch (e) {
            toast.error(`Invalid YAML: ${e instanceof Error ? e.message : 'unknown error'}`)
          }
        }

        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Header */}
            <DetailHeader
              backTo="/protocols"
              backLabel="Protocols"
              title={protocolData.name}
              titleClassName="font-mono"
              status={protocolData.is_active ? 'active' : 'disabled'}
              statusLabel={protocolData.is_active ? 'Active' : 'Inactive'}
              actions={
                <>
                  <Tag mono>v{protocolData.version}</Tag>
                  <Button variant="ghost" onClick={() => setShowEdit(true)}>
                    Edit
                  </Button>
                  {!isGlobal && (
                    protocolData.is_active ? (
                      <Button
                        variant="danger"
                        onClick={() => setDeactivateConfirm(true)}
                      >
                        Deactivate
                      </Button>
                    ) : (
                      <Button
                        variant="secondary"
                        onClick={() => activateProtocol.mutate(protocolData.id, {
                          onSuccess: () => toast.success('Protocol reactivated'),
                          onError: (e) => toast.error(e.message),
                        })}
                        className="text-huddleroom-status-green"
                      >
                        Reactivate
                      </Button>
                    )
                  )}
                </>
              }
            />

            {/* Description + triggers card */}
            {(protocolData.description || protocolData.triggers.length > 0) && (
              <div className="bg-huddleroom-surface border border-huddleroom-border rounded-md" style={{ padding: 20 }}>
                {protocolData.description && (
                  <div>
                    <SectionLabel>Description</SectionLabel>
                    <div className="text-[13px] text-huddleroom-text-muted" style={{ lineHeight: 1.5 }}>{protocolData.description}</div>
                  </div>
                )}

                {protocolData.triggers.length > 0 && (
                  <div style={{ marginTop: protocolData.description ? 12 : 0, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {protocolData.triggers.map((t, i) => {
                      const label = typeof t === 'string' ? t : String((t as Record<string, unknown>).event_type ?? JSON.stringify(t))
                      return (
                        <Tag key={i} mono>{label}</Tag>
                      )
                    })}
                  </div>
                )}
              </div>
            )}

            {/* Definition editor + diagram (2-column split) */}
            <div className="protocol-detail-grid">
              {/* Left: Editor */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div data-testid="protocol-definition-editor" className="border border-huddleroom-border rounded-[3px] overflow-hidden">
                  <React.Suspense fallback={<div className="bg-huddleroom-depth rounded-[3px]" style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><span className="text-huddleroom-text-muted text-[11px]">loading editor...</span></div>}>
                    <Editor
                      height="400px"
                      defaultLanguage="yaml"
                      theme="vs"
                      value={editDefinition}
                      onChange={(v) => setEditDefinition(v ?? '')}
                      options={{
                        fontSize: 12,
                        fontFamily: MONO,
                        minimap: { enabled: false },
                        wordWrap: 'on',
                        lineNumbers: 'on',
                        scrollBeyondLastLine: false,
                        padding: { top: 8, bottom: 8 },
                        readOnly: isGlobal,
                      }}
                    />
                  </React.Suspense>
                </div>
                {!isGlobal && (
                  <Button
                    variant="primary"
                    onClick={handleSaveDefinition}
                    disabled={updateProtocol.isPending}
                    className="self-start"
                  >
                    {updateProtocol.isPending ? 'Saving...' : 'Save definition'}
                  </Button>
                )}
              </div>

              {/* Right: Diagram */}
              <div data-testid="protocol-graph" className="border border-huddleroom-border rounded-md bg-huddleroom-depth" style={{
                height: 440,
                overflow: 'hidden', position: 'relative',
              }}>
                {parsed.states.size === 0 ? (
                  <div className="text-huddleroom-text-muted text-xs" style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%',
                  }}>
                    no state diagram available
                  </div>
                ) : (
                  <React.Suspense fallback={<div className="text-huddleroom-text-muted text-xs" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>Loading diagram…</div>}>
                    <ProtocolGraph parsed={parsed} currentState={currentInst?.current_state} />
                  </React.Suspense>
                )}
              </div>
            </div>

            {/* Instance list */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <StructuralLabel>Instances</StructuralLabel>
                <Select
                  aria-label="Filter protocol instances by status"
                  value={statusFilter}
                  onChange={(e) => {
                    setSearchParams((prev) => {
                      const next = new URLSearchParams(prev)
                      if (e.target.value && e.target.value !== 'all') next.set('instanceStatus', e.target.value)
                      else next.delete('instanceStatus')
                      return next
                    }, { replace: true })
                  }}
                  className="w-[120px]"
                >
                  <option value="all">all statuses</option>
                  <option value="active">active</option>
                  <option value="paused">paused</option>
                  <option value="completed">completed</option>
                  <option value="failed">failed</option>
                </Select>
              </div>

              {instancesLoading ? (
                <div className="text-huddleroom-text-muted text-xs bg-huddleroom-surface rounded-md border border-huddleroom-border" style={{ padding: 16 }}>
                  loading instances...
                </div>
              ) : instances.length === 0 ? (
                <div className="bg-huddleroom-surface rounded-md border border-huddleroom-border">
                  <EmptyState
                    title="No instances yet"
                    body="Runs of this protocol will appear here when it is triggered."
                    className="px-4"
                  />
                </div>
              ) : (
                <>
                {instances.length > 50 && (
                  <div className="text-[11px] text-huddleroom-text-muted" style={{ marginBottom: 4 }}>
                    showing 50 of {instances.length}
                  </div>
                )}
                <div data-testid="protocol-instance-list" className="border border-huddleroom-border rounded-md overflow-hidden bg-huddleroom-surface">
                  {instances.slice(0, 50).map((inst, idx) => (
                    <div key={inst.id}>
                      <div
                        role="button"
                        tabIndex={0}
                        className={idx < Math.min(instances.length, 50) - 1 ? 'border-b border-huddleroom-depth' : ''}
                        style={{
                          display: 'grid', gridTemplateColumns: 'auto 1fr auto auto auto 72px', gap: 12, padding: '12px 16px', minHeight: 32,
                          alignItems: 'center',
                          cursor: 'pointer',
                          transition: 'background 120ms',
                        }}
                        onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.background = UI_COLORS.depth }}
                        onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.background = UI_COLORS.surface }}
                        onClick={() => {
                          setSelectedInstanceId(inst.id === selectedInstanceId ? null : inst.id)
                          setExpandedInstance(inst.id === expandedInstance ? null : inst.id)
                        }}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelectedInstanceId(inst.id === selectedInstanceId ? null : inst.id); setExpandedInstance(inst.id === expandedInstance ? null : inst.id) } }}
                      >
                        <ChevronRight size={12} className="text-huddleroom-text-muted" style={{
                          transform: expandedInstance === inst.id ? 'rotate(90deg)' : 'rotate(0deg)',
                          transition: 'transform 120ms',
                        }} />
                        <div className="text-[11px] text-huddleroom-text-muted font-mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis' }} title={inst.id}>
                          {inst.id.slice(0, 12)}…
                        </div>
                        <span className="text-[11px] font-semibold rounded-[2px] text-huddleroom-text-muted font-mono" style={{
                          padding: '2px 6px', background: UI_COLORS.border,
                        }}>
                          {inst.current_state}
                        </span>
                        <span
                          className={`status-${inst.status} text-[11px] font-mono`}
                          style={{ minWidth: 60 }}
                          aria-label={instanceStatusAnnouncement(inst.status)}
                        >
                          {inst.status}
                        </span>
                        <span className="text-[11px] text-huddleroom-text-muted font-mono">
                          {absolute(inst.started_at)}
                        </span>
                        {(inst.status === 'active' || inst.status === 'paused') && (
                          <div style={{ display: 'flex', gap: 4 }} onClick={(e) => e.stopPropagation()}>
                            {inst.status === 'active' ? (
                              <Button
                                variant="secondary"
                                size="sm"
                                aria-label={`Pause protocol instance ${inst.id.slice(0, 12)}`}
                                title="Pause instance"
                                onClick={() => {
                                  pauseInstance.mutate(inst.id, {
                                    onSuccess: () => toast.success('Instance paused'),
                                    onError: (e) => toast.error(e.message),
                                  })
                                }}
                                className="text-[10px]"
                                style={{ minHeight: 32, padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 2 }}
                              >
                                <Pause size={12} />
                              </Button>
                            ) : (
                              <Button
                                variant="secondary"
                                size="sm"
                                aria-label={`Resume protocol instance ${inst.id.slice(0, 12)}`}
                                title="Resume instance"
                                onClick={() => {
                                  resumeInstance.mutate(inst.id, {
                                    onSuccess: () => toast.success('Instance resumed'),
                                    onError: (e) => toast.error(e.message),
                                  })
                                }}
                                className="text-[10px]"
                                style={{ minHeight: 32, padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 2 }}
                              >
                                <Play size={12} />
                              </Button>
                            )}
                            <Button
                              variant="danger"
                              size="sm"
                              aria-label={`Abandon protocol instance ${inst.id.slice(0, 12)}`}
                              title="Abandon instance"
                              onClick={() => setAbandonConfirm(inst.id)}
                              className="text-[10px]"
                              style={{ minHeight: 32, padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 2 }}
                            >
                              <X size={12} />
                            </Button>
                          </div>
                        )}
                      </div>

                      {/* Expanded transition history */}
                      {expandedInstance === inst.id && (
                        <InstanceTransitionHistory projectId={pid} instanceId={inst.id} />
                      )}
                    </div>
                  ))}
                </div>
                </>
              )}
            </div>

            {/* Edit modal */}
            <ProtocolFormModal
              open={showEdit}
              onClose={() => setShowEdit(false)}
              initialData={protocolData}
              mode="edit"
              pid={pid}
            />

            {/* Deactivate protocol confirm dialog */}
            <ConfirmDialog
              open={deactivateConfirm}
              onOpenChange={setDeactivateConfirm}
              title="Deactivate this protocol?"
              consequence="It will no longer be available to agents."
              confirmLabel="Deactivate"
              onConfirm={() => {
                deactivateProtocol.mutate(protocolData.id, {
                  onSuccess: () => { toast.success('Protocol deactivated'); setDeactivateConfirm(false) },
                  onError: (e) => toast.error(e.message),
                })
              }}
              isPending={deactivateProtocol.isPending}
            />

            {/* Abandon instance confirm dialog */}
            <ConfirmDialog
              open={abandonConfirm !== null}
              onOpenChange={(open) => { if (!open) setAbandonConfirm(null) }}
              title="Abandon this instance?"
              consequence="This terminates the running protocol instance. Progress will be lost."
              confirmLabel="Abandon"
              onConfirm={() => {
                if (abandonConfirm) {
                  abandonInstance.mutate(abandonConfirm, {
                    onSuccess: () => { toast.success('Instance abandoned'); setAbandonConfirm(null) },
                    onError: (e) => toast.error(e.message),
                  })
                }
              }}
              isPending={abandonInstance.isPending}
            />
          </div>
        )
      }}
    </QueryState>
  )
}

// ─── InstanceTransitionHistory ───────────────────────────────────────────────

function InstanceTransitionHistory({ projectId, instanceId }: { projectId: string | null; instanceId: string }) {
  const { data: transitions } = useProtocolInstanceTransitions(projectId, instanceId)

  return (
    <div className="bg-huddleroom-depth border-t border-huddleroom-border" style={{ padding: '8px 16px' }}>
      {!transitions || transitions.length === 0 ? (
        <div className="text-[11px] text-huddleroom-text-muted font-mono">no transitions</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {transitions.map((t) => {
            const label = t.transition_name ?? t.event_type
            const timestamp = t.transitioned_at ?? t.created_at
            // fallback to deprecated fields during deprecation window — see schemas/protocol.py
            return (
              <div key={t.id} className="text-[11px] text-huddleroom-text-muted font-mono">
                {t.from_state} → {t.to_state} {label ? `(${label})` : ''} | {absolute(timestamp)}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ─── ProtocolsPage ───────────────────────────────────────────────────────────

export function ProtocolsPage() {
  useDocumentTitle('Protocols')
  const { protocolId } = useParams<{ protocolId?: string }>()
  if (protocolId) return <ProtocolDetailView protocolId={protocolId} />
  return <ProtocolListView />
}
