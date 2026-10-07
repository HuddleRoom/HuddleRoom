import React, { useState, useEffect, useMemo, useCallback } from 'react'
import { useParams, useNavigate, useSearchParams } from 'react-router-dom'
const Editor = React.lazy(() => import('@monaco-editor/react'))
const GraphDiagram = React.lazy(() => import('./GraphDiagram').then((m) => ({ default: m.GraphDiagram })))
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
  useGraphs, useGraph, useGraphsAll,
  useCreateGraph, useUpdateGraph, useDeactivateGraph, useActivateGraph,
  useGraphRuns, useGraphRunsAll,
  useGraphRunSteps,
  usePauseRun, useResumeRun, useAbandonRun,
} from '@/api/graphs'
import type { Graph, GraphRun, GraphRunStep } from '@/lib/types'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'
import { absolute } from '@/lib/time'

// ─── Constants / helpers ──────────────────────────────────────────────────────

const MONO = 'var(--huddleroom-font-mono)'

function readableStatus(status: string): string {
  return status.replace(/_/g, ' ')
}

function runStatusAnnouncement(status: string): string {
  if (status === 'paused') {
    return 'Graph run status paused. Live execution is stopped until an operator resumes it.'
  }
  if (status === 'active') {
    return 'Graph run status active. Live execution is currently running.'
  }
  return `Graph run status ${readableStatus(status)}.`
}

// ─── Shared styles ────────────────────────────────────────────────────────────

// ─── Graph form state ──────────────────────────────────────────────────────

interface GraphFormState {
  name: string
  version: string
  description: string
  definition: string
  triggers: string
}

const BLANK_FORM: GraphFormState = {
  name: '',
  version: '1.0.0',
  description: '',
  definition: `start_node: pending
nodes:
  pending:
    type: initial
  active: {}
  completed:
    type: terminal
  failed:
    type: terminal
edges:
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

function graphToForm(graph: Graph): GraphFormState {
  return {
    name: graph.name,
    version: graph.version,
    description: graph.description ?? '',
    definition: (() => {
      try {
        return typeof graph.definition === 'object' && graph.definition !== null
          ? yaml.dump(graph.definition)
          : '';
      } catch {
        toast.error('Graph definition is invalid and cannot be displayed')
        return '';
      }
    })(),
    triggers: (graph.triggers ?? []).map((t) =>
      typeof t === 'string' ? t : String((t as Record<string, unknown>).event_type ?? JSON.stringify(t))
    ).join(', '),
  }
}

// ─── GraphFormModal ────────────────────────────────────────────────────────

function GraphFormModal({
  open,
  onClose,
  initialData,
  mode,
  pid,
}: {
  open: boolean
  onClose: () => void
  initialData?: Graph | null
  mode: 'create' | 'edit'
  pid: string | null
}) {
  const [form, setForm] = useState<GraphFormState>(BLANK_FORM)
  const createGraph = useCreateGraph(pid)
  const updateGraph = useUpdateGraph(pid)

  useEffect(() => {
    if (open) {
      setForm(initialData ? graphToForm(initialData) : BLANK_FORM)
    }
  }, [open, initialData])

  function set(patch: Partial<GraphFormState>) {
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
        createGraph.mutate({
          name: payload.name,
          version: payload.version,
          description: payload.description ?? undefined,
          definition: payload.definition,
          triggers: payload.triggers,
        }, {
          onSuccess: () => { toast.success('Graph created'); onClose() },
          onError: (e) => toast.error(e.message),
        })
      } else if (initialData) {
        updateGraph.mutate({ graphId: initialData.id, data: {
          name: payload.name,
          version: payload.version,
          description: payload.description ?? undefined,
          definition: payload.definition,
          triggers: payload.triggers,
        } }, {
          onSuccess: () => { toast.success('Graph updated'); onClose() },
          onError: (e) => toast.error(e.message),
        })
      }
    } catch (e) {
      toast.error(`Invalid YAML: ${e instanceof Error ? e.message : 'unknown error'}`)
    }
  }

  const isPending = createGraph.isPending || updateGraph.isPending
  const canSubmit = form.name.trim() && form.version.trim()

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => { if (!v) onClose() }}
      title={mode === 'create' ? 'New graph' : 'Edit graph'}
      description={
        mode === 'create'
          ? 'Create a new graph with name, version, YAML definition, and triggers.'
          : `Edit the definition, version, and triggers for ${initialData?.name ?? 'this graph'}.`
      }
      size="md"
      footer={{
        primaryLabel: isPending ? 'Saving...' : mode === 'create' ? 'Create graph' : 'Save changes',
        primaryType: 'submit',
        formId: 'graph-form',
        isPending,
        primaryDisabled: !canSubmit,
      }}
    >
      <form id="graph-form" onSubmit={(e) => { e.preventDefault(); handleSubmit() }}>
            {/* Row: name + version */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 14 }}>
              <div>
                <Input id="graph-name" label="Name *" value={form.name} placeholder="graph name"
                  onChange={(e) => set({ name: e.target.value })}
                />
              </div>
              <div>
                <Input id="graph-version" label="Version *" value={form.version} placeholder="1.0.0"
                  onChange={(e) => set({ version: e.target.value })}
                />
              </div>
            </div>

            <Textarea id="graph-description" label="Description" rows={3}
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

            <Input id="graph-triggers" label="Triggers (comma-separated)" value={form.triggers} placeholder="start, stop"
              onChange={(e) => set({ triggers: e.target.value })}
            />
      </form>
    </Dialog>
  )
}

// ─── Parse definition to get nodes and edges ───────────────────────────

export interface ParsedNodeInfo {
  id: string
  isInitial: boolean
  isTerminal: boolean
}

export interface ParsedFlow {
  nodes: Map<string, ParsedNodeInfo>
  edges: Array<{ from: string; to: string; event?: string }>
  startNode?: string
}

function parseGraphDefinition(definition: Record<string, unknown>): ParsedFlow {
  const nodes = new Map<string, ParsedNodeInfo>()
  const edges: Array<{ from: string; to: string; event?: string }> = []
  let startNode: string | undefined

  // Parse nodes
  const nodesObj = definition.nodes as Record<string, unknown> | undefined
  if (nodesObj && typeof nodesObj === 'object') {
    Object.entries(nodesObj).forEach(([name, config]) => {
      const cfg = config as Record<string, unknown> | undefined
      const type = (cfg?.type as string | undefined) ?? ''

      nodes.set(name, {
        id: name,
        isInitial: type === 'initial',
        isTerminal: type === 'terminal' || type === 'end' || (cfg?.is_terminal as boolean | undefined) === true,
      })
    })
  }

  // Parse start node
  const startNodeVal = definition.start_node as string | undefined
  if (startNodeVal) {
    startNode = startNodeVal
  } else {
    // Look for start node by type
    nodes.forEach((info, name) => {
      if (info.isInitial) startNode = name
    })
  }

  // Parse edges
  const edgesArr = definition.edges as Array<Record<string, unknown>> | undefined
  if (Array.isArray(edgesArr)) {
    edgesArr.forEach((t) => {
      const from = (t.from as string | undefined) ?? (t.from_node as string | undefined)
      const to = (t.to as string | undefined) ?? (t.to_node as string | undefined)
      const eventRaw = t.event ?? t.event_type
      const event: string | undefined = typeof eventRaw === 'string'
        ? eventRaw
        : typeof eventRaw === 'object' && eventRaw !== null
          ? String((eventRaw as Record<string, unknown>).event_type ?? '')
          : undefined
      if (from && to) {
        // Ensure nodes exist
        if (!nodes.has(from)) {
          nodes.set(from, { id: from, isInitial: false, isTerminal: false })
        }
        if (!nodes.has(to)) {
          nodes.set(to, { id: to, isInitial: false, isTerminal: false })
        }
        edges.push({ from, to, event })
      }
    })
  }

  return { nodes, edges, startNode }
}

// ─── GraphListView ─────────────────────────────────────────────────────────

function GraphListView() {
  const navigate = useNavigate()
  const pid = useUIStore((s) => s.activeProjectId)
  const [showCreate, setShowCreate] = useState(false)
  const [includeInactive, setIncludeInactive] = useState(false)
  const [search, setSearch] = useState('')
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  const { items: graphs, isLoading, isError: graphsError, refetch: graphsRefetch } = useGraphsAll(pid, includeInactive)
  const deactivateGraph = useDeactivateGraph(pid)
  const activateGraph = useActivateGraph(pid)
  const filteredGraphs = useMemo(() => {
    const query = search.trim().toLowerCase()
    if (!query) {
      return graphs
    }
    return graphs.filter((graph) => {
      const haystack = `${graph.name} ${graph.version} ${graph.description ?? ''}`.toLowerCase()
      return haystack.includes(query)
    })
  }, [graphs, search])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <PageHeader title="Graphs" />
        <Button
          variant="primary"
          onClick={() => setShowCreate(true)}
        >
          <Plus size={12} /> New graph
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
          aria-label="Search graphs"
          placeholder="Search graphs"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ maxWidth: 240 }}
        />
      </div>

      {/* Graph list */}
      <QueryState
        query={{
          isLoading,
          isError: graphsError,
          refetch: graphsRefetch,
          data: filteredGraphs,
        }}
        skeleton="rows"
        skeletonCount={3}
        errorLabel="Failed to load graphs"
        emptyLabel={search.trim() ? 'No graphs match this search' : 'No graphs'}
        emptyDetail={search.trim() ? undefined : 'Runs appear here when graphs are triggered.'}
      >
        {(displayGraphs) => (
          <div className="border border-huddleroom-border rounded-md overflow-hidden bg-huddleroom-surface">
            {displayGraphs.map((p, idx) => {
              const isGlobal = !p.project_id
              return (
                <div key={p.id}
                  role="button"
                  tabIndex={0}
                  data-testid={`graph-card-${p.id}`}
                  className={idx < displayGraphs.length - 1 ? 'border-b border-huddleroom-depth' : ''}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 16, padding: '12px 16px', minHeight: 32,
                    transition: 'background 120ms',
                    cursor: 'pointer',
                  }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.background = UI_COLORS.depth }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.background = UI_COLORS.surface }}
                  onClick={() => navigate(`/graphs/${p.id}`)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); navigate(`/graphs/${p.id}`) } }}
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
                      onClick={() => navigate(`/graphs/${p.id}`)}
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
                            activateGraph.mutate(p.id, {
                              onSuccess: () => toast.success('Graph reactivated'),
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
      <GraphFormModal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        mode="create"
        pid={pid}
      />

      {/* Deactivate confirmation */}
      <ConfirmDialog
        open={!!deleteConfirm}
        onOpenChange={(v) => { if (!v) setDeleteConfirm(null) }}
        title="Deactivate this graph?"
        consequence="It will no longer be available to agents."
        confirmLabel={deactivateGraph.isPending ? 'Deactivating...' : 'Deactivate'}
        isPending={deactivateGraph.isPending}
        onConfirm={() => {
          if (!deleteConfirm) return
          deactivateGraph.mutate(deleteConfirm, {
            onSuccess: () => { toast.success('Graph deactivated'); setDeleteConfirm(null) },
            onError: (e) => toast.error(e.message),
          })
        }}
      />
    </div>
  )
}

// ─── GraphDetailView ───────────────────────────────────────────────────────

function GraphDetailView({ graphId }: { graphId: string }) {
  const pid = useUIStore((s) => s.activeProjectId)
  const [showEdit, setShowEdit] = useState(false)
  const [editDefinition, setEditDefinition] = useState('')
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [searchParams, setSearchParams] = useSearchParams()
  const statusFilter = searchParams.get('runStatus') ?? 'all'
  const [expandedRun, setExpandedRun] = useState<string | null>(null)
  const [abandonConfirm, setAbandonConfirm] = useState<string | null>(null)
  const [deactivateConfirm, setDeactivateConfirm] = useState(false)

  const { data: graph, isLoading, isError: graphError, refetch: graphRefetch } = useGraph(pid, graphId)
  const { items: runs, isLoading: runsLoading } = useGraphRunsAll(pid, {
    graph_id: graphId,
    status: statusFilter === 'all' ? undefined : statusFilter,
  })

  const updateGraph = useUpdateGraph(pid)
  const deactivateGraph = useDeactivateGraph(pid)
  const activateGraph = useActivateGraph(pid)
  const pauseRun = usePauseRun(pid)
  const resumeRun = useResumeRun(pid)
  const abandonRun = useAbandonRun(pid)

  useEffect(() => {
    if (graph?.definition) {
      try {
        setEditDefinition(yaml.dump(graph.definition))
      } catch {
        toast.error('Graph definition is invalid and cannot be displayed')
        setEditDefinition('')
      }
    }
  }, [graph])

  const parsed = useMemo(
    () => parseGraphDefinition(graph?.definition ?? {}),
    [graph?.definition]
  )
  const currentRun = selectedRunId ? runs.find((i) => i.id === selectedRunId) : null

  return (
    <QueryState
      query={{ isLoading, isError: graphError, refetch: graphRefetch, data: graph }}
      skeleton="rows"
      skeletonCount={4}
      errorLabel="Failed to load graph"
      emptyLabel="Graph not found"
    >
      {(graphData) => {
        const isGlobal = !graphData.project_id

        function handleSaveDefinition() {
          try {
            const definition = yaml.load(editDefinition) as Record<string, unknown>
            updateGraph.mutate(
              { graphId, data: { definition } },
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
              backTo="/graphs"
              backLabel="Graphs"
              title={graphData.name}
              titleClassName="font-mono"
              status={graphData.is_active ? 'active' : 'disabled'}
              statusLabel={graphData.is_active ? 'Active' : 'Inactive'}
              actions={
                <>
                  <Tag mono>v{graphData.version}</Tag>
                  <Button variant="ghost" onClick={() => setShowEdit(true)}>
                    Edit
                  </Button>
                  {!isGlobal && (
                    graphData.is_active ? (
                      <Button
                        variant="danger"
                        onClick={() => setDeactivateConfirm(true)}
                      >
                        Deactivate
                      </Button>
                    ) : (
                      <Button
                        variant="secondary"
                        onClick={() => activateGraph.mutate(graphData.id, {
                          onSuccess: () => toast.success('Graph reactivated'),
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
            {(graphData.description || graphData.triggers.length > 0) && (
              <div className="bg-huddleroom-surface border border-huddleroom-border rounded-md" style={{ padding: 20 }}>
                {graphData.description && (
                  <div>
                    <SectionLabel>Description</SectionLabel>
                    <div className="text-[13px] text-huddleroom-text-muted" style={{ lineHeight: 1.5 }}>{graphData.description}</div>
                  </div>
                )}

                {graphData.triggers.length > 0 && (
                  <div style={{ marginTop: graphData.description ? 12 : 0, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {graphData.triggers.map((t, i) => {
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
            <div className="graph-detail-grid">
              {/* Left: Editor */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div data-testid="graph-definition-editor" className="border border-huddleroom-border rounded-[3px] overflow-hidden">
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
                    disabled={updateGraph.isPending}
                    className="self-start"
                  >
                    {updateGraph.isPending ? 'Saving...' : 'Save definition'}
                  </Button>
                )}
              </div>

              {/* Right: Diagram */}
              <div data-testid="graph-diagram" className="border border-huddleroom-border rounded-md bg-huddleroom-depth" style={{
                height: 440,
                overflow: 'hidden', position: 'relative',
              }}>
                {parsed.nodes.size === 0 ? (
                  <div className="text-huddleroom-text-muted text-xs" style={{
                    display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%',
                  }}>
                    no graph diagram available
                  </div>
                ) : (
                  <React.Suspense fallback={<div className="text-huddleroom-text-muted text-xs" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>Loading diagram…</div>}>
                    <GraphDiagram parsed={parsed} currentNode={currentRun?.current_node} />
                  </React.Suspense>
                )}
              </div>
            </div>

            {/* Run list */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <StructuralLabel>Runs</StructuralLabel>
                <Select
                  aria-label="Filter graph runs by status"
                  value={statusFilter}
                  onChange={(e) => {
                    setSearchParams((prev) => {
                      const next = new URLSearchParams(prev)
                      if (e.target.value && e.target.value !== 'all') next.set('runStatus', e.target.value)
                      else next.delete('runStatus')
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

              {runsLoading ? (
                <div className="text-huddleroom-text-muted text-xs bg-huddleroom-surface rounded-md border border-huddleroom-border" style={{ padding: 16 }}>
                  loading runs...
                </div>
              ) : runs.length === 0 ? (
                <div className="bg-huddleroom-surface rounded-md border border-huddleroom-border">
                  <EmptyState
                    title="No runs yet"
                    body="Runs of this graph will appear here when it is triggered."
                    className="px-4"
                  />
                </div>
              ) : (
                <>
                {runs.length > 50 && (
                  <div className="text-[11px] text-huddleroom-text-muted" style={{ marginBottom: 4 }}>
                    showing 50 of {runs.length}
                  </div>
                )}
                <div data-testid="graph-run-list" className="border border-huddleroom-border rounded-md overflow-hidden bg-huddleroom-surface">
                  {runs.slice(0, 50).map((run, idx) => (
                    <div key={run.id}>
                      <div
                        role="button"
                        tabIndex={0}
                        className={idx < Math.min(runs.length, 50) - 1 ? 'border-b border-huddleroom-depth' : ''}
                        style={{
                          display: 'grid', gridTemplateColumns: 'auto 1fr auto auto auto 72px', gap: 12, padding: '12px 16px', minHeight: 32,
                          alignItems: 'center',
                          cursor: 'pointer',
                          transition: 'background 120ms',
                        }}
                        onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.background = UI_COLORS.depth }}
                        onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.background = UI_COLORS.surface }}
                        onClick={() => {
                          setSelectedRunId(run.id === selectedRunId ? null : run.id)
                          setExpandedRun(run.id === expandedRun ? null : run.id)
                        }}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setSelectedRunId(run.id === selectedRunId ? null : run.id); setExpandedRun(run.id === expandedRun ? null : run.id) } }}
                      >
                        <ChevronRight size={12} className="text-huddleroom-text-muted" style={{
                          transform: expandedRun === run.id ? 'rotate(90deg)' : 'rotate(0deg)',
                          transition: 'transform 120ms',
                        }} />
                        <div className="text-[11px] text-huddleroom-text-muted font-mono" style={{ overflow: 'hidden', textOverflow: 'ellipsis' }} title={run.id}>
                          {run.id.slice(0, 12)}…
                        </div>
                        <span className="text-[11px] font-semibold rounded-[2px] text-huddleroom-text-muted font-mono" style={{
                          padding: '2px 6px', background: UI_COLORS.border,
                        }}>
                          {run.current_node}
                        </span>
                        <span
                          className={`status-${run.status} text-[11px] font-mono`}
                          style={{ minWidth: 60 }}
                          aria-label={runStatusAnnouncement(run.status)}
                        >
                          {run.status}
                        </span>
                        <span className="text-[11px] text-huddleroom-text-muted font-mono">
                          {absolute(run.started_at)}
                        </span>
                        {(run.status === 'active' || run.status === 'paused') && (
                          <div style={{ display: 'flex', gap: 4 }} onClick={(e) => e.stopPropagation()}>
                            {run.status === 'active' ? (
                              <Button
                                variant="secondary"
                                size="sm"
                                aria-label={`Pause graph run ${run.id.slice(0, 12)}`}
                                title="Pause run"
                                onClick={() => {
                                  pauseRun.mutate(run.id, {
                                    onSuccess: () => toast.success('Run paused'),
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
                                aria-label={`Resume graph run ${run.id.slice(0, 12)}`}
                                title="Resume run"
                                onClick={() => {
                                  resumeRun.mutate(run.id, {
                                    onSuccess: () => toast.success('Run resumed'),
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
                              aria-label={`Abandon graph run ${run.id.slice(0, 12)}`}
                              title="Abandon run"
                              onClick={() => setAbandonConfirm(run.id)}
                              className="text-[10px]"
                              style={{ minHeight: 32, padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 2 }}
                            >
                              <X size={12} />
                            </Button>
                          </div>
                        )}
                      </div>

                      {/* Expanded step history */}
                      {expandedRun === run.id && (
                        <RunStepHistory projectId={pid} runId={run.id} />
                      )}
                    </div>
                  ))}
                </div>
                </>
              )}
            </div>

            {/* Edit modal */}
            <GraphFormModal
              open={showEdit}
              onClose={() => setShowEdit(false)}
              initialData={graphData}
              mode="edit"
              pid={pid}
            />

            {/* Deactivate graph confirm dialog */}
            <ConfirmDialog
              open={deactivateConfirm}
              onOpenChange={setDeactivateConfirm}
              title="Deactivate this graph?"
              consequence="It will no longer be available to agents."
              confirmLabel="Deactivate"
              onConfirm={() => {
                deactivateGraph.mutate(graphData.id, {
                  onSuccess: () => { toast.success('Graph deactivated'); setDeactivateConfirm(false) },
                  onError: (e) => toast.error(e.message),
                })
              }}
              isPending={deactivateGraph.isPending}
            />

            {/* Abandon run confirm dialog */}
            <ConfirmDialog
              open={abandonConfirm !== null}
              onOpenChange={(open) => { if (!open) setAbandonConfirm(null) }}
              title="Abandon this run?"
              consequence="This terminates the running graph run. Progress will be lost."
              confirmLabel="Abandon"
              onConfirm={() => {
                if (abandonConfirm) {
                  abandonRun.mutate(abandonConfirm, {
                    onSuccess: () => { toast.success('Run abandoned'); setAbandonConfirm(null) },
                    onError: (e) => toast.error(e.message),
                  })
                }
              }}
              isPending={abandonRun.isPending}
            />
          </div>
        )
      }}
    </QueryState>
  )
}

// ─── RunStepHistory ───────────────────────────────────────────────

function RunStepHistory({ projectId, runId }: { projectId: string | null; runId: string }) {
  const { data: steps } = useGraphRunSteps(projectId, runId)

  return (
    <div className="bg-huddleroom-depth border-t border-huddleroom-border" style={{ padding: '8px 16px' }}>
      {!steps || steps.length === 0 ? (
        <div className="text-[11px] text-huddleroom-text-muted font-mono">no steps</div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {steps.map((t) => {
            const label = t.edge_name
            const timestamp = t.stepped_at
            return (
              <div key={t.id} className="text-[11px] text-huddleroom-text-muted font-mono">
                {t.from_node} → {t.to_node} {label ? `(${label})` : ''} | {absolute(timestamp)}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ─── GraphsPage ───────────────────────────────────────────────────────────

export function GraphsPage() {
  useDocumentTitle('Graphs')
  const { graphId } = useParams<{ graphId?: string }>()
  if (graphId) return <GraphDetailView graphId={graphId} />
  return <GraphListView />
}
