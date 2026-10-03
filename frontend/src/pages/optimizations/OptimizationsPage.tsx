import React, { useState } from 'react'
import { LazyMonacoEditor as Editor } from '@/lib/lazyMonaco'
import { toast } from 'sonner'
import { Trash2 } from 'lucide-react'
import { useUIStore } from '@/stores/ui'
import { useOptimizations, useUpdateOptimization, useDeleteOptimization } from '@/api/optimizations'
import type { Optimization, OptimizationStatus } from '@/lib/types'
import { Button, QueryState, UI_COLORS, PageHeader, ConfirmDialog } from '@/components/common/uiPrimitives'
import { Callout } from '@/components/common/Callout'
import { EmptyState } from '@/components/common/EmptyState'
import { Tabs, TabPanel } from '@/components/common/Tabs'
import { Tag } from '@/components/common/Tag'
import { STATUS_COLORS } from '@/lib/statusColors'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'

const MONO = 'ui-monospace, Menlo, Monaco, monospace'

const STATUS_COLOR: Record<OptimizationStatus, string> = {
  proposed: STATUS_COLORS.amber,
  requires_approval: STATUS_COLORS.amber,
  approved: STATUS_COLORS.blue,
  active: STATUS_COLORS.green,
  shadow: STATUS_COLORS.neutral,
  disabled: STATUS_COLORS.neutral,
  rejected: STATUS_COLORS.red,
}

const TYPE_COLOR: Record<string, string> = {
  hook: STATUS_COLORS.amber,
  rule: STATUS_COLORS.blue,
  shortcut: STATUS_COLORS.neutral,
}

function TypeBadge({ type }: { type: string }) {
  return (
    <span style={{ color: TYPE_COLOR[type] || UI_COLORS.border, fontSize: 11, fontFamily: MONO, textTransform: 'uppercase' }}>
      {type}
    </span>
  )
}

interface OptimizationCardProps {
  opt: Optimization
  tab: 'proposed' | 'active'
  onStatusChange: (id: string, newStatus: OptimizationStatus) => void
  onDelete: (id: string) => void
  isDeleting?: boolean
  deleteConfirm: string | null
  setDeleteConfirm: (id: string | null) => void
}

function OptimizationCard({ opt, tab, onStatusChange, onDelete, isDeleting, deleteConfirm, setDeleteConfirm }: OptimizationCardProps) {
  const handleStatusTransition = (newStatus: OptimizationStatus) => {
    onStatusChange(opt.id, newStatus)
  }

  return (
    <div data-testid={`optimization-card-${opt.id}`} style={{
      background: UI_COLORS.surface, border: `1px solid ${UI_COLORS.border}`, borderRadius: 4, padding: 12, marginBottom: 12,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 8 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <TypeBadge type={opt.type} />
          <span style={{ color: STATUS_COLOR[opt.status], fontSize: 11, fontFamily: MONO, textTransform: 'uppercase' }}>
            {opt.status}
          </span>
        </div>
        <Button
          variant="danger"
          size="sm"
          aria-label="Delete optimization"
          onClick={() => setDeleteConfirm(opt.id)}
          disabled={isDeleting}
          style={{ display: 'flex', alignItems: 'center', gap: 5 }}
        >
          <Trash2 size={14} />
          Delete
        </Button>
      </div>

      {opt.pattern_id && (
        <div style={{ color: UI_COLORS.textMuted, fontSize: 11, fontFamily: MONO, marginBottom: 8 }}>
          from pattern: <Tag mono>{opt.pattern_id.substring(0, 8)}</Tag>
        </div>
      )}

      {tab === 'active' && (
        <div style={{ display: 'flex', gap: 12, fontSize: 12, fontFamily: MONO, marginBottom: 8 }}>
          <span style={{ color: UI_COLORS.textPrimary }}>{opt.fire_count} fires</span>
          {opt.error_rate > 0 && <span style={{ color: UI_COLORS.danger }}>{opt.error_rate}% errors</span>}
        </div>
      )}

      <div data-testid="optimization-generated-code" style={{ marginBottom: 10 }}>
        <React.Suspense fallback={<div style={{ height: 120, background: UI_COLORS.depth, borderRadius: 3, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><span style={{ color: UI_COLORS.textMuted, fontSize: 11 }}>Loading editor…</span></div>}>
          <Editor
            height={120}
            language="typescript"
            value={opt.generated_code}
            theme="vs-dark"
            options={{
              readOnly: true,
              minimap: { enabled: false },
              lineNumbers: 'off',
              folding: false,
              scrollBeyondLastLine: false,
              fontSize: 12,
              fontFamily: MONO,
            }}
          />
        </React.Suspense>
      </div>

      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {tab === 'proposed' && opt.status === 'proposed' && (
          <>
            <Button variant="primary" size="sm" onClick={() => handleStatusTransition('requires_approval')}>
              Submit for Approval
            </Button>
            <Button variant="secondary" size="sm" onClick={() => handleStatusTransition('approved')}>
              Approve
            </Button>
            <Button variant="danger" size="sm" onClick={() => handleStatusTransition('rejected')}>
              Reject
            </Button>
          </>
        )}
        {tab === 'proposed' && opt.status === 'requires_approval' && (
          <>
            <Button variant="secondary" size="sm" onClick={() => handleStatusTransition('approved')}>
              Approve
            </Button>
            <Button variant="danger" size="sm" onClick={() => handleStatusTransition('rejected')}>
              Reject
            </Button>
          </>
        )}
        {tab === 'proposed' && opt.status === 'approved' && (
          <Button variant="primary" size="sm" onClick={() => handleStatusTransition('active')}>
            Activate
          </Button>
        )}
        {tab === 'active' && opt.status === 'active' && (
          <>
            <Button variant="secondary" size="sm" onClick={() => handleStatusTransition('shadow')}>
              Move to Shadow
            </Button>
            <Button variant="danger" size="sm" onClick={() => handleStatusTransition('disabled')}>
              Disable
            </Button>
          </>
        )}
      </div>
    </div>
  )
}

export function OptimizationsPage() {
  useDocumentTitle('Optimizations')
  const { activeProjectId } = useUIStore()
  const [tab, setTab] = useState<'proposed' | 'active' | 'metrics'>('proposed')
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  const allOptimizations = useOptimizations(activeProjectId)
  const updateMutation = useUpdateOptimization(activeProjectId)
  const deleteMutation = useDeleteOptimization(activeProjectId)

  const handleStatusChange = (id: string, newStatus: OptimizationStatus) => {
    updateMutation.mutate({ id, data: { status: newStatus } }, {
      onSuccess: () => {
        toast.success(`Optimization updated to ${newStatus}`)
        setDeleteConfirm(null)
      },
      onError: (err) => {
        toast.error(`Failed to update optimization: ${err instanceof Error ? err.message : 'Unknown error'}`)
      },
    })
  }

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id, {
      onSuccess: () => {
        toast.success('Optimization deleted')
        setDeleteConfirm(null)
      },
      onError: (err) => {
        toast.error(`Failed to delete optimization: ${err instanceof Error ? err.message : 'Unknown error'}`)
      },
    })
  }

  const allItems = allOptimizations.data?.items || []
  const proposedItems = allItems.filter((o) => ['proposed', 'requires_approval', 'approved'].includes(o.status))
  const activeItems = allItems.filter((o) => o.status === 'active')

  const isLoading = allOptimizations.isLoading

  return (
    <div>
      <PageHeader title="Optimizations" />

      <Callout variant="info" className="mb-4">
        Pattern detection is not available yet. New optimization proposals will not be created automatically.
      </Callout>

      <Tabs
        idPrefix="optimizations"
        activeId={tab}
        onChange={(id) => setTab(id as 'proposed' | 'active' | 'metrics')}
        className="mb-5"
        tabs={[
          { id: 'proposed', label: 'Proposed' },
          { id: 'active', label: 'Active' },
          { id: 'metrics', label: 'Metrics' },
        ]}
      />

      <TabPanel tabId="proposed" activeId={tab} idPrefix="optimizations">
        <QueryState
          query={{
            isLoading: isLoading,
            isError: allOptimizations.isError,
            data: { items: proposedItems },
            refetch: allOptimizations.refetch,
          }}
          skeleton="cards"
          skeletonCount={3}
          errorLabel="Failed to load optimizations"
        >
          {(data) => (
            data.items.length === 0 ? (
              <EmptyState
                title="No optimizations yet"
                body="Proposed optimizations from observed patterns appear here."
              />
            ) : (
              <>
                {data.items.map((opt) => (
                  <OptimizationCard
                    key={opt.id}
                    opt={opt}
                    tab="proposed"
                    onStatusChange={handleStatusChange}
                    onDelete={handleDelete}
                    isDeleting={deleteMutation.isPending}
                    deleteConfirm={deleteConfirm}
                    setDeleteConfirm={setDeleteConfirm}
                  />
                ))}
              </>
            )
          )}
        </QueryState>
      </TabPanel>

      <TabPanel tabId="active" activeId={tab} idPrefix="optimizations">
        <QueryState
          query={{
            isLoading: isLoading,
            isError: allOptimizations.isError,
            data: { items: activeItems },
            refetch: allOptimizations.refetch,
          }}
          skeleton="cards"
          skeletonCount={3}
          errorLabel="Failed to load optimizations"
        >
          {(data) => (
            data.items.length === 0 ? (
              <EmptyState
                title="No active optimizations"
                body="Active optimizations will appear here once activated."
              />
            ) : (
              <>
                {data.items.map((opt) => (
                  <OptimizationCard
                    key={opt.id}
                    opt={opt}
                    tab="active"
                    onStatusChange={handleStatusChange}
                    onDelete={handleDelete}
                    isDeleting={deleteMutation.isPending}
                    deleteConfirm={deleteConfirm}
                    setDeleteConfirm={setDeleteConfirm}
                  />
                ))}
              </>
            )
          )}
        </QueryState>
      </TabPanel>

      <TabPanel tabId="metrics" activeId={tab} idPrefix="optimizations">
        <EmptyState
          title="No cost data yet"
          body="Cost data appears after an optimization executes."
        />
      </TabPanel>

      <ConfirmDialog
        open={deleteConfirm !== null}
        onOpenChange={(open) => { if (!open) setDeleteConfirm(null) }}
        title="Delete this optimization?"
        consequence="This permanently removes the optimization and cannot be undone."
        confirmLabel="Delete"
        onConfirm={async () => {
          if (deleteConfirm) {
            await deleteMutation.mutateAsync(deleteConfirm)
            toast.success('Optimization deleted')
            setDeleteConfirm(null)
          }
        }}
        isPending={deleteMutation?.isPending}
      />
    </div>
  )
}
