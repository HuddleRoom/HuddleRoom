import React, { useState, useMemo } from 'react'
import { toast } from 'sonner'
import { Dialog } from '@/components/common/Dialog'
import { Plus, ChevronDown, ChevronRight, Pencil, Trash2 } from 'lucide-react'
import {
  UI_COLORS,
  UI_FONT_FAMILY,
  Field,
  Input,
  Textarea,
  Button,
  Card,
  QueryState,
  PageHeader,
  ConfirmDialog,
  StructuralLabel,
} from '@/components/common/uiPrimitives'
import { Callout } from '@/components/common/Callout'
import { EmptyState } from '@/components/common/EmptyState'
import { Tag } from '@/components/common/Tag'
import { TextareaThenMonaco } from '@/components/common/TextareaThenMonaco'
import { STATUS_COLORS } from '@/lib/statusColors'
import { useUIStore } from '@/stores/ui'
import { useHooksList, useCreateHook, useUpdateHook, useDeleteHook } from '@/api/hooks'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'
import type { Hook, HookStatus } from '@/lib/types'

const STATUS_COLOR: Record<HookStatus, string> = {
  proposed: STATUS_COLORS.amber,
  active: STATUS_COLORS.green,
  shadow: STATUS_COLORS.neutral,
  disabled: '#64748B',
}

const EMPTY_SECTION_TEXT: Record<HookStatus, { title: string; body: string }> = {
  proposed: { title: 'No proposed hooks', body: 'Hooks proposed by agents will appear here for review.' },
  active: { title: 'No active hooks', body: 'Activated hooks will appear here.' },
  shadow: { title: 'No shadow hooks', body: 'Hooks running in shadow mode will appear here.' },
  disabled: { title: 'No disabled hooks', body: 'Disabled hooks will appear here.' },
}

interface HookFormState {
  name: string
  trigger_event: string
  description: string
  code: string
}

const BLANK_FORM: HookFormState = {
  name: '',
  trigger_event: '',
  description: '',
  code: '',
}

function getTransitionButtons(status: HookStatus): Array<{ label: string; targetStatus: HookStatus }> {
  switch (status) {
    case 'proposed':
      return [
        { label: 'Activate', targetStatus: 'active' },
        { label: 'Disable', targetStatus: 'disabled' },
      ]
    case 'active':
      return [
        { label: 'Shadow', targetStatus: 'shadow' },
        { label: 'Disable', targetStatus: 'disabled' },
      ]
    case 'shadow':
      return [
        { label: 'Activate', targetStatus: 'active' },
        { label: 'Disable', targetStatus: 'disabled' },
      ]
    case 'disabled':
      return [{ label: 'Re-propose', targetStatus: 'proposed' }]
    default:
      return []
  }
}

export function HooksPage() {
  useDocumentTitle('Hooks')
  const projectId = useUIStore((s) => s.activeProjectId)
  const { data: hooksData, isLoading, isError: hooksError, refetch: hooksRefetch } = useHooksList(projectId)
  const createMutation = useCreateHook(projectId)
  const updateMutation = useUpdateHook(projectId)
  const deleteMutation = useDeleteHook(projectId)

  const [showModal, setShowModal] = useState(false)
  const [editHook, setEditHook] = useState<Hook | null>(null)
  const [formState, setFormState] = useState<HookFormState>(BLANK_FORM)
  const [collapsed, setCollapsed] = useState<Record<HookStatus, boolean>>({
    proposed: false,
    active: false,
    shadow: false,
    disabled: false,
  })
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  const grouped = useMemo(() => {
    const g: Record<HookStatus, Hook[]> = {
      proposed: [],
      active: [],
      shadow: [],
      disabled: [],
    }
    ;(hooksData?.items ?? []).forEach((h) => g[h.status].push(h))
    return g
  }, [hooksData?.items])

  const handleOpenModal = (hook?: Hook) => {
    if (hook) {
      setEditHook(hook)
      setFormState({
        name: hook.name,
        trigger_event: hook.trigger_event,
        description: hook.description || '',
        code: hook.code,
      })
    } else {
      setEditHook(null)
      setFormState(BLANK_FORM)
    }
    setShowModal(true)
  }

  const handleCloseModal = () => {
    setShowModal(false)
    setEditHook(null)
    setFormState(BLANK_FORM)
  }

  const handleSaveHook = async () => {
    if (!formState.name.trim() || !formState.trigger_event.trim() || !formState.code.trim()) {
      toast.error('Name, trigger event, and code are required')
      return
    }

    try {
      if (editHook) {
        await updateMutation.mutateAsync({
          id: editHook.id,
          data: {
            name: formState.name,
            trigger_event: formState.trigger_event,
            description: formState.description,
            code: formState.code,
          },
        })
        toast.success('Hook updated')
      } else {
        await createMutation.mutateAsync({
          name: formState.name,
          trigger_event: formState.trigger_event,
          description: formState.description,
          code: formState.code,
        })
        toast.success('Hook created')
      }
      handleCloseModal()
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save hook')
    }
  }

  const handleTransitionStatus = async (hookId: string, targetStatus: HookStatus) => {
    try {
      await updateMutation.mutateAsync({
        id: hookId,
        data: { status: targetStatus },
      })
      toast.success(`Hook status updated to ${targetStatus}`)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to update hook')
    }
  }

  const handleDeleteConfirm = async () => {
    if (!deleteConfirm) return
    try {
      await deleteMutation.mutateAsync(deleteConfirm)
      toast.success('Hook deleted')
      setDeleteConfirm(null)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to delete hook')
    }
  }

  return (
    <div>
      {/* Availability notice */}
      <Callout variant="info" className="mb-4">
        Hook execution is not available yet. Hooks created here are saved for review and will not run automatically.
      </Callout>

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <PageHeader title="Hooks" />
        <Button
          variant="primary"
          size="sm"
          style={{ opacity: projectId ? 1 : 0.4, display: 'flex', alignItems: 'center', gap: 5 }}
          onClick={() => handleOpenModal()}
          disabled={!projectId}
        >
          <Plus size={14} />
          New hook
        </Button>
      </div>

      {/* Sections */}
      <QueryState
        query={{
          isLoading: isLoading,
          isError: hooksError,
          refetch: hooksRefetch,
          data: { hooks: Object.values(grouped).flat() },
        }}
        skeleton="list"
        skeletonCount={4}
        errorLabel="Failed to load hooks"
        emptyLabel="No hooks"
        emptyDetail="Hooks appear here once generated or added."
      >
        {() => {
          const allGroupsEmpty = Object.values(grouped).every((arr) => arr.length === 0)

          if (allGroupsEmpty) {
            return <EmptyState title="No hooks yet" body="Hooks appear here once generated or added." />
          }

          return (
            <div>
              {(['proposed', 'active', 'shadow', 'disabled'] as const).map((status) => {
                const items = grouped[status]
                const isCollapsed = collapsed[status]

                return (
                  <div key={status} style={{ marginBottom: 20 }}>
                    {/* Section Header */}
                    <div
                      style={{
                        display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer',
                        paddingBottom: 8, borderBottom: `1px solid ${UI_COLORS.border}`,
                      }}
                      onClick={() => setCollapsed((prev) => ({ ...prev, [status]: !prev[status] }))}
                    >
                      {isCollapsed ? <ChevronRight size={16} /> : <ChevronDown size={16} />}
                      <StructuralLabel>{status}</StructuralLabel>
                      <span style={{ color: UI_COLORS.textMuted, fontSize: 12, fontFamily: UI_FONT_FAMILY }}>
                        ({items.length})
                      </span>
                    </div>

                    {/* Section Content */}
                    {!isCollapsed && (
                      <div style={{ marginTop: 12 }}>
                        {items.length === 0 ? (
                          <EmptyState title={EMPTY_SECTION_TEXT[status].title} body={EMPTY_SECTION_TEXT[status].body} />
                        ) : (
                          items.map((hook) => (
                            <Card
                              key={hook.id}
                              data-testid={`hook-row-${hook.id}`}
                              padding="md"
                              style={{ marginBottom: 8 }}
                            >
                              {/* Hook Name */}
                              <div style={{ color: UI_COLORS.textPrimary, fontSize: 14, fontFamily: UI_FONT_FAMILY, marginBottom: 4, fontWeight: 500, overflowWrap: 'anywhere' }}>
                                {hook.name}
                              </div>

                              {/* Trigger Event */}
                              <div style={{ marginBottom: 4, overflowWrap: 'anywhere' }}>
                                <Tag mono>{hook.trigger_event}</Tag>
                              </div>

                              {/* Description */}
                              {hook.description && (
                                <div style={{ color: UI_COLORS.textMuted, fontSize: 12, fontFamily: UI_FONT_FAMILY, marginBottom: 8, overflowWrap: 'anywhere' }}>
                                  {hook.description}
                                </div>
                              )}

                              {/* Stats */}
                              <div style={{ color: UI_COLORS.textMuted, fontSize: 11, fontFamily: UI_FONT_FAMILY, marginBottom: 8 }}>
                                {hook.execution_count} runs · {hook.error_count} errors
                              </div>

                              {/* Status Badge */}
                              <div style={{ marginBottom: 8 }}>
                                <span style={{ color: STATUS_COLOR[hook.status], fontSize: 11, fontFamily: UI_FONT_FAMILY, textTransform: 'uppercase', fontWeight: 500 }}>
                                  {hook.status}
                                </span>
                              </div>

                              {/* Action Buttons */}
                              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                                {/* Transition Buttons */}
                                {getTransitionButtons(hook.status).map((btn) => (
                                  <Button
                                    key={btn.targetStatus}
                                    variant="secondary"
                                    size="sm"
                                    aria-label={`${btn.label} hook ${hook.name}`}
                                    onClick={() => handleTransitionStatus(hook.id, btn.targetStatus)}
                                  >
                                    {btn.label}
                                  </Button>
                                ))}

                                {/* Edit Button */}
                                <Button
                                  variant="secondary"
                                  size="sm"
                                  onClick={() => handleOpenModal(hook)}
                                >
                                  <Pencil size={14} />
                                </Button>

                                {/* Delete Button */}
                                <Button
                                  variant="secondary"
                                  size="sm"
                                  onClick={() => setDeleteConfirm(hook.id)}
                                  style={{ display: 'flex', alignItems: 'center', gap: 5 }}
                                >
                                  <Trash2 size={14} />
                                </Button>
                              </div>
                            </Card>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )
        }}
      </QueryState>

      <ConfirmDialog
        open={deleteConfirm !== null}
        onOpenChange={(open) => { if (!open) setDeleteConfirm(null) }}
        title="Delete hook?"
        consequence="This permanently removes the hook and cannot be undone."
        confirmLabel="Delete"
        onConfirm={handleDeleteConfirm}
        isPending={deleteMutation.isPending}
      />

      {/* Modal */}
      <Dialog
        open={showModal}
        onOpenChange={(v) => { if (!v) handleCloseModal(); else setShowModal(v) }}
        title={editHook ? 'Edit hook' : 'New hook'}
        description="Create or edit a hook."
        size="lg"
        footer={{
          primaryLabel: editHook ? 'Update' : 'Create',
          primaryType: 'submit',
          formId: 'hook-form',
        }}
      >
        <form id="hook-form" onSubmit={(e) => { e.preventDefault(); handleSaveHook() }} style={{ fontFamily: UI_FONT_FAMILY }}>
          <Input
            label="Name"
            type="text"
            value={formState.name}
            onChange={(e) => setFormState((prev) => ({ ...prev, name: e.target.value }))}
            placeholder="e.g. sync_task_status"
          />

          <Input
            label="Trigger Event"
            type="text"
            value={formState.trigger_event}
            onChange={(e) => setFormState((prev) => ({ ...prev, trigger_event: e.target.value }))}
            placeholder="e.g. task.created"
          />

          <Textarea
            label="Description"
            value={formState.description}
            onChange={(e) => setFormState((prev) => ({ ...prev, description: e.target.value }))}
            rows={3}
            placeholder="Optional description"
          />

          <Field label="Code">
            <TextareaThenMonaco
              height="300px"
              language="python"
              value={formState.code}
              onChange={(v) => setFormState((prev) => ({ ...prev, code: v }))}
              options={{
                minimap: { enabled: false },
                fontSize: 12,
                scrollBeyondLastLine: false,
                fontFamily: UI_FONT_FAMILY,
                theme: 'vs-dark',
              }}
            />
          </Field>
        </form>
      </Dialog>
    </div>
  )
}
