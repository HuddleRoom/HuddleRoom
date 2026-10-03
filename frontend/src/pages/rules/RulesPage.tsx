import React, { useState, useEffect } from 'react'
import {
  DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors,
  DragEndEvent,
} from '@dnd-kit/core'
import {
  SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy,
  useSortable, arrayMove,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import * as yaml from 'js-yaml'
import { toast } from 'sonner'
import { Plus, GripVertical, Pencil, Trash2 } from 'lucide-react'
import { useUIStore } from '@/stores/ui'
import { useRules, useCreateRule, useUpdateRule, useDeleteRule, useReorderRules } from '@/api/rules'
import type { RoutingRule } from '@/lib/types'
import { Button, Input, Textarea, QueryState, UI_COLORS, PageHeader, ConfirmDialog } from '@/components/common/uiPrimitives'
import { Dialog } from '@/components/common/Dialog'
import { Callout } from '@/components/common/Callout'
import { EmptyState } from '@/components/common/EmptyState'
import { TextareaThenMonaco } from '@/components/common/TextareaThenMonaco'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'

const MONO = 'ui-monospace, Menlo, Monaco, "Cascadia Code", monospace'

function SortableRow({ rule, onEdit, onDelete, updateRule }: {
  rule: RoutingRule
  onEdit: (rule: RoutingRule) => void
  onDelete: (id: string) => void
  updateRule: ReturnType<typeof useUpdateRule>
}) {
  const { attributes, listeners, setNodeRef, transform, transition } = useSortable({ id: rule.id })
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    background: UI_COLORS.surface,
    border: `1px solid ${UI_COLORS.border}`,
    borderRadius: 3,
    padding: '10px 12px',
    marginBottom: 8,
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    fontFamily: MONO,
    fontSize: 12,
  }

  const conditionsRaw = JSON.stringify(rule.conditions)
  const conditionsPreview = conditionsRaw.slice(0, 60) + (conditionsRaw.length > 60 ? '...' : '')

  return (
    <>
      <div ref={setNodeRef} style={style as React.CSSProperties} data-testid={`rule-row-${rule.id}`}>
        <div
          {...attributes}
          {...listeners}
          style={{ cursor: 'grab', color: UI_COLORS.textMuted, flex: 0 }}
          aria-label="Drag to reorder"
        >
          <GripVertical size={16} />
        </div>

        <div style={{ flex: 0, color: UI_COLORS.textMuted, minWidth: 30 }}>
          {rule.priority}
        </div>

        <div style={{ flex: 1, minWidth: 120, color: UI_COLORS.textPrimary }}>
          {rule.name}
        </div>

        <div style={{ flex: 1, minWidth: 100, color: UI_COLORS.textMuted }}>
          {rule.on_event}
        </div>

        <div style={{ flex: 0 }}>
          <label style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', minWidth: 44, minHeight: 44 }}>
            <input
              type="checkbox"
              checked={rule.enabled}
              aria-label={rule.enabled ? 'Disable rule' : 'Enable rule'}
              onChange={(e) => {
                updateRule.mutate({
                  id: rule.id,
                  data: { enabled: e.target.checked },
                })
              }}
              style={{ width: 16, height: 16, accentColor: UI_COLORS.primary, cursor: 'pointer' }}
            />
          </label>
        </div>

        <div
          style={{ flex: 1, minWidth: 200, color: UI_COLORS.textMuted, fontSize: 11 }}
          title={conditionsRaw}
        >
          {conditionsPreview}
        </div>

        <Button
          variant="secondary"
          size="sm"
          onClick={() => onEdit(rule)}
          aria-label={`Edit rule: ${rule.name}`}
          style={{ padding: '4px 8px', flex: 0 }}
        >
          <Pencil size={12} />
        </Button>

        <Button
          variant="danger"
          size="sm"
          onClick={() => setShowDeleteConfirm(true)}
          aria-label={`Delete rule: ${rule.name}`}
          style={{ padding: '4px 8px', flex: 0 }}
        >
          <Trash2 size={12} />
        </Button>
      </div>

      <ConfirmDialog
        open={showDeleteConfirm}
        onOpenChange={setShowDeleteConfirm}
        title="Delete rule?"
        consequence={`"${rule.name}" will be permanently deleted. This can't be undone.`}
        confirmLabel="Delete rule"
        onConfirm={() => { onDelete(rule.id); setShowDeleteConfirm(false) }}
      />
    </>
  )
}

export function RulesPage() {
  useDocumentTitle('Rules')
  const activeProjectId = useUIStore((s) => s.activeProjectId)
  const rulesQuery = useRules(activeProjectId)
  const createRule = useCreateRule(activeProjectId)
  const updateRule = useUpdateRule(activeProjectId)
  const deleteRule = useDeleteRule(activeProjectId)
  const reorderRules = useReorderRules(activeProjectId)

  const [showModal, setShowModal] = useState(false)
  const [editRule, setEditRule] = useState<RoutingRule | null>(null)
  const [localRules, setLocalRules] = useState<RoutingRule[]>([])
  const [formName, setFormName] = useState('')
  const [formOnEvent, setFormOnEvent] = useState('')
  const [formDescription, setFormDescription] = useState('')
  const [formEnabled, setFormEnabled] = useState(true)
  const [formConditions, setFormConditions] = useState('{}')
  const [formActions, setFormActions] = useState('{}')

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  )

  // Initialize local rules from query
  useEffect(() => {
    setLocalRules(rulesQuery.data?.items ?? [])
  }, [activeProjectId, rulesQuery.data])

  const handleOpenNewRule = () => {
    setEditRule(null)
    setFormName('')
    setFormOnEvent('')
    setFormDescription('')
    setFormEnabled(true)
    setFormConditions('{}')
    setFormActions('{}')
    setShowModal(true)
  }

  const handleOpenEditRule = (rule: RoutingRule) => {
    setEditRule(rule)
    setFormName(rule.name)
    setFormOnEvent(rule.on_event)
    setFormDescription(rule.description || '')
    setFormEnabled(rule.enabled)
    setFormConditions(yaml.dump(rule.conditions))
    setFormActions(yaml.dump(rule.actions))
    setShowModal(true)
  }

  const handleSubmit = async () => {
    if (!formName.trim() || !formOnEvent.trim()) {
      toast.error('Name and on_event are required')
      return
    }

    let parsedConditions: Record<string, unknown>
    let parsedActions: Record<string, unknown>

    try {
      const condResult = yaml.load(formConditions)
      parsedConditions = (condResult && typeof condResult === 'object') ? condResult as Record<string, unknown> : {}
    } catch {
      toast.error('Invalid YAML in conditions')
      return
    }

    try {
      const actResult = yaml.load(formActions)
      parsedActions = (actResult && typeof actResult === 'object') ? actResult as Record<string, unknown> : {}
    } catch {
      toast.error('Invalid YAML in actions')
      return
    }

    const data = {
      name: formName,
      on_event: formOnEvent,
      description: formDescription || undefined,
      enabled: formEnabled,
      conditions: parsedConditions,
      actions: parsedActions,
    }

    if (editRule) {
      updateRule.mutate(
        { id: editRule.id, data },
        {
          onSuccess: () => {
            toast.success('Rule updated')
            setShowModal(false)
          },
        },
      )
    } else {
      createRule.mutate(data, {
        onSuccess: () => {
          toast.success('Rule created')
          setShowModal(false)
        },
      })
    }
  }

  const handleDragEnd = (e: DragEndEvent) => {
    const { active, over } = e
    if (!over || active.id === over.id) return

    const oldIndex = localRules.findIndex((r) => r.id === active.id)
    const newIndex = localRules.findIndex((r) => r.id === over.id)

    if (oldIndex === -1 || newIndex === -1) return

    const newOrder = arrayMove(localRules, oldIndex, newIndex)
    setLocalRules(newOrder)

    const ruleIds = newOrder.map((r) => r.id)
    reorderRules.mutate(ruleIds, {
      onError: () => {
        setLocalRules(rulesQuery.data?.items || [])
      },
    })
  }

  const handleDeleteRule = (id: string) => {
    deleteRule.mutate(id, {
      onSuccess: () => {
        toast.success('Rule deleted')
      },
    })
  }

  return (
    <div>
      {/* Availability notice */}
      <Callout variant="info" className="mb-4">
        Routing rule evaluation is not available yet. Rules created here are saved for review and will not run automatically.
      </Callout>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <PageHeader title="Rules" />
        <Button
          variant="primary"
          size="sm"
          onClick={handleOpenNewRule}
          style={{ opacity: activeProjectId ? 1 : 0.4, display: 'flex', alignItems: 'center', gap: 5 }}
          disabled={!activeProjectId}
        >
          <Plus size={14} />
          New rule
        </Button>
      </div>

      {/* Rules Table */}
      <QueryState
        query={{
          isLoading: rulesQuery.isLoading,
          isError: rulesQuery.isError,
          data: rulesQuery.data,
          refetch: rulesQuery.refetch,
        }}
        skeleton="list"
        skeletonCount={3}
        errorLabel="Failed to load rules"
        emptyLabel="No routing rules"
        emptyDetail="Create a rule to route events without an LLM call."
      >
        {(data) => (
          (data.items ?? []).length === 0 ? (
            <EmptyState
              title="No routing rules"
              body="Create a rule to route events without an LLM call."
              action={{ label: 'New rule', onClick: handleOpenNewRule }}
            />
          ) : (
            <DndContext
              sensors={sensors}
              collisionDetection={closestCenter}
              onDragEnd={handleDragEnd}
            >
              <SortableContext items={localRules.map((r) => r.id)} strategy={verticalListSortingStrategy}>
                {localRules.map((rule) => (
                  <SortableRow
                    key={rule.id}
                    rule={rule}
                    onEdit={handleOpenEditRule}
                    onDelete={handleDeleteRule}
                    updateRule={updateRule}
                  />
                ))}
              </SortableContext>
            </DndContext>
          )
        )}
      </QueryState>

      {/* Create/Edit Modal */}
      <Dialog
        open={showModal}
        onOpenChange={setShowModal}
        title={editRule ? 'Edit rule' : 'New rule'}
        description="Create or edit a rule."
        size="lg"
        footer={{
          primaryLabel: editRule ? 'Update' : 'Create',
          primaryType: 'submit',
          formId: 'rule-form',
          isPending: createRule.isPending || updateRule.isPending,
        }}
      >
          <form id="rule-form" onSubmit={(e) => { e.preventDefault(); handleSubmit() }} style={{ fontFamily: MONO }}>
            <Input
              label="Name"
              type="text"
              value={formName}
              onChange={(e) => setFormName(e.target.value)}
            />

            <Input
              label="On Event"
              type="text"
              value={formOnEvent}
              onChange={(e) => setFormOnEvent(e.target.value)}
              placeholder="e.g. task.created"
            />

            <Textarea
              label="Description"
              value={formDescription}
              onChange={(e) => setFormDescription(e.target.value)}
              rows={3}
            />

            <div style={{ marginBottom: 14 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', minHeight: 44, paddingTop: 10, paddingBottom: 10 }}>
                <input
                  type="checkbox"
                  checked={formEnabled}
                  onChange={(e) => setFormEnabled(e.target.checked)}
                  style={{ width: 16, height: 16, accentColor: UI_COLORS.primary, cursor: 'pointer' }}
                />
                <span style={{ fontSize: 13, color: UI_COLORS.textPrimary }}>Enabled</span>
              </label>
            </div>

            <div style={{ marginBottom: 14 }}>
              <span style={{ display: 'block', color: UI_COLORS.textMuted, fontSize: 11, marginBottom: 4, fontFamily: MONO }}>
                Conditions (YAML)
              </span>
              <TextareaThenMonaco
                height="180px"
                language="yaml"
                value={formConditions}
                onChange={(v) => setFormConditions(v || '{}')}
                options={{
                  minimap: { enabled: false },
                  fontSize: 12,
                  lineNumbers: 'off',
                  scrollBeyondLastLine: false,
                  fontFamily: MONO,
                  wordWrap: 'on',
                  theme: 'vs-dark',
                }}
              />
            </div>

            <div style={{ marginBottom: 14 }}>
              <span style={{ display: 'block', color: UI_COLORS.textMuted, fontSize: 11, marginBottom: 4, fontFamily: MONO }}>
                Actions (YAML)
              </span>
              <TextareaThenMonaco
                height="180px"
                language="yaml"
                value={formActions}
                onChange={(v) => setFormActions(v || '{}')}
                options={{
                  minimap: { enabled: false },
                  fontSize: 12,
                  lineNumbers: 'off',
                  scrollBeyondLastLine: false,
                  fontFamily: MONO,
                  wordWrap: 'on',
                  theme: 'vs-dark',
                }}
              />
            </div>
          </form>
      </Dialog>
    </div>
  )
}
