import React, { useState, useEffect, useMemo, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import * as Collapsible from '@radix-ui/react-collapsible'
import * as DropdownMenu from '@radix-ui/react-dropdown-menu'
import { toast } from 'sonner'
import { Plus, ChevronDown, Send, CheckCircle2, Circle, MessageSquare, Users, ListChecks, Zap, RotateCw, MoreHorizontal, Check } from 'lucide-react'
import { useQueryClient } from '@tanstack/react-query'
import { Button, Card, Input, Textarea, Select, QueryState, UI_COLORS, PageHeader, ConfirmDialog, SkeletonList, SkeletonRow, StatusBadge, StructuralLabel } from '@/components/common/uiPrimitives'
import { Dialog as RadixDialog } from '@/components/common/Dialog'
import { FilterChips } from '@/components/common/FilterChips'
import { Tag } from '@/components/common/Tag'
import { DetailHeader } from '@/components/common/DetailHeader'
import { ErrorRecord } from '@/components/common/ErrorRecord'
import { Record } from '@/components/common/Record'

import { useUIStore } from '@/stores/ui'
import { useAllAgents } from '@/api/agents'
import { useMeetingWSStore } from '@/stores/meeting-ws'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'
import { STATUS_COLORS as BASE_STATUS_COLORS } from '@/lib/statusColors'
import {
  useMeetings, useMeeting, useMeetingTurns, useMeetingDecisions,
  useMeetingActionItems, useMeetingAgenda,
  useMeetingFinalReview, useCompleteMeetingFinalReview,
  useCopyMeeting, useSubmitHumanTurn, useEndMeeting, useCancelMeeting, useResumeMeeting,
  useGrantTurn, useAdvanceAgenda, useVetoDecision, useAddAgendaItem,
  useUpdateActionItem, meetingKeys,
} from '@/api/meetings'
import { getToken } from '@/lib/api-client'
import type { Meeting, MeetingTurn, MeetingDecision, MeetingActionItem, MeetingFinalReview, AgendaItem, Agent, MeetingStatus, MeetingWSEvent } from '@/lib/types'
import { absolute } from '@/lib/time'

// ─── Constants / helpers ──────────────────────────────────────────────────────

// ponytail: MONO kept as inline style (not a Tailwind class) because
// MeetingsPage.test.tsx asserts on the literal `style="font-family:..."` markup
// for these two spots; converting to `font-mono` would break those assertions.
const MONO = 'var(--huddleroom-font-mono)'

const STATUS_FILTER_OPTIONS = [
  { id: 'all', label: 'All' },
  { id: 'scheduled', label: 'Scheduled' },
  { id: 'preparing', label: 'Preparing' },
  { id: 'active', label: 'Active' },
  { id: 'concluding', label: 'Concluding' },
  { id: 'concluded', label: 'Concluded' },
  { id: 'cancelled', label: 'Cancelled' },
]

function readableDecidedBy(decidedBy: string) {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(decidedBy)
    ? decidedBy.slice(0, 8)
    : decidedBy
}

function compactJson(value: unknown) {
  return JSON.stringify(value, null, 2)
}

function eventLabel(event: MeetingWSEvent) {
  const trace = event.payload?.trace as Record<string, unknown> | undefined
  const kind = typeof trace?.kind === 'string' ? trace.kind : event.event_type
  const stage = typeof trace?.stage === 'string' ? `:${trace.stage}` : ''
  return `${kind}${stage}`
}

// ─── Shared styles ────────────────────────────────────────────────────────────

// ─── Grant Turn Modal ─────────────────────────────────────────────────────────

function GrantTurnModal({
  open,
  onClose,
  meeting,
  agentList,
}: {
  open: boolean
  onClose: () => void
  meeting: Meeting | undefined
  agentList: Agent[]
}) {
  const [selectedAgentId, setSelectedAgentId] = useState<string>('')
  const grantTurn = useGrantTurn(meeting?.project_id ?? null, meeting?.id)

  const participantIds = new Set(meeting?.participant_agent_ids ?? [])
  const participants = agentList.filter((a: Agent) => participantIds.has(a.id))

  function handleSubmit() {
    if (!selectedAgentId || !meeting) return
    grantTurn.mutate(selectedAgentId, {
      onSuccess: () => { toast.success('Turn granted'); onClose() },
      onError: (e) => toast.error(e.message),
    })
  }

  return (
    <RadixDialog
      open={open}
      onOpenChange={(v) => { if (!v) onClose() }}
      size="sm"
      title="Grant turn"
      description="Select a participant agent to grant the turn."
      footer={{
        primaryLabel: 'Grant turn',
        onPrimary: handleSubmit,
        isPending: grantTurn.isPending,
        primaryDisabled: !selectedAgentId,
      }}
    >
      <Select
        label="Select participant"
        value={selectedAgentId}
        onChange={(e) => setSelectedAgentId(e.target.value)}
      >
        <option value="">— choose agent —</option>
        {participants.map((a: Agent) => (
          <option key={a.id} value={a.id}>{a.name}</option>
        ))}
      </Select>
    </RadixDialog>
  )
}

// ─── Add Agenda Item Modal ────────────────────────────────────────────────────

function AddAgendaItemModal({
  open,
  onClose,
  meeting,
}: {
  open: boolean
  onClose: () => void
  meeting: Meeting | undefined
}) {
  const [title, setTitle] = useState<string>('')
  const [description, setDescription] = useState<string>('')
  const [question, setQuestion] = useState<string>('')
  const addAgendaItem = useAddAgendaItem(meeting?.project_id ?? null, meeting?.id)

  function handleSubmit() {
    if (!title.trim() || !meeting) return
    addAgendaItem.mutate({
      title: title.trim(),
      description: description.trim() || undefined,
      question: question.trim() || undefined,
      order: (meeting.agenda_items?.length ?? 0),
    }, {
      onSuccess: () => { toast.success('Agenda item added'); onClose() },
      onError: (e) => toast.error(e.message),
    })
  }

  const canSubmit = title.trim()

  return (
    <RadixDialog
      open={open}
      onOpenChange={(v) => { if (!v) onClose() }}
      size="sm"
      title="Agenda item"
      description="Add a new agenda item with title, description, and question."
      footer={{
        primaryLabel: 'Add item',
        primaryType: 'submit',
        formId: 'agenda-item-form',
        isPending: addAgendaItem.isPending,
        primaryDisabled: !canSubmit,
      }}
    >
      <form id="agenda-item-form" onSubmit={(e) => { e.preventDefault(); handleSubmit() }} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        <Input
          label="Title *"
          value={title}
          placeholder="agenda item title"
          onChange={(e) => setTitle(e.target.value)}
        />

        <Textarea
          label="Description"
          rows={2}
          value={description}
          placeholder="optional description"
          onChange={(e) => setDescription(e.target.value)}
        />

        <Textarea
          label="Question"
          rows={2}
          value={question}
          placeholder="optional question for discussion"
          onChange={(e) => setQuestion(e.target.value)}
        />
      </form>
    </RadixDialog>
  )
}

// ─── MeetingListView ──────────────────────────────────────────────────────────

function MeetingListView() {
  const navigate = useNavigate()
  const projectId = useUIStore((s) => s.activeProjectId)
  const [statusFilter, setStatusFilter] = useState<MeetingStatus | undefined>(undefined)

  const { data: meetingsData, isLoading, isError: meetingsError, refetch: meetingsRefetch, fetchNextPage, hasNextPage, isFetchingNextPage } = useMeetings(projectId, statusFilter)
  const meetings = meetingsData?.pages.flatMap(p => p.items) ?? []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <PageHeader title="Meetings" />
        <Button variant="primary" onClick={() => navigate('/meetings/new')}>
          <Plus size={12} /> New meeting
        </Button>
      </div>

      {/* Status filter chips */}
      <FilterChips
        ariaLabel="Meeting status"
        options={STATUS_FILTER_OPTIONS}
        activeId={statusFilter ?? 'all'}
        onChange={(id) => setStatusFilter(id === 'all' ? undefined : (id as MeetingStatus))}
      />

      {/* Meetings list */}
      <QueryState
        query={{
          isLoading,
          isError: meetingsError,
          refetch: meetingsRefetch,
          data: meetings,
        }}
        skeleton="rows"
        skeletonCount={3}
        errorLabel="Failed to load meetings"
        emptyLabel="No meetings yet"
        emptyDetail="Meetings appear here once scheduled or convened."
      >
        {(displayMeetings) => (
          <div className="bg-huddleroom-surface border border-huddleroom-border rounded-[6px]" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            {displayMeetings.map((meeting, idx) => (
              <div key={meeting.id}
                data-testid={`meeting-card-${meeting.id}`}
                className={idx < displayMeetings.length - 1 ? 'border-b border-huddleroom-depth' : ''}
                style={{
                  padding: '12px 16px',
                  display: 'flex', alignItems: 'center', gap: 12, cursor: 'pointer',
                  transition: 'background 120ms',
                }}
                onClick={() => navigate(`/meetings/${meeting.id}`)}
                onMouseEnter={(e) => { (e.currentTarget as HTMLDivElement).style.background = UI_COLORS.depth }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLDivElement).style.background = 'transparent' }}
              >
                {/* Title + status */}
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="text-[13px] text-huddleroom-text-primary font-medium" style={{ marginBottom: 2 }}>
                    {meeting.title}
                  </div>
                  <StatusBadge status={meeting.status} />
                </div>

                {/* Meeting type tag */}
                {meeting.meeting_type && (
                  <Tag mono className="flex-shrink-0">{meeting.meeting_type}</Tag>
                )}

                {/* Created date */}
                <span className="text-xs text-huddleroom-text-muted flex-shrink-0" style={{ minWidth: 100, textAlign: 'right' }}>
                  {absolute(meeting.created_at)}
                </span>
              </div>
            ))}
          </div>
        )}
      </QueryState>

      {hasNextPage && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '12px 0' }}>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => fetchNextPage()}
            disabled={isFetchingNextPage}
          >
            {isFetchingNextPage ? 'Loading...' : 'Load more'}
          </Button>
        </div>
      )}
    </div>
  )
}

// ─── MeetingLiveView (detail view) ────────────────────────────────────────────

function FinalReviewSection({
  projectId,
  meetingId,
  review,
}: {
  projectId: string | null
  meetingId: string
  review: MeetingFinalReview
}) {
  const [decisionsMade, setDecisionsMade] = useState(review.decisions_made)
  const [decisionsClear, setDecisionsClear] = useState(review.decisions_clear)
  const [itemsNeeded, setItemsNeeded] = useState(true)
  const [itemsText, setItemsText] = useState(review.suggested_action_items.join('\n'))
  const completeReview = useCompleteMeetingFinalReview(projectId, meetingId)
  const actionItems = itemsText.split('\n').map((item) => item.trim()).filter(Boolean)
  const isInvalid = itemsNeeded && actionItems.length === 0
  const reviewerId = review.reviewer_id?.slice(0, 8)

  function submitReview() {
    completeReview.mutate({
      decisions_made: decisionsMade,
      decisions_clear: decisionsClear,
      action_items_needed: itemsNeeded,
      action_items: itemsNeeded ? actionItems : [],
    }, {
      onError: (error) => toast.error(error.message),
    })
  }

  const questions = [
    ['Were decisions made?', decisionsMade, setDecisionsMade],
    ['Are decisions clearly stated?', decisionsClear, setDecisionsClear],
    ['Are action items needed?', itemsNeeded, setItemsNeeded],
  ] as const

  return (
    <section
      aria-label="Meeting final review"
      aria-busy={completeReview.isPending}
      className="bg-huddleroom-surface border border-huddleroom-border rounded-[4px]"
      style={{ padding: 12 }}
    >
      <h2 className="text-huddleroom-text-muted text-xs font-semibold" style={{ margin: '0 0 4px' }}>
        Final review
      </h2>
      <div className="text-huddleroom-text-muted text-xs" style={{ marginBottom: 10 }}>
        Requested reviewer{' '}
        <span style={{ fontFamily: MONO }}>
          {review.reviewer_kind}{reviewerId ? ` / ${reviewerId}` : ''}
        </span>
      </div>
      <fieldset style={{ border: 0, padding: 0, margin: `0 0 ${itemsNeeded ? 10 : 12}px` }}>
        <legend className="text-huddleroom-text-muted text-xs" style={{ marginBottom: 6, padding: 0 }}>
          Review questions
        </legend>
        <div style={{ display: 'grid', gap: 8 }}>
          {questions.map(([label, checked, setChecked]) => (
            <label key={label} className="text-huddleroom-text-primary text-[13px]" style={{ display: 'flex', alignItems: 'center', gap: 8, minHeight: 44, paddingTop: 10, paddingBottom: 10 }}>
              <input
                type="checkbox"
                checked={checked}
                disabled={completeReview.isPending}
                onChange={(event) => setChecked(event.target.checked)}
                style={{ width: 16, height: 16 }}
              />
              {label}
            </label>
          ))}
        </div>
      </fieldset>
      {itemsNeeded && (
        <Textarea
          label="Action items"
          aria-label="Final review action items"
          rows={3}
          value={itemsText}
          disabled={completeReview.isPending}
          onChange={(event) => setItemsText(event.target.value)}
        />
      )}
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button
          variant="primary"
          size="sm"
          disabled={completeReview.isPending || isInvalid}
          onClick={submitReview}
        >
          {completeReview.isPending
            ? 'submitting...'
            : itemsNeeded ? 'Add action items and conclude' : 'Close with no action items'}
        </Button>
      </div>
    </section>
  )
}

function MeetingLiveView({ meetingId }: { meetingId: string }) {
  const navigate = useNavigate()
  const qc = useQueryClient()
  const projectId = useUIStore((s) => s.activeProjectId)

  // Data queries
  const { data: meeting, isLoading: meetingLoading, isError: meetingError, error: meetingErrorObj } = useMeeting(projectId, meetingId)
  const { data: apiTurns } = useMeetingTurns(projectId, meetingId)
  const { data: decisions, isLoading: decisionsLoading } = useMeetingDecisions(projectId, meetingId)
  const { data: actionItems, isLoading: actionItemsLoading } = useMeetingActionItems(projectId, meetingId)
  const { data: agenda } = useMeetingAgenda(projectId, meetingId)
  const { data: finalReview } = useMeetingFinalReview(projectId, meetingId, meeting?.status === 'concluding')
  const { items: agentListForLiveView } = useAllAgents()

  // WS store
  const wsTurns = useMeetingWSStore((s) => s.turns)
  const wsEvents = useMeetingWSStore((s) => s.events)
  const wsConnected = useMeetingWSStore((s) => s.connected)
  const wsConnect = useMeetingWSStore((s) => s.connect)
  const wsDisconnect = useMeetingWSStore((s) => s.disconnect)

  // Mutations
  const submitHumanTurn = useSubmitHumanTurn(projectId, meetingId)
  const copyMeeting = useCopyMeeting()
  const endMeeting = useEndMeeting()
  const cancelMeeting = useCancelMeeting()
  const resumeMeeting = useResumeMeeting()

  // Modals
  const [editorContent, setEditorContent] = useState('')
  const [showGrantTurn, setShowGrantTurn] = useState(false)
  const [showAddAgendaItem, setShowAddAgendaItem] = useState(false)
  const [cancelConfirm, setCancelConfirm] = useState(false)
  const [endConfirm, setEndConfirm] = useState(false)
  const [verbose, setVerbose] = useState(false)

  // Scroll ref
  const scrollRef = useRef<HTMLDivElement>(null)

  // Connect WS on mount
  useEffect(() => {
    const token = getToken()
    wsConnect(meetingId, token)
    return () => wsDisconnect()
  }, [meetingId, wsConnect, wsDisconnect])

  // Poll for updates while viewing
  // ponytail: WS connection gates the poll interval. Decisions/agenda/action-item changes
  // are not currently pushed over /ws/meetings/:id (no backend producer for those event
  // types exists today), so this falls back to polling when WS is unavailable. Full
  // removal is future backend work once all event types are produced.
  useEffect(() => {
    if (wsConnected) return
    const interval = setInterval(() => {
      if (meeting) {
        qc.invalidateQueries({ queryKey: meetingKeys.detail(projectId, meetingId) })
        qc.invalidateQueries({ queryKey: meetingKeys.turns(projectId, meetingId) })
        qc.invalidateQueries({ queryKey: meetingKeys.agenda(projectId, meetingId) })
        qc.invalidateQueries({ queryKey: meetingKeys.decisions(projectId, meetingId) })
        qc.invalidateQueries({ queryKey: meetingKeys.actionItems(projectId, meetingId) })
      }
    }, 5000)
    return () => clearInterval(interval)
  }, [projectId, meetingId, qc, meeting, wsConnected])

  // Merge turns: API + WS
  const allTurns = useMemo(() => {
    const map = new Map<string, MeetingTurn>()
    for (const t of apiTurns ?? []) map.set(t.id, t)
    for (const t of wsTurns) map.set(t.id, t)
    return [...map.values()].sort((a, b) => a.created_at.localeCompare(b.created_at))
  }, [apiTurns, wsTurns])

  // Auto-scroll on new turns only when already at the bottom
  useEffect(() => {
    const el = scrollRef.current
    if (!el) return
    const isAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60
    if (isAtBottom) {
      el.scrollTop = el.scrollHeight
    }
  }, [allTurns.length])

  // Build agent lookup map
  const agentMap = useMemo(() => {
    const map = new Map<string, Agent>()
    agentListForLiveView.forEach((a: Agent) => map.set(a.id, a))
    return map
  }, [agentListForLiveView])

  function handleSubmitHumanTurn() {
    if (!editorContent.trim()) return
    submitHumanTurn.mutate(editorContent, {
      onSuccess: () => { setEditorContent(''); toast.success('Message sent') },
      onError: (e) => toast.error(e.message),
    })
  }

  function handleEndMeeting() {
    if (!meeting) return
    endMeeting.mutate(meeting.id, {
      onSuccess: () => {
        toast.success('Meeting ended')
        qc.invalidateQueries({ queryKey: ['meetings'] })
      },
      onError: (e) => toast.error(e.message),
    })
  }

  function handleCopyMeeting() {
    if (!meeting) return
    copyMeeting.mutate(meeting.id, {
      onSuccess: (copy) => {
        toast.success('Meeting copied')
        navigate(`/meetings/${copy.id}`)
      },
      onError: (e) => toast.error(e.message),
    })
  }

  function handleCancelMeeting() {
    if (!meeting) return
    cancelMeeting.mutate(meeting.id, {
      onSuccess: () => {
        toast.success('Meeting cancelled')
        qc.invalidateQueries({ queryKey: ['meetings'] })
      },
      onError: (e) => toast.error(e.message),
    })
  }

  function handleResumeMeeting() {
    if (!meeting) return
    resumeMeeting.mutate(meeting.id, {
      onSuccess: () => toast.success('Meeting resumed'),
      onError: (e) => toast.error(e.message),
    })
  }

  if (meetingLoading) {
    return (
      <div role="status" aria-busy="true" aria-label="Loading">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {Array.from({ length: 4 }).map((_, i) => (
            <SkeletonRow key={i} />
          ))}
        </div>
      </div>
    )
  }

  if (meetingError || !meeting) {
    return (
      <ErrorRecord
        error={meetingErrorObj}
        entity="meeting"
        action={<Button variant="ghost" onClick={() => navigate('/meetings')}>Back to meetings</Button>}
      />
    )
  }

  const meetingData = meeting
  const isActive = meetingData.status === 'active' || meetingData.status === 'scheduled' || meetingData.status === 'preparing'
  const canEndMeeting = meetingData.status === 'active' || meetingData.status === 'concluding'
  const canCancelMeeting = meetingData.status !== 'concluded' && meetingData.status !== 'cancelled'

  return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100%' }}>
            {/* Top row: back + title + status + controls */}
            <div className="border-b border-huddleroom-border" style={{ paddingBottom: 12 }}>
              <DetailHeader
                backTo="/meetings"
                backLabel="Meetings"
                title={meetingData.title}
                status={meetingData.status}
                actions={
                  <>
                    {meetingData.resume_state?.failed && (
                      <>
                        <StatusBadge status="failed" label="Model error" />
                        <Button
                          variant="primary"
                          size="sm"
                          disabled={resumeMeeting.isPending}
                          onClick={handleResumeMeeting}
                        >
                          <RotateCw size={12} />
                          {resumeMeeting.isPending ? 'Resuming…' : 'Resume'}
                        </Button>
                      </>
                    )}
                    {canEndMeeting && (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => setEndConfirm(true)}
                      >
                        End meeting
                      </Button>
                    )}
                    {canCancelMeeting && (
                      <Button
                        variant="danger"
                        size="sm"
                        onClick={() => setCancelConfirm(true)}
                      >
                        Cancel
                      </Button>
                    )}
                    <DropdownMenu.Root>
                      <DropdownMenu.Trigger asChild>
                        <Button variant="secondary" size="sm" aria-label="More meeting actions">
                          <MoreHorizontal size={14} />
                        </Button>
                      </DropdownMenu.Trigger>
                      <DropdownMenu.Portal>
                        <DropdownMenu.Content
                          align="end"
                          className="bg-huddleroom-surface border border-huddleroom-border rounded-[4px]"
                          style={{ padding: 4, minWidth: 160, zIndex: 50 }}
                        >
                          <DropdownMenu.Item
                            className="text-huddleroom-text-primary text-xs rounded-[2px]"
                            style={{ padding: '6px 10px', cursor: 'pointer', outline: 'none' }}
                            disabled={copyMeeting.isPending}
                            onSelect={handleCopyMeeting}
                          >
                            {copyMeeting.isPending ? 'Copying…' : 'Copy meeting'}
                          </DropdownMenu.Item>
                          <DropdownMenu.CheckboxItem
                            className="text-huddleroom-text-primary text-xs rounded-[2px]"
                            style={{ padding: '6px 10px', cursor: 'pointer', outline: 'none' }}
                            checked={verbose}
                            onCheckedChange={setVerbose}
                          >
                            <DropdownMenu.ItemIndicator>
                              <Check size={12} />
                            </DropdownMenu.ItemIndicator>
                            Verbose
                          </DropdownMenu.CheckboxItem>
                        </DropdownMenu.Content>
                      </DropdownMenu.Portal>
                    </DropdownMenu.Root>
                  </>
                }
              />
            </div>

            {/* Model error message */}
            {meetingData.resume_state?.failed && (
              <div className="border-b border-huddleroom-border" style={{ paddingBottom: 12 }}>
                <ErrorRecord
                  error={meetingData.resume_state.error ?? 'The last turn failed to generate.'}
                />
              </div>
            )}

            {/* Main content: left panel (transcript) + right panel (sidebar) */}
            <div className="flex flex-col min-[900px]:flex-row" style={{ display: 'flex', flex: 1, gap: 12, minHeight: 0 }}>
              {/* LEFT PANEL: Transcript + Input */}
              <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 12 }}>
                {meeting?.status === 'concluding' && finalReview && (
                  <FinalReviewSection key={meetingId} projectId={projectId} meetingId={meetingId} review={finalReview} />
                )}

                {/* Transcript */}
                <div
                  ref={scrollRef}
                  data-testid="meeting-transcript"
                  className="max-w-none bg-huddleroom-surface border border-huddleroom-border rounded-[4px]"
                  style={{
                    flex: 1, overflowY: 'auto', padding: '16px',
                    display: 'flex', flexDirection: 'column', gap: 12,
                  }}
                >
                  {allTurns.length === 0 ? (
                    <div className="text-huddleroom-text-muted text-[13px]" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 100 }}>
                      no messages yet
                    </div>
                  ) : (
                    allTurns.map((turn) => {
                      if (turn.is_human_turn) {
                        return (
                          <div key={turn.id} style={{ display: 'flex', justifyContent: 'flex-end' }}>
                            <div className="max-w-none bg-white border border-huddleroom-border rounded-md p-3" style={{ wordBreak: 'break-word' }}>
                              <div className="max-w-[75ch] text-[13px] text-huddleroom-text-primary" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.4 }}>
                                {turn.content}
                              </div>
                            </div>
                          </div>
                        )
                      }

                      // System turn (speaker_agent_id is null/undefined AND is_human_turn is false)
                      if (turn.speaker_agent_id == null) {
                        return (
                          <div key={turn.id} style={{ display: 'flex', justifyContent: 'center' }}>
                            <div className="max-w-none">
                              <div className="bg-white border border-huddleroom-border rounded-md p-3" style={{ wordBreak: 'break-word' }}>
                                <div className="max-w-[75ch] text-[13px] text-huddleroom-text-muted italic" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.4 }}>
                                  {turn.content}
                                </div>
                              </div>
                            </div>
                          </div>
                        )
                      }

                      // Agent turn
                      const agent = agentMap.get(turn.speaker_agent_id)
                      return (
                        <div key={turn.id} style={{ display: 'flex', justifyContent: 'flex-start' }}>
                          <div className="max-w-none">
                            <div className="text-huddleroom-text-primary text-[13px] font-semibold mb-1">
                              {agent?.name || turn.speaker_agent_id.slice(0, 8)}
                            </div>
                            <div className="bg-white border border-huddleroom-border rounded-md p-3" style={{ wordBreak: 'break-word' }}>
                              <div className="max-w-[75ch] text-[13px] text-huddleroom-text-primary" style={{ whiteSpace: 'pre-wrap', lineHeight: 1.4 }}>
                                {turn.content}
                              </div>
                            </div>
                          </div>
                        </div>
                      )
                    })
                  )}
                </div>

                {verbose && (
                  <div
                    data-testid="meeting-verbose-stream"
                    className="bg-huddleroom-surface border border-huddleroom-border rounded-[4px]"
                    style={{
                      maxHeight: 260, overflowY: 'auto', padding: '10px 12px',
                      display: 'flex', flexDirection: 'column', gap: 8,
                    }}
                  >
                    {wsEvents.length === 0 ? (
                      <div className="text-huddleroom-text-muted text-xs">
                        no websocket events yet
                      </div>
                    ) : (
                      wsEvents.map((event, index) => (
                        <details key={event.id ?? `${event.emitted_at ?? 'event'}-${index}`} className="border-b border-huddleroom-border" style={{ paddingBottom: 8 }}>
                          <summary className="text-huddleroom-text-primary text-xs" style={{ cursor: 'pointer' }}>
                            <span className="text-huddleroom-text-muted">{absolute(event.emitted_at)}</span>
                            {' · '}
                            <span className="text-huddleroom-text-muted">{event.event_type}</span>
                            {' · '}
                            {eventLabel(event)}
                            {event.source ? <> · <span className="text-huddleroom-text-muted">{event.source}</span></> : null}
                          </summary>
                          <pre className="bg-huddleroom-depth border border-huddleroom-border rounded-[3px] text-huddleroom-text-primary text-[11px] font-mono" style={{
                            margin: '8px 0 0', padding: 10, maxHeight: 220, overflow: 'auto',
                            lineHeight: 1.4,
                            whiteSpace: 'pre-wrap', wordBreak: 'break-word',
                          }}>
                            {compactJson(event)}
                          </pre>
                        </details>
                      ))
                    )}
                  </div>
                )}

                {/* Human input */}
                {isActive && (
                  <div data-testid="meeting-human-input" className="bg-huddleroom-surface border border-huddleroom-border rounded-[4px]" style={{ overflow: 'hidden' }}>
                    <Textarea
                      rows={4}
                      value={editorContent}
                      onChange={(e) => setEditorContent(e.target.value)}
                      onKeyDown={(e) => {
                        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                          e.preventDefault()
                          handleSubmitHumanTurn()
                        }
                      }}
                    />
                    <div className="bg-huddleroom-surface border-t border-huddleroom-border" style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, padding: '8px 12px' }}>
                      <Button
                        variant="primary"
                        size="sm"
                        data-testid="meeting-submit-human-turn"
                        disabled={!editorContent.trim() || submitHumanTurn.isPending}
                        onClick={handleSubmitHumanTurn}
                      >
                        <Send size={12} /> Send
                      </Button>
                    </div>
                  </div>
                )}
              </div>

              {/* RIGHT PANEL: Sidebar */}
              <div
                className="w-full min-[900px]:w-[340px] bg-huddleroom-surface border border-huddleroom-border rounded-[4px]"
                style={{ flexShrink: 0, overflowY: 'auto', display: 'flex', flexDirection: 'column' }}
              >
                {/* Agenda section */}
                <AgendaSection
                  agenda={agenda ?? []}
                  meeting={meetingData}
                  onAddItem={() => setShowAddAgendaItem(true)}
                />

                {/* Participants section */}
                <ParticipantsSection
                  meeting={meetingData}
                  agents={agentListForLiveView}
                  onGrantTurn={() => setShowGrantTurn(true)}
                />

                {/* Decisions section */}
                <DecisionsSection
                  decisions={decisions ?? []}
                  actionItems={actionItems ?? []}
                  meeting={meetingData}
                  agents={agentListForLiveView}
                  isLoading={decisionsLoading}
                />

                {/* Action items section */}
                <ActionItemsSection
                  actionItems={actionItems ?? []}
                  meeting={meetingData}
                  isLoading={actionItemsLoading}
                />
              </div>
            </div>

            {/* Modals */}
            <GrantTurnModal open={showGrantTurn} onClose={() => setShowGrantTurn(false)} meeting={meetingData} agentList={agentListForLiveView} />
            <AddAgendaItemModal open={showAddAgendaItem} onClose={() => setShowAddAgendaItem(false)} meeting={meetingData} />
            <ConfirmDialog
              open={cancelConfirm}
              onOpenChange={setCancelConfirm}
              title="Cancel this meeting?"
              consequence="This ends the live meeting. Any in-progress turns will be lost."
              confirmLabel="Cancel meeting"
              onConfirm={() => { handleCancelMeeting(); setCancelConfirm(false) }}
              isPending={cancelMeeting?.isPending}
            />
            <ConfirmDialog
              open={endConfirm}
              onOpenChange={setEndConfirm}
              title="End this meeting?"
              consequence="This moves the meeting straight to final review — no further turns will be taken."
              confirmLabel="End meeting"
              onConfirm={() => { handleEndMeeting(); setEndConfirm(false) }}
              isPending={endMeeting?.isPending}
            />
          </div>
  )
}

// ─── Sidebar sections ─────────────────────────────────────────────────────────

function AgendaSection({
  agenda,
  meeting,
  onAddItem,
}: {
  agenda: AgendaItem[]
  meeting: Meeting
  onAddItem: () => void
}) {
  const [open, setOpen] = useState(true)
  const advanceAgenda = useAdvanceAgenda(meeting.project_id, meeting.id)

  const activeItem = agenda.find((a) => a.status === 'active')

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen}>
      <Collapsible.Trigger className="border-huddleroom-border" style={{
        background: 'none', borderWidth: '0 0 1px 0', borderStyle: 'solid',
        padding: '10px 16px', cursor: 'pointer', display: 'flex',
        alignItems: 'center', justifyContent: 'space-between', width: '100%',
      }}>
        <StructuralLabel>Agenda</StructuralLabel>
        <ChevronDown size={12} style={{ transform: open ? 'rotate(0)' : 'rotate(-90deg)', transition: 'transform 120ms' }} />
      </Collapsible.Trigger>
      <Collapsible.Content className="border-b border-huddleroom-border">
        <div data-testid="meeting-agenda" style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 8 }}>
          {agenda.length === 0 ? (
            <div className="text-xs text-huddleroom-text-muted">no agenda items</div>
          ) : (
            <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
              {agenda.map((item) => (
                <li key={item.id} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 8px' }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div className="text-[13px] text-huddleroom-text-primary" style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {item.title}
                    </div>
                    <div className="text-xs text-huddleroom-text-muted">
                      {item.resolution_summary || item.resolution_kind || item.status}
                    </div>
                  </div>

                  {item.id === activeItem?.id && meeting.status === 'active' && (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => {
                        advanceAgenda.mutate(item.id, {
                          onSuccess: () => toast.success('Agenda advanced'),
                          onError: (e) => toast.error(e.message),
                        })
                      }}
                      className="text-xs"
                      style={{ flexShrink: 0, padding: '2px 8px' }}
                    >
                      Advance
                    </Button>
                  )}
                </li>
              ))}
            </ul>
          )}

          {(meeting.status === 'scheduled' || meeting.status === 'preparing') && (
            <Button
              variant="secondary"
              size="sm"
              onClick={onAddItem}
              className="text-xs"
              style={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: 4 }}
            >
              <Plus size={10} /> add
            </Button>
          )}
        </div>
      </Collapsible.Content>
    </Collapsible.Root>
  )
}

// Local participant pill — mirrors RoomPanel's AgentPill (not shared, per design spec).
function ParticipantAgentPill({ adapterType, provider }: { adapterType?: Agent['adapter_type']; provider?: string }) {
  const text = adapterType && provider ? `${adapterType} · ${provider}` : adapterType ?? provider
  if (!text) return null
  return (
    <span className="inline-flex items-center rounded-full bg-huddleroom-depth px-2 py-0.5 font-mono text-micro font-medium leading-none text-huddleroom-text-secondary">
      {text}
    </span>
  )
}

function ParticipantsSection({
  meeting,
  agents,
  onGrantTurn,
}: {
  meeting: Meeting
  agents: Agent[]
  onGrantTurn: () => void
}) {
  const [open, setOpen] = useState(true)
  const agentMap = useMemo(() => {
    const map = new Map<string, Agent>()
    agents.forEach((a) => map.set(a.id, a))
    return map
  }, [agents])

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen}>
      <Collapsible.Trigger className="border-huddleroom-border" style={{
        background: 'none', borderWidth: '0 0 1px 0', borderStyle: 'solid',
        padding: '10px 16px', cursor: 'pointer', display: 'flex',
        alignItems: 'center', justifyContent: 'space-between', width: '100%',
      }}>
        <StructuralLabel>Participants</StructuralLabel>
        <ChevronDown size={12} style={{ transform: open ? 'rotate(0)' : 'rotate(-90deg)', transition: 'transform 120ms' }} />
      </Collapsible.Trigger>
      <Collapsible.Content className="border-b border-huddleroom-border">
        <div data-testid="meeting-participants" role="list" style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column' }}>
          {meeting.participant_agent_ids.length === 0 ? (
            <div className="text-xs text-huddleroom-text-muted">no participants</div>
          ) : (
            meeting.participant_agent_ids.map((agentId) => {
              const agent = agentMap.get(agentId)
              return (
                <div key={agentId} className="flex items-center gap-x-3 py-1.5">
                  {agent ? (
                    <>
                      <span className="text-body font-semibold text-huddleroom-text-primary">{agent.name}</span>
                      <ParticipantAgentPill adapterType={agent.adapter_type} provider={agent.provider} />
                    </>
                  ) : (
                    <span className="font-mono text-body text-huddleroom-text-muted">{agentId.slice(0, 8)}</span>
                  )}
                </div>
              )
            })
          )}

          {meeting.status === 'active' && (
            <Button
              variant="secondary"
              size="sm"
              onClick={onGrantTurn}
              className="text-xs"
              style={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: 4 }}
            >
              <Zap size={10} /> Grant turn
            </Button>
          )}
        </div>
      </Collapsible.Content>
    </Collapsible.Root>
  )
}

function resolveAgentNames(ids: string[] | undefined, agentMap: Map<string, Agent>): string | undefined {
  if (!ids || ids.length === 0) return undefined
  return ids.map((id) => agentMap.get(id)?.name || id.slice(0, 8)).join(', ')
}

function DecisionsSection({
  decisions,
  actionItems,
  meeting,
  agents,
  isLoading,
}: {
  decisions: MeetingDecision[]
  actionItems: MeetingActionItem[]
  meeting: Meeting
  agents: Agent[]
  isLoading?: boolean
}) {
  const [open, setOpen] = useState(true)
  const [vetoConfirm, setVetoConfirm] = useState<string | null>(null)
  const vetoDecision = useVetoDecision(meeting.project_id, meeting.id)
  const agentMap = useMemo(() => {
    const map = new Map<string, Agent>()
    agents.forEach((a) => map.set(a.id, a))
    return map
  }, [agents])

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen}>
      <Collapsible.Trigger className="border-huddleroom-border" style={{
        background: 'none', borderWidth: '0 0 1px 0', borderStyle: 'solid',
        padding: '10px 16px', cursor: 'pointer', display: 'flex',
        alignItems: 'center', justifyContent: 'space-between', width: '100%',
      }}>
        <StructuralLabel>Decisions ({decisions.length})</StructuralLabel>
        <ChevronDown size={12} style={{ transform: open ? 'rotate(0)' : 'rotate(-90deg)', transition: 'transform 120ms' }} />
      </Collapsible.Trigger>
      <Collapsible.Content className="border-b border-huddleroom-border">
        <div data-testid="meeting-decisions" style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 8 }}>
          {isLoading ? (
            <SkeletonList count={2} />
          ) : decisions.length === 0 ? (
            <div className="text-xs text-huddleroom-text-muted">no decisions</div>
          ) : (
            decisions.map((dec) => {
              const linkedItems = actionItems.filter((item) => item.depends_on_decision_id === dec.id)
              return (
                <div key={dec.id} className="bg-huddleroom-depth rounded-[2px]" style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: '8px' }}>
                  <Record
                    keyWidth={84}
                    rows={[
                      { key: 'Question', value: dec.question },
                      { key: 'Chosen', value: (dec.chosen_option || dec.content) ? <span className="font-semibold">{dec.chosen_option || dec.content}</span> : undefined },
                      { key: 'Rationale', value: dec.rationale },
                      { key: 'Rejected', value: dec.alternatives_rejected?.length ? dec.alternatives_rejected.join(', ') : undefined },
                      { key: 'Agreed', value: resolveAgentNames(dec.participants_agreed, agentMap) },
                      { key: 'Dissent', value: resolveAgentNames(dec.dissent, agentMap) },
                      {
                        key: 'Actions',
                        value: linkedItems.length
                          ? (
                            <div className="space-y-1">
                              {linkedItems.map((item) => (
                                <div key={item.id}>→ {item.description}</div>
                              ))}
                            </div>
                          )
                          : undefined,
                      },
                      { key: 'Veto', value: dec.veto_reason ? <span className="text-huddleroom-status-amber">{dec.veto_reason}</span> : undefined },
                    ]}
                  />
                  <div className="text-xs text-huddleroom-text-muted">by <span style={{ fontFamily: MONO }}>{agentMap.get(dec.decided_by)?.name || readableDecidedBy(dec.decided_by)}</span></div>
                  {meeting.status === 'active' && (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={() => setVetoConfirm(dec.id)}
                      className="text-huddleroom-danger text-xs"
                      style={{ alignSelf: 'flex-start', padding: '2px 6px' }}
                    >
                      Veto
                    </Button>
                  )}
                </div>
              )
            })
          )}
        </div>
      </Collapsible.Content>
      <ConfirmDialog
        open={vetoConfirm !== null}
        onOpenChange={(open) => { if (!open) setVetoConfirm(null) }}
        title="Veto this decision?"
        consequence="This blocks the proposed decision from being accepted."
        confirmLabel="Veto"
        onConfirm={() => { if (vetoConfirm) { vetoDecision.mutate(vetoConfirm, {
          onSuccess: () => toast.success('Decision vetoed'),
          onError: (e) => toast.error(e.message),
        }) }; setVetoConfirm(null) }}
        isPending={vetoDecision?.isPending}
      />
    </Collapsible.Root>
  )
}

function ActionItemsSection({
  actionItems,
  meeting,
  isLoading,
}: {
  actionItems: MeetingActionItem[]
  meeting: Meeting
  isLoading?: boolean
}) {
  const [open, setOpen] = useState(false)
  const updateActionItem = useUpdateActionItem(meeting.project_id, meeting.id)
  const unlinkedItems = actionItems.filter((item) => !item.depends_on_decision_id)

  return (
    <Collapsible.Root open={open} onOpenChange={setOpen}>
      <Collapsible.Trigger className="border-huddleroom-border" style={{
        background: 'none', borderWidth: '0 0 1px 0', borderStyle: 'solid',
        padding: '10px 16px', cursor: 'pointer', display: 'flex',
        alignItems: 'center', justifyContent: 'space-between', width: '100%',
      }}>
        <StructuralLabel>Action items ({unlinkedItems.length})</StructuralLabel>
        <ChevronDown size={12} style={{ transform: open ? 'rotate(0)' : 'rotate(-90deg)', transition: 'transform 120ms' }} />
      </Collapsible.Trigger>
      <Collapsible.Content>
        <div style={{ padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 8 }}>
          {isLoading ? (
            <SkeletonList count={2} />
          ) : unlinkedItems.length === 0 ? (
            <div className="text-xs text-huddleroom-text-muted">no action items</div>
          ) : (
            unlinkedItems.map((item) => (
              <div key={item.id} className="bg-huddleroom-depth rounded-[2px]" style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '8px' }}>
                <button style={{
                  background: 'none', border: 'none', padding: 0, cursor: 'pointer', display: 'flex', alignItems: 'center',
                  color: item.status === 'done' ? BASE_STATUS_COLORS.green : UI_COLORS.textMuted,
                }}
                  aria-label={item.status === 'done' ? `Mark action item "${item.description}" as in progress` : `Mark action item "${item.description}" as done`}
                  title={item.status === 'done' ? 'Mark as in progress' : 'Mark as done'}
                  onClick={() => {
                    const newStatus = item.status === 'done' ? 'in_progress' : 'done'
                    updateActionItem.mutate({ actionItemId: item.id, data: { status: newStatus } }, {
                      onSuccess: () => toast.success('Action item updated'),
                      onError: (e) => toast.error(e.message),
                    })
                  }}
                >
                  {item.status === 'done' ? <CheckCircle2 size={14} /> : <Circle size={14} />}
                </button>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="text-[13px] text-huddleroom-text-primary" style={{
                    textDecoration: item.status === 'done' ? 'line-through' : 'none',
                  }}>
                    {item.description}
                  </div>
                  <div className="text-xs text-huddleroom-text-muted">
                    {item.assignee_agent_id ? <>assigned: <span className="font-mono">{item.assignee_agent_id.slice(0, 8)}</span></> : 'unassigned'}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </Collapsible.Content>
    </Collapsible.Root>
  )
}

// ─── MeetingsPage (entry point) ───────────────────────────────────────────────

export function MeetingsPage() {
  useDocumentTitle('Meetings')
  const { meetingId } = useParams<{ meetingId?: string }>()
  if (meetingId) return <MeetingLiveView meetingId={meetingId} />
  return <MeetingListView />
}
