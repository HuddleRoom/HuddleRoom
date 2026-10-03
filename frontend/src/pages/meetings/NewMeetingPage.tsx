import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Plus, X } from 'lucide-react'
import { Button, Input, Textarea, Select, UI_COLORS, PageHeader, NoProjectSelected } from '@/components/common/uiPrimitives'
import { Panel } from '@/components/common/Panel'
import { useUIStore } from '@/stores/ui'
import { useAllActiveAgents } from '@/api/agents'
import { useCreateMeeting } from '@/api/meetings'
import type { CreateMeetingRequest } from '@/api/meetings'
import type { Agent } from '@/lib/types'

interface NewMeetingFormState {
  title: string
  meeting_type: string
  turn_strategy: string
  deadlock_strategy: string
  max_duration_minutes: number
  auto_start: boolean
  scheduled_at: string
  signal_check_enabled: boolean
  participant_agent_ids: string[]
  organizer_agent_id: string
  planner_agent_id: string
  agenda_items: { title: string; description: string; question: string }[]
}

const BLANK_FORM: NewMeetingFormState = {
  title: '',
  meeting_type: 'decision',
  turn_strategy: 'round_robin',
  deadlock_strategy: 'human_intervention',
  max_duration_minutes: 30,
  auto_start: true,
  scheduled_at: '',
  signal_check_enabled: false,
  participant_agent_ids: [],
  organizer_agent_id: '',
  planner_agent_id: '',
  agenda_items: [{ title: '', description: '', question: '' }],
}

export function NewMeetingPage() {
  const navigate = useNavigate()
  const projectId = useUIStore((s) => s.activeProjectId)
  const setCreateProjectOpen = useUIStore((s) => s.setCreateProjectOpen)
  const { items: agentList } = useAllActiveAgents()
  const createMeeting = useCreateMeeting(projectId)
  const [form, setForm] = useState<NewMeetingFormState>(BLANK_FORM)

  function set(patch: Partial<NewMeetingFormState>) {
    setForm((p) => ({ ...p, ...patch }))
  }

  function toggleParticipant(agentId: string) {
    setForm((p) => {
      const exists = p.participant_agent_ids.includes(agentId)
      return {
        ...p,
        participant_agent_ids: exists
          ? p.participant_agent_ids.filter((id) => id !== agentId)
          : [...p.participant_agent_ids, agentId],
      }
    })
  }

  function addAgendaItem() {
    setForm((p) => ({
      ...p,
      agenda_items: [...p.agenda_items, { title: '', description: '', question: '' }],
    }))
  }

  function removeAgendaItem(index: number) {
    setForm((p) => ({
      ...p,
      agenda_items: p.agenda_items.filter((_, i) => i !== index),
    }))
  }

  function updateAgendaItem(index: number, patch: Partial<{ title: string; description: string; question: string }>) {
    setForm((p) => {
      const items = [...p.agenda_items]
      items[index] = { ...items[index], ...patch }
      return { ...p, agenda_items: items }
    })
  }

  function handleSubmit() {
    const payload: CreateMeetingRequest = {
      title: form.title.trim(),
      meeting_type: form.meeting_type,
      turn_strategy: form.turn_strategy,
      deadlock_strategy: form.deadlock_strategy,
      max_duration_minutes: form.max_duration_minutes,
      auto_start: form.auto_start,
      signal_check_enabled: form.signal_check_enabled,
      participant_agent_ids: form.participant_agent_ids,
      agenda_items: form.agenda_items
        .filter((item) => item.title.trim())
        .map((item, idx) => ({
          title: item.title.trim(),
          description: item.description.trim() || null,
          question: item.question.trim() || null,
          order: idx,
        })),
      ...(form.scheduled_at ? { scheduled_at: form.scheduled_at } : {}),
      ...(form.organizer_agent_id ? { organizer_agent_id: form.organizer_agent_id } : {}),
      ...(form.planner_agent_id ? { planner_agent_id: form.planner_agent_id } : {}),
    }

    createMeeting.mutate(payload, {
      onSuccess: (meeting) => {
        toast.success('Meeting created')
        navigate(`/meetings/${meeting.id}`)
      },
      onError: (e) => toast.error(e.message),
    })
  }

  const isPending = createMeeting.isPending
  const canSubmit = form.title.trim().length > 0 && form.participant_agent_ids.length > 0

  if (!projectId) {
    return (
      <NoProjectSelected
        pageTitle="New meeting"
        pageIntro="Meetings are scoped to the currently selected project. Choose a workspace first, then set up the meeting's type, strategy, participants, and agenda."
        title="Select a project to create a meeting"
        detail="The top-bar switcher is the entry point for project-scoped work. Once you choose a project, this page will let you configure and launch a new meeting for it."
        primaryAction={{ label: 'New project', onClick: () => setCreateProjectOpen(true) }}
      />
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <PageHeader title="New meeting" />

      <Panel header={{ title: 'Details' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 14 }}>
          <Input
            label="Title *"
            value={form.title}
            placeholder="meeting title"
            onChange={(e) => set({ title: e.target.value })}
          />
          <Select
            label="Meeting Type *"
            value={form.meeting_type}
            onChange={(e) => set({ meeting_type: e.target.value })}
          >
            <option value="decision">decision</option>
            <option value="review">review</option>
            <option value="standup">standup</option>
            <option value="escalation">escalation</option>
            <option value="adhoc">adhoc</option>
          </Select>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 14 }}>
          <Select
            label="Turn Strategy"
            value={form.turn_strategy}
            onChange={(e) => set({ turn_strategy: e.target.value })}
          >
            <option value="round_robin">round_robin</option>
            <option value="agenda_driven">agenda_driven</option>
            <option value="moderated">moderated</option>
            <option value="organizer_controlled">organizer_controlled</option>
          </Select>
          <Select
            label="Deadlock Strategy"
            value={form.deadlock_strategy}
            onChange={(e) => set({ deadlock_strategy: e.target.value })}
          >
            <option value="human_intervention">human_intervention</option>
            <option value="majority_rules">majority_rules</option>
            <option value="table_item">table_item</option>
            <option value="escalate">escalate</option>
          </Select>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 14 }}>
          <Input
            label="Max Duration (minutes)"
            type="number"
            value={form.max_duration_minutes}
            onChange={(e) => set({ max_duration_minutes: parseInt(e.target.value, 10) || 30 })}
          />
          <Input
            label="Scheduled at (optional)"
            type="datetime-local"
            value={form.scheduled_at}
            onChange={(e) => set({ scheduled_at: e.target.value })}
          />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 14 }}>
          <Select
            label="Organizer agent"
            value={form.organizer_agent_id}
            onChange={(e) => set({ organizer_agent_id: e.target.value })}
          >
            <option value="">— none —</option>
            {agentList.map((a: Agent) => (
              <option key={a.id} value={a.id}>{a.name}</option>
            ))}
          </Select>
          <Select
            label="Planner agent"
            value={form.planner_agent_id}
            onChange={(e) => set({ planner_agent_id: e.target.value })}
          >
            <option value="">— none —</option>
            {agentList.map((a: Agent) => (
              <option key={a.id} value={a.id}>{a.name}</option>
            ))}
          </Select>
        </div>

        <div style={{ display: 'flex', gap: 24 }}>
          <label className="text-huddleroom-text-muted text-xs" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', minHeight: 44, paddingTop: 10, paddingBottom: 10 }}>
            <input
              type="checkbox"
              checked={form.auto_start}
              onChange={(e) => set({ auto_start: e.target.checked })}
              style={{ width: 16, height: 16, cursor: 'pointer', accentColor: UI_COLORS.primary }}
            />
            Auto start
          </label>
          <label className="text-huddleroom-text-muted text-xs" style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', minHeight: 44, paddingTop: 10, paddingBottom: 10 }}>
            <input
              type="checkbox"
              checked={form.signal_check_enabled}
              onChange={(e) => set({ signal_check_enabled: e.target.checked })}
              style={{ width: 16, height: 16, cursor: 'pointer', accentColor: UI_COLORS.primary }}
            />
            Signal check enabled
          </label>
        </div>
      </Panel>

      <Panel header={{ title: 'Participants and agenda' }}>
        <div style={{ marginBottom: 14 }}>
          <label className="text-huddleroom-text-muted text-xs" style={{ display: 'block', marginBottom: 4 }}>Participant agents *</label>
          <div className="bg-huddleroom-surface border border-huddleroom-border rounded-[3px]" style={{
            maxHeight: 200, overflowY: 'auto', padding: 8,
          }}>
            {agentList.length === 0 ? (
              <div className="text-huddleroom-text-muted text-xs" style={{ padding: '8px 4px' }}>no agents available</div>
            ) : (
              agentList.map((agent: Agent) => (
                <label key={agent.id} className="rounded-[2px]" style={{
                  display: 'flex', alignItems: 'center', gap: 8, padding: '4px 8px',
                  cursor: 'pointer', minHeight: 44,
                }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLLabelElement).style.background = UI_COLORS.depth }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLLabelElement).style.background = 'none' }}
                >
                  <input
                    type="checkbox"
                    checked={form.participant_agent_ids.includes(agent.id)}
                    onChange={() => toggleParticipant(agent.id)}
                    style={{ width: 16, height: 16, cursor: 'pointer', accentColor: UI_COLORS.primary }}
                  />
                  <span className="text-xs text-huddleroom-text-primary">{agent.name}</span>
                </label>
              ))
            )}
          </div>
        </div>

        <div>
          <label className="text-huddleroom-text-muted text-xs" style={{ display: 'block', marginBottom: 4 }}>Agenda items (optional)</label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {form.agenda_items.map((item, idx) => (
              <div key={idx} className="bg-huddleroom-surface border border-huddleroom-border rounded-[3px]" style={{
                padding: 12, display: 'flex', flexDirection: 'column', gap: 8,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span className="text-huddleroom-text-muted text-[11px]">item {idx + 1}</span>
                  {form.agenda_items.length > 1 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeAgendaItem(idx)}
                      className="text-huddleroom-danger text-xs"
                      style={{ padding: 0 }}
                    >
                      <X size={12} /> remove
                    </Button>
                  )}
                </div>

                <Input
                  value={item.title}
                  placeholder="agenda item title"
                  onChange={(e) => updateAgendaItem(idx, { title: e.target.value })}
                />

                <Input
                  value={item.description}
                  placeholder="description (optional)"
                  onChange={(e) => updateAgendaItem(idx, { description: e.target.value })}
                />

                <Input
                  value={item.question}
                  placeholder="question (optional)"
                  onChange={(e) => updateAgendaItem(idx, { question: e.target.value })}
                />
              </div>
            ))}

            <Button
              variant="secondary"
              size="sm"
              onClick={addAgendaItem}
              style={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: 6 }}
            >
              <Plus size={12} /> add agenda item
            </Button>
          </div>
        </div>
      </Panel>

      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
        <Button variant="secondary" onClick={() => navigate('/meetings')}>
          Cancel
        </Button>
        <Button
          variant="primary"
          disabled={!canSubmit || isPending}
          onClick={handleSubmit}
        >
          {isPending ? 'Creating...' : 'Create meeting'}
        </Button>
      </div>
    </div>
  )
}
