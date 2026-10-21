import { describe, expect, it } from 'vitest'
import type {
  NeedsYouReference,
  OrchestrationAuthorityDecisionRecord,
  OrchestrationBlocker,
  OrchestrationGate,
  OrchestrationMeetingCommitment,
  OrchestrationProcessRunRecord,
  OrchestrationWarningRecord,
} from '@/lib/types'
import * as presentation from './needsYouPresentation'

type QueueRow =
  | { kind: 'blocker'; key: string; blocker: OrchestrationBlocker }
  | { kind: 'lm-retry'; key: string; process: OrchestrationProcessRunRecord; message: string }
  | { kind: 'answer'; key: string; decision: OrchestrationAuthorityDecisionRecord; process: OrchestrationProcessRunRecord | undefined }
  | { kind: 'answer-batch'; key: string; decisions: OrchestrationAuthorityDecisionRecord[]; process: OrchestrationProcessRunRecord | undefined }
  | { kind: 'goal-definition-questions'; key: string; decisions: OrchestrationAuthorityDecisionRecord[]; process: OrchestrationProcessRunRecord | undefined }
  | { kind: 'recovery-proposal'; key: string; decision: OrchestrationAuthorityDecisionRecord }
  | { kind: 'gate'; key: string; gate: OrchestrationGate; description: string | undefined }
  | { kind: 'meeting-commitment'; key: string; commitment: OrchestrationMeetingCommitment }
  | { kind: 'warning'; key: string; warning: OrchestrationWarningRecord; process: OrchestrationProcessRunRecord | undefined; canAcknowledge: boolean }

type PresentedRow = {
  title: string
  explanation: string
  nextStep: string
  actionOwner: 'owner' | 'maintainer' | 'unknown'
  details: string
  references: NeedsYouReference[]
}

type PresentationUnderTest = typeof presentation & {
  conciseNeedsYouTitle: (candidate: string | null, fallback: string) => string
  presentNeedsYouRow: (row: QueueRow, references: readonly NeedsYouReference[]) => PresentedRow
  queueOccurrence: (
    row: QueueRow,
    allDecisions: readonly OrchestrationAuthorityDecisionRecord[],
  ) => { key: string; occurredAt: string | null }
  compareQueueOccurrences: (
    a: { key: string; occurredAt: string | null },
    b: { key: string; occurredAt: string | null },
  ) => number
  validateNeedsYouEdit: (
    snapshot: {
      rowKey: string
      decisionIds: readonly string[]
      decisions: readonly OrchestrationAuthorityDecisionRecord[]
    },
    authoritative: readonly OrchestrationAuthorityDecisionRecord[] | undefined,
  ) => 'current' | 'terminal' | 'stale' | 'unavailable'
  validateRuntimeAnswerReceipt: (
    reviewed: OrchestrationAuthorityDecisionRecord,
    submitted: { selectedOption: string; reason: string | null },
    result: {
      decision: OrchestrationAuthorityDecisionRecord
      process?: OrchestrationProcessRunRecord | null
      continuation_applied?: boolean
    },
  ) => 'accepted' | 'stale'
}

const helpers = presentation as PresentationUnderTest
const BASE_TIME = '2026-10-14T08:00:00Z'

function process(overrides: Partial<OrchestrationProcessRunRecord> = {}): OrchestrationProcessRunRecord {
  return {
    id: 'process-1', goal_id: 'goal-1', run_id: 'run-1', process_type: 'goal_definition',
    process_version: 1, status: 'running', trigger_reason: 'test', input_snapshot: {}, outputs: {},
    skipped_by: null, override_reason: null, superseded_by_id: null,
    started_at: BASE_TIME, completed_at: null, created_at: BASE_TIME, updated_at: BASE_TIME,
    ...overrides,
  }
}

function decision(overrides: Partial<OrchestrationAuthorityDecisionRecord> = {}): OrchestrationAuthorityDecisionRecord {
  return {
    id: 'decision-1', goal_id: 'goal-1', run_id: 'run-1', decision_key: 'owner_question',
    title: 'Release approval', status: 'pending', authority: 'human', authority_agent_id: null,
    source_process_run_id: null, question: 'Should the release proceed?',
    context: 'The verification evidence is incomplete.', options: ['approve', 'wait'],
    recommendation: null, consequences: 'Waiting delays the release.', selected_option: null, reason: null,
    decided_by_user_id: null, decided_by_agent_id: null, overrides_recommendation: false,
    created_warning_id: null, related_gate_id: null, related_action_id: null,
    asked_at: BASE_TIME, decided_at: null, created_at: BASE_TIME, updated_at: BASE_TIME,
    ...overrides,
  }
}

function warning(overrides: Partial<OrchestrationWarningRecord> = {}): OrchestrationWarningRecord {
  return {
    id: 'warning-1', goal_id: 'goal-1', run_id: 'run-1', warning_type: 'coverage_gap', severity: 'warning',
    message: 'Coverage evidence is incomplete.', source_process_run_id: null, related_gate_id: null,
    related_action_id: null, related_agent_id: null, source_agent_review_id: null,
    related_authority_decision_id: null, acknowledged_by: null, acknowledged_at: null, active: true,
    resolved_by: null, resolved_reason: null, resolved_at: null, blocks_completion: false,
    created_at: BASE_TIME, updated_at: BASE_TIME,
    ...overrides,
  }
}

function gate(overrides: Partial<OrchestrationGate> = {}): OrchestrationGate {
  return {
    id: 'gate-1', run_id: 'run-1', success_criterion_key: 'ship', gate_type: 'evidence',
    required_evidence: {}, status: 'failed', failure_reason: 'No verification evidence was submitted.',
    created_at: BASE_TIME, updated_at: BASE_TIME, accepted_at: null, failed_at: BASE_TIME,
    ...overrides,
  }
}

function commitment(overrides: Partial<OrchestrationMeetingCommitment> = {}): OrchestrationMeetingCommitment {
  return {
    id: 'commitment-1', meeting_id: 'meeting-1', meeting_title: 'Release review',
    description: 'Document the deployment decision.', state: 'task_in_flight', task_id: 'task-1',
    graph_run_id: null, reason: null, created_at: BASE_TIME,
    ...overrides,
  }
}

function recoveryDecision(overrides: Partial<OrchestrationAuthorityDecisionRecord> = {}): OrchestrationAuthorityDecisionRecord {
  return decision({
    id: 'recovery-1', decision_key: 'blocker_recovery:blocker-1', title: 'Fix proposal',
    question: 'Approve the proposed recovery?', context: null, options: ['approve_fix', 'reject'],
    recovery: {
      kind: 'recovery_proposal_v1', proposal_id: 'proposal-1', version: 1, status: 'pending_approval',
      problem: 'No reviewer can verify the completed work.', general_updates: ['Add a reviewer'],
      after_approval: 'The verification step restarts.', global_effect: false,
      exact_details: null, evidence: null, error: null, source_blocker_id: 'blocker-1',
      source_action_id: null, source_decision_id: null,
    },
    ...overrides,
  })
}

function reference(overrides: Partial<NeedsYouReference> = {}): NeedsYouReference {
  return {
    kind: 'task', id: '10000000-0000-4000-8000-000000000001', project_id: 'project-1',
    goal_id: 'goal-1', label: 'Release checklist',
    ...overrides,
  }
}

function rowsByKind(): Record<QueueRow['kind'], QueueRow> {
  const baselineProcess = process()
  return {
    blocker: { kind: 'blocker', key: 'blocker:evidence', blocker: { kind: 'evidence_gap', reason: 'Verification is blocked.' } },
    'lm-retry': { kind: 'lm-retry', key: 'lm-retry:process-1', process: baselineProcess, message: 'The model request failed.' },
    answer: { kind: 'answer', key: 'answer:decision-1', decision: decision(), process: undefined },
    'answer-batch': {
      kind: 'answer-batch', key: 'answer-batch:process-1:review-1,review-2',
      decisions: [
        decision({ id: 'review-1', decision_key: 'agent_definition_review:proposal:agent-1' }),
        decision({ id: 'review-2', decision_key: 'agent_definition_review:proposal:agent-2' }),
      ],
      process: process({ process_type: 'agent_definition_review' }),
    },
    'goal-definition-questions': {
      kind: 'goal-definition-questions', key: 'goal-definition-questions:process-1:round:2',
      decisions: [decision({ id: 'question-1', decision_key: 'goal_definition:adaptive:2:0:objective' })],
      process: baselineProcess,
    },
    'recovery-proposal': { kind: 'recovery-proposal', key: 'recovery-proposal:recovery-1', decision: recoveryDecision() },
    gate: { kind: 'gate', key: 'gate:gate-1', gate: gate(), description: 'Ship verified work.' },
    'meeting-commitment': { kind: 'meeting-commitment', key: 'meeting-commitment:commitment-1', commitment: commitment() },
    warning: { kind: 'warning', key: 'warning:warning-1', warning: warning(), process: undefined, canAcknowledge: true },
  }
}

describe('SPR210 concise Needs You titles', () => {
  it('uses the useful source title unchanged and the deterministic fallback when it is absent', () => {
    expect(helpers.conciseNeedsYouTitle('Release approval', 'Owner guidance needed')).toBe('Release approval')
    expect(helpers.conciseNeedsYouTitle(null, 'Owner guidance needed')).toBe('Owner guidance needed')
  })

  it('bounds a multiword title to exactly 100 Unicode codepoints at a word boundary', () => {
    const title = helpers.conciseNeedsYouTitle(`${'word '.repeat(24)}ending`, 'Owner guidance needed')

    expect(Array.from(title)).toHaveLength(100)
    expect(title).toBe(`${'word '.repeat(20).trimEnd()}…`)
  })

  it('uses a hard codepoint boundary for one long emoji token without splitting a surrogate pair', () => {
    const title = helpers.conciseNeedsYouTitle('🧪'.repeat(120), 'Owner guidance needed')

    expect(Array.from(title)).toHaveLength(100)
    expect(title).toBe(`${'🧪'.repeat(99)}…`)
    expect(title).not.toContain('�')
  })

  it.each([
    ['blocker', 'Work is blocked'],
    ['lm-retry', 'Model request failed'],
    ['answer', 'Owner guidance needed'],
    ['answer-batch', 'Agent definitions need review'],
    ['goal-definition-questions', 'Goal clarification needed'],
    ['recovery-proposal', 'Recovery proposal'],
    ['gate', 'Verification failed'],
    ['meeting-commitment', 'Meeting action needs attention'],
    ['warning', 'Warning needs attention'],
  ] as const)('uses the %s fallback heading when the source title is technical, missing, or repeats its body', (kind, expected) => {
    const rows = rowsByKind()
    const row = rows[kind]
    if (row.kind === 'answer') row.decision.title = 'Runtime decision: run:raw-run-id:kind:ask_human:release'
    else if (row.kind === 'gate') row.description = undefined
    else if (row.kind === 'meeting-commitment') row.commitment.reason = row.commitment.description
    else if (row.kind === 'warning') row.warning.message = 'Warning needs attention'

    expect(helpers.presentNeedsYouRow(row, []).title).toBe(expected)
  })

  it('keeps a distinct useful source title while the complete substantive question remains in Details', () => {
    const row = rowsByKind().answer
    if (row.kind !== 'answer') throw new Error('answer fixture expected')

    const result = helpers.presentNeedsYouRow(row, [])

    expect(result.title).toBe('Release approval')
    expect(result.details).toContain('Should the release proceed?')
    expect(`${result.title} ${result.explanation}`).not.toContain('Should the release proceed? Should the release proceed?')
  })
})

describe('SPR211 verified reference presentation', () => {
  it('routes only a current-project, bound artifact to its exact plan candidate hash', () => {
    const artifact = reference({ kind: 'artifact', id: 'artifact-review-4', goal_id: 'goal-1', label: 'Plan candidate v4' })

    expect(presentation.needsYouReferencePath(artifact, 'project-1'))
      .toBe('/orchestration/goal-1#plan-review-artifact-review-4')
    expect(presentation.needsYouReferencePath({ ...artifact, project_id: 'foreign-project' }, 'project-1')).toBeNull()
    expect(presentation.needsYouReferencePath({ ...artifact, goal_id: null }, 'project-1')).toBeNull()
    expect(presentation.needsYouReferencePath({ ...artifact, label: null }, 'project-1')).toBeNull()
  })

  it('substitutes only exact typed identifier tokens and neutralizes unknown IDs and runtime keys outside Details', () => {
    const taskId = '10000000-0000-4000-8000-000000000001'
    const agentId = '20000000-0000-4000-8000-000000000002'
    const unknownId = '30000000-0000-4000-8000-000000000003'
    const runtimeKey = 'run:raw-run-id:kind:ask_human:deployment'
    const row = rowsByKind().answer
    if (row.kind !== 'answer') throw new Error('answer fixture expected')
    row.decision.title = runtimeKey
    row.decision.question = `Review ${taskId} with ${agentId}; unknown ${unknownId}; preserve x${taskId}y only in Details.`
    const references = [
      reference({ id: taskId, kind: 'task', label: 'Shared label' }),
      reference({ id: agentId, kind: 'agent', label: 'Shared label' }),
    ]

    const result = helpers.presentNeedsYouRow(row, references)
    const primary = `${result.title} ${result.explanation} ${result.nextStep}`

    expect(primary.match(/Shared label/g)).toHaveLength(2)
    expect(primary).toContain('Reference unavailable')
    expect(primary).not.toContain(taskId)
    expect(primary).not.toContain(agentId)
    expect(primary).not.toContain(unknownId)
    expect(primary).not.toContain(runtimeKey)
    expect(result.details).toContain(taskId)
    expect(result.details).toContain(agentId)
    expect(result.details).toContain(unknownId)
    expect(result.details).toContain(runtimeKey)
    expect(result.references.map(({ kind, id }) => `${kind}:${id}`)).toEqual([`task:${taskId}`, `agent:${agentId}`])
  })

  it('renders unavailable typed labels neutrally, ignores malformed metadata, and never mutates source records', () => {
    const unavailableId = '40000000-0000-4000-8000-000000000004'
    const row = rowsByKind().warning
    if (row.kind !== 'warning') throw new Error('warning fixture expected')
    row.warning.message = `Task ${unavailableId} needs review.`
    const before = structuredClone(row)
    const references = [
      reference({ id: unavailableId, label: null }),
      { kind: 'task', id: '', project_id: '', goal_id: null, label: 42 } as unknown as NeedsYouReference,
    ]

    const result = helpers.presentNeedsYouRow(row, references)

    expect(result.explanation).toContain('Task unavailable')
    expect(result.references).toEqual([references[0]])
    expect(row).toEqual(before)
    expect(() => helpers.presentNeedsYouRow(row, [])).not.toThrow()
  })

  it('keeps structured recovery/capability contracts authoritative over alarming prose', () => {
    const recovery = rowsByKind()['recovery-proposal']
    if (recovery.kind !== 'recovery-proposal') throw new Error('recovery fixture expected')
    recovery.decision.recovery!.problem = 'Malformed parser invariant caused a HuddleRoom product bug.'
    const ordinary = rowsByKind().warning
    if (ordinary.kind !== 'warning') throw new Error('warning fixture expected')
    ordinary.warning.message = 'Malformed parser invariant caused a HuddleRoom product bug.'
    ordinary.warning.warning_type = 'coverage_gap'
    const malformedOutput = rowsByKind().warning
    if (malformedOutput.kind !== 'warning') throw new Error('warning fixture expected')
    malformedOutput.warning.warning_type = 'malformed_output'

    const recoveryResult = helpers.presentNeedsYouRow(recovery, [])
    const ordinaryResult = helpers.presentNeedsYouRow(ordinary, [])
    const malformedResult = helpers.presentNeedsYouRow(malformedOutput, [])

    expect(recoveryResult.title).toBe('Recovery proposal')
    expect(recoveryResult.actionOwner).toBe('owner')
    expect(ordinaryResult.actionOwner).not.toBe('maintainer')
    expect(malformedResult.actionOwner).not.toBe('maintainer')
  })
})

describe('SPR212 queue occurrences', () => {
  it('normalizes offset and SQLite-naive timestamps to the same UTC instant and sends invalid dates after dated rows', () => {
    const answer = rowsByKind().answer
    const warningRow = rowsByKind().warning
    const gateRow = rowsByKind().gate
    if (answer.kind !== 'answer' || warningRow.kind !== 'warning' || gateRow.kind !== 'gate') throw new Error('fixture mismatch')
    answer.decision.asked_at = '2026-10-14T10:00:00+02:00'
    warningRow.warning.created_at = '2026-10-14T08:00:00'
    gateRow.gate.failed_at = 'not-a-date'
    gateRow.gate.created_at = 'also-not-a-date'

    const occurrences = [answer, warningRow, gateRow].map((row) => helpers.queueOccurrence(row, []))

    expect(occurrences[0].occurredAt).toBe('2026-10-14T08:00:00.000Z')
    expect(occurrences[1].occurredAt).toBe('2026-10-14T08:00:00.000Z')
    expect(occurrences[2].occurredAt).toBeNull()
    expect([...occurrences].sort(helpers.compareQueueOccurrences).map(({ key }) => key)).toEqual([
      'answer:decision-1',
      'warning:warning-1',
      'gate:gate-1',
    ])
  })

  it('rejects impossible calendar dates while accepting leap-day offset and naive UTC instants', () => {
    const occurrence = (id: string, createdAt: string) => helpers.queueOccurrence({
      kind: 'warning', key: '', warning: warning({ id, created_at: createdAt }), process: undefined, canAcknowledge: true,
    }, [])

    expect(occurrence('feb-30', '2026-02-30T08:00:00Z').occurredAt).toBeNull()
    expect(occurrence('non-leap', '2025-02-29T08:00:00Z').occurredAt).toBeNull()
    expect(occurrence('leap-offset', '2024-02-29T10:30:00+02:30').occurredAt).toBe('2024-02-29T08:00:00.000Z')
    expect(occurrence('leap-naive', '2024-02-29T08:00:00').occurredAt).toBe('2024-02-29T08:00:00.000Z')
  })

  it('keeps untyped blocker ids and timestamps out of its stable undated identity', () => {
    const blocker = (overrides: Record<string, unknown>): QueueRow => ({
      kind: 'blocker', key: '', blocker: {
        kind: 'external_wait', reason: 'Waiting for an external dependency.', ...overrides,
      } as OrchestrationBlocker,
    })
    const first = helpers.queueOccurrence(blocker({
      process_id: 'process-arbitrary-a', source_action_id: 'action-arbitrary-a',
      created_at: '2026-10-14T11:00:00Z', updated_at: '2026-10-14T12:00:00Z',
    }), [])
    const second = helpers.queueOccurrence(blocker({
      process_id: 'process-arbitrary-b', source_action_id: 'action-arbitrary-b',
      created_at: '2030-01-01T00:00:00Z', updated_at: '2030-01-02T00:00:00Z',
    }), [])

    expect(first).toEqual(second)
    expect(first.occurredAt).toBeNull()
    expect(first.key).toMatch(/^blocker:external_wait:content:/)
    for (const leaked of ['process-arbitrary', 'action-arbitrary', '2026-', '2030-']) expect(first.key).not.toContain(leaked)

    const typed = helpers.queueOccurrence(blocker({
      task_id: 'task-declared', gate_id: 'gate-declared', process_id: 'process-arbitrary',
    }), [])
    expect(typed.key).toBe('blocker:external_wait:task_id:task-declared|gate_id:gate-declared')
  })

  it('sorts all nine kinds globally by newest occurrence, codepoint key ties, then undated key', () => {
    const rows = rowsByKind()
    const setDate = (row: QueueRow, value: string | null) => {
      if (row.kind === 'answer') row.decision.asked_at = value ?? 'invalid'
      else if (row.kind === 'answer-batch' || row.kind === 'goal-definition-questions') row.decisions[0].asked_at = value ?? 'invalid'
      else if (row.kind === 'recovery-proposal') row.decision.asked_at = value ?? 'invalid'
      else if (row.kind === 'gate') { row.gate.failed_at = value; row.gate.created_at = value ?? 'invalid' }
      else if (row.kind === 'meeting-commitment') row.commitment.created_at = value
      else if (row.kind === 'warning') row.warning.created_at = value ?? 'invalid'
      else if (row.kind === 'lm-retry') row.process.created_at = value ?? 'invalid'
    }
    setDate(rows.warning, '2026-10-14T09:00:00Z')
    setDate(rows.gate, '2026-10-14T09:00:00Z')
    setDate(rows.answer, '2026-10-14T08:50:00Z')
    setDate(rows['answer-batch'], '2026-10-14T08:40:00Z')
    setDate(rows['goal-definition-questions'], '2026-10-14T08:30:00Z')
    setDate(rows['recovery-proposal'], '2026-10-14T08:20:00Z')
    setDate(rows['lm-retry'], '2026-10-14T08:10:00Z')
    setDate(rows['meeting-commitment'], '2026-10-14T08:00:00Z')

    const sorted = Object.values(rows)
      .map((row) => helpers.queueOccurrence(row, []))
      .sort(helpers.compareQueueOccurrences)
      .map(({ key }) => key)

    expect(sorted).toEqual([
      'gate:gate-1',
      'warning:warning-1',
      'answer:decision-1',
      'answer-batch:process-1:review-1,review-2',
      'goal-definition-questions:process-1:round:2',
      'recovery-proposal:recovery-1',
      'lm-retry:process-1:checkpoint',
      'meeting-commitment:commitment-1',
      'blocker:evidence_gap:content:f925cb08',
    ])
  })

  it('uses full decision history for a partial clarification round occurrence without changing source order', () => {
    const answered = decision({
      id: 'question-0', status: 'answered', decision_key: 'goal_definition:adaptive:4:0:objective',
      source_process_run_id: 'process-1',
      asked_at: '2026-10-14T09:30:00Z', decided_at: '2026-10-14T09:35:00Z', selected_option: 'First answer',
    })
    const pending = decision({
      id: 'question-1', decision_key: 'goal_definition:adaptive:4:1:constraints',
      source_process_run_id: 'process-1',
      asked_at: '2026-10-14T09:20:00Z', updated_at: '2026-10-14T11:00:00Z',
    })
    const row: QueueRow = {
      kind: 'goal-definition-questions', key: 'goal-definition-questions:process-1:round:4',
      decisions: [pending], process: process(),
    }
    const before = structuredClone([answered, pending])

    const occurrence = helpers.queueOccurrence(row, [pending, answered])

    expect(occurrence).toEqual({
      key: 'goal-definition-questions:process-1:round:4',
      occurredAt: '2026-10-14T09:30:00.000Z',
    })
    expect([answered, pending]).toEqual(before)
  })
})

describe('SPR212 retained-edit authority validation', () => {
  function snapshotOf(...decisions: OrchestrationAuthorityDecisionRecord[]) {
    return {
      rowKey: `answer:${decisions.map((item) => item.id).join(',')}`,
      decisionIds: decisions.map((item) => item.id),
      decisions,
    }
  }

  it('accepts unchanged authority while ignoring display labels and timestamps', () => {
    const reviewed = decision({
      id: 'runtime-q1', runtime_identity: 'runtime-q1', contract_version: 'contract-v1',
    })
    const authoritative = {
      ...reviewed,
      title: 'A fresher display label',
      asked_at: '2026-10-14T10:00:00Z',
      created_at: '2026-10-14T10:00:00Z',
      updated_at: '2026-10-14T10:01:00Z',
    }

    expect(helpers.validateNeedsYouEdit(snapshotOf(reviewed), [authoritative])).toBe('current')
  })

  it('does not invalidate a legacy ordinary question for context-only progress changes', () => {
    const reviewed = decision({
      id: 'legacy-q1', runtime_identity: null, contract_version: null,
      decision_key: 'owner_question', context: JSON.stringify({ progress: 1, label: 'Draft' }),
    })
    const authoritative = {
      ...reviewed,
      context: JSON.stringify({ progress: 99, label: 'Current display' }),
      updated_at: '2026-10-14T10:01:00Z',
    }

    expect(helpers.validateNeedsYouEdit(snapshotOf(reviewed), [authoritative])).toBe('current')
  })

  it('returns terminal only for an authoritative matching record that is no longer pending', () => {
    const reviewed = decision({
      id: 'runtime-q1', runtime_identity: 'runtime-q1', contract_version: 'contract-v1',
    })
    const answered = {
      ...reviewed,
      status: 'answered' as const,
      selected_option: 'approve',
      reason: 'Answered elsewhere.',
      decided_at: '2026-10-14T10:00:00Z',
    }

    expect(helpers.validateNeedsYouEdit(snapshotOf(reviewed), [answered])).toBe('terminal')
  })

  it.each([
    ['goal ownership', { goal_id: 'other-goal' }],
    ['runtime identity', { runtime_identity: 'runtime-q1-replaced' }],
    ['contract version', { contract_version: 'contract-v2' }],
  ] as const)('returns stale when a versioned decision changes %s', (_name, changed) => {
    const reviewed = decision({
      id: 'runtime-q1', runtime_identity: 'runtime-q1', contract_version: 'contract-v1',
    })

    expect(helpers.validateNeedsYouEdit(snapshotOf(reviewed), [{ ...reviewed, ...changed }])).toBe('stale')
  })

  it.each([
    ['question', { question: 'A replacement question' }],
    ['options', { options: ['wait', 'reject'] }],
    ['recommendation', { recommendation: 'wait' }],
    ['consequences', { consequences: 'A replacement consequence.' }],
    ['specialized recovery contract', {
      recovery: {
        kind: 'recovery_proposal_v1' as const,
        proposal_id: 'replacement-proposal', version: 2,
        status: 'pending_approval' as const,
        problem: 'Replacement problem', general_updates: [], after_approval: 'Retry.',
        global_effect: false, exact_details: null, evidence: null, error: null,
        source_blocker_id: null, source_action_id: null, source_decision_id: null,
      },
    }],
  ] as const)('returns stale when a legacy null-version decision changes its %s', (_name, changed) => {
    const reviewed = decision({
      id: 'legacy-q1', runtime_identity: null, contract_version: null,
      recovery: {
        kind: 'recovery_proposal_v1', proposal_id: 'proposal-1', version: 1,
        status: 'pending_approval', problem: 'Original problem', general_updates: [],
        after_approval: 'Retry.', global_effect: false, exact_details: null,
        evidence: null, error: null, source_blocker_id: null,
        source_action_id: null, source_decision_id: null,
      },
    })

    expect(helpers.validateNeedsYouEdit(snapshotOf(reviewed), [{ ...reviewed, ...changed }])).toBe('stale')
  })

  it.each([
    ['the authority query is unconfirmed', undefined],
    ['the reviewed record is missing', [decision({ id: 'different-decision' })]],
  ] as const)('returns unavailable when %s', (_name, authoritative) => {
    const reviewed = decision({ id: 'runtime-q1', runtime_identity: 'runtime-q1', contract_version: 'contract-v1' })

    expect(helpers.validateNeedsYouEdit(
      snapshotOf(reviewed),
      authoritative as readonly OrchestrationAuthorityDecisionRecord[] | undefined,
    )).toBe('unavailable')
  })

  it('validates the whole frozen group instead of only its currently active member', () => {
    const first = decision({ id: 'round-q1', decision_key: 'goal_definition:adaptive:3:0:objective' })
    const second = decision({ id: 'round-q2', decision_key: 'goal_definition:adaptive:3:1:constraints' })

    expect(helpers.validateNeedsYouEdit(snapshotOf(first, second), [first, {
      ...second, status: 'answered', selected_option: 'external answer', decided_at: '2026-10-14T10:00:00Z',
    }])).toBe('terminal')
  })

  it('marks an atomic frozen group stale when any member changes contract', () => {
    const first = decision({ id: 'batch-q1', contract_version: 'batch:v1:q1' })
    const second = decision({ id: 'batch-q2', contract_version: 'batch:v1:q2' })

    expect(helpers.validateNeedsYouEdit(snapshotOf(first, second), [first, {
      ...second, contract_version: 'batch:v2:q2',
    }])).toBe('stale')
  })
})

describe('SPR212 runtime answer receipt validation', () => {
  const reviewed = decision({
    id: 'runtime-q1', goal_id: 'goal-1', run_id: 'run-1', decision_key: 'runtime:question:1',
    runtime_identity: 'runtime-identity-q1', contract_version: 'contract-q1:v1',
    options: [{ key: 'acknowledge' }], recommendation: null,
  })
  const submitted = { selectedOption: 'acknowledge', reason: '  reviewed explanation\n' }
  const answered = {
    ...reviewed,
    status: 'answered' as const,
    selected_option: 'acknowledge',
    reason: 'reviewed explanation',
    decided_at: '2026-10-14T10:00:00Z',
  }

  it('accepts an identical persisted replay even when continuation was not applied again', () => {
    expect(helpers.validateRuntimeAnswerReceipt(reviewed, submitted, {
      decision: answered, process: null, continuation_applied: false,
    })).toBe('accepted')
  })

  it.each([
    ['id', { id: 'other-decision' }],
    ['goal ownership', { goal_id: 'other-goal' }],
    ['run identity', { run_id: 'other-run' }],
    ['decision key', { decision_key: 'runtime:question:other' }],
    ['runtime identity', { runtime_identity: 'runtime-identity-q1-v2' }],
    ['contract version', { contract_version: 'contract-q1:v2' }],
    ['status', { status: 'pending' as const }],
    ['persisted option', { selected_option: 'reject' }],
    ['normalized reason', { reason: 'different explanation' }],
  ] as const)('rejects a receipt with mismatched %s', (_name, changed) => {
    expect(helpers.validateRuntimeAnswerReceipt(reviewed, submitted, {
      decision: { ...answered, ...changed }, process: null, continuation_applied: true,
    })).toBe('stale')
  })

  it('keeps null distinct from an empty persisted reason', () => {
    expect(helpers.validateRuntimeAnswerReceipt(reviewed, {
      selectedOption: 'acknowledge', reason: null,
    }, {
      decision: { ...answered, reason: '' }, process: null, continuation_applied: false,
    })).toBe('stale')
  })

  it('accepts an explicitly submitted empty optional reason when the backend persists an empty string', () => {
    expect(helpers.validateRuntimeAnswerReceipt(reviewed, {
      selectedOption: 'acknowledge', reason: '',
    }, {
      decision: { ...answered, reason: '' }, process: null, continuation_applied: false,
    })).toBe('accepted')
  })

  it('rejects a non-runtime reviewed record', () => {
    expect(helpers.validateRuntimeAnswerReceipt({ ...reviewed, runtime_identity: null }, submitted, {
      decision: answered, process: null, continuation_applied: true,
    })).toBe('stale')
  })

  it('rejects a missing decision receipt instead of inferring success from continuation', () => {
    expect(helpers.validateRuntimeAnswerReceipt(reviewed, submitted, {
      decision: null,
      process: null,
      continuation_applied: true,
    } as unknown as Parameters<typeof helpers.validateRuntimeAnswerReceipt>[2])).toBe('stale')
  })
})

describe('SPR285 closeout_blocked presentation', () => {
  const closeoutRow = (message: string | null = 'Goal completion is blocked: 1 cause remains.'): QueueRow => ({
    kind: 'warning', key: 'warning:closeout-1',
    warning: warning({ id: 'closeout-1', warning_type: 'closeout_blocked', severity: 'blocker', message }),
    process: undefined, canAcknowledge: false,
  })

  it('titles the diagnostic and explains that resolving it alone does not fix the causes', () => {
    const result = helpers.presentNeedsYouRow(closeoutRow(), [])
    expect(result.title).toBe('Completion is blocked')
    expect(result.explanation).toContain('Goal completion is blocked: 1 cause remains.')
    expect(result.explanation).toContain(
      'The listed causes need action. Resolving this notice alone does not fix them; it returns until they are resolved.',
    )
    expect(result.nextStep).toBe('Address each listed cause. Closeout resumes automatically.')
    expect(result.actionOwner).toBe('owner')
    expect(`${result.title} ${result.explanation} ${result.nextStep}`).not.toContain('Reference unavailable')
  })

  it('does not duplicate the guidance when the server message is empty', () => {
    const result = helpers.presentNeedsYouRow(closeoutRow(null), [])
    expect(result.explanation.match(/Resolving this notice alone/g)).toHaveLength(1)
  })

  it('leaves other warning types on the generic copy', () => {
    const result = helpers.presentNeedsYouRow(rowsByKind().warning, [])
    expect(result.title).not.toBe('Completion is blocked')
    expect(result.explanation).not.toContain('Resolving this notice alone')
    expect(result.nextStep).not.toContain('Closeout resumes automatically')
  })
})
