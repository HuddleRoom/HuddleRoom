import { describe, expect, it } from 'vitest'
import type { OrchestrationAuthorityDecisionRecord, OrchestrationGoal, OrchestrationProcessRunRecord } from '@/lib/types'
import {
  BASELINE_PROCESS_TYPES,
  errorRecord,
  gateOverrideDecisionLabel,
  goalStatusLabel,
  nowLine,
  processDescription,
  processLabel,
  runStatusLabel,
  statusPhrase,
  warningSeverityLabel,
  warningTypeLabel,
} from './humanize'
import { ApiError } from '@/lib/api-client'

describe('orchestration display labels', () => {
  it('keeps the baseline processes in execution order with human labels', () => {
    expect(BASELINE_PROCESS_TYPES).toEqual([
      'goal_definition',
      'manager_selection',
      'agent_definition_review',
      'team_hierarchy',
      'effectiveness_review',
      'goal_closeout',
    ])
    expect(Object.fromEntries(BASELINE_PROCESS_TYPES.map((type) => [type, processLabel(type)]))).toEqual({
      goal_definition: 'Goal definition',
      manager_selection: 'Manager selection',
      agent_definition_review: 'Agent definition review',
      team_hierarchy: 'Team hierarchy',
      effectiveness_review: 'Effectiveness review',
      goal_closeout: 'Goal closeout',
    })
  })

  it.each([
    ['running', { label: 'In progress', tone: 'in_progress' }],
    ['waiting_decision', { label: 'Needs you', tone: 'needs-you' }],
    ['completed', { label: 'Done', tone: 'done' }],
    ['skipped', { label: 'Skipped', tone: 'skipped' }],
  ])('humanizes the %s process status', (status, expected) => {
    expect(statusPhrase(status)).toEqual(expected)
  })

  it.each([
    ['active', 'Active'],
    ['blocked', 'Blocked'],
    ['paused', 'Paused'],
    ['completed', 'Completed'],
    ['cancelled', 'Cancelled'],
  ])('humanizes the %s goal status', (status, expected) => {
    expect(goalStatusLabel(status)).toBe(expected)
  })

  it.each([
    ['running', 'Running'],
    ['blocked', 'Blocked'],
    ['paused', 'Paused'],
    ['completed', 'Completed'],
    ['cancelled', 'Cancelled'],
  ])('humanizes the %s run status', (status, expected) => {
    expect(runStatusLabel(status)).toBe(expected)
  })

  it.each([
    ['recommendation', 'Recommendation'],
    ['warning', 'Warning'],
    ['blocker', 'Blocker'],
    ['hard_stop', 'Hard stop'],
  ])('humanizes the %s warning severity', (severity, expected) => {
    expect(warningSeverityLabel(severity)).toBe(expected)
  })

  it('uses readable words for warning types', () => {
    expect(warningTypeLabel('coverage_gap')).toBe('Coverage gap')
  })

  it('keeps unknown enum values readable without throwing', () => {
    expect(processLabel('future_process')).toBe('future_process')
    expect(statusPhrase('future_status')).toEqual({ label: 'future_status', tone: 'idle' })
    expect(goalStatusLabel('future_goal')).toBe('future_goal')
    expect(runStatusLabel('future_run')).toBe('future_run')
    expect(warningSeverityLabel('future_severity')).toBe('future_severity')
  })

  it.each([
    ['accept', 'Accept'],
    ['reject', 'Reject'],
  ])('humanizes gate override decision %s', (value, expected) => {
    expect(gateOverrideDecisionLabel(value as 'accept' | 'reject')).toBe(expected)
  })

  it('gives every baseline process a non-empty, distinct fallback description for the outcome-card empty state', () => {
    const descriptions = BASELINE_PROCESS_TYPES.map(processDescription)
    for (const description of descriptions) {
      expect(typeof description).toBe('string')
      expect(description.trim().length).toBeGreaterThan(0)
    }
    expect(new Set(descriptions).size).toBe(descriptions.length)
  })
})

describe('nowLine', () => {
  const ISO = '2026-08-05T08:00:00Z'

  function process(overrides: Partial<OrchestrationProcessRunRecord> = {}): OrchestrationProcessRunRecord {
    return {
      id: 'process-1', goal_id: 'goal-1', run_id: 'run-1', process_type: 'goal_definition',
      process_version: 1, status: 'completed', trigger_reason: 'test', input_snapshot: {}, outputs: {},
      skipped_by: null, override_reason: null, superseded_by_id: null,
      started_at: ISO, completed_at: ISO, created_at: ISO, updated_at: ISO,
      ...overrides,
    }
  }

  function decision(overrides: Partial<OrchestrationAuthorityDecisionRecord> = {}): OrchestrationAuthorityDecisionRecord {
    return {
      id: 'decision-1', goal_id: 'goal-1', run_id: 'run-1', decision_key: 'some_decision',
      title: 'Approve', status: 'pending', authority: 'human', authority_agent_id: null,
      source_process_run_id: null, question: 'Approve?', context: null, options: ['approve'],
      recommendation: null, consequences: null, selected_option: null, reason: null,
      decided_by_user_id: null, decided_by_agent_id: null, overrides_recommendation: false,
      created_warning_id: null, related_gate_id: null, related_action_id: null,
      asked_at: ISO, decided_at: null, created_at: ISO, updated_at: ISO,
      ...overrides,
    }
  }

  function text(segments: ReturnType<typeof nowLine>) {
    return segments.map((segment) => segment.text).join('')
  }

  it('names the running step and the next (not-yet-started) one, humanized', () => {
    const goal: Pick<OrchestrationGoal, 'status'> = { status: 'active' }
    const processes = [
      process({ process_type: 'goal_definition' }),
      process({ process_type: 'manager_selection' }),
      process({ process_type: 'agent_definition_review' }),
      process({ id: 'running', process_type: 'team_hierarchy', status: 'running' }),
    ]
    expect(nowLine({ goal, processes, checkpointItems: [] })).toEqual([{ text: 'Structuring the team · next: Effectiveness review' }])
  })

  it('names the waiting-decision step, the pending question count, and the next (not-yet-started) step', () => {
    const goal: Pick<OrchestrationGoal, 'status'> = { status: 'active' }
    const waiting = process({ id: 'waiting', process_type: 'agent_definition_review', status: 'waiting_decision' })
    const processes = [
      process({ process_type: 'goal_definition' }),
      process({ process_type: 'manager_selection' }),
      waiting,
    ]
    const checkpointItems = [
      decision({ id: 'q1', source_process_run_id: 'waiting' }),
      decision({ id: 'q2', source_process_run_id: 'waiting' }),
    ]
    expect(nowLine({ goal, processes, checkpointItems })).toEqual([{ text: 'Reviewing agent definitions · 2 questions for you · next: Team hierarchy' }])
  })

  // Regression (review round 1, finding 2): a waiting-decision step picked by
  // priority isn't necessarily sequence-adjacent to what's actually running —
  // "next: X" while X is already running elsewhere is a contradiction, so it
  // must be suppressed rather than shown.
  it('suppresses "next" when the sequence-adjacent step is already running elsewhere', () => {
    const goal: Pick<OrchestrationGoal, 'status'> = { status: 'active' }
    const processes = [
      process({ process_type: 'goal_definition' }),
      process({ id: 'waiting', process_type: 'manager_selection', status: 'waiting_decision' }),
      process({ id: 'running', process_type: 'agent_definition_review', status: 'running' }),
    ]
    const checkpointItems = [decision({ id: 'q1', source_process_run_id: 'waiting' })]
    expect(nowLine({ goal, processes, checkpointItems })).toEqual([{ text: 'Selecting a manager · 1 question for you' }])
  })

  it('reports sign-off pending once every step is terminal but the goal is still open', () => {
    const goal: Pick<OrchestrationGoal, 'status'> = { status: 'active' }
    const processes = BASELINE_PROCESS_TYPES.map((type) => process({ id: type, process_type: type, status: 'completed' }))
    expect(text(nowLine({ goal, processes, checkpointItems: [] }))).toBe('All steps complete — sign-off pending.')
  })

  it.each([
    ['completed', 'All steps complete — goal closed.'],
    ['cancelled', 'This goal was cancelled.'],
  ] as const)('reports terminal copy for a %s goal', (status, expected) => {
    const goal: Pick<OrchestrationGoal, 'status'> = { status }
    expect(text(nowLine({ goal, processes: [process({ status: 'completed' })], checkpointItems: [] }))).toBe(expected)
  })

  it('reports not-started-yet for a first-run goal with no processes', () => {
    const goal: Pick<OrchestrationGoal, 'status'> = { status: 'active' }
    expect(text(nowLine({ goal, processes: [], checkpointItems: [] }))).toBe('Not started yet.')
  })

  // Task 12b: while the baseline gate is unauthorized, the now-line names the
  // gate instead of the (nonexistent, since nothing has run yet) step.
  it('reports the baseline gate is waiting for Start baseline when unauthorized with no processes yet', () => {
    const goal: Pick<OrchestrationGoal, 'status'> = { status: 'active' }
    const run = { phase: 'baseline', baseline_authorized: false }
    expect(text(nowLine({ goal, processes: [], checkpointItems: [], run }))).toBe('Baseline is waiting — press Start baseline to begin.')
  })

  it('reports the baseline gate is waiting for Continue baseline when unauthorized mid-baseline (e.g. after a Pause)', () => {
    const goal: Pick<OrchestrationGoal, 'status'> = { status: 'active' }
    const run = { phase: 'baseline', baseline_authorized: false }
    const processes = [process({ process_type: 'goal_definition', status: 'completed' })]
    expect(text(nowLine({ goal, processes, checkpointItems: [], run }))).toBe('Baseline is waiting — press Continue baseline to resume.')
  })

  it('ignores the baseline gate once authorized (falls through to normal step narration)', () => {
    const goal: Pick<OrchestrationGoal, 'status'> = { status: 'active' }
    const run = { phase: 'baseline', baseline_authorized: true }
    expect(text(nowLine({ goal, processes: [], checkpointItems: [], run }))).toBe('Not started yet.')
  })
})

describe('errorRecord classifier', () => {
  // ApiError status-based classification
  it('classifies 401 ApiError as auth', () => {
    const err = new ApiError(401, 'Unauthorized')
    const result = errorRecord(err)
    expect(result.errorClass).toBe('auth')
    expect(result.what).toBe("The orchestrator's language-model key was rejected.")
    expect(result.why).toBe('The provider returned an authentication error — the key is missing, invalid, or revoked.')
    expect(result.doThis).toBe('Update the key in Settings.')
    expect(result.details).toMatch(/^HTTP 401 — Unauthorized/)
  })

  it('classifies 429 ApiError as rate_limit', () => {
    const err = new ApiError(429, 'Too Many Requests')
    const result = errorRecord(err)
    expect(result.errorClass).toBe('rate_limit')
    expect(result.what).toBe('The language-model provider is rate-limiting requests.')
    expect(result.why).toBe('Too many requests were sent in a short window.')
    expect(result.doThis).toBe('Wait a moment, then retry.')
    expect(result.details).toMatch(/^HTTP 429 — Too Many Requests/)
  })

  it('classifies 404 ApiError as not_found with default entity', () => {
    const err = new ApiError(404, 'Not Found')
    const result = errorRecord(err)
    expect(result.errorClass).toBe('not_found')
    expect(result.what).toBe("This item doesn't exist.")
    expect(result.why).toBe('It may have been deleted, or the link is stale.')
    expect(result.doThis).toBe('Go back to the list.')
    expect(result.details).toMatch(/^HTTP 404 — Not Found/)
  })

  it('classifies 404 ApiError as not_found with custom entity', () => {
    const err = new ApiError(404, 'Not Found')
    const result = errorRecord(err, { entity: 'goal' })
    expect(result.errorClass).toBe('not_found')
    expect(result.what).toBe("This goal doesn't exist.")
    expect(result.doThis).toBe('Go back to goals.')
  })

  it('classifies 500 ApiError as server', () => {
    const err = new ApiError(500, 'Internal Server Error')
    const result = errorRecord(err)
    expect(result.errorClass).toBe('server')
    expect(result.what).toBe('The orchestrator could not answer.')
    expect(result.why).toBe('The server returned an unexpected error.')
    expect(result.doThis).toBe('Retry.')
    expect(result.details).toMatch(/^HTTP 500 — Internal Server Error/)
  })

  it('classifies unknown ApiError (409) with message as what', () => {
    const err = new ApiError(409, 'An active goal already exists')
    const result = errorRecord(err)
    expect(result.errorClass).toBe('unknown')
    expect(result.what).toBe('An active goal already exists')
    expect(result.why).toBe('No further detail is available.')
    expect(result.doThis).toBe('Check Details, or try again.')
    expect(result.details).toMatch(/^HTTP 409 — An active goal already exists/)
  })

  // Regex-based string classification
  it('classifies string with "unauthorized" as auth', () => {
    const result = errorRecord('unauthorized')
    expect(result.errorClass).toBe('auth')
    expect(result.what).toBe("The orchestrator's language-model key was rejected.")
  })

  it('classifies string with "invalid api key" as auth', () => {
    const result = errorRecord('Provider rejected the request: invalid api key for account 42')
    expect(result.errorClass).toBe('auth')
  })

  it('classifies string with "rate limit exceeded" as rate_limit', () => {
    const result = errorRecord('rate limit exceeded')
    expect(result.errorClass).toBe('rate_limit')
    expect(result.what).toBe('The language-model provider is rate-limiting requests.')
  })

  it('classifies string with "timeout" as timeout', () => {
    const result = errorRecord('Request timed out')
    expect(result.errorClass).toBe('timeout')
    expect(result.what).toBe('The request took too long and was dropped.')
    expect(result.why).toBe("The provider or network didn't respond in time.")
  })

  it('classifies string with "404" as not_found', () => {
    const result = errorRecord('404 not found')
    expect(result.errorClass).toBe('not_found')
  })

  it('classifies string with "500" as server', () => {
    const result = errorRecord('500 internal server error')
    expect(result.errorClass).toBe('server')
    expect(result.what).toBe('The orchestrator could not answer.')
  })

  it('classifies TypeError with "Failed to fetch" as timeout', () => {
    const err = new TypeError('Failed to fetch')
    const result = errorRecord(err)
    expect(result.errorClass).toBe('timeout')
    expect(result.details).toContain('Failed to fetch')
  })

  it('classifies fully-unrecognized string as unknown', () => {
    const err = 'Something completely unexpected'
    const result = errorRecord(err)
    expect(result.errorClass).toBe('unknown')
    expect(result.what).toBe('Something completely unexpected')
    expect(result.why).toBe('No further detail is available.')
    expect(result.doThis).toBe('Check Details, or try again.')
  })

  it('includes HTTP status and message in details', () => {
    const err = new ApiError(401, 'Unauthorized')
    const result = errorRecord(err)
    expect(result.details).toMatch(/^HTTP 401 — Unauthorized/)
  })

  it('includes stack trace in details when available', () => {
    const err = new Error('Test error')
    const result = errorRecord(err)
    expect(result.details).toContain('Test error')
    expect(result.details).toContain('Error:')
  })
})
