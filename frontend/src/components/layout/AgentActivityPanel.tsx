'use client'

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { parseUtc } from '@/lib/time'
import { useAgentResponseStore } from '@/stores/agent-response'
import { useWSStore } from '@/stores/ws'
import { orderedCalls, reconcileAgentTabs, AgentCallRecord, friendlyOperation, requestPreview, assembleStream, responsePresentation } from '@/lib/agentResponse'
import { Tabs } from '@/components/common/Tabs'
import { StatusBadge } from '@/components/common/uiPrimitives'

// Bound structured rendering separately from the captured original in Details.
function ReadableValue({ value }: { value: unknown }) {
  const budget = { remaining: 512 }
  const render = (current: unknown, depth: number): React.ReactNode => {
    if (--budget.remaining < 0 || depth > 12) return <span>Further content is available in Details.</span>
    if (current === null) return <span>null</span>
    if (current === undefined) return <span>Unavailable</span>
    if (typeof current !== 'object') return <p style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{String(current)}</p>
    if (Array.isArray(current)) {
      if (!current.length) return <span>Empty list</span>
      return <ul style={{ margin: '8px 0', paddingLeft: 20 }}>{current.slice(0, 128).map((item, index) => (
        <li key={index}>{render(item, depth + 1)}</li>
      ))}{current.length > 128 && <li>Further content is available in Details.</li>}</ul>
    }
    const entries = Object.entries(current)
    if (!entries.length) return <span>Empty object</span>
    return <dl style={{ margin: '8px 0' }}>{entries.slice(0, 128).map(([key, item]) => (
      <React.Fragment key={key}>
        <dt className="font-semibold" style={{ marginTop: 8 }}>{friendlyOperation(key)}</dt>
        <dd style={{ margin: '4px 0 0 12px' }}>{render(item, depth + 1)}</dd>
      </React.Fragment>
    ))}{entries.length > 128 && <dd>Further content is available in Details.</dd>}</dl>
  }
  return <>{render(value, 0)}</>
}

function formatDuration(startedAt: string, terminatedAt?: string): string {
  const ms = Math.max(0, (terminatedAt ? parseUtc(terminatedAt).getTime() : Date.now()) - parseUtc(startedAt).getTime())
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}m`
}

function OriginalValue({ value }: { value: unknown }) {
  return <pre style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', margin: '4px 0 12px' }}>{JSON.stringify(value, null, 2) ?? 'Unavailable'}</pre>
}

function terminalExplanation(call: AgentCallRecord): string | null {
  if (!call.terminal) return null
  const status = call.terminal.status
  const explanations: Record<string, string> = {
    failed: 'This call failed.', timeout: 'This call timed out.', timed_out: 'This call timed out.',
    cancelled: 'This call was cancelled.', canceled: 'This call was cancelled.', reset: 'This call was reset.',
  }
  const explanation = explanations[status]
  if (explanation || call.terminal.error) return [explanation, call.terminal.error].filter(Boolean).join(' ')
  return null
}

function orchestrationDecisionSummary(call: AgentCallRecord, value: unknown): React.ReactNode | null {
  if (
    call.operation !== 'decision' ||
    !value ||
    typeof value !== 'object' ||
    Array.isArray(value)
  ) return null

  const envelope = value as Record<string, unknown>
  const decision = envelope.decision
  if (!decision || typeof decision !== 'object' || Array.isArray(decision)) return null
  const fields = decision as Record<string, unknown>
  if (
    fields.action_type !== 'recover_blocker' ||
    typeof fields.reason !== 'string' ||
    !fields.reason.includes('plan_author_unavailable') ||
    !Array.isArray(fields.evidence_refs)
  ) return null

  const count = fields.evidence_refs.length
  return <div data-testid="orchestration-decision-summary">
    <div className="font-semibold">Start managed recovery</div>
    <p style={{ margin: '4px 0 0' }}>Planning is blocked because the approved team setup has no eligible planning agent.</p>
    <p className="text-xs text-huddleroom-text-muted" style={{ margin: '4px 0 0' }}>Based on {count} supporting record{count === 1 ? '' : 's'}.</p>
  </div>
}

function CallBlock({ call, onVisibleContent }: { call: AgentCallRecord; onVisibleContent: () => void }) {
  const isTerminal = !!call.terminal
  const [reasoningOpen, setReasoningOpen] = useState(!isTerminal)
  const wasTerminal = useRef(isTerminal)
  const previousVisible = useRef<string | null>(null)
  const reasoning = assembleStream(call.items, 'reasoning')
  const stderr = assembleStream(call.items, 'stderr')
  const response = responsePresentation(call)
  const decisionSummary = response.value === undefined ? null : orchestrationDecisionSummary(call, response.value)
  const tools = call.items.filter((item) => item.type !== 'output')
  const explanation = terminalExplanation(call)
  const preview = requestPreview(call.requestDisplay, 240)

  useEffect(() => {
    if (isTerminal && !wasTerminal.current) setReasoningOpen(false)
    wasTerminal.current = isTerminal
  }, [isTerminal])

  // Native disclosure changes themselves never count as new stream content.
  const visibleSignature = JSON.stringify([
    response.value, explanation, call.sequenceGap, call.outputEvicted, response.finalLimited,
    reasoningOpen ? reasoning : null,
  ])
  useEffect(() => {
    if (previousVisible.current !== visibleSignature) onVisibleContent()
    previousVisible.current = visibleSignature
  }, [visibleSignature, onVisibleContent])

  return (
    <div data-call-id={call.callId} className="border-b border-huddleroom-border" style={{ paddingBottom: 12, marginBottom: 12, minWidth: 0, overflowWrap: 'anywhere' }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', marginBottom: 8 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="text-sm font-semibold text-huddleroom-primary">{friendlyOperation(call.operation)}</div>
          <div className="text-xs text-huddleroom-text-muted">{call.actorLabel} · {formatDuration(call.callStartedAt, call.terminal?.emittedAt)}</div>
        </div>
        <StatusBadge status={call.terminal?.status || 'running'} label={call.terminal?.status || 'Running'} dot />
      </div>
      {call.parentCallId && <p className="text-xs text-huddleroom-text-muted" style={{ margin: '8px 0' }}>Fallback or repair of an earlier call.</p>}
      <p data-testid="call-request-preview" className="text-[13px] text-huddleroom-text-secondary" style={{ whiteSpace: 'pre-wrap', margin: '8px 0 12px' }}>
        {call.requestDisplay.kind === 'unavailable' ? 'Request unavailable; the start of this call was not observed.' :
          call.requestDisplay.kind === 'continuation' ? preview || 'Continuation after tool call' : preview || 'No request text captured.'}
      </p>
      {call.requestDisplay.truncated && <p className="text-xs text-huddleroom-text-muted">[Request truncated]</p>}
      {call.sequenceGap && <p className="text-xs text-huddleroom-danger">Sequence gap detected; some activity is unavailable.</p>}
      {call.outputEvicted && <p className="text-xs text-huddleroom-text-muted">Earlier output was evicted to keep this live view bounded.</p>}
      {call.observationTruncated && <p className="text-xs text-huddleroom-text-muted">Some observed output was truncated.</p>}
      {explanation && <p className="text-[13px] text-huddleroom-danger" style={{ whiteSpace: 'pre-wrap' }}>{explanation}</p>}
      {call.terminal?.errorTruncated && <p className="text-xs text-huddleroom-text-muted">The captured failure explanation was truncated.</p>}
      {reasoning && <details data-testid="call-thinking" open={reasoningOpen} style={{ marginBottom: 12 }}>
        <summary className="font-semibold text-xs" style={{ cursor: 'pointer', marginBottom: 8 }} onClick={(event) => {
          // Prevent a native toggle from racing the running-to-terminal state change.
          event.preventDefault()
          const nextOpen = !reasoningOpen
          previousVisible.current = JSON.stringify([response.value, explanation, call.sequenceGap, call.outputEvicted, response.finalLimited, nextOpen ? reasoning : null])
          setReasoningOpen(nextOpen)
        }}>Thinking</summary>
        <div style={{ fontSize: 14, lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>{reasoning}</div>
      </details>}
      {response.value !== undefined && <div data-testid="call-response" style={{ fontSize: 14, lineHeight: 1.5, whiteSpace: 'pre-wrap', marginBottom: 12 }}>{decisionSummary || <ReadableValue value={response.value} />}</div>}
      {response.mismatch && <p className="text-xs text-huddleroom-text-muted">The final response differs from the provisional stream; the final response is shown. The provisional response is in Details.</p>}
      {response.finalLimited && <p className="text-xs text-huddleroom-text-muted">The captured final response was truncated; available streamed text is retained.</p>}
      {response.toolOnly && <p data-testid="call-response" className="text-[13px] text-huddleroom-text-secondary">This call returned tool calls without response text. See Details.</p>}
      {response.empty && <p data-testid="call-response" className="text-[13px] text-huddleroom-text-secondary">No response text was returned by this call.</p>}
      {!isTerminal && response.value === undefined && <p className="text-xs text-huddleroom-text-muted">Waiting for response text…</p>}
      <details data-testid="call-details" style={{ marginTop: 12 }}>
        <summary className="font-semibold text-xs" style={{ cursor: 'pointer' }}>Details</summary>
        <div className="font-mono text-xs" style={{ marginTop: 8, overflowWrap: 'anywhere' }}>
          <dl style={{ margin: '8px 0' }}>
            {Object.entries({ Operation: call.operation, 'Invocation kind': `(${call.invocationKind})`, 'Model or runtime': call.modelOrRuntime || 'Unavailable', 'Call ID': call.callId, 'Parent call ID': call.parentCallId || 'None' }).map(([label, value]) => <React.Fragment key={label}><dt className="font-semibold">{label}</dt><dd style={{ margin: '0 0 8px' }}>{value}</dd></React.Fragment>)}
          </dl>
          <div className="font-semibold">Full captured request</div><div>Request kind: {call.requestDisplay.kind}</div><ReadableValue value={call.requestDisplay.content} /><div className="font-semibold" style={{ marginTop: 8 }}>Captured original request</div><OriginalValue value={call.requestDisplay.content} /><div>Request truncated: {String(call.requestDisplay.truncated)}</div>
          {response.value !== undefined && <><div className="font-semibold">Captured original response</div><OriginalValue value={call.finalResponse?.content ?? response.value} /></>}
          {(response.mismatch || response.finalLimited) && <><div className="font-semibold">Provisional response</div><OriginalValue value={response.provisional} /></>}
          {stderr && <><div className="font-semibold">Stderr diagnostics</div><pre style={{ whiteSpace: 'pre-wrap', margin: '4px 0 12px' }}>{stderr}</pre></>}
          {tools.map((item) => <div key={item.sequence} style={{ marginBottom: 12 }}>
            <div className="font-semibold">{item.type === 'tool_started' ? 'Tool arguments' : 'Tool result'}: {item.name}</div>
            {item.type === 'tool_finished' && <div>Outcome: {item.outcome}</div>}
            <OriginalValue value={item.type === 'tool_started' ? item.arguments : item.result} />
            {item.truncated && <div>Captured tool content was truncated.</div>}
          </div>)}
          {call.finalResponse && <><div className="font-semibold">Final response envelope and truncation flags</div><OriginalValue value={call.finalResponse} /></>}
          {call.terminal && <><div className="font-semibold">Terminal metadata and truncation flags</div><OriginalValue value={call.terminal} /></>}
          {call.terminalStatePreserved && <p>Terminal call state is preserved.</p>}
        </div>
      </details>
    </div>
  )
}

interface AgentActivityPanelProps {
  projectId: string | null
}

export function AgentActivityPanel({ projectId }: AgentActivityPanelProps) {
  const state = useAgentResponseStore()
  const wsState = useWSStore()
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const programmaticScrollTop = useRef<number | null>(null)
  const followRef = useRef(true)

  const [selectedActor, setSelectedActor] = useState<string | null>(null)
  const [focusedActor, setFocusedActor] = useState<string | null>(null)
  const [visibleActors, setVisibleActors] = useState<string[]>([])
  const [followEnabled, setFollowEnabled] = useState(true)
  const [newUpdatesCount, setNewUpdatesCount] = useState(0)
  const pauseFollow = useCallback(() => {
    followRef.current = false
    setFollowEnabled(false)
  }, [])
  const scrollToLatest = useCallback(() => {
    const el = scrollContainerRef.current
    if (!el) return
    el.scrollTop = el.scrollHeight
    programmaticScrollTop.current = el.scrollTop
  }, [])
  const onVisibleContent = useCallback(() => {
    if (followRef.current) scrollToLatest()
    else setNewUpdatesCount((count) => count + 1)
  }, [scrollToLatest])

  const allCalls = useMemo(() => orderedCalls(state.calls), [state.calls])
  const nonterminalCount = allCalls.filter((c) => !c.terminal).length

  // Reconcile actor tabs when candidateRevision changes
  useEffect(() => {
    const { visibleActorIds, selectedActorId, focusedActorId } = reconcileAgentTabs(
      state.calls,
      visibleActors,
      selectedActor,
      focusedActor
    )
    setVisibleActors(visibleActorIds)
    setSelectedActor(selectedActorId)
    setFocusedActor(focusedActorId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.candidateRevision, selectedActor, focusedActor])

  const handleNewUpdatesClick = () => {
    followRef.current = true
    setFollowEnabled(true)
    setNewUpdatesCount(0)
    scrollToLatest()
  }

  // Determine which calls to show based on selected actor tab
  const displayedCalls = selectedActor
    ? allCalls.filter((c) => c.actorId === selectedActor)
    : allCalls

  // Build tab list
  const tabs = [
    { id: 'all', label: 'All' },
    ...(visibleActors.includes('orchestrator') ? [{ id: 'orchestrator', label: 'Orchestrator' }] : []),
    ...visibleActors
      .filter((a) => a !== 'orchestrator')
      .slice(0, 3)
      .map((actorId) => {
        const call = Object.values(state.calls).find((c) => c.actorId === actorId)
        return { id: actorId, label: call?.actorLabel || 'Unknown agent' }
      }),
  ]

  const overflowActors = visibleActors
    .filter((a) => a !== 'orchestrator' && !tabs.some((t) => t.id === a))
    .map((actorId) => {
      const call = Object.values(state.calls).find((c) => c.actorId === actorId)
      return { id: actorId, label: call?.actorLabel || 'Unknown agent' }
    })

  const connectivityLabel = wsState.connected ? 'Connected' : 'Reconnecting live activity…'
  const connectivityStatus = wsState.connected ? 'active' : (wsState.connecting || wsState.retrying ? 'preparing' : 'active')

  return (
    <div onFocusCapture={(event) => {
      const target = event.target as HTMLElement
      const id = target.getAttribute('id')
      if (id?.startsWith('agent-activity-tab-')) {
        const actor = id.slice('agent-activity-tab-'.length)
        setFocusedActor(actor === 'all' ? null : actor)
        pauseFollow()
      }
    }} className="bg-huddleroom-surface border border-huddleroom-border rounded-[6px]" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Header */}
      <div className="border-b border-huddleroom-border" style={{ padding: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <h3 className="text-sm font-semibold" style={{ margin: 0 }}>Calls</h3>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <StatusBadge
              status={connectivityStatus}
              label={connectivityLabel}
              dot={true}
            />
            <span className="text-xs text-huddleroom-text-muted">{nonterminalCount} active</span>
          </div>
        </div>

        {/* Tabs */}
        <Tabs
          tabs={tabs}
          activeId={selectedActor || 'all'}
          onChange={(id) => {
            setSelectedActor(id === 'all' ? null : id)
            setFocusedActor(id === 'all' ? null : id)
            pauseFollow()
          }}
          idPrefix="agent-activity"
          ariaLabel="Agent activity calls"
          withPanels={false}
          className="mb-2"
        />

        {/* More agents select */}
        {overflowActors.length > 0 && (
          <select
            aria-label="More agents"
            value=""
            onChange={(e) => {
              if (e.target.value) {
                const selectedId = e.target.value
                // Find least-recent visible agent (exclude orch, selected, focused)
                const actorRecency = new Map<string, string>()
                for (const call of orderedCalls(state.calls)) {
                  const existing = actorRecency.get(call.actorId)
                  if (!existing || call.callStartedAt > existing) {
                    actorRecency.set(call.actorId, call.callStartedAt)
                  }
                }

                const leastRecentVisible = visibleActors
                  .filter((a) => a !== 'orchestrator' && a !== selectedActor && a !== focusedActor)
                  .sort((a, b) => {
                    const aTime = actorRecency.get(a) || ''
                    const bTime = actorRecency.get(b) || ''
                    return aTime.localeCompare(bTime) // ascending (least recent first)
                  })[0]

                if (leastRecentVisible) {
                  const newVisible = visibleActors
                    .filter((a) => a !== leastRecentVisible)
                    .concat(selectedId)
                  setVisibleActors(newVisible)
                  setSelectedActor(selectedId)
                  setFocusedActor(selectedId)
                  pauseFollow()
                }
                e.target.value = ''
              }
            }}
            className="text-xs border border-huddleroom-border rounded-[4px] bg-huddleroom-surface"
            style={{
              padding: '4px 8px',
              cursor: 'pointer',
              width: '100%',
            }}
          >
            <option value="" disabled>
              More agents ({overflowActors.length})
            </option>
            {overflowActors.map((agent) => (
              <option key={agent.id} value={agent.id}>
                {agent.label}
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Content area with scroll and follow control */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <div
          ref={scrollContainerRef}
          data-testid="calls-scroll"
          onFocusCapture={pauseFollow}
          onWheel={pauseFollow}
          onTouchMove={pauseFollow}
          onPointerDown={(event) => { if ((event.target as HTMLElement).closest('summary')) pauseFollow() }}
          onKeyDown={(event) => { if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(event.key)) pauseFollow() }}
          onScroll={(event) => {
            const top = event.currentTarget.scrollTop
            if (programmaticScrollTop.current !== null && Math.abs(top - programmaticScrollTop.current) < 1) return
            programmaticScrollTop.current = null
            pauseFollow()
          }}
          style={{
            flex: 1,
            overflowY: 'auto',
            overflowAnchor: 'none',
            padding: 12,
          }}
        >
          <div>
            {displayedCalls.length === 0 && (
              <p className="text-[13px] text-huddleroom-text-muted" style={{ margin: 0 }}>No activity yet.</p>
            )}
            {displayedCalls.map((call) => (
              <CallBlock key={call.callId} call={call} onVisibleContent={onVisibleContent} />
            ))}
          </div>
        </div>

        {/* Follow button and new updates counter */}
        <div className="border-t border-huddleroom-border" style={{ padding: 12, display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            aria-pressed={followEnabled}
            onClick={pauseFollow}
            className="text-xs border border-huddleroom-border rounded-[4px] bg-huddleroom-surface"
            style={{ padding: '4px 8px', cursor: 'pointer' }}
          >
            {followEnabled ? 'Follow' : 'Follow paused'}
          </button>
          {!followEnabled && <button
            onClick={handleNewUpdatesClick}
            className="text-xs border border-huddleroom-border rounded-[4px] bg-huddleroom-surface-blue text-huddleroom-primary font-medium"
            style={{ padding: '4px 8px', cursor: 'pointer' }}
          >Jump to latest{newUpdatesCount > 0 ? ` (${newUpdatesCount} new update${newUpdatesCount === 1 ? '' : 's'})` : ''}</button>}
        </div>
      </div>
    </div>
  )
}
