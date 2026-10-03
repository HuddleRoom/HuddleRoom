'use client'

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import { useAgentResponseStore } from '@/stores/agent-response'
import { useWSStore } from '@/stores/ws'
import { orderedCalls, reconcileAgentTabs, AgentCallRecord, AgentCallItem, JsonValue } from '@/lib/agentResponse'
import { Tabs } from '@/components/common/Tabs'
import { LedgerRow } from '@/components/common/LedgerRow'
import { StatusBadge } from '@/components/common/uiPrimitives'
import { cn } from '@/components/common/uiPrimitives'

// Local helper to render values without JSON.stringify
function ReadableValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) return null

  if (typeof value === 'string') {
    return <p style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{value}</p>
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return <p style={{ whiteSpace: 'pre-wrap', margin: 0 }}>{String(value)}</p>
  }

  if (Array.isArray(value)) {
    return (
      <ul style={{ margin: '0.5em 0', paddingLeft: '1.5em' }}>
        {value.map((item, idx) => (
          <li key={idx}>
            <ReadableValue value={item} />
          </li>
        ))}
      </ul>
    )
  }

  if (typeof value === 'object') {
    const humanizeKey = (key: string): string => {
      return key
        .split('_')
        .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ')
    }

    const entries = Object.entries(value as Record<string, unknown>)
    return (
      <dl style={{ margin: '0.5em 0' }}>
        {entries.map(([key, val]) => (
          <React.Fragment key={key}>
            <dt className="font-semibold" style={{ marginTop: '0.5em' }}>{humanizeKey(key)}</dt>
            <dd style={{ margin: '0.25em 0 0 1em' }}>
              <ReadableValue value={val} />
            </dd>
          </React.Fragment>
        ))}
      </dl>
    )
  }

  return null
}

// Format duration in ms to readable string; freeze at terminal time
function formatDuration(startedAt: string, terminatedAt?: string): string {
  const start = new Date(startedAt).getTime()
  const end = terminatedAt ? new Date(terminatedAt).getTime() : Date.now()
  const ms = end - start
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}m`
}

// Render a single call block
function CallBlock({ call }: { call: AgentCallRecord }) {
  const [reasoningOpen, setReasoningOpen] = useState(!call.terminal)
  const [toolsOpen, setToolsOpen] = useState(false)

  useEffect(() => {
    // Auto-close reasoning on terminal
    if (call.terminal) {
      setReasoningOpen(false)
    }
  }, [call.terminal])

  const reasoningItems = call.items.filter((item) => item.type === 'output' && item.stream === 'reasoning')
  const outputItems = call.items.filter((item) => item.type === 'output' && item.stream === 'output')
  const toolItems = call.items.filter((item) => item.type === 'tool_started' || item.type === 'tool_finished')

  return (
    <div
      className="border-b border-huddleroom-border"
      style={{
        paddingBottom: 12,
        marginBottom: 12,
      }}
    >
      {/* Call header */}
      <div style={{ marginBottom: 12, display: 'flex', gap: 8, alignItems: 'flex-start' }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <LedgerRow
            iso={call.callStartedAt}
            action={call.operation}
            detail={
              <>
                {call.actorLabel} · ({call.invocationKind}) · {formatDuration(call.callStartedAt, call.terminal?.emittedAt)}
                {call.parentCallId && <> · ← {call.parentCallId}</>}
              </>
            }
          >
            {/* Request section */}
            <div style={{ marginBottom: 12, paddingBottom: 8 }}>
              <details open style={{ display: 'block' }}>
                <summary className="font-semibold text-xs" style={{ cursor: 'pointer', marginBottom: 8 }}>
                  Request
                </summary>
                {call.requestDisplay.kind === 'unavailable' && (
                  <p className="text-[13px] text-huddleroom-text-secondary" style={{ margin: 0 }}>
                    Request unavailable; the start of this call was not observed.
                  </p>
                )}
                {call.requestDisplay.kind === 'continuation' && (
                  <p className="text-[13px] text-huddleroom-text-secondary" style={{ margin: 0 }}>
                    Continuation after tool call
                  </p>
                )}
                {call.requestDisplay.kind === 'prompt' && (
                  <div className="text-[13px]">
                    <ReadableValue value={call.requestDisplay.content} />
                  </div>
                )}
                {call.requestDisplay.truncated && (
                  <p className="text-xs text-huddleroom-text-muted" style={{ margin: '8px 0 0 0' }}>
                    [Request truncated]
                  </p>
                )}
              </details>
            </div>

            {/* Sequence gap notice */}
            {call.sequenceGap && (
              <p className="text-xs text-huddleroom-danger" style={{ margin: '8px 0' }}>
                Sequence gap detected; some activity is unavailable.
              </p>
            )}

            {/* Output evicted notice */}
            {call.outputEvicted && (
              <p className="text-xs text-huddleroom-text-muted" style={{ margin: '8px 0' }}>
                Earlier output was evicted to keep this live view bounded.
              </p>
            )}

            {/* Reasoning disclosure */}
            {reasoningItems.length > 0 && (
              <details open={reasoningOpen} onToggle={(e) => setReasoningOpen(e.currentTarget.open)}>
                <summary className="font-semibold text-xs" style={{ cursor: 'pointer', marginBottom: 8 }}>
                  Reasoning
                </summary>
                {reasoningItems.map((item) => (
                  <div key={item.sequence} className="text-[13px]" style={{ marginBottom: 8 }}>
                    <ReadableValue value={(item as any).text} />
                  </div>
                ))}
              </details>
            )}

            {/* Output items */}
            {outputItems.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                {outputItems.map((item) => (
                  <div key={item.sequence} className="text-[13px]" style={{ marginBottom: 8 }}>
                    <ReadableValue value={(item as any).text} />
                  </div>
                ))}
              </div>
            )}

            {/* Tools disclosure */}
            {toolItems.length > 0 && (
              <details open={toolsOpen} onToggle={(e) => setToolsOpen(e.currentTarget.open)}>
                <summary className="font-semibold text-xs" style={{ cursor: 'pointer', marginBottom: 8 }}>
                  Tools ({toolItems.length})
                </summary>
                {toolItems.map((item) => (
                  <div key={item.sequence} style={{ marginBottom: 12, paddingLeft: 12 }}>
                    {item.type === 'tool_started' && (
                      <div>
                        <div className="font-semibold text-xs">Call: {item.name}</div>
                        {item.arguments && (
                          <div style={{ marginTop: 4 }}>
                            <ReadableValue value={item.arguments} />
                          </div>
                        )}
                        {item.truncated && (
                          <p className="text-xs text-huddleroom-text-muted" style={{ margin: '4px 0 0 0' }}>
                            [Arguments truncated]
                          </p>
                        )}
                      </div>
                    )}
                    {item.type === 'tool_finished' && (
                      <div>
                        <div className="font-semibold text-xs">Result: {item.name}</div>
                        <div className="text-xs text-huddleroom-text-muted" style={{ marginTop: 2 }}>
                          Outcome: <span className="font-medium text-huddleroom-text-primary">{item.outcome}</span>
                        </div>
                        {item.result && (
                          <div style={{ marginTop: 4 }}>
                            <ReadableValue value={item.result} />
                          </div>
                        )}
                        {item.truncated && (
                          <p className="text-xs text-huddleroom-text-muted" style={{ margin: '4px 0 0 0' }}>
                            [Result truncated]
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </details>
            )}

            {/* Terminal state preserved notice */}
            {call.terminalStatePreserved && (
              <p className="text-xs text-huddleroom-text-muted" style={{ margin: '8px 0' }}>
                Terminal call state is preserved.
              </p>
            )}
          </LedgerRow>
        </div>
        {call.terminal ? (
          <StatusBadge status={call.terminal.status} label={call.terminal.status} dot={true} />
        ) : (
          <StatusBadge status="running" label="Running" dot={true} />
        )}
      </div>
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
  const contentRef = useRef<HTMLDivElement>(null)

  const [selectedActor, setSelectedActor] = useState<string | null>(null)
  const [focusedActor, setFocusedActor] = useState<string | null>(null)
  const [visibleActors, setVisibleActors] = useState<string[]>([])
  const [followEnabled, setFollowEnabled] = useState(true)
  const [newUpdatesCount, setNewUpdatesCount] = useState(0)
  const [lastContentHeight, setLastContentHeight] = useState(0)

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
  }, [state.candidateRevision])

  // Follow behavior: detect scroll distance from bottom
  useEffect(() => {
    if (!scrollContainerRef.current) return

    const handleScroll = () => {
      const el = scrollContainerRef.current
      if (!el) return

      const scrollTop = el.scrollTop
      const scrollHeight = el.scrollHeight
      const clientHeight = el.clientHeight
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight

      if (distanceFromBottom > 60) {
        setFollowEnabled(false)
      } else if (distanceFromBottom <= 60 && !followEnabled) {
        setFollowEnabled(true)
        setNewUpdatesCount(0)
      }
    }

    const el = scrollContainerRef.current
    el.addEventListener('scroll', handleScroll)
    return () => el.removeEventListener('scroll', handleScroll)
  }, [followEnabled])

  // Auto-scroll when follow is enabled and new items arrive
  useEffect(() => {
    if (!followEnabled || !scrollContainerRef.current) return

    const newHeight = contentRef.current?.scrollHeight || 0
    if (newHeight > lastContentHeight) {
      const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

      if (prefersReducedMotion) {
        scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight
      } else {
        scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight
      }

      setLastContentHeight(newHeight)
    }
  }, [followEnabled, lastContentHeight, allCalls])

  // Track new updates when follow is disabled
  useEffect(() => {
    if (followEnabled) {
      setNewUpdatesCount(0)
    } else {
      const newHeight = contentRef.current?.scrollHeight || 0
      if (newHeight > lastContentHeight) {
        setNewUpdatesCount((c) => c + 1)
        setLastContentHeight(newHeight)
      }
    }
  }, [allCalls, followEnabled, lastContentHeight])

  const handleNewUpdatesClick = () => {
    setFollowEnabled(true)
    setNewUpdatesCount(0)
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight
    }
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
    <div className="bg-huddleroom-surface border border-huddleroom-border rounded-[6px]" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
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
            setFollowEnabled(true)
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
          onFocusCapture={() => {
            // Focusing inside content disables follow
            setFollowEnabled(false)
          }}
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: 12,
          }}
        >
          <div ref={contentRef}>
            {displayedCalls.length === 0 && (
              <p className="text-[13px] text-huddleroom-text-muted" style={{ margin: 0 }}>No activity yet.</p>
            )}
            {displayedCalls.map((call) => (
              <CallBlock key={call.callId} call={call} />
            ))}
          </div>
        </div>

        {/* Follow button and new updates counter */}
        <div className="border-t border-huddleroom-border" style={{ padding: 12, display: 'flex', gap: 8, alignItems: 'center' }}>
          <button
            aria-pressed={followEnabled}
            onClick={() => {
              setFollowEnabled(!followEnabled)
              if (!followEnabled && scrollContainerRef.current) {
                scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight
              }
            }}
            className={cn(
              'text-xs border border-huddleroom-border rounded-[4px]',
              followEnabled ? 'bg-huddleroom-surface font-semibold' : 'bg-huddleroom-depth font-normal'
            )}
            style={{
              padding: '4px 8px',
              cursor: 'pointer',
            }}
          >
            Follow
          </button>

          {newUpdatesCount > 0 && !followEnabled && (
            <button
              onClick={handleNewUpdatesClick}
              className="text-xs border border-huddleroom-border rounded-[4px] bg-huddleroom-surface-blue text-huddleroom-primary font-medium"
              style={{
                padding: '4px 8px',
                cursor: 'pointer',
              }}
            >
              {newUpdatesCount} new update{newUpdatesCount !== 1 ? 's' : ''}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
