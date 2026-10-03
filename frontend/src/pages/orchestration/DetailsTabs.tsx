import { forwardRef, useEffect, useImperativeHandle, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import * as Collapsible from '@radix-ui/react-collapsible'
import {
  useOrchestrationMemorySection,
  useRerunLastOrchestrationBaseline,
  useStepOrchestrationBaseline,
} from '@/api/orchestration'
import { Button, SkeletonRow } from '@/components/common/uiPrimitives'
import { Panel } from '@/components/common/Panel'
import { Record as RecordList } from '@/components/common/Record'
import { Tabs, TabPanel } from '@/components/common/Tabs'
import type {
  OrchestrationAuthorityDecisionRecord,
  OrchestrationBaselineProcessType,
  OrchestrationGoalDetail,
  OrchestrationMemoryOverview,
  OrchestrationProcessRunRecord,
  OrchestrationWarningRecord,
} from '@/lib/types'
import { ActivityLog, type ActivityScope } from './ActivityLog'
import { ConversationPanel } from './ConversationPanel'
import { BASELINE_PROCESS_TYPES, errorRecord, processLabel } from './humanize'
import { useGoalAnnouncer } from './goalAnnouncer'

// requestAnimationFrame doesn't exist in the vitest unit-test environment
// (`environment: 'node'`, no browser globals) — same reason SecondarySection
// below needs a getComputedStyle/requestAnimationFrame guard. Runs the
// callback synchronously there instead of dropping it.
function afterLayout(callback: () => void) {
  if (typeof requestAnimationFrame === 'function') requestAnimationFrame(callback)
  else callback()
}

// `document.getElementById` isn't implemented by the lightweight vitest test
// DOM, so imperative tab-focusing walks down from a container ref instead —
// works against both a real DOM subtree and the test one.
function findDescendantById(root: HTMLElement | null, id: string): HTMLElement | null {
  if (!root) return null
  for (const child of Array.from(root.childNodes)) {
    const element = child as HTMLElement
    if (typeof element.getAttribute === 'function' && element.getAttribute('id') === id) return element
    const match = findDescendantById(element, id)
    if (match) return match
  }
  return null
}

// Generic disclosure primitive. Falls back to a native <details> when the
// environment can't run the Radix Collapsible primitive (the lightweight
// vitest DOM lacks getComputedStyle/requestAnimationFrame) — same guard
// BaselineDashboard.tsx used before this moved here with the Debug tab.
export function SecondarySection({ title, children }: { title: string; children: ReactNode }) {
  if (typeof getComputedStyle !== 'function' || typeof requestAnimationFrame !== 'function') {
    return <details className="rounded-md border border-huddleroom-border bg-huddleroom-surface">
      <summary className="min-h-11 cursor-pointer px-4 py-3 text-sm font-medium text-huddleroom-text-primary">{title}</summary>
      <div className="border-t border-huddleroom-border p-4">{children}</div>
    </details>
  }
  return <Collapsible.Root className="rounded-md border border-huddleroom-border bg-huddleroom-surface">
    <Collapsible.Trigger asChild><Button type="button" variant="secondary" className="min-h-11 w-full justify-between border-0">{title}</Button></Collapsible.Trigger>
    <Collapsible.Content className="border-t border-huddleroom-border p-4">{children}</Collapsible.Content>
  </Collapsible.Root>
}

function BaselineDebugPanel({ projectId, goalId }: { projectId: string; goalId: string }) {
  const step = useStepOrchestrationBaseline(projectId)
  const rerun = useRerunLastOrchestrationBaseline(projectId)
  const [processType, setProcessType] = useState<OrchestrationBaselineProcessType>('goal_definition')
  const busy = step.isPending || rerun.isPending
  return <SecondarySection title="Baseline debug">
    <label className="flex flex-col gap-1 text-sm text-huddleroom-text-primary"><span className="text-xs text-huddleroom-text-muted">Process</span>
      <select aria-label="Orchestrator step type" className="min-h-11 rounded-md border border-huddleroom-border bg-huddleroom-depth px-2" value={processType} disabled={busy} onChange={(event) => setProcessType(event.target.value as OrchestrationBaselineProcessType)}>
        {BASELINE_PROCESS_TYPES.map((type) => <option key={type} value={type}>{processLabel(type)}</option>)}
      </select>
    </label>
    <div className="mt-3 flex flex-wrap gap-2">
      <Button type="button" disabled={busy} onClick={() => step.mutate({ goalId, processType })}>{step.isPending ? 'Advancing…' : 'Advance process'}</Button>
      <Button type="button" variant="secondary" disabled={busy} onClick={() => rerun.mutate({ goalId, processType })}>{rerun.isPending ? 'Re-running…' : 'Re-run process'}</Button>
    </div>
  </SecondarySection>
}

type MemoryQuery = {
  data: OrchestrationMemoryOverview | undefined
  isLoading: boolean
  isError: boolean
}

function MemoryTabContent({ projectId, goalId, memory }: { projectId: string; goalId: string; memory: MemoryQuery }) {
  const [openSection, setOpenSection] = useState<string | null>(null)
  const section = useOrchestrationMemorySection(projectId, goalId, openSection ?? undefined)
  const { announceQueue, announceError } = useGoalAnnouncer()

  useEffect(() => { if (memory.isLoading) announceQueue('Loading orchestrator memory.') }, [memory.isLoading, announceQueue])
  useEffect(() => { if (memory.isError || (!memory.isLoading && !memory.data)) announceError('Failed to load memory.') }, [memory.isError, memory.isLoading, memory.data, announceError])
  useEffect(() => {
    if (!openSection || section.isLoading !== true) return
    const title = memory.data?.toc.find((entry) => entry.section_key === openSection)?.title ?? openSection
    announceQueue(`Loading ${title}.`)
  }, [openSection, section.isLoading, memory.data, announceQueue])
  useEffect(() => { if (section.isError) announceError('Failed to load section.') }, [section.isError, announceError])

  if (memory.isLoading) return <div aria-busy="true" aria-label="Loading orchestrator memory" className="space-y-2"><SkeletonRow /><SkeletonRow /><SkeletonRow /></div>
  if (memory.isError || !memory.data) return <p className="text-sm text-huddleroom-status-red">Failed to load memory.</p>

  const preface = memory.data.preface
  const rows: [string, string][] = [
    ['Objective', preface.objective ?? '—'],
    ['Goal status', preface.goal_status],
    ['Goal weight', preface.goal_weight],
    ['Run status', preface.run_status ?? '—'],
    ['Current process', preface.current_process ? `${processLabel(preface.current_process.process_type)} · ${preface.current_process.status}` : '—'],
    ['Manager', preface.manager ?? '—'],
    ['Hierarchy', preface.hierarchy ?? '—'],
    ['Constraints', preface.constraints ?? '—'],
  ]

  return <div className="flex flex-col gap-4">
    <RecordList rows={rows.map(([key, value]) => ({ key, value }))} />

    {preface.active_warnings.length > 0 && <div>
      <h3 className="text-xs font-medium text-huddleroom-text-muted">Active warnings</h3>
      <ul className="mt-1 divide-y divide-huddleroom-border">
        {preface.active_warnings.map((warning, index) => (
          <li key={index} className="py-1 text-sm text-huddleroom-text-secondary">
            {errorRecord(warning.message ?? warning.warning_type).what}{warning.acknowledged ? ' (acknowledged)' : ''}
          </li>
        ))}
      </ul>
    </div>}

    {memory.data.always_loaded.length > 0 && <div className="flex flex-col gap-3">
      <h3 className="text-xs font-medium text-huddleroom-text-muted">Always loaded</h3>
      {memory.data.always_loaded.map((loaded) => (
        <article key={loaded.id} className="rounded border border-huddleroom-border p-3">
          <h4 className="text-sm font-semibold text-huddleroom-text-primary">{loaded.title}</h4>
          {loaded.summary && <p className="mt-1 text-xs text-huddleroom-text-muted">{loaded.summary}</p>}
          <p className="mt-2 whitespace-pre-wrap text-sm text-huddleroom-text-secondary">{loaded.body}</p>
        </article>
      ))}
    </div>}

    {memory.data.toc.length > 0 && <div className="flex flex-col gap-2">
      <h3 className="text-xs font-medium text-huddleroom-text-muted">More sections</h3>
      {memory.data.toc.map((entry) => {
        const open = openSection === entry.section_key
        return <details key={entry.section_key} open={open} className="rounded border border-huddleroom-border">
          <summary
            className="min-h-11 cursor-pointer px-3 py-2 text-sm font-medium text-huddleroom-text-primary"
            onClick={(event) => { event.preventDefault(); setOpenSection((current) => (current === entry.section_key ? null : entry.section_key)) }}
          >
            {entry.title}
          </summary>
          {open && <div className="border-t border-huddleroom-border p-3">
            {section.isLoading ? <div aria-busy="true" aria-label={`Loading ${entry.title}`}><SkeletonRow /></div>
              : section.isError ? <p className="text-sm text-huddleroom-status-red">Failed to load section.</p>
              : <p className="whitespace-pre-wrap text-sm text-huddleroom-text-secondary">{section.data?.body}</p>}
          </div>}
        </details>
      })}
    </div>}
  </div>
}

export type DetailsTabKey = 'ledger' | 'plan' | 'gates' | 'delegations' | 'memory' | 'debug'

export interface DetailsTabsHandle {
  // Switches to the Ledger tab scoped to the currently selected step.
  // Threaded down to ProcessFocus's "View activity for this step" control.
  showStepActivity: () => void
  // Shows a cited decision even when the operator was viewing another tab or
  // had narrowed the ledger to a step or event kind.
  showDecisionActivity: (decisionId: string) => void
  // Switches to the Debug tab. Threaded down to the action row's "Danger
  // zone" text link. No-op when the Debug tab isn't rendered (!debug).
  showDebugTab: () => void
}

export interface DetailsTabsProps {
  projectId: string
  goalId: string
  detail: OrchestrationGoalDetail
  processes: readonly OrchestrationProcessRunRecord[]
  decisions: readonly OrchestrationAuthorityDecisionRecord[]
  warnings: readonly OrchestrationWarningRecord[]
  selectedStepType: OrchestrationBaselineProcessType | null
  memory: MemoryQuery
  debug: boolean
  decisionFocusId?: string | null
  decisionNavigation?: string | null
  goalPlanContent: ReactNode
  gatesContent: ReactNode
  delegationsContent: ReactNode
  suggestionsContent: ReactNode
  debugContent?: ReactNode
}

const BASE_TABS: { key: DetailsTabKey; label: string }[] = [
  { key: 'ledger', label: 'Ledger' },
  { key: 'plan', label: 'Plan' },
  { key: 'gates', label: 'Gates' },
  { key: 'delegations', label: 'Delegations' },
  { key: 'memory', label: 'Memory' },
]

export const DetailsTabs = forwardRef<DetailsTabsHandle, DetailsTabsProps>(function DetailsTabs({
  projectId, goalId, detail, processes, decisions, warnings, selectedStepType, memory, debug,
  goalPlanContent, gatesContent, delegationsContent, suggestionsContent, debugContent, decisionFocusId = null, decisionNavigation = null,
}, ref) {
  const tabs = debug ? [...BASE_TABS, { key: 'debug' as const, label: 'Debug' }] : BASE_TABS
  const [activeTab, setActiveTab] = useState<DetailsTabKey>('ledger')
  const [activityScope, setActivityScope] = useState<ActivityScope>('goal')
  const [focusDecisionId, setFocusDecisionId] = useState<string | null>(null)
  const [focusDecisionPending, setFocusDecisionPending] = useState(false)
  const sectionRef = useRef<HTMLElement | null>(null)
  const handledDecisionNavigation = useRef<string | null>(null)

  function showDecisionActivity(decisionId: string) {
    setActiveTab('ledger')
    setActivityScope('goal')
    setFocusDecisionId(decisionId)
    setFocusDecisionPending(true)
  }

  useLayoutEffect(() => {
    if (!decisionFocusId || !decisionNavigation) {
      handledDecisionNavigation.current = null
      return
    }
    if (handledDecisionNavigation.current === decisionNavigation) return
    handledDecisionNavigation.current = decisionNavigation
    showDecisionActivity(decisionFocusId)
  }, [decisionFocusId, decisionNavigation])

  useImperativeHandle(ref, () => ({
    showStepActivity: () => {
      setFocusDecisionId(null)
      setFocusDecisionPending(false)
      setActiveTab('ledger')
      setActivityScope('step')
      afterLayout(() => {
        const tab = findDescendantById(sectionRef.current, 'details-tab-ledger')
        // Optional call (not just optional-chained access): the lightweight
        // vitest test DOM's elements don't implement scrollIntoView at all.
        tab?.scrollIntoView?.({ block: 'nearest' })
        tab?.focus()
      })
    },
    showDecisionActivity,
    showDebugTab: () => {
      if (!debug) return
      setActiveTab('debug')
      afterLayout(() => {
        const tab = findDescendantById(sectionRef.current, 'details-tab-debug')
        tab?.scrollIntoView?.({ block: 'nearest' })
        tab?.focus()
      })
    },
  }))

  return (
    <section ref={sectionRef} aria-labelledby="details-tabs-heading" className="overflow-hidden rounded-md border border-huddleroom-border bg-huddleroom-surface">
      <h2 id="details-tabs-heading" className="sr-only">Details</h2>
      <div className="px-2 pt-2">
        <Tabs
          tabs={tabs.map((tab) => ({ id: tab.key, label: tab.label }))}
          activeId={activeTab}
          onChange={(id) => setActiveTab(id as DetailsTabKey)}
          idPrefix="details"
          ariaLabel="Goal details"
          className="overflow-x-auto"
        />
      </div>

      {/* All panels stay mounted (hidden via the native `hidden` attribute)
          rather than unmounting on tab switch: preserves scroll/expand state
          (Memory's fetch-on-open toc, Gates override forms) across tab
          switches, and keeps content reachable by renderToStaticMarkup-based
          tests without simulating a click. */}
      <TabPanel tabId="ledger" activeId={activeTab} idPrefix="details" className="p-4">
        <Panel>
          <div className="flex flex-col gap-4">
            <ConversationPanel projectId={projectId} goalId={goalId} />
            <ActivityLog
              processes={processes} decisions={decisions} warnings={warnings} detail={detail} selectedStepType={selectedStepType}
              scope={activityScope}
              onScopeChange={(scope) => { setFocusDecisionId(null); setFocusDecisionPending(false); setActivityScope(scope) }}
              debug={debug}
              focusDecisionId={focusDecisionId}
              focusDecisionPending={focusDecisionPending}
              onDecisionFocused={() => setFocusDecisionPending(false)}
            />
          </div>
        </Panel>
      </TabPanel>
      <TabPanel tabId="plan" activeId={activeTab} idPrefix="details" className="p-4">
        <Panel>
          <div className="flex flex-col gap-4">
            {goalPlanContent}
            {suggestionsContent}
          </div>
        </Panel>
      </TabPanel>
      <TabPanel tabId="gates" activeId={activeTab} idPrefix="details" className="p-4">
        <Panel>{gatesContent}</Panel>
      </TabPanel>
      <TabPanel tabId="delegations" activeId={activeTab} idPrefix="details" className="p-4">
        <Panel>{delegationsContent}</Panel>
      </TabPanel>
      <TabPanel tabId="memory" activeId={activeTab} idPrefix="details" className="p-4">
        <Panel>
          <MemoryTabContent projectId={projectId} goalId={goalId} memory={memory} />
        </Panel>
      </TabPanel>
      {debug && (
        <TabPanel tabId="debug" activeId={activeTab} idPrefix="details" className="p-4">
          <Panel>
            <div className="flex flex-col gap-4">
              {debugContent}
              <BaselineDebugPanel projectId={projectId} goalId={goalId} />
            </div>
          </Panel>
        </TabPanel>
      )}
    </section>
  )
})
