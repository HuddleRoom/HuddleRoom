import type { AdapterType } from '@/lib/types'

export interface RoomPanelAgentRow {
  agentId: string
  role: string
  adapterType?: AdapterType
  provider?: string
  activity: string
  working: boolean
}

export interface RoomPanelMeetingRow {
  question: string
  chosenOption: string
  dissentCount: number
}

export interface RoomPanelProps {
  goalTitle: string
  agentRows: RoomPanelAgentRow[]
  meetingRow: RoomPanelMeetingRow | null
  gates: { total: number; accepted: number; open: number; failed: number } | null
}

// Only pill on the page — local span, not a shared primitive (design spec §3).
function AgentPill({ adapterType, provider }: { adapterType?: AdapterType; provider?: string }) {
  const text = adapterType && provider ? `${adapterType} · ${provider}` : adapterType ?? provider
  if (!text) return null
  return (
    <span className="inline-flex items-center rounded-full bg-huddleroom-depth px-2 py-0.5 font-mono text-micro font-medium leading-none text-huddleroom-text-secondary">
      {text}
    </span>
  )
}

export function RoomPanel({ goalTitle, agentRows, meetingRow, gates }: RoomPanelProps) {
  const hasBelow = agentRows.length > 0 || meetingRow !== null || gates !== null
  const verified = gates !== null && gates.open === 0 && gates.failed === 0

  return (
    <section
      aria-labelledby="room-panel-title"
      className="w-full rounded-lg border border-huddleroom-border bg-white px-5 py-4 shadow-[0_1px_2px_rgba(31,34,38,.06),0_12px_32px_rgba(31,34,38,.06)] max-md:px-4 max-md:py-3"
    >
      <h2
        id="room-panel-title"
        className={`m-0 text-title font-semibold text-huddleroom-text-primary ${hasBelow ? 'border-b border-huddleroom-border pb-3 mb-3' : ''}`}
      >
        {goalTitle}
      </h2>

      {!hasBelow && (
        <p className="m-0 pt-1 text-body text-huddleroom-text-muted">Agents appear here when work is delegated.</p>
      )}

      {agentRows.length > 0 && (
        <div role="list" aria-label="Delegated agents" className="space-y-2">
          {agentRows.map((row) => (
            <div key={row.agentId} role="listitem" className="flex flex-wrap items-center gap-x-3 gap-y-0.5 py-1.5">
              <span aria-hidden className={`h-2 w-2 shrink-0 rounded-full ${row.working ? 'bg-huddleroom-primary' : 'bg-huddleroom-faint'}`} />
              <span className="flex shrink-0 items-center gap-1.5">
                <span className="max-w-[160px] truncate text-body font-semibold text-huddleroom-text-primary">{row.role}</span>
                <AgentPill adapterType={row.adapterType} provider={row.provider} />
              </span>
              <span className="min-w-0 flex-1 truncate text-body text-huddleroom-text-muted" title={row.activity}>
                {row.activity}
              </span>
            </div>
          ))}
        </div>
      )}

      {meetingRow && (
        <div className="mt-3 space-y-1 border-t border-huddleroom-border pt-3">
          <p className="m-0 text-body text-huddleroom-text-secondary">{meetingRow.question}</p>
          <p className="m-0 text-body text-huddleroom-text-primary">
            <span className="font-semibold">Decided:</span> {meetingRow.chosenOption}
            {meetingRow.dissentCount > 0 && (
              <span className="text-huddleroom-text-muted"> · {meetingRow.dissentCount} dissent{meetingRow.dissentCount === 1 ? '' : 's'}</span>
            )}
          </p>
        </div>
      )}

      {gates && (
        <div className="mt-3 border-t border-huddleroom-border pt-3">
          <p className={`m-0 flex items-center gap-1.5 text-body font-medium ${verified ? 'text-huddleroom-status-green' : 'text-huddleroom-status-amber'}`}>
            <span aria-hidden className="text-[10px] leading-none">●</span>
            {verified
              ? `${gates.accepted} of ${gates.total} criteria. Independent proof accepted.`
              : `${gates.accepted} of ${gates.total} criteria verified.`}
          </p>
        </div>
      )}
    </section>
  )
}
