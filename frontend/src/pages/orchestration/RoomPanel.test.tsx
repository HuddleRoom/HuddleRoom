import React from 'react'
import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { RoomPanel, type RoomPanelAgentRow } from './RoomPanel'

function agentRow(overrides: Partial<RoomPanelAgentRow> = {}): RoomPanelAgentRow {
  return {
    agentId: 'agent-1',
    role: 'Validator',
    activity: 'Reviewing evidence',
    working: true,
    ...overrides,
  }
}

describe('RoomPanel', () => {
  it('renders the empty state with no agent, meeting, or verified rows', () => {
    const markup = renderToStaticMarkup(
      <RoomPanel goalTitle="Ship the release" agentRows={[]} meetingRow={null} gates={null} />,
    )

    expect(markup).toContain('Ship the release')
    expect(markup).toContain('Agents appear here when work is delegated.')
    expect(markup).not.toContain('role="list"')
    expect(markup).not.toContain('Decided:')
    expect(markup).not.toContain('criteria')
  })

  it('renders partial agent rows: pill only when adapterType+provider present, dot state reflects working', () => {
    const rows: RoomPanelAgentRow[] = [
      agentRow({ agentId: 'agent-1', role: 'Validator', adapterType: 'api', provider: 'openai', working: true }),
      agentRow({ agentId: 'agent-2', role: 'Planner', adapterType: undefined, provider: undefined, working: false }),
    ]
    const markup = renderToStaticMarkup(
      <RoomPanel goalTitle="Ship the release" agentRows={rows} meetingRow={null} gates={null} />,
    )

    expect(markup).toContain('Validator')
    expect(markup).toContain('Planner')
    expect(markup).toContain('api · openai')
    expect(markup).not.toContain('Agents appear here when work is delegated.')
    expect(markup).not.toContain('Decided:')
    expect(markup).not.toContain('criteria')

    // Exactly one pill rendered (only agent-1 has adapterType+provider).
    const pillMatches = markup.match(/bg-huddleroom-depth px-2 py-0\.5/g) ?? []
    expect(pillMatches).toHaveLength(1)

    // Dot state differs: working agent gets bg-huddleroom-primary, idle gets bg-huddleroom-faint.
    expect(markup).toContain('bg-huddleroom-primary')
    expect(markup).toContain('bg-huddleroom-faint')
  })

  it('renders at most one pill per agent row', () => {
    const rows: RoomPanelAgentRow[] = [
      agentRow({ agentId: 'agent-1', role: 'Validator', adapterType: 'api', provider: 'openai' }),
      agentRow({ agentId: 'agent-2', role: 'Planner', adapterType: 'cli', provider: 'anthropic' }),
      agentRow({ agentId: 'agent-3', role: 'Reviewer', adapterType: undefined, provider: undefined }),
    ]
    const markup = renderToStaticMarkup(
      <RoomPanel goalTitle="Ship the release" agentRows={rows} meetingRow={null} gates={null} />,
    )

    const pillMatches = markup.match(/bg-huddleroom-depth px-2 py-0\.5/g) ?? []
    expect(pillMatches).toHaveLength(2)
  })

  it('shows a green verified message when all gates are accepted', () => {
    const markup = renderToStaticMarkup(
      <RoomPanel
        goalTitle="Ship the release"
        agentRows={[]}
        meetingRow={null}
        gates={{ total: 3, accepted: 3, open: 0, failed: 0 }}
      />,
    )

    expect(markup).toContain('3 of 3 criteria. Independent proof accepted.')
    expect(markup).toContain('text-huddleroom-status-green')
    expect(markup).not.toContain('text-huddleroom-status-amber')
  })

  it('shows an amber in-progress message when some gates are open', () => {
    const markup = renderToStaticMarkup(
      <RoomPanel
        goalTitle="Ship the release"
        agentRows={[]}
        meetingRow={null}
        gates={{ total: 3, accepted: 1, open: 2, failed: 0 }}
      />,
    )

    expect(markup).toContain('1 of 3 criteria verified.')
    expect(markup).toContain('text-huddleroom-status-amber')
    expect(markup).not.toContain('text-huddleroom-status-green')
    expect(markup).not.toContain('Independent proof accepted.')
  })

  it('treats a failed gate as not-verified (amber), not green', () => {
    const markup = renderToStaticMarkup(
      <RoomPanel
        goalTitle="Ship the release"
        agentRows={[]}
        meetingRow={null}
        gates={{ total: 3, accepted: 2, open: 0, failed: 1 }}
      />,
    )

    expect(markup).toContain('2 of 3 criteria verified.')
    expect(markup).toContain('text-huddleroom-status-amber')
  })

  it('renders the meeting question and bolds only "Decided:", omitting the dissent clause at 0', () => {
    const markup = renderToStaticMarkup(
      <RoomPanel
        goalTitle="Ship the release"
        agentRows={[]}
        meetingRow={{ question: 'Which approach should we take?', chosenOption: 'Ship the API-first design.', dissentCount: 0 }}
        gates={null}
      />,
    )

    expect(markup).toContain('Which approach should we take?')
    expect(markup).toMatch(/<span class="font-semibold">Decided:<\/span> Ship the API-first design\./)
    expect(markup).not.toContain('dissent')
  })

  it('renders a dissent count clause when dissentCount is 2', () => {
    const markup = renderToStaticMarkup(
      <RoomPanel
        goalTitle="Ship the release"
        agentRows={[]}
        meetingRow={{ question: 'Which approach should we take?', chosenOption: 'Ship the API-first design.', dissentCount: 2 }}
        gates={null}
      />,
    )

    expect(markup).toContain('· 2 dissents')
  })

  it('renders a singular dissent count clause when dissentCount is 1', () => {
    const markup = renderToStaticMarkup(
      <RoomPanel
        goalTitle="Ship the release"
        agentRows={[]}
        meetingRow={{ question: 'Which approach should we take?', chosenOption: 'Ship the API-first design.', dissentCount: 1 }}
        gates={null}
      />,
    )

    expect(markup).toContain('· 1 dissent')
    expect(markup).not.toContain('1 dissents')
  })

  it('renders agent, meeting, and verified rows together when all are present', () => {
    const markup = renderToStaticMarkup(
      <RoomPanel
        goalTitle="Ship the release"
        agentRows={[agentRow()]}
        meetingRow={{ question: 'Which approach should we take?', chosenOption: 'Ship the API-first design.', dissentCount: 0 }}
        gates={{ total: 1, accepted: 1, open: 0, failed: 0 }}
      />,
    )

    expect(markup).toContain('role="list"')
    expect(markup).toContain('Decided:')
    expect(markup).toContain('1 of 1 criteria. Independent proof accepted.')
  })
})
