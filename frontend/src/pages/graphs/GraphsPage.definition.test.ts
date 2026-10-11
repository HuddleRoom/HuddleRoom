import { describe, expect, it } from 'vitest'
import yaml from 'js-yaml'
import { BLANK_FORM, parseGraphDefinition } from './GraphsPage'

describe('graph definitions', () => {
  it('renders node-level edges and engine start and terminal nodes', () => {
    const flow = parseGraphDefinition({
      start_node: 'opened',
      nodes: {
        opened: { edges: [{ to: 'review', trigger_event: 'test.passed' }] },
        review: { edges: [{ to: 'merged', trigger_event: 'code.pr_merged' }] },
        merged: { edges: [] },
      },
      terminal_nodes: { success: ['merged'], failure: [] },
    })

    expect(flow.edges).toEqual([
      { from: 'opened', to: 'review', event: 'test.passed' },
      { from: 'review', to: 'merged', event: 'code.pr_merged' },
    ])
    expect(flow.startNode).toBe('opened')
    expect(flow.nodes.get('opened')?.isInitial).toBe(true)
    expect(flow.nodes.get('merged')?.isTerminal).toBe(true)
  })

  it('creates starter YAML with executable node-level edges and terminal groups', () => {
    const definition = yaml.load(BLANK_FORM.definition) as Record<string, unknown>
    expect(definition).not.toHaveProperty('edges')
    expect(definition.terminal_nodes).toEqual({ success: ['completed'], failure: ['failed'] })
    expect(parseGraphDefinition(definition).edges).toEqual([
      { from: 'pending', to: 'active', event: 'start' },
      { from: 'active', to: 'completed', event: 'done' },
      { from: 'active', to: 'failed', event: 'error' },
    ])
  })
})
