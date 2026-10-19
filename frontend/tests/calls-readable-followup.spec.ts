import { expect, test } from '@playwright/test'

// Synthetic browser fixtures: no provider, application server, or database.
for (const viewport of [{ name: 'desktop', width: 1440, height: 900 }, { name: 'mobile', width: 375, height: 812 }]) {
  test(`SPR189/190 readable screenshot regressions at ${viewport.name}`, async ({ page }, testInfo) => {
    const projectId = 'calls-readable-fixture'
    const now = new Date().toISOString()
    const recordId = 'fea808f0-ada9-488a-8962-1b1a919255be'
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await page.setViewportSize(viewport)
    await page.addInitScript((id) => localStorage.setItem('huddleroom-ui', JSON.stringify({ state: { activeProjectId: id, sidebarCollapsed: true }, version: 1 })), projectId)
    let socket: import('@playwright/test').WebSocketRoute | undefined
    await page.routeWebSocket('**/ws/**', (route) => { socket = route })
    await page.route('**/api/v1/**', (route) => {
      const path = new URL(route.request().url()).pathname
      if (path === '/api/v1/config') return route.fulfill({ json: { auth_enabled: false } })
      if (path === '/api/v1/projects') return route.fulfill({ json: { items: [{ id: projectId, name: 'Synthetic Calls fixture', workspace_path: '/mock', config: {}, created_at: now, updated_at: now }], next_cursor: null } })
      if (path.endsWith('/orchestration/chat')) return route.fulfill({ json: { items: [], total: 0, omitted: 0, allowance: { enabled: false, unlimited: true }, goal_allowances: {}, steering: {} } })
      if (path.endsWith('/orchestration/health')) return route.fulfill({ json: { debug_enabled: false, status: 'ok' } })
      if (route.request().method() !== 'GET') throw new Error(`Unmocked mutation ${path}`)
      return route.fulfill({ json: { items: [], next_cursor: null } })
    })
    await page.goto('/dashboard/')
    await expect.poll(() => !!socket).toBe(true)
    if (viewport.name === 'mobile') await page.getByRole('button', { name: /Activity/ }).click()
    await page.getByRole('tab', { name: 'Calls', exact: true }).click()
    await expect(page.getByText('No live calls captured in this view.')).toBeVisible()
    await page.screenshot({ path: `test-results/g6/calls-empty-${viewport.name}.png` })
    const emit = (callId: string, sequence: number, type: string, payload: Record<string, unknown>, actor = 'orchestrator') => socket!.send(JSON.stringify({
      id: `${callId}-${sequence}`, project_id: projectId, call_id: callId, sequence, event_type: `agent_response.${type}`, emitted_at: now,
      call_started_at: now, invocation_kind: 'cli_main', operation: actor === 'orchestrator' ? 'recovery' : 'planning', parent_call_id: null,
      actor_kind: actor === 'orchestrator' ? 'system' : 'agent', actor_id: actor, payload,
    }))
    const snapshot = (content: unknown) => ({ content, tool_calls: [], usage: null, content_truncated: false, tool_calls_truncated: false, usage_truncated: false, tool_count: 0, text_present: true })
    const recovery = { kind: 'evidence_query', query: { kind: 'agent_execution', agent_id: recordId } }
    emit('recovery', 0, 'started', { actor_label: 'Orchestrator', model_or_runtime: 'codex CLI', request_display: { kind: 'prompt', content: { description: 'Diagnose why planning is blocked' }, truncated: false } })
    emit('recovery', 1, 'output', { stream: 'output', text: JSON.stringify({ content: JSON.stringify(recovery), tool_calls: [] }), message_id: 'r', message_complete: true })
    emit('recovery', 2, 'terminal', { status: 'completed', error: null, error_truncated: false, metadata: null, metadata_truncated: false, final_response: snapshot(JSON.stringify(recovery)) })
    await page.getByRole('tab', { name: 'Orchestrator', exact: true }).click()
    const response = page.getByTestId('call-response').last()
    await expect(response).toHaveText('Requested agent execution evidence.')
    await expect(page.getByTestId('call-request-preview').last()).toContainText('Diagnose why planning is blocked')
    await expect(response).not.toContainText(recordId)
    await expect(page.getByText(/final response differs/)).not.toBeVisible()
    await expect(page.getByTestId('call-details').last()).not.toHaveAttribute('open', '')
    await page.screenshot({ path: testInfo.outputPath(`readable-${viewport.name}-recovery.png`), fullPage: true })
    const plan = { schema_version: 1, plan_kind: 'implementation', summary: 'Improve Calls using supported completed events.', plan_items: [
      { id: 'contracts', title: 'Verify event contracts', work_function: 'research', scope: 'Existing CLI', deliverable: 'Provider-free fixtures', agent_id: recordId, forbidden_work: ['No new adapter'], why_runnable: 'Ready' },
      { id: 'render', title: 'Render readable activity', work_function: 'implementation', scope: 'Calls panel', deliverable: 'Readable requests and results', depends_on: ['contracts'], why_runnable: 'Contract established' },
    ] }
    emit('planning', 0, 'started', { actor_label: 'Team Lead', model_or_runtime: 'codex CLI', request_display: { kind: 'prompt', content: JSON.stringify({ title: 'Calls followup', description: 'Improve readability of requests, recovery evidence, and Team Lead plans.' }), truncated: false } }, 'lead')
    emit('planning', 1, 'output', { stream: 'reasoning', text: 'Checking public summary and tool lifecycle events.', message_id: 'reasoning', message_complete: true }, 'lead')
    emit('planning', 2, 'tool_started', { name: 'command_execution', arguments: { command: 'python -V' }, truncated: false }, 'lead')
    emit('planning', 3, 'output', { stream: 'output', text: 'I will use the existing event contract.', message_id: 'commentary', message_complete: true }, 'lead')
    await page.getByRole('tab', { name: 'Team Lead', exact: true }).click()
    await expect(page.getByTestId('call-thinking').last()).toHaveAttribute('open', '')
    await expect(page.getByTestId('call-tool-progress').last()).toContainText('command_execution: started')
    await page.screenshot({ path: testInfo.outputPath(`readable-${viewport.name}-progress.png`), fullPage: true })
    emit('planning', 4, 'output', { stream: 'output', text: JSON.stringify(plan), message_id: 'final', message_complete: true }, 'lead')
    emit('planning', 5, 'terminal', { status: 'completed', error: null, error_truncated: false, metadata: null, metadata_truncated: false, final_response: snapshot(JSON.stringify(plan)) }, 'lead')
    await expect(page.getByTestId('call-commentary').last()).toContainText('I will use the existing event contract.')
    await expect(response).toContainText('Improve Calls using supported completed events.')
    await expect(response).toContainText('No new adapter')
    await expect(response).toContainText('Depends On')
    await expect(response).not.toContainText(recordId)
    await expect(response).not.toContainText('Schema Version')
    await expect(page.getByTestId('call-thinking').last()).not.toHaveAttribute('open', '')
    expect(await response.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true)
    await page.screenshot({ path: testInfo.outputPath(`readable-${viewport.name}-plan.png`), fullPage: true })
    await page.getByTestId('call-details').last().locator('summary').click()
    await expect(page.getByTestId('call-details').last()).toContainText(recordId)
    await page.screenshot({ path: testInfo.outputPath(`readable-${viewport.name}-details.png`), fullPage: true })
    expect(errors).toEqual([])
  })
}
