import { expect, test } from '@playwright/test'
import { gotoDashboard, selectSeedProject } from './support/pages/app'

test('activity strip is visible below 1280px while the persistent rail is hidden', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 800 })
  await gotoDashboard(page)
  await selectSeedProject(page)

  await expect(page.getByRole('button', { name: 'Activity' })).toBeVisible()
  await expect(page.locator('.agent-activity-panel-container')).not.toBeVisible()
})

test('agent activity panel is always visible at 1280px+, even with zero calls', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await gotoDashboard(page)
  await selectSeedProject(page)

  await expect(page.locator('.agent-activity-panel-container')).toBeVisible()
  await expect(page.getByRole('tab', { name: 'Ask', selected: true })).toBeVisible()
  const callsTab = page.getByRole('tab', { name: 'Calls' })
  await expect(callsTab).toBeVisible()
  await callsTab.click()
  await expect(page.getByText('No live calls captured in this view.')).toBeVisible()
})


for (const viewport of [{ name: 'desktop', width: 1440, height: 900 }, { name: 'mobile', width: 375, height: 812 }]) {
  test(`live readable Calls preserve scroll and focus at ${viewport.name}`, async ({ page }, testInfo) => {
    const projectId = 'spr189-browser-project'
    const now = new Date().toISOString()
    await page.setViewportSize(viewport)
    await page.addInitScript((id) => {
      localStorage.setItem('huddleroom-ui', JSON.stringify({ state: { activeProjectId: id, sidebarCollapsed: true }, version: 1 }))
    }, projectId)
    let socket: import('@playwright/test').WebSocketRoute | undefined
    let rejectReconnect = false
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await page.routeWebSocket('**/ws/**', (route) => {
      if (rejectReconnect) route.close({ code: 1013, reason: 'Fixture holds reconnect' })
      else socket = route
    })
    await page.route('**/api/**', (route) => {
      const request = route.request()
      const path = new URL(request.url()).pathname
      if (path === '/api/v1/config') return route.fulfill({ json: { auth_enabled: false } })
      if (path === '/api/v1/projects') return route.fulfill({ json: { items: [{ id: projectId, name: 'Calls fixture project',
        workspace_path: '/mock/calls', config: {}, created_at: now, updated_at: now }], next_cursor: null } })
      if (path.endsWith('/orchestration/chat')) return route.fulfill({ json: { items: [], total: 0, omitted: 0,
        allowance: { enabled: false, unlimited: true }, goal_allowances: {}, steering: {} } })
      if (path.endsWith('/orchestration/health')) return route.fulfill({ json: { debug_enabled: false, status: 'ok' } })
      if (request.method() !== 'GET') throw new Error(`Unmocked mutation ${request.method()} ${path}`)
      return route.fulfill({ json: { items: [], next_cursor: null } })
    })
    await page.goto('/dashboard')
    await expect.poll(() => !!socket).toBe(true)
    if (viewport.name === 'mobile') await page.getByRole('button', { name: /Activity/ }).click()
    await page.getByRole('tab', { name: 'Calls', exact: true }).click()
    const scroll = page.getByTestId('calls-scroll')
    const emit = (callId: string, sequence: number, type: string, payload: Record<string, unknown>, actor = 'orchestrator') => socket!.send(JSON.stringify({
      id: `${callId}-${sequence}-${type}`, project_id: projectId, call_id: callId, sequence, event_type: `agent_response.${type}`,
      emitted_at: new Date().toISOString(), call_started_at: now, invocation_kind: 'cli_main', operation: 'goal_analysis',
      parent_call_id: null, actor_kind: actor === 'orchestrator' ? 'system' : 'agent', actor_id: actor, payload,
    }))
    const snapshot = (content: string) => ({ content, tool_calls: [], usage: { total_tokens: 12 }, content_truncated: false,
      tool_calls_truncated: false, usage_truncated: false, tool_count: 0, text_present: true })
    for (let repetition = 0; repetition < 2; repetition++) {
      const callId = `live-${repetition}`
      emit(callId, 0, 'started', { actor_label: 'Orchestrator', model_or_runtime: 'codex CLI',
        request_display: { kind: 'prompt', content: 'Explain the active goal '.repeat(30), truncated: false } })
      emit(callId, 1, 'output', { stream: 'output', text: `Visible response${repetition}\n` + 'Readable paragraph text. '.repeat(140) })
      emit(callId, 2, 'output', { stream: 'reasoning', text: 'Provider thinking text' })
      await page.getByRole('tab', { name: 'Orchestrator', exact: true }).click()
      const response = page.getByTestId('call-response').last()
      const thinking = page.getByTestId('call-thinking').last()
      const details = page.getByTestId('call-details').last()
      await expect(response).toContainText(`Visible response${repetition}`)
      await expect(thinking).toHaveAttribute('open', '')
      await expect(details).not.toHaveAttribute('open', '')
      expect(await response.evaluate((node) => getComputedStyle(node).fontSize)).toBe('14px')
      expect(await response.evaluate((node) => node.scrollWidth <= node.clientWidth)).toBe(true)
      await page.screenshot({ path: testInfo.outputPath(`calls-${viewport.name}-running-${repetition}.png`), fullPage: true })
      await scroll.evaluate((node) => { node.scrollTop = 0; node.dispatchEvent(new Event('scroll')) })
      const pausedPosition = await scroll.evaluate((node) => node.scrollTop)
      await page.screenshot({ path: testInfo.outputPath(`calls-${viewport.name}-header-thinking-${repetition}.png`), fullPage: true })
      const thinkingSummary = thinking.locator('summary')
      await thinkingSummary.focus()
      await thinkingSummary.press('Enter')
      await expect(thinking).not.toHaveAttribute('open', '')
      const disclosurePosition = await scroll.evaluate((node) => node.scrollTop)
      emit(callId, 3, 'output', { stream: 'reasoning', text: ' more actual thinking' })
      emit(callId, 4, 'output', { stream: 'output', text: '\nMore visible output while paused. '.repeat(40) })
      emit(`hidden-${repetition}`, 0, 'started', { actor_label: 'Hidden agent', model_or_runtime: 'other',
        request_display: { kind: 'prompt', content: 'hidden task', truncated: false } }, 'hidden-actor')
      emit(`hidden-${repetition}`, 1, 'output', { stream: 'output', text: 'hidden output '.repeat(100) }, 'hidden-actor')
      await expect(response).toContainText('More visible output while paused.')
      expect(await scroll.evaluate((node) => node.scrollTop)).toBe(disclosurePosition)
      expect(pausedPosition).toBe(0)
      await expect(thinkingSummary).toBeFocused()
      await expect(thinking).not.toHaveAttribute('open', '')
      await expect(page.getByRole('tab', { name: 'Orchestrator', exact: true })).toHaveAttribute('aria-selected', 'true')
      await page.getByRole('button', { name: /Jump to latest/ }).click()
      await expect.poll(() => scroll.evaluate((node) => node.scrollHeight - node.scrollTop - node.clientHeight)).toBeLessThan(3)
      emit(callId, 5, 'output', { stream: 'output', text: '\nLatest visible passage. '.repeat(30) })
      await expect(response).toContainText('Latest visible passage.')
      await expect.poll(() => scroll.evaluate((node) => node.scrollHeight - node.scrollTop - node.clientHeight)).toBeLessThan(3)
      emit(callId, 7, 'output', { stream: 'stderr', text: 'Diagnostic only' })
      await expect(page.getByText('Sequence gap detected; some activity is unavailable.').last()).toBeVisible()
      // Reopen running Thinking then terminal must collapse exactly once.
      await thinkingSummary.click()
      await expect(thinking).toHaveAttribute('open', '')
      emit(callId, 8, 'terminal', { status: 'completed', error: null, error_truncated: false,
        metadata: null, metadata_truncated: false, final_response: snapshot(`Authoritative final${repetition}`) })
      await expect(response).toContainText(`Authoritative final${repetition}`)
      await expect(thinking).not.toHaveAttribute('open', '')
      await page.screenshot({ path: testInfo.outputPath(`calls-${viewport.name}-terminal-closed-${repetition}.png`), fullPage: true })
      await thinkingSummary.click()
      await expect(thinking).toHaveAttribute('open', '')
      emit(`hidden-${repetition}`, 2, 'output', { stream: 'output', text: 'unrelated update after terminal' }, 'hidden-actor')
      await expect(thinking).toHaveAttribute('open', '')
      await details.locator('summary').click()
      await expect(details).toHaveAttribute('open', '')
      await expect(details).toContainText(`Visible response${repetition}`)
      await expect(details).toContainText('codex CLI')
      await page.screenshot({ path: testInfo.outputPath(`calls-${viewport.name}-details-${repetition}.png`), fullPage: true })
      rejectReconnect = true
      const closingSocket = socket!
      socket = undefined
      closingSocket.close({ code: 1013, reason: 'Fixture disconnect' })
      await expect(page.getByText(/Reconnecting live activity…/).first()).toBeVisible()
      await expect(response).toContainText(`Authoritative final${repetition}`)
      rejectReconnect = false
      await expect.poll(() => !!socket).toBe(true)
    }
    expect(errors).toEqual([])
    if (viewport.name === 'mobile') {
      await page.keyboard.press('Escape')
      await expect(page.getByRole('button', { name: /Activity/ })).toBeFocused()
    }
  })
}
