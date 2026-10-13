import { expect, test } from '@playwright/test'
import { baselineDashboardState, detail, goal, run } from './support/orchestration-fixtures'

const projectId = 'project-chat'
const goalA = { ...goal, id: 'goal-alpha', project_id: projectId, objective: 'Ship Alpha', status: 'active' }
const goalB = { ...goalA, id: 'goal-beta', objective: 'Ship Beta' }
const now = '2026-10-13T10:00:00Z'

for (const viewport of [{ name: 'desktop', width: 1440, height: 900 }, { name: 'mobile', width: 375, height: 812 }]) {
  test(`unified conversation frozen lifecycle journey at ${viewport.name}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport)
    await page.addInitScript((id) => {
      localStorage.setItem('huddleroom-ui', JSON.stringify({ state: { activeProjectId: id, sidebarCollapsed: true }, version: 1 }))
    }, projectId)
    const submissions: Array<{ content: string; context_goal_id?: string | null; action_id?: string | null; client_request_id: string }> = []
    const items: Array<Record<string, unknown>> = []
    const baseline = baselineDashboardState()
    const allowance = { enabled: true, unlimited: false, limit: 1000, remaining: 900 }
    const goalAllowance = { enabled: true, limit: 1000, used: 100, remaining: 900 }
    const action = { action_id: 'action-pause-alpha', operation: 'pause', project_id: projectId, goal_id: goalA.id,
      run_id: 'run-alpha', target_id: null, frozen_versions: { state: 'frozen-alpha' }, payload: {},
      status: 'awaiting_confirmation', requires_confirmation: true,
      summary: 'Pauses work; baseline authorization is cleared during baseline.\n' + JSON.stringify({
        goal_status: 'active', run_id: 'run-alpha', run_status: 'running', phase: 'authorized',
        contract: { ...goalA, budget: { max_tokens: 20000 } }, authorization: null,
      }) }
    let confirmed = false
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await page.routeWebSocket('**/ws/**', (socket) => socket.close())
    // Every API request is intercepted; no model or domain action reaches the live server.
    await page.route('**/api/**', async (route) => {
      const req = route.request()
      const path = new URL(req.url()).pathname
      if (path === '/api/v1/config') return route.fulfill({ json: { auth_enabled: false } })
      if (path === '/api/v1/projects') return route.fulfill({ json: { items: [{ id: projectId, name: 'Chat test project',
        workspace_path: '/mock/workspace', config: {}, created_at: now, updated_at: now }], next_cursor: null } })
      if (path.endsWith('/orchestration/chat')) {
        if (req.method() === 'GET') return route.fulfill({ json: { items, total: items.length, omitted: 0,
          allowance, goal_allowances: { [goalA.id]: goalAllowance, [goalB.id]: goalAllowance }, steering: {} } })
        const body = req.postDataJSON() as typeof submissions[number]
        submissions.push(body)
        const response = { id: `turn-${items.length + 1}`, actor_id: 'user-test', client_request_id: body.client_request_id,
          content: body.content, answer: 'Alpha is active and awaiting validation.', status: 'completed', created_at: now,
          context_goal_id: body.context_goal_id ?? null, run_id: body.context_goal_id === goalA.id ? 'run-alpha' : 'run-beta',
          historical: false, source_references: [], source_goal_ids: body.context_goal_id ? [body.context_goal_id] : [],
          citations: [{ type: 'goal', id: goalB.id, goal_id: goalB.id, label: 'Go to Beta' }],
          off_topic: false, error: null, goal_turn: null as Record<string, unknown> | null, actions: [] as Array<typeof action>, receipts: [] as Array<Record<string, unknown>>,
          accounting_snapshot: { scope: 'goal_conversation', goal_id: goalA.id, calls_admitted: 1 } }
        if (body.content === 'What is the state?') response.goal_turn = {
          response_id: 'response-1', feedback_eligible: true, feedback: null, investigation: null, proposed_steering: null,
        }
        if (body.content.startsWith('Prioritize')) {
          response.answer = 'Direction queued for the current run.'
          response.receipts = [{ action_id: 'action-steer', operation: 'submit_steering', project_id: projectId,
            goal_id: goalA.id, run_id: 'run-alpha', status: 'pending', message: response.answer, error_code: null }]
        } else if (body.content.startsWith('Pause')) {
          response.answer = 'Review Pause for Alpha.'
          response.actions = [action]
        } else if (body.action_id) {
          expect(body.action_id).toBe(action.action_id)
          expect(body.content).toBe('yes')
          confirmed = true
          response.answer = 'Alpha paused.'
          response.context_goal_id = goalA.id
          response.run_id = 'run-alpha'
          response.receipts = [{ action_id: action.action_id, operation: 'pause', project_id: projectId,
            goal_id: goalA.id, run_id: 'run-alpha', status: 'completed', message: response.answer, error_code: null }]
          for (const prior of items) if ((prior.actions as Array<typeof action>)?.some((item) => item.action_id === action.action_id)) {
            prior.actions = [{ ...action, status: 'completed' }]
          }
        }
        items.push(response)
        return route.fulfill({ json: response })
      }
      if (path.endsWith('/orchestration/goals')) return route.fulfill({ json: { items: [goalA, goalB], next_cursor: null } })
      const goalId = path.match(/\/orchestration\/goals\/([^/]+)/)?.[1]
      if (goalId && path.endsWith(`/${goalId}`)) {
        const selected = goalId === goalA.id ? goalA : goalB
        const payload = detail('active', 'running')
        return route.fulfill({ json: { ...payload, goal: selected, run: { ...run, id: goalId === goalA.id ? 'run-alpha' : 'run-beta',
          goal_id: goalId, status: confirmed && goalId === goalA.id ? 'paused' : 'running', phase: 'authorized' },
          decisions: [], actions: [], gates: [], evidence: [], agent_suggestions: [] } })
      }
      if (path.endsWith('/memory')) return route.fulfill({ json: baseline.memory })
      if (path.endsWith('/checkpoint')) return route.fulfill({ json: { items: [], deferred_count: 0, total_pending: 0 } })
      if (/\/(processes|agent-reviews|warnings|decisions)$/.test(path)) return route.fulfill({ json: [] })
      if (path.endsWith('/orchestration/health')) return route.fulfill({ json: { debug_enabled: false, status: 'ok' } })
      if (req.method() !== 'GET') throw new Error(`Unexpected unmocked mutation ${req.method()} ${path}`)
      return route.fulfill({ json: { items: [], next_cursor: null } })
    })

    await page.goto(`/dashboard/orchestration/${goalA.id}`)
    if (viewport.name === 'mobile') await page.getByRole('button', { name: /Activity/ }).click()
    await page.getByRole('tab', { name: 'Calls', exact: true }).click()
    if (viewport.name === 'mobile') await page.keyboard.press('Escape')
    await page.getByRole('button', { name: 'Open conversation', exact: true }).click()
    const composer = page.getByRole('textbox', { name: 'Message the orchestrator' })
    await expect(composer).toBeVisible()
    await expect(composer).toBeFocused()
    await expect(page.getByRole('textbox', { name: 'Message the orchestrator' })).toHaveCount(1)
    await composer.fill('What is the state?')
    await composer.press('Control+Enter')
    await expect(page.getByText('Alpha is active and awaiting validation.', { exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Helpful', exact: true })).toBeVisible()
    await page.screenshot({ path: testInfo.outputPath(`spr185-${viewport.name}-feedback.png`), fullPage: true })
    await composer.fill('Prioritize verification')
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect(page.getByText('Direction queued for the current run.', { exact: true }).first()).toBeVisible()
    await composer.fill('Pause this goal')
    await page.getByRole('button', { name: 'Send', exact: true }).click()
    await expect(page.getByRole('button', { name: /Review pause/i })).toBeVisible()
    await page.getByRole('link', { name: 'Go to Beta', exact: true }).first().click()
    await expect(page).toHaveURL(/orchestration\/goal-beta/)
    await page.getByRole('button', { name: /Review pause/i }).click()
    const dialog = page.getByRole('alertdialog')
    await expect(dialog).toContainText('Ship Alpha')
    await expect(dialog).toContainText('run-alpha')
    await expect(dialog).not.toContainText('Ship Beta')
    await dialog.getByRole('button', { name: /Confirm/i }).click()
    await expect(page.getByText('Alpha paused.', { exact: true }).first()).toBeVisible()
    expect(submissions.map((item) => item.content)).toEqual(['What is the state?', 'Prioritize verification', 'Pause this goal', 'yes'])
    expect(submissions.slice(0, 3).every((item) => item.context_goal_id === goalA.id)).toBe(true)
    expect(submissions[3].action_id).toBe(action.action_id)
    await expect(composer).toHaveCount(1)
    const box = await composer.boundingBox()
    expect(box).not.toBeNull()
    expect(box!.x).toBeGreaterThanOrEqual(0)
    expect(box!.x + box!.width).toBeLessThanOrEqual(viewport.width)
    await page.screenshot({ path: testInfo.outputPath(`spr185-${viewport.name}.png`), fullPage: true })
    expect(errors).toEqual([])
    if (viewport.name === 'mobile') {
      await page.keyboard.press('Escape')
      await expect(composer).toBeHidden()
      await expect(page.getByRole('button', { name: /Activity/ })).toBeFocused()
    }
  })
}
