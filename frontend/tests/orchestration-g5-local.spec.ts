import path from 'node:path'
import { expect, test, type Page } from '@playwright/test'
import type { OrchestrationStatusLine, OrchestrationSupervision, OrchestrationSupervisionCondition, OrchestrationWorkflowFacts } from '../src/lib/types'
import { baselineProcess, detail, goal, interceptOrchestration, supervisionFor } from './support/orchestration-fixtures'

// Provider-free: mocked API + WebSocket only; any non-GET other than the allowed one fails the test.
const NOW = '2026-07-15T10:00:00Z'
const ref = (kind: 'task' | 'process', id: string, label: string) => ({ kind, id, project_id: goal.project_id, goal_id: goal.id, label }) as never

function workflow(over: Partial<OrchestrationWorkflowFacts> = {}): OrchestrationWorkflowFacts {
  return {
    run_id: 'run-release', control_version: 1, phase: 'baseline', phase_label: 'Baseline setup', process: null, process_type: 'goal_definition',
    process_status: 'running', task: null, operation_summary: 'Recorded: Goal definition is running.', occurred_at: NOW,
    pending_decision_id: null, additional_pending_count: 0, application: null, additional_application_count: 0,
    guidance: { kind: 'none', label: '', decision_id: null, reason: null }, ...over,
  }
}

interface Scenario {
  goalStatus: string; runStatus: string; authorized: boolean; condition: OrchestrationSupervisionCondition
  workflow: OrchestrationWorkflowFacts | null; currentState?: OrchestrationStatusLine['current_state']
  supervision?: Partial<OrchestrationSupervision>; processStatus?: 'running' | 'waiting_decision' | 'completed'; objective?: string
}

function build(s: Scenario) {
  const base = detail(s.goalStatus as never, s.runStatus as never)
  const status_line: OrchestrationStatusLine = {
    current_state: s.currentState ?? { kind: 'unknown', summary: 'Recorded state only.', activities: [] },
    attention_summary: null, last_material_event: null, workflow: s.workflow,
  }
  return {
    ...base,
    ...(s.objective ? { goal: { ...base.goal, objective: s.objective } } : {}),
    run: { ...base.run, phase: 'baseline', condition: s.condition, baseline_authorized: s.authorized, control_version: 1, active_blockers: [] },
    supervision: { ...supervisionFor(s.condition), workers: [], status_line, ...s.supervision },
  }
}

async function setup(page: Page, s: Scenario, allowed: string[] = []) {
  const mutations: string[] = []
  page.on('request', (r) => { if (new URL(r.url()).pathname.startsWith('/api/') && r.method() !== 'GET') mutations.push(`${r.method()} ${new URL(r.url()).pathname}`) })
  await page.addInitScript((id) => localStorage.setItem('huddleroom-ui', JSON.stringify({ state: { activeProjectId: id, sidebarCollapsed: true }, version: 1 })), goal.project_id)
  await page.routeWebSocket('**/ws/**', () => {})
  const now = new Date().toISOString()
  // Lowest priority catch-all (registered first).
  await page.route('**/api/v1/**', (route) => {
    const path = new URL(route.request().url()).pathname
    if (route.request().method() !== 'GET') return route.fulfill({ status: 500, json: { detail: `Unmocked mutation ${path}` } })
    if (path === '/api/v1/config') return route.fulfill({ json: { auth_enabled: false } })
    if (path === '/api/v1/projects') return route.fulfill({ json: { items: [{ id: goal.project_id, name: 'G5 fixture', workspace_path: '/mock', config: {}, created_at: now, updated_at: now }], next_cursor: null } })
    if (path.endsWith('/orchestration/health')) return route.fulfill({ json: { debug_enabled: true, status: 'ok' } })
    if (path.endsWith('/orchestration/chat')) return route.fulfill({ json: { items: [], total: 0, omitted: 0, allowance: { enabled: false, unlimited: true }, goal_allowances: {}, steering: {} } })
    return route.fulfill({ json: { items: [], next_cursor: null } })
  })
  await interceptOrchestration(page)
  const payload = build(s)
  const status = s.processStatus ?? 'running'
  await page.route('**/api/v1/projects/*/orchestration/goals/goal-release', (route) =>
    route.request().method() === 'GET' ? route.fulfill({ json: payload }) : route.fallback())
  await page.route('**/api/v1/projects/*/orchestration/goals/goal-release/processes', (route) =>
    route.fulfill({ json: [baselineProcess({ status })] }))
  await page.route('**/api/v1/projects/*/orchestration/goals/goal-release/baseline/authorize', (route) =>
    route.fulfill({ json: { ...payload, run: { ...payload.run, baseline_authorized: true } } }))
  return { assertMutations: () => expect(mutations).toEqual(allowed) }
}

const open = async (page: Page, name: string, vp: string) => {
  await page.goto(`/dashboard/orchestration/${goal.id}`)
  await expect(page.getByRole('heading', { name: 'Current operation' })).toBeVisible()
  return async () => {
    await page.screenshot({ path: path.resolve(process.cwd(), `test-results/g5/g5-${name}-${vp}.png`), fullPage: true })
    // Two sizes of tolerance: documentElement and scrollingElement must not overflow horizontally.
    expect(await page.evaluate(() => { const e = document.scrollingElement!; return e.scrollWidth <= e.clientWidth })).toBe(true)
  }
}
const op = (page: Page) => page.locator('section[aria-labelledby="current-operation-title"]')
async function answerInNeedsYou(page: Page) {
  const button = page.getByRole('button', { name: 'Answer in Needs You' })
  await expect(button).toBeVisible()
  expect((await button.boundingBox())!.height).toBeGreaterThanOrEqual(44)
  await button.click()
  await expect(page.locator('#needs-you-queue-heading')).toBeFocused()
}
const noActions = /^(Start baseline|Continue baseline|Authorize|Start work)/

for (const vp of [{ name: '1440x900', width: 1440, height: 900 }, { name: '390x844', width: 390, height: 844 }, { name: '375x812', width: 375, height: 812 }]) {
  test.describe(`G5 ${vp.name}`, () => {
    test.use({ viewport: { width: vp.width, height: vp.height } })

    test('a baseline running, no worker', async ({ page }) => {
      const long = 'LongTaskName'.repeat(12)
      const h = await setup(page, { goalStatus: 'active', runStatus: 'running', authorized: true, condition: 'working',
        workflow: workflow({ task: ref('task', 'task-validation', long) }), currentState: { kind: 'working', summary: 'Baseline step recorded as running.', activities: [] } })
      const shot = await open(page, 'a-running-no-worker', vp.name)
      await expect(op(page)).toContainText('Phase: Baseline setup')
      await expect(op(page)).toContainText('Recorded: Goal definition is running.')
      await expect(page.getByText('No active worker confirmed')).toHaveCount(0)
      await expect(op(page)).not.toContainText(/worker (is )?(live|active)|is live|currently active/i)
      await expect(page.getByRole('button', { name: noActions })).toHaveCount(0)
      await shot(); h.assertMutations()
    })

    test('b paused with pending decision and saved answer', async ({ page }) => {
      const h = await setup(page, { goalStatus: 'paused', runStatus: 'paused', authorized: false, condition: 'paused',
        workflow: workflow({ pending_decision_id: 'direction-release', application: { action_id: 'a1', decision_ids: ['d1'], state: 'pending', reason: null },
          guidance: { kind: 'resume', label: 'Resume goal', decision_id: null, reason: null } }),
        currentState: { kind: 'paused', summary: 'Goal is paused.', activities: [] }, supervision: { pending_direction: supervisionFor('needs_you').pending_direction } })
      const shot = await open(page, 'b-paused-pending', vp.name)
      await expect(op(page).locator('span').filter({ hasText: /^●\s*Paused$/ })).toBeVisible()
      await expect(op(page)).toContainText('Waiting for your decision: "Should this release proceed to deployment?"')
      await expect(op(page)).toContainText('Paused — new work is held. Work already started is not cancelled and may still finish.')
      await expect(op(page)).toContainText('Saved · not applied — Not in effect yet.')
      await expect(op(page)).toContainText('Status refreshed')
      await expect(op(page)).not.toContainText('Goal is paused.')
      await expect(op(page)).not.toContainText('Data as of')
      await op(page).screenshot({ path: path.resolve(process.cwd(), `test-results/g5/g5-b2-card-${vp.name}.png`) })
      await answerInNeedsYou(page)
      await expect(op(page)).toContainText('Next: Resume goal')
      await expect(page.getByRole('button', { name: noActions })).toHaveCount(0)
      await expect(page.getByRole('button', { name: /Authorize remaining baseline/ })).toHaveCount(0)
      await shot()
      await page.getByRole('tab', { name: 'Debug' }).click()
      await expect(page.getByRole('heading', { name: 'Pending owner decision' })).toBeVisible()
      await expect(page.getByRole('heading', { name: 'Runtime waits' })).toBeVisible()
      await page.screenshot({ path: path.resolve(process.cwd(), `test-results/g5/g5-b-debug-tab-${vp.name}.png`), fullPage: true })
      h.assertMutations()
    })

    test('c pending decision, baseline unauthorized', async ({ page }) => {
      const h = await setup(page, { goalStatus: 'active', runStatus: 'running', authorized: false, condition: 'needs_you', processStatus: 'waiting_decision',
        workflow: workflow({ process_status: 'waiting_decision', pending_decision_id: 'decision-human-object',
          guidance: { kind: 'review_decision', label: 'Answer decision', decision_id: 'decision-human-object', reason: null } }) })
      const shot = await open(page, 'c-pending-decision', vp.name)
      await expect(op(page)).toContainText('Next: Answer decision')
      await expect(op(page)).toContainText('Waiting for your decision: Decision details not loaded yet')
      await op(page).screenshot({ path: path.resolve(process.cwd(), `test-results/g5/g5-c2-card-${vp.name}.png`) })
      await answerInNeedsYou(page)
      await expect(page.getByRole('button', { name: noActions })).toHaveCount(0)
      await expect(page.getByRole('button', { name: /Authorize/ })).toHaveCount(0)
      await shot(); h.assertMutations()
    })

    test('d saved answer blocked', async ({ page }) => {
      const h = await setup(page, { goalStatus: 'active', runStatus: 'running', authorized: true, condition: 'needs_attention',
        workflow: workflow({ application: { action_id: 'a1', decision_ids: ['d1'], state: 'blocked', reason: 'authority revoked' },
          guidance: { kind: 'review_blocker', label: 'Review blocker', decision_id: null, reason: null } }) })
      const shot = await open(page, 'd-application-blocked', vp.name)
      await expect(op(page)).toContainText('Blocked — authority revoked')
      await shot(); h.assertMutations()
    })

    test('e authorize remaining baseline by keyboard', async ({ page }) => {
      const h = await setup(page, { goalStatus: 'active', runStatus: 'running', authorized: false, condition: 'waiting', processStatus: 'completed',
        workflow: workflow({ guidance: { kind: 'authorize_baseline', label: 'Authorize remaining baseline', decision_id: null, reason: null } }) },
      ['POST /api/v1/projects/' + goal.project_id + '/orchestration/goals/goal-release/baseline/authorize'])
      const shot = await open(page, 'e-authorize', vp.name)
      const button = page.getByRole('button', { name: 'Authorize remaining baseline' })
      await expect(button).toBeVisible()
      await expect(button).toBeEnabled()
      await expect(page.getByText('Baseline is waiting — press Authorize remaining baseline to continue.')).toBeVisible()
      await shot()
      await page.locator('body').click({ position: { x: 1, y: 1 } })
      let focused = false
      for (let i = 0; i < 80 && !focused; i++) {
        await page.keyboard.press('Tab')
        focused = await button.evaluate((el) => el === document.activeElement)
      }
      expect(focused, 'authorize button reachable by Tab').toBe(true)
      const posted = page.waitForRequest((r) => r.method() === 'POST' && r.url().endsWith('/baseline/authorize'))
      await page.keyboard.press('Enter')
      await posted
      await page.waitForTimeout(500)
      h.assertMutations()
    })

    test('f completed', async ({ page }) => {
      const h = await setup(page, { goalStatus: 'completed', runStatus: 'completed', authorized: true, condition: 'completed', processStatus: 'completed',
        workflow: workflow({ phase_label: 'Completed', application: null, guidance: { kind: 'none', label: '', decision_id: null, reason: null } }),
        currentState: { kind: 'completed', summary: 'Goal completed.', activities: [] } })
      const shot = await open(page, 'f-completed', vp.name)
      await expect(op(page).locator('span').filter({ hasText: /^●\s*Completed$/ })).toBeVisible()
      await expect(op(page)).not.toContainText(/Saved · not applied|Blocked —|Failed —/)
      await expect(op(page)).not.toContainText('Next:')
      await expect(page.getByRole('button', { name: /^(Pause|Resume|Cancel|Start|Continue|Authorize)/ })).toHaveCount(0)
      await shot(); h.assertMutations()
    })

    test('d2 failed application and unknown recorded time', async ({ page }) => {
      const h = await setup(page, { goalStatus: 'active', runStatus: 'running', authorized: true, condition: 'needs_attention',
        workflow: workflow({ occurred_at: null, application: { action_id: 'a1', decision_ids: ['d1'], state: 'failed', reason: 'executor error' } }) })
      const shot = await open(page, 'd2-application-failed', vp.name)
      await expect(op(page)).toContainText('Failed — executor error')
      await expect(op(page)).toContainText('Recorded time unknown')
      await expect(op(page)).not.toContainText('Next:')
      await shot(); h.assertMutations()
    })

    test('i long guidance label on authorize button wraps', async ({ page }) => {
      const label = 'Authorize remaining baseline ' + 'and continue every outstanding process step carefully '.repeat(3)
      const h = await setup(page, { goalStatus: 'active', runStatus: 'running', authorized: false, condition: 'waiting', processStatus: 'completed',
        workflow: workflow({ guidance: { kind: 'authorize_baseline', label, decision_id: null, reason: null } }) })
      const shot = await open(page, 'i-long-guidance-button', vp.name)
      const button = page.getByRole('button', { name: /^Authorize remaining baseline and continue/ })
      await expect(button).toBeVisible()
      const box = (await button.boundingBox())!
      expect(box.x + box.width).toBeLessThanOrEqual(vp.width)
      await button.scrollIntoViewIfNeeded()
      await page.locator('section[aria-labelledby="baseline-process-chain-heading"]').screenshot({ path: path.resolve(process.cwd(), `test-results/g5/g5-i2-button-${vp.name}.png`) })
      await shot(); h.assertMutations()
    })

    test('g legacy workflow null', async ({ page }) => {
      const h = await setup(page, { goalStatus: 'active', runStatus: 'running', authorized: true, condition: 'working', workflow: null,
        currentState: { kind: 'working', summary: 'Working.', activities: [] } })
      const shot = await open(page, 'g-legacy-null', vp.name)
      await expect(op(page)).toContainText('Workflow details unavailable.')
      await shot(); h.assertMutations()
    })

    test('h long agent and task names wrap; top bar says Live connection', async ({ page }) => {
      const long = 'UnbrokenAgentName'.repeat(10)
      const h = await setup(page, { goalStatus: 'active', runStatus: 'running', authorized: true, condition: 'working',
        workflow: workflow({ task: ref('task', 'task-validation', long), process: ref('process', 'process-goal-definition', long) }),
        currentState: { kind: 'working', summary: 'Working.', activities: [{ key: 'k', state: 'reported_active', summary: 'Reported activity', subject: ref('task', 't', long), actor: ref('task', 'ag', long), observed_at: NOW }] } })
      const shot = await open(page, 'h-long-names', vp.name)
      await expect(page.getByText('Live connection', { exact: true }).first()).toBeVisible()
      await shot(); h.assertMutations()
    })
  })
}

// ---- G6 compact controls (SPR235) ----
const LONG = Array.from({ length: 3 }, (_, i) => `Paragraph ${i + 1}: ` + 'ship the evidence-backed release with full validation across every environment. '.repeat(2).trim()).join('\n\n')
const g6 = (name: string) => path.resolve(process.cwd(), `test-results/g6/g6-${name}.png`)
const bar = (page: Page) => page.getByTestId('orchestration-sticky-bar')

const inViewport = async (page: Page, loc: ReturnType<Page['locator']>) => {
  const b = (await loc.boundingBox())!
  const vp = page.viewportSize()!
  expect(b, 'has box').toBeTruthy()
  expect(b.y).toBeGreaterThanOrEqual(0); expect(b.x).toBeGreaterThanOrEqual(0)
  expect(b.y + b.height).toBeLessThanOrEqual(vp.height); expect(b.x + b.width).toBeLessThanOrEqual(vp.width)
}
const noHScroll = (page: Page) => page.evaluate(() => {
  const e = document.scrollingElement!
  const sc = document.querySelector('[data-testid="orchestration-sticky-bar"]')
  let n = sc?.parentElement ?? null; let ok = e.scrollWidth <= e.clientWidth
  while (n) { if (n.scrollWidth > n.clientWidth + 1 && getComputedStyle(n).overflowX !== 'visible') ok = false; n = n.parentElement }
  return ok
})
const scrollAllToBottom = (page: Page) => page.evaluate(() => {
  let n: Element | null = document.querySelector('[data-testid="orchestration-sticky-bar"]')
  while (n) { n.scrollTop = n.scrollHeight; n = n.parentElement }
  window.scrollTo(0, document.body.scrollHeight)
})

for (const vp of [{ name: '1440x900', width: 1440, height: 900 }, { name: '375x812', width: 375, height: 812 }]) {
  test.describe(`G6 compact controls (SPR235) ${vp.name}`, () => {
    test.use({ viewport: { width: vp.width, height: vp.height } })
    const desktop = vp.width >= 768

    async function common(page: Page, shot: string, controls: RegExp) {
      await expect(bar(page)).toBeVisible()
      const btn = bar(page).getByRole('button', { name: controls })
      await expect(btn).toBeVisible()
      await inViewport(page, btn); await inViewport(page, bar(page))
      expect(await noHScroll(page)).toBe(true)
      await page.screenshot({ path: g6(`${shot}-${vp.name}-viewport`) })
      await scrollAllToBottom(page)
      await page.waitForTimeout(100)
      if (desktop) { await inViewport(page, bar(page)); await page.screenshot({ path: g6(`${shot}-${vp.name}-scrolled`) }) }
      expect(await noHScroll(page)).toBe(true)
    }

    test('a long multi-paragraph objective, running', async ({ page }) => {
      const h = await setup(page, { goalStatus: 'active', runStatus: 'running', authorized: true, condition: 'working', objective: LONG,
        workflow: workflow(), currentState: { kind: 'working', summary: 'Working.', activities: [] } })
      await open(page, 'g6a', vp.name)
      await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1)
      await common(page, 'a-long-objective', /^Pause goal/)
      const summary = page.locator('summary', { hasText: /full objective/ })
      await summary.focus(); await page.keyboard.press('Enter')
      const det = summary.locator('..')
      await expect(det).toHaveAttribute('open', '')
      await expect(det).toContainText('Paragraph 3:')
      await expect(det.locator('p')).toHaveText(LONG)
      await page.screenshot({ path: g6(`a-disclosure-open-${vp.name}`) })
      await page.keyboard.press('Enter')
      await expect(det).not.toHaveAttribute('open', '')
      if (desktop) { // Tab onward from the sticky bar: focused element is never hidden under it.
        await page.evaluate(() => (document.querySelector('[data-testid="orchestration-sticky-bar"]') as HTMLElement).focus())
        for (let i = 0; i < 25; i++) {
          await page.keyboard.press('Tab')
          const covered = await page.evaluate(() => {
            const a = document.activeElement as HTMLElement | null; const b = document.querySelector('[data-testid="orchestration-sticky-bar"]')!
            if (!a || a === document.body || b.contains(a)) return false
            const r = a.getBoundingClientRect(); const s = b.getBoundingClientRect()
            return r.bottom > s.top && r.top < s.bottom && r.right > s.left && r.left < s.right
          })
          expect(covered, `Tab step ${i} covered by sticky bar`).toBe(false)
        }
      }
      h.assertMutations()
    })

    test('b paused with pending decision and saved answer', async ({ page }) => {
      const h = await setup(page, { goalStatus: 'paused', runStatus: 'paused', authorized: false, condition: 'paused', objective: LONG,
        workflow: workflow({ pending_decision_id: 'direction-release', application: { action_id: 'a1', decision_ids: ['d1'], state: 'pending', reason: null },
          guidance: { kind: 'resume', label: 'Resume goal', decision_id: null, reason: null } }),
        currentState: { kind: 'paused', summary: 'Goal is paused.', activities: [] }, supervision: { pending_direction: supervisionFor('needs_you').pending_direction } })
      await open(page, 'g6b', vp.name)
      await common(page, 'b-paused', /^Resume goal/)
      await expect(bar(page)).toContainText('Goal: Paused')
      await expect(bar(page).getByRole('button', { name: /^Pause goal/ })).toHaveCount(0)
      await expect(page.getByRole('button', { name: 'Answer in Needs You' })).toBeVisible()
      await expect(op(page)).toContainText('Saved · not applied')
      h.assertMutations()
    })

    test('c blocked saved answer (needs attention)', async ({ page }) => {
      const h = await setup(page, { goalStatus: 'active', runStatus: 'running', authorized: true, condition: 'needs_attention', objective: LONG,
        workflow: workflow({ application: { action_id: 'a1', decision_ids: ['d1'], state: 'blocked', reason: 'authority revoked' },
          guidance: { kind: 'review_blocker', label: 'Review blocker', decision_id: null, reason: null } }) })
      await open(page, 'g6c', vp.name)
      await common(page, 'c-blocked', /^Pause goal/)
      await expect(op(page)).toContainText('Blocked — authority revoked')
      h.assertMutations()
    })
  })
}
