import { expect, test } from '@playwright/test'
import { gotoDashboard, selectSeedProject } from './support/pages/app'
import { detail, goal, interceptOrchestration, openOrchestrationGoal, run } from './support/orchestration-fixtures'

// Phase 8 Verify bar: WS badge (TopBar) flips Live -> Reconnecting/Off within
// ~12s of the socket dropping (client heartbeat deadline is 10s, checked
// every 2s) and recovers once the network is restored.
test('WS badge flips to Reconnecting/Off when offline and recovers on reconnect', async ({ page, context }) => {
  await gotoDashboard(page)
  await selectSeedProject(page)
  await expect(page.getByText('Live', { exact: true })).toBeVisible({ timeout: 15_000 })

  await context.setOffline(true)
  await expect(page.getByText(/^(Reconnecting|Off)$/)).toBeVisible({ timeout: 13_000 })

  await context.setOffline(false)
  await expect(page.getByText('Live', { exact: true })).toBeVisible({ timeout: 20_000 })
})

// Phase 8 Verify bar: a delayed /api/v1/config response shows the neutral
// shell ("Connecting to HuddleRoom…") instead of a blank page, then resolves
// to the normal dashboard once config arrives.
test('shows the shell instead of a blank page while /config is delayed', async ({ page }) => {
  await page.route('**/api/v1/config', async (route) => {
    await new Promise((resolve) => setTimeout(resolve, 2_000))
    await route.continue()
  })

  await page.goto('/dashboard')
  await expect(page.getByText('Connecting to HuddleRoom…')).toBeVisible({ timeout: 1_000 })
  await expect(page.getByText('Connecting to HuddleRoom…')).toBeHidden({ timeout: 5_000 })
  await expect(page.getByRole('combobox', { name: 'Project switcher' })).toBeVisible()
})

// Phase 8 Verify bar: hot query hooks (tasks/sessions/protocol-instances
// counts, recent events) gate refetchInterval on WS connection state — zero
// polling requests should fire while the socket is connected.
test('has zero polling requests to hot query endpoints while the socket is connected', async ({ page }) => {
  await gotoDashboard(page)
  await selectSeedProject(page)
  await expect(page.getByText('Live', { exact: true })).toBeVisible({ timeout: 15_000 })

  const hotEndpoint = /\/tasks\/count|\/sessions\/count|\/protocol-instances\/count|\/api\/v1\/events\b/
  const seen = new Set<string>()
  const repeated: string[] = []
  page.on('request', (request) => {
    const url = new URL(request.url())
    const key = `${url.pathname}${url.search}`
    if (!hotEndpoint.test(key)) return
    if (seen.has(key)) repeated.push(key)
    seen.add(key)
  })

  await page.waitForTimeout(12_000)
  expect(repeated, `unexpected repeated polling requests while WS connected: ${repeated.join(', ')}`).toEqual([])
})

// Phase 8 Verify bar: the goal-page ledger caps its initial render at 50 rows
// with a "Show more" affordance, even when 200+ entries exist.
test('caps the goal ledger at 50 entries with Show more', async ({ page }) => {
  const totalActions = 219 // + 1 seed decision = 220 ledger entries
  const bigActions = Array.from({ length: totalActions }, (_, i) => ({
    id: `action-bulk-${i}`,
    run_id: run.id,
    decision_id: null,
    idempotency_key: `run:bulk:${i}`,
    action_type: 'bulk_note',
    request: { reason: `Bulk ledger entry ${i}` },
    target_type: null,
    target_id: null,
    status: 'completed',
    error: null,
    created_at: new Date(Date.UTC(2026, 6, 15, 9, 30, 0) + i * 1000).toISOString(),
    updated_at: new Date(Date.UTC(2026, 6, 15, 9, 30, 0) + i * 1000).toISOString(),
  }))

  await interceptOrchestration(page, {})
  // Registered after interceptOrchestration's own handler, so this route is
  // consulted first (last-registered wins) — fulfil only the goal-detail GET
  // ourselves with a bulked-up ledger, and fall back for every other request
  // orchestration.spec.ts's harness already handles.
  await page.route('**/api/v1/projects/*/orchestration/goals**', async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const suffix = url.pathname.split('/orchestration/goals')[1]
    if (request.method() !== 'GET' || suffix !== `/${goal.id}`) {
      await route.fallback()
      return
    }
    const base = detail()
    await route.fulfill({ json: { ...base, actions: [...base.actions, ...bigActions] } })
  })

  await openOrchestrationGoal(page)

  const ledger = page.locator('section[aria-labelledby="details-tabs-heading"]')
  const rows = ledger.getByRole('listitem')
  await expect(rows).toHaveCount(50)
  const showMore = ledger.getByRole('button', { name: 'Show more' })
  await expect(showMore).toBeVisible()

  await showMore.click()
  await expect(rows).not.toHaveCount(50)
  expect(await rows.count()).toBeGreaterThan(200)
  await expect(showMore).toBeHidden()
})
