import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { expect, test, type Page } from '@playwright/test'
import { neverObserves } from './helpers/poll'
import { captureReceipts } from './helpers/receipts'
import { confirmPause, continueBaseline, decisionRow, goalStatusBadge, pauseGoal, resumeGoal } from './helpers/selectors'

// Second live run, driven step by step by tests/live_acceptance/test_live_run_l2.py against an
// already-running server (one test per UI step, picked with --grep). Env: LIVE_ACCEPTANCE_PROJECT_ID,
// LIVE_ACCEPTANCE_GOAL_ID, LIVE_ACCEPTANCE_OUTPUT (JSON + screenshots).

const projectId = process.env.LIVE_ACCEPTANCE_PROJECT_ID ?? ''
const goalId = process.env.LIVE_ACCEPTANCE_GOAL_ID ?? ''
const out = process.env.LIVE_ACCEPTANCE_OUTPUT || 'test-results-live'
const ACCEPTED = 'Pause accepted. Existing work may finish.'
const TH_TITLE = /Approve team hierarchy/i

async function openGoal(page: Page) {
  expect(projectId && goalId, 'LIVE_ACCEPTANCE_PROJECT_ID and LIVE_ACCEPTANCE_GOAL_ID are required').toBeTruthy()
  await page.addInitScript((id) => {
    localStorage.setItem('huddleroom-ui', JSON.stringify({ state: { activeProjectId: id, sidebarCollapsed: false }, version: 1 }))
  }, projectId)
  await page.goto(`/dashboard/orchestration/${goalId}`)
  await page.waitForLoadState('networkidle').catch(() => {})
}

const writeJson = (name: string, data: unknown) => {
  mkdirSync(out, { recursive: true })
  writeFileSync(join(out, `${name}.json`), JSON.stringify(data, null, 2))
}
const shot = (page: Page, name: string) => page.screenshot({ path: join(out, `${name}.png`), fullPage: true })
const hdr = (page: Page) => page.getByTestId('orchestration-goal-header')

test('resume', async ({ page }) => {
  test.setTimeout(120_000)
  await openGoal(page)
  await resumeGoal(page).click()
  await expect(resumeGoal(page)).toHaveCount(0, { timeout: 30_000 })
  await shot(page, 'after-resume')
  writeJson('resume', { done: true })
})

test('continue-baseline', async ({ page }) => {
  test.setTimeout(120_000)
  await openGoal(page)
  await continueBaseline(page).click({ timeout: 60_000 })
  await expect(continueBaseline(page)).toHaveCount(0, { timeout: 30_000 })
  await shot(page, 'after-continue-baseline')
  writeJson('continue-baseline', { done: true })
})

test('answer-th', async ({ page }) => {
  test.setTimeout(150_000)
  const receipts = captureReceipts(page)
  await openGoal(page)
  const row = decisionRow(page, TH_TITLE).first()
  await expect(row).toBeVisible({ timeout: 60_000 })
  await row.getByRole('button', { name: /^Accept/i }).first().click()
  await expect(decisionRow(page, TH_TITLE)).toHaveCount(0, { timeout: 60_000 })
  await shot(page, 'after-answer-th')
  writeJson('answer-th', { answer_posts: receipts.all.filter((r) => r.kind === 'answer-post') })
})

test('start-work', async ({ page }) => {
  test.setTimeout(120_000)
  await openGoal(page)
  const start = page.getByRole('button', { name: 'Start work' })
  await start.click({ timeout: 60_000 })
  await expect(start).toHaveCount(0, { timeout: 30_000 })
  await shot(page, 'after-start-work')
  writeJson('start-work', { done: true })
})

test('pause', async ({ page }) => {
  test.setTimeout(120_000)
  const receipts = captureReceipts(page)
  await openGoal(page)
  await pauseGoal(page).click()
  const t0 = Date.now()
  await confirmPause(page).click()
  await expect(page.getByText(ACCEPTED).first()).toBeVisible({ timeout: 20_000 })
  const clickToAcceptedMs = Date.now() - t0
  await shot(page, 'pause-accepted')
  await expect(goalStatusBadge(hdr(page), 'Paused').first()).toBeVisible({ timeout: 10_000 })
  const [badgeNever, pauseNeverEnabled] = await Promise.all([
    neverObserves(async () => (await goalStatusBadge(hdr(page), 'Active').count()) > 0
      || (await goalStatusBadge(hdr(page), 'Running').count()) > 0, 15_000),
    neverObserves(async () => (await pauseGoal(page).count()) > 0 && (await pauseGoal(page).isEnabled()), 15_000),
  ])
  writeJson('pause', {
    click_to_accepted_ms: clickToAcceptedMs, never_active_or_running: badgeNever, pause_goal_never_enabled: pauseNeverEnabled,
    pause_post: receipts.all.find((r) => r.kind === 'pause-post') ?? null,
  })
  expect(clickToAcceptedMs).toBeLessThan(5_000)
  expect(badgeNever).toBe(true)
  expect(pauseNeverEnabled).toBe(true)
})
