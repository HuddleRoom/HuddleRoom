import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { expect, test, type Page } from '@playwright/test'
import { dropAfterCommit, failWith500, holdThenRelease, reloadMidRequest } from './helpers/faults'
import { neverObserves, observesWithin } from './helpers/poll'
import { captureReceipts } from './helpers/receipts'
import {
  approveIn, confirmPause, continueBaseline, decisionRow, goalStatusBadge, pauseGoal, resumeGoal, savedAnswers,
} from './helpers/selectors'

// One live run, driven step by step by tests/live_acceptance/test_live_run.py against an already-running
// server (one test per step group, picked with --grep). Env: LIVE_ACCEPTANCE_PROJECT_ID,
// LIVE_ACCEPTANCE_GOAL_ID, LIVE_ACCEPTANCE_OUTPUT (JSON + screenshots), LIVE_ACCEPTANCE_DECISION_TITLE
// (optional regex source for the decision row of request-changes).

const projectId = process.env.LIVE_ACCEPTANCE_PROJECT_ID ?? ''
const goalId = process.env.LIVE_ACCEPTANCE_GOAL_ID ?? ''
const out = process.env.LIVE_ACCEPTANCE_OUTPUT || 'test-results-live'
const ACCEPTED = 'Pause accepted. Existing work may finish.'
const BATCH_TITLE = /Agent definitions need review/i
const TH_TITLE = 'Approve team hierarchy'
const ANSWER = /\/decisions\/[^/?]+\/answer(\?.*)?$/
const BATCH = /\/decisions\/agent-definition-review\/batch-answer(\?.*)?$/
// Per-decision readback the UI makes to reconcile a dropped answer (single decision, uuid segment).
const READBACK = /\/goals\/[^/]+\/decisions\/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}(\?.*)?$/i
// Background refetches of the dashboard: goal detail, decisions list, checkpoint, processes, warnings.
// Not the per-decision readback above.
const BACKGROUND = /\/goals\/[^/]+(\/decisions(\/checkpoint)?|\/processes|\/warnings)?(\?.*)?$/i
const PAUSE_POST = /\/goals\/[^/]+\/pause-control(\?.*)?$/

async function openGoal(page: Page) {
  expect(projectId && goalId, 'LIVE_ACCEPTANCE_PROJECT_ID and LIVE_ACCEPTANCE_GOAL_ID are required').toBeTruthy()
  // The active project lives in persisted UI state (zustand persist, key huddleroom-ui).
  await page.addInitScript((id) => {
    localStorage.setItem('huddleroom-ui', JSON.stringify({ state: { activeProjectId: id, sidebarCollapsed: false }, version: 1 }))
  }, projectId)
  await page.goto(`/dashboard/orchestration/${goalId}`)
}

const writeJson = (name: string, data: unknown) => {
  mkdirSync(out, { recursive: true })
  writeFileSync(join(out, `${name}.json`), JSON.stringify(data, null, 2))
}
const shot = (page: Page, name: string) => page.screenshot({ path: join(out, `${name}.png`), fullPage: true })
const checking = (page: Page) => page.getByText('Checking saved answer')
const hdr = (page: Page) => page.getByTestId('orchestration-goal-header')
const batchRow = (page: Page) => decisionRow(page, BATCH_TITLE)

/** Approve the whole agent-definition batch: the bulk button if offered, else every article's Approve. */
async function approveBatch(page: Page) {
  const row = batchRow(page)
  await expect(row.first()).toBeVisible({ timeout: 60_000 })
  const bulk = row.first().getByRole('button', { name: /^Approve all remaining \(\d+\)$/ })
  if (await bulk.count()) {
    await bulk.click()
    return
  }
  const articles = row.first().getByRole('article')
  const n = await articles.count()
  for (let i = 0; i < n; i += 1) await approveIn(articles.nth(i)).click()
}

/** Goal header badges 'Active'/'Running' must never appear and 'Pause goal' must never be enabled for ms. */
async function pauseNegatives(page: Page, ms: number) {
  const [badgeNever, pauseNeverEnabled] = await Promise.all([
    neverObserves(async () => (await goalStatusBadge(hdr(page), 'Active').count()) > 0
      || (await goalStatusBadge(hdr(page), 'Running').count()) > 0, ms),
    neverObserves(async () => (await pauseGoal(page).count()) > 0 && (await pauseGoal(page).isEnabled()), ms),
  ])
  return { never_active_or_running: badgeNever, pause_goal_never_enabled: pauseNeverEnabled, poll_ms: ms }
}

const posts = (r: ReturnType<typeof captureReceipts>, kind: 'answer-post' | 'pause-post') => r.all.filter((x) => x.kind === kind)

test('approve-batch', async ({ page }) => {
  test.setTimeout(240_000)
  const receipts = captureReceipts(page)
  const seen: Array<{ url: string; status: number; body: string }> = []
  page.on('response', async (res) => {
    if (/\/decisions|\/processes/.test(res.url())) {
      seen.push({ url: res.url(), status: res.status(), body: (await res.text().catch(() => '')).slice(0, 4000) })
    }
  })
  // Always written (finally), also on failure: whatever was measured so far.
  const result: Record<string, unknown> = {}
  let faults: Array<{ release: () => Promise<void> }> = []
  try {
    await openGoal(page)
    const t0 = Date.now()
    await page.waitForLoadState('networkidle').catch(() => {})
    result.idle_wait_ms = Date.now() - t0
    const appeared = await page.getByText(BATCH_TITLE).first().waitFor({ timeout: 20_000 }).then(() => true, () => false)
    if (!appeared) {
      await shot(page, 'batch-row-missing')
      result.failed = 'batch row did not render within 20 s'
      throw new Error('Agent definitions batch row did not render (see approve-batch.json)')
    }
    // Phase A (answer uncertain): the server commits the batch answer, the client sees a reset. The UI then reads
    // each decision back; those reads are held 1.5 s (no 500) so "Checking saved answer" is observable.
    const drop = await dropAfterCommit(page, BATCH)
    const readbackHold = await holdThenRelease(page, READBACK, 1500)
    faults = [drop, readbackHold]
    const saved = savedAnswers(page)
    // Background refetches (decisions list, checkpoint, processes, warnings, goal detail; not the per-decision
    // readback) are held 300 ms then answered 500 (first 6 matching requests), but only from the moment the batch POST has been sent, so the
    // UI's pre-submit confirmation read passes and its own post-answer refetches hit the fault.
    let armed = false
    page.on('request', (req) => { if (req.method() === 'POST' && BATCH.test(req.url())) armed = true })
    const background = (url: URL) => armed && BACKGROUND.test(url.pathname + url.search)
    const bg = await failWith500(page, background, 6)
    const bgHold = await holdThenRelease(page, background, 300)
    faults.push(bg, bgHold)
    try {
      await approveBatch(page)
      result.checking_observed = await observesWithin(async () => (await checking(page).count()) > 0, 20_000, 50)
      await expect(saved).toBeVisible({ timeout: 30_000 })
      await expect(saved.getByText('Application pending.')).toBeVisible({ timeout: 5_000 })
      await shot(page, 'saved-after-drop')
      result.saved_count = await saved.getByText('Saved.').count()
      // Recorded, not asserted: with the refetches failing the row only leaves once a later refetch succeeds.
      result.batch_row_gone = await expect(batchRow(page)).toHaveCount(0, { timeout: 20_000 }).then(() => true, () => false)
      // Phase B (saved despite background refresh): the faults above are still armed; hold the assertions for 10 s.
      const [savedStayed, checkingReturned] = await Promise.all([
        neverObserves(async () => !(await saved.isVisible()), 10_000),
        observesWithin(async () => (await checking(page).count()) > 0, 10_000),
      ])
      await shot(page, 'saved-under-background-500')
      Object.assign(result, { saved_stayed: savedStayed, checking_with_saved: checkingReturned })
    } finally {
      Object.assign(result, { drop_hits: drop.hits, readback_hold_hits: readbackHold.hits,
        refetch_hits: bg.hits, hold_hits: bgHold.hits })
    }
    expect(receipts.postCount('answer-post')).toBe(1)
    expect(result.checking_with_saved).toBe(false)
    expect(result.saved_stayed).toBe(true)
  } finally {
    writeJson('approve-batch', {
      ...result, answer_posts: posts(receipts, 'answer-post'), decision_gets: receipts.all.filter((r) => r.kind === 'decision-get'),
      receipt: receipts.lastAnswerReceipt(), responses: seen,
    })
    for (const f of faults) await f.release()
  }
})

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

test('pause', async ({ page }) => {
  test.setTimeout(120_000)
  const receipts = captureReceipts(page)
  await openGoal(page)
  await pauseGoal(page).click()
  await shot(page, 'pause-confirm-open')
  const t0 = Date.now()
  await confirmPause(page).click()
  await expect(page.getByText(ACCEPTED).first()).toBeVisible({ timeout: 20_000 })
  const clickToAcceptedMs = Date.now() - t0
  await shot(page, 'pause-accepted')
  await expect(goalStatusBadge(hdr(page), 'Paused').first()).toBeVisible({ timeout: 10_000 })
  const neg = await pauseNegatives(page, 15_000)
  await shot(page, 'pause-after-poll')
  writeJson('pause', { click_to_accepted_ms: clickToAcceptedMs, ...neg, pause_post: posts(receipts, 'pause-post')[0] ?? null })
  expect(clickToAcceptedMs).toBeLessThan(5_000)
  expect(neg.never_active_or_running).toBe(true)
  expect(neg.pause_goal_never_enabled).toBe(true)
})

test('pause-drop', async ({ page }) => {
  test.setTimeout(120_000)
  const receipts = captureReceipts(page)
  await openGoal(page)
  const drop = await dropAfterCommit(page, PAUSE_POST)
  await pauseGoal(page).click()
  await confirmPause(page).click()
  await expect(goalStatusBadge(hdr(page), 'Paused').first()).toBeVisible({ timeout: 60_000 })
  await shot(page, 'pause-drop-paused')
  const neg = await pauseNegatives(page, 5_000)
  writeJson('pause-drop', { ...neg, hits: drop.hits, pause_posts: posts(receipts, 'pause-post') })
  await drop.release()
  expect(neg.never_active_or_running).toBe(true)
  expect(neg.pause_goal_never_enabled).toBe(true)
})

test('pause-reload', async ({ page }) => {
  test.setTimeout(120_000)
  const receipts = captureReceipts(page)
  await openGoal(page)
  const fault = await reloadMidRequest(page, PAUSE_POST)
  await pauseGoal(page).click()
  await confirmPause(page).click()
  await fault.reload()
  await expect(goalStatusBadge(hdr(page), 'Paused').first()).toBeVisible({ timeout: 60_000 })
  await shot(page, 'pause-reload-paused')
  const neg = await pauseNegatives(page, 10_000)
  writeJson('pause-reload', { ...neg, hits: fault.hits, pause_posts: posts(receipts, 'pause-post') })
  await fault.release()
  expect(neg.never_active_or_running).toBe(true)
  expect(neg.pause_goal_never_enabled).toBe(true)
})

test('request-changes', async ({ page }) => {
  test.setTimeout(180_000)
  const receipts = captureReceipts(page)
  await openGoal(page)
  const row = decisionRow(page, new RegExp(process.env.LIVE_ACCEPTANCE_DECISION_TITLE || TH_TITLE, 'i')).first()
  await expect(row).toBeVisible({ timeout: 60_000 })
  await row.getByRole('button', { name: 'Choose differently\u2026' }).click()
  await row.getByRole('radio', { name: /request_changes/i }).check({ timeout: 10_000 })
  await row.getByRole('textbox', { name: /^Reason/ }).fill('Live acceptance: please revise the proposed team.')
  await shot(page, 'request-changes-form')
  const fault = await reloadMidRequest(page, ANSWER)
  await row.getByRole('button', { name: 'Submit answer' }).click()
  await fault.reload()
  const answer = posts(receipts, 'answer-post')[0]
  const decisionUrl = (answer?.url ?? '').replace(/\/answer(\?.*)?$/, '')
  await expect(page.getByRole('region', { name: 'Needs you' })).toBeVisible({ timeout: 60_000 })
  await shot(page, 'request-changes-after-reload')
  // Not asserted: the Saved banner is client state and is gone after a reload.
  const readback = decisionUrl ? await page.request.get(decisionUrl) : null
  writeJson('request-changes', {
    answer_posts: posts(receipts, 'answer-post'), hits: fault.hits, post_url: answer?.url ?? null,
    readback_status: readback?.status() ?? null, readback: readback ? await readback.json().catch(() => null) : null,
    saved_visible_after_reload: await savedAnswers(page).isVisible(),
  })
  await fault.release()
  expect(receipts.postCount('answer-post')).toBe(1)
})
