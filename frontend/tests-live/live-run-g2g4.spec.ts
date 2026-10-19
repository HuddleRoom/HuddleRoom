import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { expect, test, type Locator, type Page } from '@playwright/test'
import { captureReceipts } from './helpers/receipts'
import { continueBaseline, decisionRow, pauseGoal, resumeGoal } from './helpers/selectors'

// G2--G4 live steps, driven individually by test_live_run_g2g4.py.  The Python
// harness supplies the IDs, feedback text derived from P1, and a supervisor URL.
const projectId = process.env.LIVE_ACCEPTANCE_PROJECT_ID ?? ''
const goalId = process.env.LIVE_ACCEPTANCE_GOAL_ID ?? ''
const out = process.env.LIVE_ACCEPTANCE_OUTPUT || 'test-results-live'
const TH_TITLE = /Approve team hierarchy/i
const planDecisionTitle = process.env.LIVE_ACCEPTANCE_PLAN_DECISION_TITLE ?? ''

async function openGoal(page: Page) {
  expect(projectId && goalId, 'LIVE_ACCEPTANCE_PROJECT_ID and LIVE_ACCEPTANCE_GOAL_ID are required').toBeTruthy()
  await page.addInitScript((id) => {
    localStorage.setItem('huddleroom-ui', JSON.stringify({ state: { activeProjectId: id, sidebarCollapsed: false }, version: 1 }))
  }, projectId)
  await page.goto(`/dashboard/orchestration/${goalId}`)
  await page.waitForLoadState('networkidle').catch(() => {})
}

function writeJson(name: string, data: unknown) {
  mkdirSync(out, { recursive: true })
  writeFileSync(join(out, `${name}.json`), JSON.stringify(data, null, 2))
}

const shot = (page: Page, name: string) => page.screenshot({ path: join(out, `${name}.png`), fullPage: true })
const hierarchyRow = (page: Page) => decisionRow(page, TH_TITLE).first()

async function choose(page: Page, row: Locator, option: RegExp, reason?: string) {
  await row.getByRole('button', { name: 'Choose differently…' }).click()
  const radio = row.getByRole('radio', { name: option }).first()
  await expect(radio).toBeVisible()
  await radio.check()
  if (reason) await row.getByRole('textbox', { name: /^Reason/ }).fill(reason)
  await row.getByRole('button', { name: /^(Submit )?answer$/i }).click()
}

test('resume', async ({ page }) => {
  test.setTimeout(120_000)
  await openGoal(page)
  await resumeGoal(page).click()
  await expect(resumeGoal(page)).toHaveCount(0, { timeout: 30_000 })
  writeJson('resume', { done: true })
})

test('continue-baseline', async ({ page }) => {
  test.setTimeout(120_000)
  await openGoal(page)
  await continueBaseline(page).click({ timeout: 60_000 })
  await expect(continueBaseline(page)).toHaveCount(0, { timeout: 30_000 })
  writeJson('continue-baseline', { done: true })
})

test('request-changes', async ({ page }) => {
  test.setTimeout(180_000)
  const receipts = captureReceipts(page)
  await openGoal(page)
  const row = hierarchyRow(page)
  await expect(row).toBeVisible({ timeout: 60_000 })
  const feedback = process.env.LIVE_ACCEPTANCE_FEEDBACK_MESSAGE
  expect(feedback, 'Python must derive feedback from P1 assignments before this UI step').toBeTruthy()
  await choose(page, row, /request.?changes/i, feedback)
  await expect(row).toHaveCount(0, { timeout: 60_000 })
  await shot(page, 'request-changes')
  writeJson('request-changes', { feedback, answer_posts: receipts.all.filter((item) => item.kind === 'answer-post') })
  expect(receipts.postCount('answer-post')).toBe(1)
})

test('inspect-revision', async ({ page }) => {
  test.setTimeout(120_000)
  await openGoal(page)
  await page.reload()
  const evidence = page.getByRole('region', { name: 'Hierarchy evidence' }).first()
  await expect(evidence.getByText('Feedback requested', { exact: true })).toBeVisible({ timeout: 60_000 })
  await expect(evidence.getByText('Outcomes', { exact: true })).toBeVisible()
  await expect(evidence.getByText('Assignments', { exact: true })).toBeVisible()
  await expect(evidence.getByText(/Responsibility coverage/).first()).toBeVisible()
  // ProcessChain filters superseded rows, so P1 has no UI card to inspect in this build.
  const p1HistoryCheck = 'not_exercised: no P1 history card exposed by the dashboard'
  await page.setViewportSize({ width: 1440, height: 900 })
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await shot(page, 'revision-1440')
  await page.setViewportSize({ width: 390, height: 844 })
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  await shot(page, 'revision-390')
  writeJson('inspect-revision', { no_horizontal_overflow_1440: true, no_horizontal_overflow_390: true, p1_history_controls: p1HistoryCheck })
})

test('approve-hierarchy', async ({ page }) => {
  test.setTimeout(120_000)
  const receipts = captureReceipts(page)
  await openGoal(page)
  const row = hierarchyRow(page)
  await expect(row).toBeVisible({ timeout: 60_000 })
  await row.getByRole('button', { name: 'Choose differently…' }).click()
  await expect(row.getByRole('radio', { name: /^approve$/i })).toHaveCount(0)
  const approveWithGaps = row.getByRole('radio', { name: /approve.?with.?documented.?gaps/i })
  await expect(approveWithGaps).toBeVisible()
  await approveWithGaps.check()
  await row.getByRole('button', { name: /^(Submit )?answer$/i }).click()
  await expect(row).toHaveCount(0, { timeout: 60_000 })
  await shot(page, 'approved-hierarchy')
  writeJson('approve-hierarchy', {
    documented_gap_option: true, plain_approve_absent: true,
    answer_posts: receipts.all.filter((item) => item.kind === 'answer-post'),
  })
  expect(receipts.postCount('answer-post')).toBe(1)
})

test('start-planning', async ({ page }) => {
  test.setTimeout(120_000)
  await openGoal(page)
  await page.getByRole('button', { name: 'Start work' }).click({ timeout: 60_000 })
  const dialog = page.getByRole('alertdialog', { name: /Start work/i })
  const planOnly = dialog.getByRole('radio', { name: /Plan only — review and verify the candidate/i })
  await expect(planOnly).toBeChecked()
  await expect(dialog.getByText(/Implementation needs a later Start/i)).toBeVisible()
  await planOnly.check()
  await dialog.getByRole('button', { name: 'Start work', exact: true }).click()
  await expect(dialog).toHaveCount(0, { timeout: 30_000 })
  await shot(page, 'start-planning')
  writeJson('start-planning', { planning_only: true })
})

test('inspect-plan', async ({ page }) => {
  test.setTimeout(120_000)
  expect(planDecisionTitle, 'Python must provide the exact plan decision title').toBeTruthy()
  await openGoal(page)
  const row = decisionRow(page, planDecisionTitle).first()
  await expect(row).toBeVisible({ timeout: 120_000 })
  await row.getByRole('link', { name: /Plan candidate v\d+/ }).click()
  const candidate = page.getByRole('region', { name: /Plan candidate v\d+/ })
  await expect(candidate).toBeVisible({ timeout: 30_000 })
  const markdown = candidate.locator('pre')
  const content = await markdown.innerText()
  expect(content.trim()).toBeTruthy()
  expect(content).not.toMatch(/Reference unavailable|Being drafted/i)
  await expect(candidate.getByRole('button', { name: /Approve plan v\d+/ })).toBeVisible()
  await expect(candidate.getByRole('button', { name: 'Request revision' })).toBeVisible()
  await page.reload()
  const reloadedCandidate = page.getByRole('region', { name: /Plan candidate v\d+/ })
  await expect(reloadedCandidate).toBeVisible()
  await expect(reloadedCandidate.getByRole('button', { name: /Approve plan v\d+/ })).toBeVisible()
  await expect(reloadedCandidate.getByRole('button', { name: 'Request revision' })).toBeVisible()
  const reloadedContent = await reloadedCandidate.locator('pre').innerText()
  expect(reloadedContent.trim()).toBeTruthy()
  expect(reloadedContent).not.toMatch(/Reference unavailable|Being drafted/i)
  await shot(page, 'plan-after-reload')
  writeJson('inspect-plan', { decision_readable: true })
})

test('approve-plan', async ({ page }) => {
  test.setTimeout(120_000)
  const receipts = captureReceipts(page)
  expect(planDecisionTitle, 'Python must provide the exact plan decision title').toBeTruthy()
  await openGoal(page)
  const row = decisionRow(page, planDecisionTitle).first()
  await expect(row).toBeVisible({ timeout: 60_000 })
  await row.getByRole('link', { name: /Plan candidate v\d+/ }).click()
  const candidate = page.getByRole('region', { name: /Plan candidate v\d+/ })
  await expect(candidate).toBeVisible()
  await candidate.getByRole('button', { name: /Approve plan v\d+/ }).click()
  await choose(page, row, /^approve$/i)
  await expect(row).toHaveCount(0, { timeout: 60_000 })
  await shot(page, 'approved-plan')
  writeJson('approve-plan', { approved: true, answer_posts: receipts.all.filter((item) => item.kind === 'answer-post') })
  expect(receipts.postCount('answer-post')).toBe(1)
})

test('pause', async ({ page }) => {
  test.setTimeout(120_000)
  await openGoal(page)
  await pauseGoal(page).click()
  await page.getByRole('button', { name: 'Confirm pause' }).click()
  writeJson('pause', { done: true })
})
