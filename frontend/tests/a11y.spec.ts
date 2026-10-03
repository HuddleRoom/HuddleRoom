import AxeBuilder from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'
import { ROUTES, SEED, gotoDashboard, openRoute, selectSeedProject } from './support/pages/app'
import { interceptOrchestration, openOrchestrationGoal } from './support/orchestration-fixtures'

// Serious/critical only — moderate/minor violations are tracked separately
// (design-system follow-ups), not a hard gate here.
const BLOCKING_IMPACTS = ['serious', 'critical']

// Known pre-existing violations, tracked in ISSUES.md "UI / Accessibility".
// Verified NOT introduced by Phase 10 — allow-listed by exact rule id per
// route so any OTHER serious/critical violation, or a NEW rule on these
// routes, still fails the test. Route key matches the `label` passed to
// expectNoBlockingViolations below; matching is exact-then-prefix, so
// 'goal-detail' also covers the steering-ledger+pending-feedback state.
const PRE_EXISTING_ALLOWLIST: Record<string, string[]> = {
  // pre-existing, tracked in ISSUES.md "UI / Accessibility"; markup untouched by Phase 10
  Dashboard: ['color-contrast'],
  // pre-existing, tracked in ISSUES.md "UI / Accessibility"; markup untouched by Phase 10
  Hooks: ['button-name'],
  // pre-existing, tracked in ISSUES.md "UI / Accessibility"; markup untouched by Phase 10
  Protocols: ['nested-interactive'],
  // pre-existing, tracked in ISSUES.md "UI / Accessibility"; markup untouched by Phase 10
  'meeting-detail': ['label', 'aria-required-children'],
  // pre-existing, tracked in ISSUES.md "UI / Accessibility"; markup untouched by Phase 10
  'protocol-detail': ['nested-interactive'],
  // pre-existing, tracked in ISSUES.md "UI / Accessibility"; markup untouched by Phase 10
  'goal-detail': ['definition-list', 'dlitem', 'listitem', 'color-contrast'],
}

async function expectNoBlockingViolations(page: Page, label: string) {
  const allowedRuleIds =
    PRE_EXISTING_ALLOWLIST[label] ?? Object.entries(PRE_EXISTING_ALLOWLIST).find(([key]) => label.startsWith(key))?.[1] ?? []
  const results = await new AxeBuilder({ page }).analyze()
  const blocking = results.violations
    .filter((violation) => BLOCKING_IMPACTS.includes(violation.impact ?? ''))
    .filter((violation) => !allowedRuleIds.includes(violation.id))
  const description = blocking.map((violation) => `${violation.id} (${violation.impact}): ${violation.nodes.map((node) => node.target.join(' ')).join(', ')}`).join('\n')
  expect(blocking, `${label} has serious/critical axe violations:\n${description}`).toEqual([])
}

test.describe('axe a11y — zero serious/critical violations', () => {
  test('no-project empty state', async ({ page }) => {
    await gotoDashboard(page, { expectProjectSwitcher: true })
    await expectNoBlockingViolations(page, 'no-project empty state')
  })

  for (const route of ROUTES) {
    test(`${route.navName} (${route.path})`, async ({ page }) => {
      await gotoDashboard(page)
      await selectSeedProject(page)
      if (route.path !== '/dashboard') await openRoute(page, route.navName, route.path)
      await expect(page.getByRole('heading', { name: route.heading }).first()).toBeVisible()
      await expectNoBlockingViolations(page, route.navName)
    })
  }

  test('meeting-detail', async ({ page }) => {
    await gotoDashboard(page)
    await selectSeedProject(page)
    await page.getByRole('link', { name: 'Meetings', exact: true }).click()
    await page.getByText(SEED.meetings.active).click()
    await expect(page).toHaveURL(/\/dashboard\/meetings\/[^/]+$/)
    await expect(page.getByTestId('meeting-agenda')).toBeVisible()
    await expectNoBlockingViolations(page, 'meeting-detail')
  })

  test('protocol-detail', async ({ page }) => {
    await gotoDashboard(page)
    await selectSeedProject(page)
    await page.getByRole('link', { name: 'Protocols', exact: true }).click()
    await page.getByText(SEED.protocols.active).click()
    await expect(page).toHaveURL(/\/dashboard\/protocols\/[^/]+$/)
    await expect(page.getByTestId('protocol-graph')).toBeVisible()
    await expectNoBlockingViolations(page, 'protocol-detail')
  })

  test('goal-detail', async ({ page }) => {
    await interceptOrchestration(page)
    await openOrchestrationGoal(page)
    await expect(page.locator('section[aria-label="Conversation"]')).toBeVisible()
    await expectNoBlockingViolations(page, 'goal-detail')
  })

  // The merged steering + feedback UI (ConversationPanel) only renders its
  // richest controls — feedback radios, a pending steering ledger row, the
  // steering directive textarea/selects, and the Withdraw/Replace buttons —
  // once a message has been answered and a direction submitted. Drive both
  // through the UI against the mocked orchestration harness so axe sees the
  // real DOM those controls produce, not just their resting/empty state.
  test('goal-detail with a pending feedback prompt and a steering ledger row', async ({ page }) => {
    await page.clock.install({ time: new Date('2026-09-16T09:00:00Z') })
    const harness = await interceptOrchestration(page, { steering: true })
    await openOrchestrationGoal(page)
    const conversation = page.locator('section[aria-label="Conversation"]')
    const steeringDialog = page.getByRole('dialog')

    await conversation.getByLabel('Advisory message').fill('What is the current release risk?')
    await conversation.getByRole('button', { name: 'Send message' }).click()
    await expect(conversation).toContainText('What is the current release risk?')
    harness.completeConversation('conversation-response-1')
    await page.clock.fastForward(2_000)
    const feedback = conversation.getByRole('region', { name: 'Feedback for message 1' })
    await expect(feedback.getByRole('button', { name: 'Needs work' })).toBeVisible()
    // Open the Needs-work reason group so the native radios (min-h-11 rows)
    // are actually in the DOM for axe, not just the initial two buttons.
    await feedback.getByRole('button', { name: 'Needs work' }).click()
    await expect(feedback.getByRole('radio', { name: 'Did not answer' })).toBeVisible()

    await conversation.getByRole('button', { name: 'Steer work' }).click()
    await steeringDialog.getByLabel('Steering directive').fill('Prioritize validation before other unstarted work.')
    await steeringDialog.getByLabel('Expected impact').fill('Validation runs before other unstarted work.')
    await steeringDialog.getByRole('button', { name: 'Submit steering' }).click()
    // Only confirm the a11y-relevant controls rendered — axe runs against
    // this steering + feedback UI, not against a specific status string.
    await expect(conversation.getByRole('button', { name: /Withdraw|Replace/ })).toBeVisible()
    await expect(feedback.getByRole('radio', { name: 'Did not answer' })).toBeVisible()

    await expectNoBlockingViolations(page, 'goal-detail with steering ledger + pending feedback')
  })
})
