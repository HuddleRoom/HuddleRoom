import { expect, test, type Page } from '@playwright/test'
import { SEED, gotoDashboard, selectSeedProject } from './support/pages/app'
import { interceptOrchestration, openOrchestrationGoal } from './support/orchestration-fixtures'

// Phase 11: no route may cause horizontal page scroll at any of these widths.
const WIDTHS = [1280, 1024, 768, 390] as const
const HEIGHT = 900

interface OverflowResult {
  contentX: boolean
  pastRight: number
  offenders: string[]
}

async function scanOverflow(page: Page): Promise<OverflowResult> {
  return page.evaluate(() => {
    const selector = (element: Element): string => {
      if (element.id) return `#${element.id}`
      const cls = typeof element.className === 'string' ? element.className.trim().split(/\s+/).slice(0, 2).join('.') : ''
      return cls ? `${element.tagName.toLowerCase()}.${cls}` : element.tagName.toLowerCase()
    }
    const contentX = document.scrollingElement !== null
      && document.scrollingElement.scrollWidth > window.innerWidth
    // An element inside its own intentional horizontal scroller (e.g. a
    // Kanban board or a scroll-snap stepper) legitimately extends past the
    // viewport — only the *page* scrolling is a real overflow bug. Walk up
    // from each offender; if we hit a scrollable ancestor other than
    // document.scrollingElement before reaching the root, it's contained by
    // design and not a violation.
    const hasInnerScroller = (element: Element): boolean => {
      let node: Element | null = element.parentElement
      while (node && node !== document.scrollingElement) {
        const style = window.getComputedStyle(node)
        const overflowX = style.overflowX
        if ((overflowX === 'auto' || overflowX === 'scroll') && node.scrollWidth > node.clientWidth) return true
        node = node.parentElement
      }
      return false
    }
    const offenders: string[] = []
    for (const element of Array.from(document.querySelectorAll('body *'))) {
      const rect = element.getBoundingClientRect()
      if (rect.right > window.innerWidth + 1 && !hasInnerScroller(element)) offenders.push(selector(element))
    }
    return { contentX, pastRight: offenders.length, offenders: offenders.slice(0, 5) }
  })
}

async function assertNoOverflow(page: Page, label: string, width: number) {
  await page.setViewportSize({ width, height: HEIGHT })
  const result = await scanOverflow(page)
  expect(result.contentX, `${label} @ ${width}px: page has horizontal scroll`).toBe(false)
  expect(
    result.pastRight,
    `${label} @ ${width}px: ${result.pastRight} element(s) extend past viewport, e.g. ${result.offenders.join(', ')}`,
  ).toBe(0)
}

test.describe('overflow scan — no horizontal scroll at any breakpoint', () => {
  for (const width of WIDTHS) {
    test(`dashboard @ ${width}px`, async ({ page }) => {
      await gotoDashboard(page)
      await selectSeedProject(page)
      await assertNoOverflow(page, 'dashboard', width)
    })

    test(`goal detail @ ${width}px`, async ({ page }) => {
      await interceptOrchestration(page)
      await openOrchestrationGoal(page)
      await expect(page.locator('section[aria-label="Conversation"]')).toBeVisible()
      await assertNoOverflow(page, 'goal detail', width)
    })

    test(`agents @ ${width}px`, async ({ page }) => {
      await gotoDashboard(page)
      await selectSeedProject(page)
      await page.getByRole('link', { name: 'Agents', exact: true }).click()
      await assertNoOverflow(page, 'agents', width)
    })

    test(`tasks @ ${width}px`, async ({ page }) => {
      await gotoDashboard(page)
      await selectSeedProject(page)
      await page.getByRole('link', { name: 'Tasks', exact: true }).click()
      await expect(page.getByText(SEED.tasks.ready)).toBeVisible()
      await assertNoOverflow(page, 'tasks', width)
    })
  }

  for (const width of WIDTHS) {
    test(`meeting detail @ ${width}px`, async ({ page }) => {
      await gotoDashboard(page)
      await selectSeedProject(page)
      await page.getByRole('link', { name: 'Meetings', exact: true }).click()
      await page.getByText(SEED.meetings.active).click()
      await expect(page).toHaveURL(/\/dashboard\/meetings\/[^/]+$/)
      await expect(page.getByTestId('meeting-agenda')).toBeVisible()
      await assertNoOverflow(page, 'meeting detail', width)
    })
  }

  // Goal detail with the steering ledger + compose form + feedback fieldset
  // visible (ConversationPanel's richest state) — checked at the two
  // narrowest breakpoints only, per Phase 11 scope.
  for (const width of [768, 390] as const) {
    test(`goal detail with steering ledger + feedback @ ${width}px`, async ({ page }) => {
      await page.clock.install({ time: new Date('2026-09-16T09:00:00Z') })
      const harness = await interceptOrchestration(page, { steering: true })
      await openOrchestrationGoal(page)
      const conversation = page.locator('section[aria-label="Conversation"]')

      await conversation.getByLabel('Advisory message').fill('What is the current release risk?')
      await conversation.getByRole('button', { name: 'Send message' }).click()
      await expect(conversation).toContainText('What is the current release risk?')
      harness.completeConversation('conversation-response-1')
      await page.clock.fastForward(2_000)
      const feedback = conversation.getByRole('region', { name: 'Feedback for message 1' })
      await expect(feedback.getByRole('button', { name: 'Needs work' })).toBeVisible()
      await feedback.getByRole('button', { name: 'Needs work' }).click()
      await expect(feedback.getByRole('radio', { name: 'Did not answer' })).toBeVisible()

      await conversation.getByRole('button', { name: 'Steer work' }).click()
      // The steering compose is a portal Dialog (outside the Conversation
      // section) — scope its controls to the dialog, not `conversation`.
      const steeringDialog = page.getByRole('dialog')
      await steeringDialog.getByLabel('Steering directive').fill('Prioritize validation before other unstarted work.')
      await steeringDialog.getByLabel('Expected impact').fill('Validation runs before other unstarted work.')
      await expect(steeringDialog.getByLabel('Steering directive')).toBeVisible()

      await assertNoOverflow(page, 'goal detail with steering', width)
    })
  }
})
