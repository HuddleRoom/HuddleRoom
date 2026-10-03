import { expect, test } from '@playwright/test'
import { gotoDashboard, selectSeedProject } from './support/pages/app'
import { interceptOrchestration } from './support/orchestration-fixtures'

// T7.1 — "Ask the orchestrator" rail e2e. Mocks the advisor endpoints (no
// real LLM). The rail is persistent only at >=1280px (P0.1), so this uses
// the project's default 1440x900 viewport.
test('rail Ask tab: ask a question, get a cited answer, allowance decrements', async ({ page }) => {
  const limit = 1000
  let remaining = limit
  const items: Array<{
    id: string
    question: string
    answer: string | null
    citations: Array<{ type: string; id: string; label: string; goal_id?: string }>
    off_topic: boolean
    status: string
    created_at: string
  }> = []

  await interceptOrchestration(page)

  await page.route('**/api/v1/projects/*/orchestration/conversation', async (route) => {
    const request = route.request()
    if (request.method() === 'GET') {
      await route.fulfill({
        json: { items, allowance: { enabled: true, unlimited: false, limit, remaining } },
      })
      return
    }
    if (request.method() === 'POST') {
      const body = request.postDataJSON() as { content: string }
      remaining -= 1
      const turn = {
        id: `turn-${items.length + 1}`,
        question: body.content,
        answer: 'Two goals are active; the release goal is blocked on independent validation.',
        citations: [
          { type: 'goal', id: 'goal-1', label: 'Goal: Ship the release' },
          { type: 'decision', id: 'decision-request-verification', goal_id: 'goal-release', label: 'Decision: Request independent validation' },
        ],
        off_topic: false,
        status: 'completed',
        created_at: '2026-09-18T08:00:00Z',
      }
      items.push(turn)
      await route.fulfill({ json: turn })
      return
    }
    await route.fallback()
  })

  await gotoDashboard(page)
  await selectSeedProject(page)

  const rail = page.locator('.agent-activity-panel-container')
  await expect(rail.getByRole('tab', { name: 'Ask' })).toBeVisible()
  await rail.getByRole('tab', { name: 'Ask' }).click()

  const composer = rail.getByRole('textbox', { name: 'Ask the orchestrator' })
  await expect(composer).toBeVisible()
  await expect(rail.getByText('Allowance: 1000 remaining of 1000')).toBeVisible()

  await composer.fill('What is the state of my project?')
  await composer.press('Control+Enter')

  await expect(rail.getByText('Two goals are active; the release goal is blocked on independent validation.')).toBeVisible()
  await expect(rail.getByRole('link', { name: 'Goal: Ship the release' })).toHaveAttribute('href', '/dashboard/orchestration/goal-1')
  await expect(rail.getByText('Allowance: 999 remaining of 1000')).toBeVisible()

  await rail.getByRole('link', { name: 'Decision: Request independent validation' }).click()
  await expect(page).toHaveURL('/dashboard/orchestration/goal-release#decision-decision-request-verification')
  const citedDecision = page.locator('#decision-decision-request-verification')
  await expect(citedDecision).toBeVisible()
  await expect(citedDecision).toBeInViewport()
  await expect(citedDecision).toBeFocused()
})
