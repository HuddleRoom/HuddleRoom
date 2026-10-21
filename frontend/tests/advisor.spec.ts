import { expect, test } from '@playwright/test'
import { gotoDashboard, openAskRail, selectSeedProject } from './support/pages/app'
import { interceptOrchestration } from './support/orchestration-fixtures'

// T7.1 — "Ask the orchestrator" rail e2e. Mocks the project chat endpoint (no
// real LLM). The rail is persistent only at >=1280px (P0.1), so this uses
// the project's default 1440x900 viewport. The conversation allowance was
// retired, so there is no remaining-allowance readout to assert any more.
test('rail Ask tab: ask a question and get a cited answer with working citation links', async ({ page }) => {
  const items: Array<Record<string, unknown>> = []

  await interceptOrchestration(page)

  await page.route('**/api/v1/projects/*/orchestration/chat', async (route) => {
    const request = route.request()
    if (request.method() === 'GET') {
      await route.fulfill({
        json: { items, total: items.length, omitted: 0, allowance: { enabled: false, unlimited: true }, goal_allowances: {}, steering: {} },
      })
      return
    }
    if (request.method() === 'POST') {
      const body = request.postDataJSON() as { content: string; client_request_id: string }
      const turn = {
        id: `turn-${items.length + 1}`,
        client_request_id: body.client_request_id,
        content: body.content,
        answer: 'Two goals are active; the release goal is blocked on independent validation.',
        citations: [
          { type: 'goal', id: 'goal-1', label: 'Goal: Ship the release' },
          { type: 'decision', id: 'decision-request-verification', goal_id: 'goal-release', label: 'Decision: Request independent validation' },
        ],
        source_references: [],
        context_goal_id: null,
        historical: false,
        goal_turn: null,
        actions: [],
        receipts: [],
        error: null,
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

  const rail = await openAskRail(page)
  const composer = rail.getByRole('textbox', { name: 'Message the orchestrator' })
  await expect(composer).toBeVisible()

  await composer.fill('What is the state of my project?')
  await composer.press('Control+Enter')

  await expect(rail.getByText('Two goals are active; the release goal is blocked on independent validation.')).toBeVisible()
  await expect(rail.getByRole('link', { name: 'Goal: Ship the release' })).toHaveAttribute('href', '/dashboard/orchestration/goal-1')

  await rail.getByRole('link', { name: 'Decision: Request independent validation' }).click()
  await expect(page).toHaveURL('/dashboard/orchestration/goal-release#decision-decision-request-verification')
  const citedDecision = page.locator('#decision-decision-request-verification')
  await expect(citedDecision).toBeVisible()
  await expect(citedDecision).toBeInViewport()
  await expect(citedDecision).toBeFocused()
})
