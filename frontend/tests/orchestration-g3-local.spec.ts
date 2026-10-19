import { expect, test } from '@playwright/test'

for (const viewport of [{ name: 'desktop', width: 1440, height: 900 }, { name: 'narrow', width: 390, height: 844 }]) {
  test(`renders the provider-free G3 review fixture at ${viewport.name}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport)
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    await page.route('**/api/v1/**', (route) => { throw new Error(`Unexpected API mutation or request: ${route.request().method()} ${route.request().url()}`) })
    await page.route('**/api/v1/projects/project-1/orchestration/goals/goal-1', (route) => route.fulfill({ json: { goal: { objective: 'Deliver review' } } }))
    await page.route('**/api/v1/projects/project-1/orchestration/chat', (route) => route.fulfill({ json: { items: [], total: 0, omitted: 0, allowance: { enabled: true, unlimited: false, limit: 1, remaining: 1 }, goal_allowances: {}, steering: {} } }))
    await page.goto('/dashboard/tests/support/g3-react-fixture.html')
    await expect(page.getByText('v3 · Current')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Plan candidate v4' })).toBeVisible()
    await expect(page.locator('[title="provenance-4"]')).toBeVisible()
    await expect(page.getByRole('heading', { name: 'Conversation with the orchestrator' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Approve plan v4' })).toBeEnabled()
    await page.getByRole('link', { name: 'Link to this candidate' }).focus()
    await expect(page.getByRole('link', { name: 'Link to this candidate' })).toBeFocused()
    await page.keyboard.press('Tab')
    await expect(page.getByRole('button', { name: 'Approve plan v4' })).toBeFocused()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    await page.screenshot({ path: testInfo.outputPath(`g3-${viewport.name}-ready.png`), fullPage: true })

    await page.goto('/dashboard/tests/support/g3-react-fixture.html?state=stale')
    await expect(page.getByText('This candidate remains readable for revision, but it cannot be approved or executed.')).toBeVisible()
    await expect(page.getByRole('button', { name: 'Approve plan v4' })).toBeDisabled()
    await expect(page.getByRole('button', { name: 'Request revision' })).toBeEnabled()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    await page.screenshot({ path: testInfo.outputPath(`g3-${viewport.name}-stale.png`), fullPage: true })
    expect(errors).toEqual([])
  })
}
