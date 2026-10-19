import { expect, test } from '@playwright/test'

for (const viewport of [{ name: 'desktop', width: 1440, height: 900 }, { name: 'narrow', width: 390, height: 844 }]) {
  test(`retained G4 real Step at ${viewport.name}`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport)
    const errors: string[] = []
    page.on('pageerror', (error) => errors.push(error.message))
    let unavailable = false
    await page.routeWebSocket('**/ws/**', () => { throw new Error('Unexpected WebSocket') })
    await page.route('**/api/v1/**', (route) => {
      const request = route.request()
      if (request.method() !== 'GET' || !new URL(request.url()).pathname.endsWith('/agents')) throw new Error(`Unexpected API ${request.method()} ${request.url()}`)
      if (unavailable) return route.fulfill({ status: 503, json: { detail: 'Current unavailable' } })
      return route.fulfill({ json: { items: Array.from({ length: 5 }, (_, i) => ({
        id: `agent-${i}`, name: `Current agent ${i + 1}`, role: 'reviewer', provider: 'current-provider', model: i === 4 ? '' : `current-model-${i + 1}`,
        adapter_type: 'cli', cli_runtime: 'codex', capabilities: ['review'], config: {}, is_active: true,
      })), next_cursor: null } })
    })
    await page.goto('/dashboard/tests/support/g4-fixture.html?state=historical')
    await expect(page.getByText(/Superseded/)).toBeVisible()
    await expect(page.getByText('Current agent 1 · reviewer')).toBeVisible()
    await expect(page.getByText('Unknown / provider default', { exact: false })).toBeVisible()
    await expect(page.getByText('Reviewed configuration', { exact: true })).toHaveCount(5)
    await expect(page.getByRole('button', { name: /Run|Retry|Skip|submit/i })).toHaveCount(0)
    await expect(page.getByText('Captured input: Recorded.', { exact: false })).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    await page.screenshot({ path: testInfo.outputPath(`g4-${viewport.name}-historical-current.png`), fullPage: true })
    const details = page.locator('summary').filter({ hasText: 'Raw data' })
    await details.focus()
    await expect(details).toBeFocused()
    await page.keyboard.press('Enter')
    await expect(details.locator('..')).toHaveAttribute('open', '')
    await page.screenshot({ path: testInfo.outputPath(`g4-${viewport.name}-keyboard-raw.png`), fullPage: true })
    await page.keyboard.press('Enter')
    unavailable = true
    await page.reload()
    await expect(page.getByText('Current settings unavailable', { exact: true })).toHaveCount(5)
    await expect(page.getByText('historic-provider / historic-model-1')).toBeVisible()
    await expect(page.getByText('Current agent 1 · reviewer')).toHaveCount(0)
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    await page.screenshot({ path: testInfo.outputPath(`g4-${viewport.name}-current-unavailable.png`), fullPage: true })
    expect(errors).toEqual([])
  })
}
