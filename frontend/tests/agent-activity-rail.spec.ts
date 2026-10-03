import { expect, test } from '@playwright/test'
import { gotoDashboard, selectSeedProject } from './support/pages/app'

test('activity strip is visible below 1280px while the persistent rail is hidden', async ({ page }) => {
  await page.setViewportSize({ width: 1024, height: 800 })
  await gotoDashboard(page)
  await selectSeedProject(page)

  await expect(page.getByRole('button', { name: 'Activity' })).toBeVisible()
  await expect(page.locator('.agent-activity-panel-container')).not.toBeVisible()
})

test('agent activity panel is always visible at 1280px+, even with zero calls', async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 })
  await gotoDashboard(page)
  await selectSeedProject(page)

  await expect(page.locator('.agent-activity-panel-container')).toBeVisible()
  await expect(page.getByRole('tab', { name: 'Ask', selected: true })).toBeVisible()
  const callsTab = page.getByRole('tab', { name: 'Calls' })
  await expect(callsTab).toBeVisible()
  await callsTab.click()
  await expect(page.getByText('No activity yet.')).toBeVisible()
})
