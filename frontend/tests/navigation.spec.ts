import { expect, test } from '@playwright/test'
import { ROUTES, expectDashboardReady, gotoDashboard, openRoute, selectSeedProject } from './support/pages/app'

test.beforeEach(async ({ page }) => {
  await gotoDashboard(page)
  await selectSeedProject(page)
})

test('main navigation reaches every primary dashboard route', async ({ page }) => {
  for (const route of ROUTES) {
    await openRoute(page, route.navName, route.path)
    await expect(page.locator('body')).not.toContainText(/^Loading$/i)
  }
})

test('dashboard quick actions navigate to work areas', async ({ page }) => {
  await page.goto('/dashboard')
  await selectSeedProject(page)
  await expectDashboardReady(page)

  await page.getByRole('button', { name: 'View ready tasks' }).click()
  await expect(page).toHaveURL(/\/dashboard\/tasks\?status=ready$/)

  await page.goto('/dashboard')
  await selectSeedProject(page)
  await expectDashboardReady(page)
  // \s required after the label — the bare sidebar nav link ("Meetings") would
  // otherwise also match and make this locator ambiguous for .click().
  await page.getByRole('link', { name: /^Meetings\s/ }).click()
  await expect(page).toHaveURL(/\/dashboard\/meetings\/?$/)

  await page.goto('/dashboard')
  await selectSeedProject(page)
  await expectDashboardReady(page)
  await page.getByRole('link', { name: /^Sessions\s/ }).click()
  await expect(page).toHaveURL(/\/dashboard\/agents\/?$/)
})

test('sidebar collapse and expand controls remain usable', async ({ page }) => {
  await page.getByRole('button', { name: 'Collapse sidebar' }).click()
  await expect(page.getByRole('button', { name: 'Expand sidebar' })).toBeVisible()
  await page.getByRole('button', { name: 'Expand sidebar' }).click()
  await expect(page.getByRole('button', { name: 'Collapse sidebar' })).toBeVisible()
})
