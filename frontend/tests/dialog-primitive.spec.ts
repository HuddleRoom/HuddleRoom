import { expect, test } from '@playwright/test'
import { gotoDashboard, selectSeedProject } from './support/pages/app'

// Phase 13: the shared Dialog primitive's contract, proved on a couple of
// representative dialogs rather than every adopter page.
test.describe('Dialog primitive contract', () => {
  test.beforeEach(async ({ page }) => {
    await gotoDashboard(page)
    await selectSeedProject(page)
  })

  test('New task dialog: sentence-case title, initial focus on first field, Escape closes', async ({ page }) => {
    await page.getByRole('link', { name: 'Tasks', exact: true }).click()
    await page.getByRole('button', { name: 'New task' }).click()

    const dialog = page.getByRole('dialog', { name: 'New task' })
    await expect(dialog).toBeVisible()
    await expect(dialog.getByRole('heading', { name: 'New task' })).toBeVisible()

    await expect(page.getByPlaceholder('task title')).toBeFocused()

    await page.keyboard.press('Escape')
    await expect(dialog).toBeHidden()
  })

  test('New project dialog: sentence-case title, initial focus on first field, Escape closes', async ({ page }) => {
    await page.getByRole('button', { name: 'New project' }).click()

    const dialog = page.getByRole('dialog', { name: 'New project' })
    await expect(dialog).toBeVisible()
    await expect(dialog.getByRole('heading', { name: 'New project' })).toBeVisible()

    await expect(page.getByLabel('Name *')).toBeFocused()

    await page.keyboard.press('Escape')
    await expect(dialog).toBeHidden()
  })

  test('no-project empty state has no dead "Open project switcher" button', async ({ page }) => {
    await gotoDashboard(page)
    await expect(page.getByRole('button', { name: 'New project' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Open project switcher' })).toHaveCount(0)
  })
})
