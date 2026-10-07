import { expect, test } from '@playwright/test'
import { SEED, gotoDashboard, selectSeedProject } from './support/pages/app'

test.beforeEach(async ({ page }) => {
  await gotoDashboard(page)
  await selectSeedProject(page)
})

test('meetings list filters and detail surfaces render seeded live data', async ({ page }) => {
  await page.getByRole('link', { name: 'Meetings', exact: true }).click()
  await expect(page.getByText(SEED.meetings.active)).toBeVisible()
  await expect(page.getByText(SEED.meetings.concluded)).toBeVisible()

  await page.getByRole('radio', { name: 'Active' }).click()
  await expect(page.getByText(SEED.meetings.active)).toBeVisible()
  await expect(page.getByText(SEED.meetings.concluded)).toBeHidden()

  await page.getByText(SEED.meetings.active).click()
  await expect(page).toHaveURL(/\/dashboard\/meetings\/[^/]+$/)
  await expect(page.getByTestId('meeting-agenda')).toContainText('Decide UI rollout')
  await expect(page.getByTestId('meeting-transcript')).toContainText('deterministic')
  await expect(page.getByTestId('meeting-human-input')).toBeVisible()

  await page.getByText(/Action items/i).click()
  await expect(page.getByText('Verify live Playwright artifacts')).toBeVisible()
})

test('new meeting page validates required title and participants', async ({ page }) => {
  await page.getByRole('link', { name: 'Meetings', exact: true }).click()
  await page.getByRole('button', { name: /new meeting/i }).click()
  await expect(page).toHaveURL(/\/meetings\/new$/)
  await expect(page.getByRole('heading', { name: 'New meeting' })).toBeVisible()
  await expect(page.getByRole('button', { name: /create meeting/i })).toBeDisabled()
  await page.getByPlaceholder('meeting title').fill(`UI created meeting ${Date.now()}`)
  // Title alone is not enough — at least one participant is also required.
  await expect(page.getByRole('button', { name: /create meeting/i })).toBeDisabled()
  await page.getByRole('button', { name: 'Cancel' }).click()
  await expect(page).toHaveURL(/\/meetings$/)
})

test('graphs support search, inactive toggle, detail view, definition editor, and run filters', async ({ page }) => {
  await page.getByRole('link', { name: 'Graphs', exact: true }).click()
  await expect(page.getByText(SEED.graphs.active)).toBeVisible()

  await page.getByRole('textbox', { name: 'Search graphs' }).fill(SEED.graphs.active)
  await expect(page.getByText(SEED.graphs.active)).toBeVisible()
  await expect(page.getByText(SEED.graphs.inactive)).toBeHidden()

  await page.getByRole('textbox', { name: 'Search graphs' }).fill('')
  await page.getByLabel(/include inactive/i).check()
  await expect(page.getByText(SEED.graphs.inactive)).toBeVisible()

  await page.getByText(SEED.graphs.active).click()
  await expect(page).toHaveURL(/\/dashboard\/graphs\/[^/]+$/)
  await expect(page.getByTestId('graph-diagram')).toBeVisible()
  await expect(page.getByTestId('graph-definition-editor')).toBeVisible()

  await page.getByRole('combobox', { name: 'Filter graph runs by status' }).selectOption('active')
  await expect(page.getByTestId('graph-run-list')).toContainText('active')
  await expect(page.getByRole('button', { name: /Pause graph run/ }).first()).toBeVisible()
})
