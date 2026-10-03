import { expect, test } from '@playwright/test'
import { SEED, gotoDashboard, selectSeedProject } from './support/pages/app'

test.beforeEach(async ({ page }) => {
  await gotoDashboard(page)
  await selectSeedProject(page)
})

test('tasks board supports filters, detail panel, create validation, and create flow', async ({ page }) => {
  await page.getByRole('link', { name: 'Tasks', exact: true }).click()
  await expect(page.getByText(SEED.tasks.ready)).toBeVisible()

  await page.getByRole('combobox', { name: 'Filter tasks by status' }).selectOption('ready')
  await expect(page.getByText(SEED.tasks.ready)).toBeVisible()
  await expect(page.getByText(SEED.tasks.backlog)).toBeHidden()

  await page.getByRole('combobox', { name: 'Filter tasks by status' }).selectOption('')
  await page.getByText(SEED.tasks.ready).click()
  await expect(page.getByTestId('task-detail-panel')).toContainText(SEED.tasks.ready)
  await expect(page.getByTestId('task-run-control')).toBeVisible()

  await page.getByRole('button', { name: 'New task' }).click()
  const createTaskDialog = page.getByRole('dialog', { name: /new task/i })
  await expect(createTaskDialog).toBeVisible()
  await expect(createTaskDialog.getByRole('button', { name: 'Create task' })).toBeDisabled()

  const title = `UI created task ${Date.now()}`
  await page.getByPlaceholder('task title').fill(title)
  await page.getByPlaceholder('optional description').fill('Created by live Playwright')
  await createTaskDialog.getByRole('button', { name: 'Create task' }).click()
  await expect(page.getByRole('dialog', { name: /new task/i })).toBeHidden()
  await expect(page.getByText(title)).toBeVisible()
})

test('tasks run control is disabled for completed tasks and available for ready tasks', async ({ page }) => {
  await page.getByRole('link', { name: 'Tasks', exact: true }).click()
  await page.getByText(SEED.tasks.done).click()
  await expect(page.getByTestId('task-run-control')).toBeDisabled()

  await page.getByText(SEED.tasks.ready).click()
  await expect(page.getByTestId('task-run-control')).toBeEnabled()
})

test('agents support create validation, edit dialog, detail route, and detail tabs', async ({ page }) => {
  await page.getByRole('link', { name: 'Agents' }).click()
  await expect(page.getByText(SEED.agents.architect)).toBeVisible()
  await page.getByRole('button', { name: /show inactive/i }).click()
  await expect(page.getByText(SEED.agents.inactive)).toBeVisible()

  await page.getByRole('button', { name: /new agent/i }).click()
  const newAgentDialog = page.getByRole('dialog', { name: /new agent/i })
  await expect(newAgentDialog).toBeVisible()
  await expect(newAgentDialog.getByRole('button', { name: 'Create agent' })).toBeDisabled()
  await page.keyboard.press('Escape')

  await page.getByRole('button', { name: `Edit agent ${SEED.agents.architect}` }).click()
  await expect(page.getByRole('dialog', { name: /edit agent/i })).toBeVisible()
  await expect(page.getByLabel('Name *')).toHaveValue(SEED.agents.architect)
  await page.keyboard.press('Escape')

  await page.getByText(SEED.agents.architect).click()
  await expect(page).toHaveURL(/\/dashboard\/agents\/[^/]+$/)
  await expect(page.getByRole('tab', { name: /^Sessions/ })).toBeVisible()
  await page.getByRole('tab', { name: /^Tasks/ }).click()
  await expect(page.getByText(SEED.tasks.ready)).toBeVisible()
})
