import { expect, test } from '@playwright/test'
import { SEED, gotoDashboard, selectSeedProject } from './support/pages/app'

test.beforeEach(async ({ page }) => {
  await gotoDashboard(page)
  await selectSeedProject(page)
})

test('knowledge supports search, detail, create validation, and disposable delete confirmation', async ({ page }) => {
  await page.getByRole('link', { name: 'Knowledge' }).click()
  await expect(page.getByText(SEED.knowledge.title)).toBeVisible()

  await page.getByRole('textbox', { name: 'Search knowledge' }).fill('rollout')
  await page.getByRole('button', { name: 'Search' }).click()
  await expect(page.getByText(SEED.knowledge.title)).toBeVisible()

  await page.getByRole('button', { name: new RegExp(`View knowledge ${SEED.knowledge.title}`) }).click()
  await expect(page.getByTestId('knowledge-detail')).toContainText('Rollout policy')
  await page.getByTestId('knowledge-detail').getByRole('button', { name: 'Close' }).click()

  await page.getByRole('button', { name: 'New item' }).click()
  await expect(page.getByRole('dialog', { name: 'New Knowledge Item' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Create' })).toBeVisible()
  await page.keyboard.press('Escape')

  await page.getByRole('textbox', { name: 'Search knowledge' }).fill('')
  await page.getByRole('button', { name: 'Search' }).click()
  const knowledgeRow = page.locator('[data-testid^="knowledge-item-"]').filter({ hasText: SEED.knowledge.disposableTitle }).first()
  const deleteButton = knowledgeRow.getByRole('button', { name: new RegExp(`Delete knowledge ${SEED.knowledge.disposableTitle}`) })
  await deleteButton.click()
  await expect(page.getByRole('alertdialog')).toBeVisible()
  await expect(page.getByRole('alertdialog')).toContainText('Delete this item?')
  await expect(page.getByRole('alertdialog').getByRole('button', { name: 'Delete' })).toBeVisible()
})

test('memory supports project and global views, shared filter, search, and delete confirmation', async ({ page }) => {
  await page.getByRole('link', { name: 'Memory' }).click()
  await expect(page.getByText(SEED.memory.content)).toBeVisible()

  await page.getByRole('checkbox', { name: 'Show shared memory only' }).check()
  await expect(page.getByText(SEED.memory.content)).toBeVisible()

  await page.getByPlaceholder('Search memories...').fill('guardrail')
  await page.getByRole('button', { name: 'Search' }).click()
  await expect(page.getByText(SEED.memory.content)).toBeVisible()
  const memoryCard = page.locator('[data-testid^="memory-item-"]').filter({ hasText: SEED.memory.content }).first()
  await memoryCard.getByRole('button', { name: 'Delete' }).click()
  await expect(page.getByRole('alertdialog')).toBeVisible()
  await expect(page.getByRole('alertdialog')).toContainText('Delete this memory?')
  await expect(page.getByRole('alertdialog').getByRole('button', { name: 'Delete' })).toBeVisible()

  await page.keyboard.press('Escape')
  await expect(page.getByRole('alertdialog')).not.toBeVisible()

  await page.getByRole('tab', { name: 'Global' }).click()
  await expect(page.getByText('Global reviewer memory for UI tests')).toBeVisible()
})

test('rules and hooks expose row controls and create and edit affordances', async ({ page }) => {
  await page.getByRole('link', { name: 'Rules' }).click()
  await expect(page.getByText(SEED.rule)).toBeVisible()
  await page.getByRole('button', { name: new RegExp(`Edit rule: ${SEED.rule}`) }).click()
  const editRuleDialog = page.getByRole('dialog', { name: 'Edit Rule' })
  await expect(editRuleDialog).toBeVisible()
  await expect(editRuleDialog.getByRole('button', { name: 'Update' })).toBeVisible()
  await page.keyboard.press('Escape')

  await page.getByRole('link', { name: 'Hooks' }).click()
  await expect(page.getByText(SEED.hooks.active)).toBeVisible()
  await page.getByRole('button', { name: `Re-propose hook ${SEED.hooks.disposable}` }).click()
  await expect(page.getByText(SEED.hooks.disposable)).toBeVisible()
  await page.getByRole('button', { name: 'New Hook' }).click()
  const newHookDialog = page.getByRole('dialog', { name: 'New hook' })
  await expect(newHookDialog).toBeVisible()
  await expect(newHookDialog.getByRole('button', { name: 'Create' })).toBeVisible()
  await page.keyboard.press('Escape')
})

test('optimizations and settings expose meaningful controls without mutating local data', async ({ page }) => {
  await page.getByRole('link', { name: 'Optimizations' }).click()
  await expect(page.getByTestId(/optimization-card-/).first()).toBeVisible()
  await expect(page.getByTestId('optimization-generated-code').first()).toBeVisible()
  await expect(page.getByRole('button', { name: /Submit for Approval|Approve|Reject|Activate/ }).first()).toBeVisible()
  await expect(page.getByRole('button', { name: /Delete optimization/ }).first()).toBeVisible()

  await page.getByRole('link', { name: 'Settings' }).click()
  await expect(page.getByTestId('new-api-key-button')).toBeVisible()
  await page.getByTestId('new-api-key-button').click()
  const createKeyDialog = page.getByRole('dialog', { name: 'New API key' })
  await expect(createKeyDialog).toBeVisible()
  await expect(createKeyDialog.getByRole('button', { name: 'Create key' })).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(page.getByTestId('settings-save-config')).toBeVisible()
})
