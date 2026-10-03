import { expect, test, type Page } from '@playwright/test'
import { CONTROL_MAP, type ControlExpectation } from './support/control-map'
import { gotoDashboard, selectSeedProject } from './support/pages/app'

async function expectControl(page: Page, control: ControlExpectation): Promise<void> {
  if (control.kind === 'role') {
    const locator = page.getByRole(control.role, { name: control.name })
    if (typeof control.name === 'string') {
      await expect(locator).toHaveCount(1)
    }
    await expect(locator.first()).toBeVisible()
    if (control.enabled === false) {
      await expect(locator.first()).toBeDisabled()
    } else if (control.enabled === true) {
      await expect(locator.first()).toBeEnabled()
    }
    return
  }

  if (control.kind === 'testId') {
    const locator = page.getByTestId(control.testId)
    await expect(locator).toHaveCount(1)
    await expect(locator).toBeVisible()
    return
  }

  await expect(page.getByText(control.text).first()).toBeVisible()
}

test.beforeEach(async ({ page }) => {
  await gotoDashboard(page)
  await selectSeedProject(page)
})

for (const route of CONTROL_MAP) {
  test(`${route.label} exposes its expected meaningful controls`, async ({ page }) => {
    await page.goto(route.path)
    await expect(page.locator('body')).not.toContainText(/^Loading$/i)

    for (const control of route.controls) {
      await expectControl(page, control)
    }
  })
}
