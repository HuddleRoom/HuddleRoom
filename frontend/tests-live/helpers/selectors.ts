import type { Locator, Page } from '@playwright/test'

export const pauseGoal = (p: Page) => p.getByRole('button', { name: 'Pause goal' })
export const confirmPause = (p: Page) => p.getByRole('button', { name: 'Confirm pause' })
export const resumeGoal = (p: Page) => p.getByRole('button', { name: 'Resume goal' })
export const continueBaseline = (p: Page) => p.getByRole('button', { name: /^(Continue baseline|Authorize (remaining )?baseline)$/ })
export const startBaseline = (p: Page) => p.getByRole('button', { name: 'Start baseline' })
export const savedAnswers = (p: Page) => p.getByRole('region', { name: 'Saved answers' })
export const needsYouHeading = (p: Page) => p.getByRole('heading', { name: 'Needs you' })
/** Live status regions (role=status); pass name to narrow by accessible name. */
export const statusRegions = (scope: Locator | Page, name?: string | RegExp) =>
  name === undefined ? scope.getByRole('status') : scope.getByRole('status', { name })
/**
 * The goal status badge is a <span>● {label}</span> (the dot is an aria-hidden child), so getByText with
 * exact never matches; it is matched by its full text instead (e.g. 'Paused'). Scope it to the goal
 * header to keep strict mode happy.
 */
export const goalStatusBadge = (scope: Locator | Page, label: string) =>
  scope.locator('span', { hasText: new RegExp(`^\\s*●?\\s*${label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*$`) })
/**
 * A Needs you queue row: a nameless <li> inside the 'Needs you' region (NeedsYouQueue.tsx), so it
 * is matched by the decision title text it contains.
 */
export const decisionRow = (p: Page, title: string | RegExp) =>
  p.getByRole('region', { name: 'Needs you' }).getByRole('listitem').filter({ hasText: title })
export const approveIn = (row: Locator) => row.getByRole('button', { name: 'Approve', exact: true })
export const rejectIn = (row: Locator) => row.getByRole('button', { name: 'Reject', exact: true })
