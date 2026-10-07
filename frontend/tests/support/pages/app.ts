import { expect, type Page } from '@playwright/test'

export interface DashboardOptions {
  expectProjectSwitcher?: boolean
}

export interface LiveCredentials {
  email: string
  password: string
}

const DASHBOARD_SUMMARY_HEADING = /Project activity|System steady/i
const DASHBOARD_PRIMARY_ACTION = /View ready tasks|View meetings|Review agents/i

export const SEED = {
  projectName: 'HuddleRoom UI Live Project',
  agents: {
    architect: 'huddleroom-ui-architect',
    reviewer: 'huddleroom-ui-reviewer',
    inactive: 'huddleroom-ui-inactive',
  },
  tasks: {
    ready: 'UI seed ready task',
    backlog: 'UI seed backlog task',
    blocked: 'UI seed blocked task',
    done: 'UI seed done task',
    disposable: 'UI disposable task',
  },
  meetings: {
    active: 'UI seed active meeting',
    concluded: 'UI seed concluded meeting',
  },
  graphs: {
    active: 'ui_seed_graph',
    inactive: 'ui_inactive_graph',
  },
  knowledge: {
    title: 'UI seed rollout policy',
    disposableTitle: 'UI disposable knowledge',
  },
  memory: {
    content: 'Remember the UI rollout guardrail',
  },
  rule: 'UI seed routing rule',
  hooks: {
    active: 'ui_seed_hook',
    disposable: 'ui_disabled_hook',
  },
  optimization: 'UI seed optimization',
} as const

export const ROUTES = [
  { path: '/dashboard', navName: 'Dashboard', heading: /Project activity|System steady|Choose a project/i },
  {
    path: '/dashboard/orchestration',
    navName: 'Orchestration',
    heading: /Orchestration/i,
  },
  { path: '/dashboard/tasks', navName: 'Tasks', heading: /Tasks|Task board/i },
  { path: '/dashboard/agents', navName: 'Agents', heading: /Agents/i },
  { path: '/dashboard/graphs', navName: 'Graphs', heading: /Graphs/i },
  { path: '/dashboard/meetings', navName: 'Meetings', heading: /Meetings/i },
  { path: '/dashboard/knowledge', navName: 'Knowledge', heading: /Knowledge/i },
  { path: '/dashboard/memory', navName: 'Memory', heading: /Memory/i },
  { path: '/dashboard/rules', navName: 'Rules', heading: /Rules/i },
  { path: '/dashboard/hooks', navName: 'Hooks', heading: /Hooks/i },
  { path: '/dashboard/optimizations', navName: 'Optimizations', heading: /Optimizations/i },
  { path: '/dashboard/settings', navName: 'Settings', heading: /Settings/i },
] as const

export async function gotoDashboard(page: Page, options: DashboardOptions = {}): Promise<void> {
  const { expectProjectSwitcher = true } = options
  await page.goto('/dashboard')
  if (expectProjectSwitcher) {
    await expect(page.getByRole('combobox', { name: 'Project switcher' })).toBeVisible()
  }
}

export async function selectSeedProject(page: Page): Promise<void> {
  const projectSwitcher = page.getByRole('combobox', { name: 'Project switcher' })
  await expect(projectSwitcher).toBeVisible()
  await projectSwitcher.selectOption({ label: SEED.projectName })
  await expect(projectSwitcher).toHaveValue(/.+/)
  await expectDashboardReady(page)
}

export async function openRoute(page: Page, navName: string, expectedPath: string): Promise<void> {
  // exact: true — sidebar nav links share a name prefix with Dashboard's StatTile
  // quick-action links (e.g. "Meetings" nav vs "Meetings 3" StatTile), which
  // otherwise collide under Playwright's default substring match.
  await page.getByRole('link', { name: navName, exact: true }).click()
  await expect(page).toHaveURL(new RegExp(`${expectedPath}/?$`))
}

export async function login(page: Page, credentials: LiveCredentials): Promise<void> {
  await page.goto('/dashboard/login')
  await page.getByLabel(/email/i).fill(credentials.email)
  await page.getByLabel(/password/i).fill(credentials.password)
  await page.getByRole('button', { name: /log in|sign in/i }).click()
}

export async function expectRouteReady(page: Page, label: string): Promise<void> {
  await expect(page.locator('body')).not.toContainText(/^Loading$/i)
  await expect(page.getByRole('link', { name: label }).or(page.getByText(label).first())).toBeVisible()
}

export async function expectDashboardReady(page: Page): Promise<void> {
  await expect(page.locator('body')).not.toContainText(/^Loading$/i)
  await expect(page.getByRole('heading', { name: DASHBOARD_SUMMARY_HEADING })).toBeVisible()
  await expect(page.getByRole('button', { name: DASHBOARD_PRIMARY_ACTION })).toBeVisible()
}

export async function closeDialog(page: Page, name: RegExp | string): Promise<void> {
  const dialog = page.getByRole('dialog', { name })
  await expect(dialog).toBeVisible()
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
}
