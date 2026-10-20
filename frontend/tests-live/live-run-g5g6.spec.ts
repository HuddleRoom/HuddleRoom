import { mkdirSync, unlinkSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { expect, test, type Page } from '@playwright/test'
import { failWith500 } from './helpers/faults'
import {
  AUTH_BTN, BAR, controls, fetchWorkflow, layoutProbe, pendingQuestion, safe, statusFacts, type Errs,
} from './helpers/g5g6'

// G5/G6 live steps, driven one at a time by the Python harness (--grep "(^|\s)<name>$").
// Steps are observers: each always writes <step>.json (even on failure) and Python judges it.
const projectId = process.env.LIVE_ACCEPTANCE_PROJECT_ID ?? ''
const goalId = process.env.LIVE_ACCEPTANCE_GOAL_ID ?? ''
const out = process.env.LIVE_ACCEPTANCE_OUTPUT || 'test-results-live'

async function openGoal(page: Page) {
  expect(projectId && goalId, 'LIVE_ACCEPTANCE_PROJECT_ID and LIVE_ACCEPTANCE_GOAL_ID are required').toBeTruthy()
  await page.addInitScript((id) => {
    localStorage.setItem('huddleroom-ui', JSON.stringify({ state: { activeProjectId: id, sidebarCollapsed: false }, version: 1 }))
  }, projectId)
  await page.goto(`/dashboard/orchestration/${goalId}`)
  await page.waitForLoadState('networkidle').catch(() => {})
  // Real breakage if the goal page never rendered.
  await page.getByRole('heading', { name: 'Current operation' }).first().waitFor({ timeout: 60_000 })
}

function writeJson(name: string, data: unknown) {
  mkdirSync(out, { recursive: true })
  writeFileSync(join(out, `${name}.json`), JSON.stringify(data, null, 2))
}

const shot = (page: Page, name: string, fullPage = false) => page.screenshot({ path: join(out, `${name}.png`), fullPage })

/** Run a step body, then always write <name>.json with workflow + ui_errors. */
async function observe(page: Page, name: string, body: (rec: Record<string, any>, errs: Errs) => Promise<void>) {
  const errs: Errs = []
  const rec: Record<string, any> = {}
  try {
    await body(rec, errs)
  } finally {
    rec.workflow = await fetchWorkflow(page, errs)
    rec.ui_errors = errs
    writeJson(name, rec)
  }
}

const callsTab = (page: Page) => page.getByRole('tab', { name: /^Calls/ })
const askTab = (page: Page) => page.getByRole('tab', { name: /^Ask/ })
const selectCalls = async (page: Page, errs: Errs) => {
  await safe(errs, 'calls tab', async () => { await callsTab(page).click({ timeout: 30_000 }) })
}
const connectivityText = (page: Page, errs: Errs) => safe(errs, 'connectivity', () =>
  page.locator('header').getByText(/^(Live connection|Connecting|Connection paused|Off|No project)$/).first().innerText({ timeout: 10_000 }))
const callIds = (page: Page) => page.evaluate(() => Array.from(document.querySelectorAll('[data-call-id]')).map((e) => e.getAttribute('data-call-id')!))
// Status line + controls + facts, the common core of the status-* steps.
async function core(page: Page, rec: Record<string, any>, errs: Errs) {
  rec.facts = await statusFacts(page, errs)
  rec.controls = await controls(page, errs)
}

test('authorize-baseline', async ({ page }) => {
  test.setTimeout(180_000)
  await observe(page, 'authorize-baseline', async (rec, errs) => {
    await openGoal(page)
    const btn = page.getByRole('button', { name: AUTH_BTN })
    rec.workflow_guidance_before = (await fetchWorkflow(page, errs))?.guidance ?? null
    rec.facts = await statusFacts(page, errs)
    rec.controls = await controls(page, errs)
    rec.authorize_button_name = await safe(errs, 'authorize name', async () => (await btn.first().innerText({ timeout: 15_000 })).trim())
    await shot(page, 'authorize-baseline-before')
    if (process.env.LIVE_ACCEPTANCE_CLICK === '1') {
      const posts: string[] = []
      page.on('request', (r) => { if (r.method() === 'POST') posts.push(r.url()) })
      const resp = page.waitForResponse((r) => r.request().method() === 'POST' && /\/baseline\/authorize/.test(r.url()), { timeout: 30_000 })
        .then((r) => r.status(), () => null)
      await btn.first().click({ timeout: 30_000 })
      rec.authorize_status = await resp
      await page.waitForTimeout(500)
      // A visible "Starting…" button means the click is still in flight: not gone.
      const starting = await page.getByRole('button', { name: /^Starting/ }).count()
      rec.button_gone = starting === 0 && await expect(btn).toHaveCount(0, { timeout: 30_000 }).then(() => true, () => false)
      rec.post_urls = posts
      // Gap G-2: where does focus go once the button unmounts?
      rec.active_element_after = await safe(errs, 'active element', () => page.evaluate((bar) => {
        const a = document.activeElement as HTMLElement | null
        return { tag: a?.tagName ?? null, text: (a?.getAttribute('aria-label') ?? a?.innerText ?? '').trim().slice(0, 80),
          id: a?.id ?? null, inside_bar: !!a && !!document.querySelector(bar)?.contains(a), is_body: a === document.body }
      }, BAR))
      await shot(page, 'authorize-baseline-after')
    }
    const settle = Number(process.env.LIVE_ACCEPTANCE_SETTLE_MS ?? 0)
    if (settle > 0) {
      await page.waitForTimeout(settle)
      await page.reload()
      await page.getByRole('heading', { name: 'Current operation' }).first().waitFor({ timeout: 60_000 })
      rec.after_settle = await statusFacts(page, errs)
    }
  })
})

test('status-baseline', async ({ page }) => {
  test.setTimeout(180_000)
  await observe(page, 'status-baseline', async (rec, errs) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await openGoal(page)
    await core(page, rec, errs)
    rec.layout = {}
    for (const [w, h] of [[1440, 900], [375, 812]]) rec.layout[w] = await layoutProbe(page, w, h, 'status-baseline', out)
  })
})

test('layout', async ({ page }) => {
  test.setTimeout(180_000)
  await observe(page, 'layout', async (rec) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await openGoal(page)
    rec.layout = {}
    for (const [w, h] of [[1440, 900], [375, 812]]) rec.layout[w] = await layoutProbe(page, w, h, 'layout', out)
  })
})

test('status-pending', async ({ page }) => {
  test.setTimeout(120_000)
  await observe(page, 'status-pending', async (rec, errs) => {
    await openGoal(page)
    await core(page, rec, errs)
    Object.assign(rec, await pendingQuestion(page, await fetchWorkflow(page, errs), errs))
    await shot(page, 'status-pending-1440')
    // Debug tab is config-gated (debug_enabled); absent is a valid observation.
    const tab = page.getByRole('tab', { name: 'Debug', exact: true })
    rec.debug_tab_present = (await tab.count()) > 0
    if (rec.debug_tab_present) {
      await safe(errs, 'debug tab', async () => { await tab.click() })
      const dbg = await safe(errs, 'debug probe', () => page.evaluate(() => {
        const heads = Array.from(document.querySelectorAll('h3, h4, dt, span, p, div')).filter((e) => e.children.length === 0)
        const find = (t: string) => heads.find((e) => (e.textContent ?? '').trim().startsWith(t)) ?? null
        const p = find('Pending owner decision'), w = find('Runtime waits')
        const grp = (e: Element | null) => e?.closest('div') ?? null
        return { pending: !!p, waits: !!w, separate: !!p && !!w && grp(p) !== grp(w) && !grp(p)!.contains(w!) && !grp(w)!.contains(p!) }
      }))
      rec.debug_pending_text = dbg?.pending ?? false
      rec.debug_runtime_waits_text = dbg?.waits ?? false
      rec.debug_separate = dbg?.separate ?? false
      await shot(page, 'status-pending-debug')
    }
  })
})

test('status-paused-pending', async ({ page }) => {
  test.setTimeout(120_000)
  await observe(page, 'status-paused-pending', async (rec, errs) => {
    await openGoal(page)
    await core(page, rec, errs)
    Object.assign(rec, await pendingQuestion(page, await fetchWorkflow(page, errs), errs))
    await shot(page, 'status-paused-pending-1440')
  })
})

test('status-saved', async ({ page }) => {
  test.setTimeout(120_000)
  await observe(page, 'status-saved', async (rec, errs) => {
    await openGoal(page)
    rec.facts = await statusFacts(page, errs)
    await page.reload()
    await page.getByRole('heading', { name: 'Current operation' }).first().waitFor({ timeout: 60_000 })
    rec.after_reload = await statusFacts(page, errs)
    await shot(page, 'status-saved-1440')
  })
})

test('status-authorize', async ({ page }) => {
  test.setTimeout(120_000)
  await observe(page, 'status-authorize', async (rec, errs) => {
    await openGoal(page)
    rec.guidance = (await fetchWorkflow(page, errs))?.guidance ?? null
    await core(page, rec, errs)
    rec.authorize_button_name = await safe(errs, 'authorize name', async () => (await page.getByRole('button', { name: AUTH_BTN }).first().innerText({ timeout: 15_000 })).trim())
    await shot(page, 'status-authorize-1440')
  })
})

test('status-th-pending', async ({ page }) => {
  test.setTimeout(120_000)
  await observe(page, 'status-th-pending', async (rec, errs) => {
    await openGoal(page)
    await core(page, rec, errs)
    Object.assign(rec, await pendingQuestion(page, await fetchWorkflow(page, errs), errs))
    await shot(page, 'status-th-pending-1440')
    const trunc = (s: string) => s.slice(0, 4000)
    const summary = page.getByRole('region', { name: 'Hierarchy evidence' }).first()
    rec.th_summary_text = await safe(errs, 'th summary', async () => {
      await summary.screenshot({ path: join(out, 'th-summary.png'), timeout: 15_000 })
      return trunc(await summary.innerText())
    })
    // Select the team-hierarchy step in the stepper (aria-label "Select Structuring the team").
    const step = page.getByRole('button', { name: 'Select Structuring the team' })
    rec.th_step_present = (await step.count()) > 0
    if (rec.th_step_present) {
      await safe(errs, 'th step', async () => {
        if ((await step.getAttribute('aria-pressed', { timeout: 5_000 })) !== 'true') await step.click({ timeout: 10_000 })
        const chain = page.locator('section[aria-labelledby="baseline-process-chain-heading"]')
        await chain.screenshot({ path: join(out, 'th-step-detail.png'), timeout: 15_000 })
        rec.th_step_text = trunc(await chain.innerText())
      })
    }
  })
})

test('status-paused-admitted', async ({ page }) => {
  test.setTimeout(180_000)
  await observe(page, 'status-paused-admitted', async (rec, errs) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await openGoal(page)
    await core(page, rec, errs)
    const body = (await safe(errs, 'body', () => page.evaluate(() => document.body.innerText))) ?? ''
    rec.live_claim_matches = [...body.matchAll(/\b(working now|is working|running now|currently running|active now)\b/gi)].map((m) => m[0])
    rec.layout = {}
    for (const [w, h] of [[1440, 900], [375, 812]]) rec.layout[w] = await layoutProbe(page, w, h, 'status-paused-admitted', out)
  })
})

// Non-blocking: Python waits for <OUTPUT>/calls-live.ready, then releases one model call.
test('calls-live', async ({ page }) => {
  const waitMs = Number(process.env.LIVE_ACCEPTANCE_CALLS_WAIT_MS ?? 300_000)
  test.setTimeout(waitMs + 120_000)
  await observe(page, 'calls-live', async (rec, errs) => {
    mkdirSync(out, { recursive: true })
    try { unlinkSync(join(out, 'calls-live.ready')) } catch { /* none yet */ }
    await page.setViewportSize({ width: 1440, height: 900 })
    await openGoal(page)
    await selectCalls(page, errs)
    // Gap G-3 evidence: Calls selected by role; check Ask does not also look selected.
    rec.calls_aria_selected = await safe(errs, 'calls sel', () => callsTab(page).getAttribute('aria-selected'))
    rec.ask_aria_selected = await safe(errs, 'ask sel', () => askTab(page).getAttribute('aria-selected'))
    rec.calls_tab_selected = rec.calls_aria_selected === 'true'
    rec.calls_tab_class = await safe(errs, 'calls class', () => callsTab(page).getAttribute('class'))
    rec.ask_tab_class = await safe(errs, 'ask class', () => askTab(page).getAttribute('class'))
    const before = await callIds(page)
    rec.calls_before = before.length
    await shot(page, 'calls-live-1440')
    // Release the harness only once Calls is selected and the stream is live.
    const note = page.getByText(/^Calls appear here live/)
    const live = page.locator('header').getByText('Live connection', { exact: true })
    const ready = await safe(errs, 'ready gate', async () => {
      await expect(callsTab(page)).toHaveAttribute('aria-selected', 'true', { timeout: 10_000 })
      await expect(note.or(live).first()).toBeVisible({ timeout: 60_000 })
      return true
    })
    rec.ready_blocked_reason = ready ? null : (errs[errs.length - 1] ?? 'ready gate failed')
    if (ready) writeFileSync(join(out, 'calls-live.ready'), String(Date.now()))
    // A Calls entry is a [data-call-id] block (CallBlock), created by an agent-response WS event.
    rec.entry_appeared_at_ms = null
    rec.entry_text = null
    const end = Date.now() + (ready ? waitMs : 0)
    while (Date.now() < end) {
      try {
        const fresh = (await callIds(page)).find((id) => !before.includes(id))
        if (fresh) {
          rec.entry_appeared_at_ms = Date.now()
          rec.entry_text = ((await page.locator(`[data-call-id="${fresh}"]`).innerText().catch(() => '')) as string).slice(0, 1000)
          break
        }
      } catch (e) { errs.push(`poll: ${String(e).split('\n')[0]}`) }
      await page.waitForTimeout(500)
    }
    rec.calls_after = await callIds(page).then((x) => x.length, () => null)
    await safe(errs, 'entry shot', () => shot(page, 'calls-live-entry'))
  })
})

test('reload-calls-empty', async ({ page }) => {
  test.setTimeout(120_000)
  await observe(page, 'reload-calls-empty', async (rec, errs) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await openGoal(page)
    await page.reload()
    await page.getByRole('heading', { name: 'Current operation' }).first().waitFor({ timeout: 60_000 })
    await selectCalls(page, errs)
    const empty = 'No live calls captured in this view.'
    const note = 'Calls appear here live while this page is open; earlier calls and calls from before a reload are not shown.'
    // The note depends on the socket being connected; give it time.
    await page.getByText(note, { exact: true }).waitFor({ timeout: 20_000 }).catch(() => {})
    rec.empty_text_present = (await page.getByText(empty, { exact: true }).count()) > 0
    rec.live_note_present = (await page.getByText(note, { exact: true }).count()) > 0
    const body = (await safe(errs, 'body', () => page.evaluate(() => document.body.innerText))) ?? ''
    rec.forbidden_matches = [...body.matchAll(/no calls (were )?(made|happened)|0 calls|no activity/gi)].map((m) => m[0])
    rec.facts = await statusFacts(page, errs)
    await shot(page, 'reload-calls-empty-1440')
  })
})

test('stale-refresh', async ({ page }) => {
  test.setTimeout(300_000)
  await observe(page, 'stale-refresh', async (rec, errs) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await openGoal(page)
    const fresh = () => page.locator('section[aria-labelledby="current-operation-title"] p').filter({ hasText: /^Status (last )?refreshed|^Status not yet/ }).first()
    rec.fresh_text_before = await safe(errs, 'fresh before', () => fresh().innerText({ timeout: 15_000 }))
    // The goal query has staleTime 10s and polls only while the WS is down, so the reliable trigger is a
    // focus/visibility event after the data has gone stale (react-query refetchOnWindowFocus).
    rec.refetch_trigger = 'wait 11s (staleTime 10s) then dispatch visibilitychange+focus'
    const trigger = async () => {
      await page.waitForTimeout(11_000)
      await page.evaluate(() => {
        document.dispatchEvent(new Event('visibilitychange'))
        window.dispatchEvent(new Event('visibilitychange'))
        window.dispatchEvent(new Event('focus'))
      })
    }
    // Detail GET only: exact path, no sub-path.  8 hits covers the query's retry and a few retriggers.
    const fault = await failWith500(page, (u) => u.pathname.endsWith(`/orchestration/goals/${goalId}`), 8)
    try {
      await trigger()
      rec.stale_text = await safe(errs, 'stale text', () => fresh().filter({ hasText: 'refresh failed' }).innerText({ timeout: 60_000 }))
      rec.controls_while_stale = await controls(page, errs)
      rec.fault_hits = fault.hits
      await shot(page, 'stale-refresh-stale')
    } finally {
      await fault.release()
    }
    await trigger()
    rec.fresh_text_after_release = await safe(errs, 'fresh after', async () => {
      await expect(fresh()).toHaveText(/^Status refreshed(?!.*refresh failed)/, { timeout: 60_000 })
      return fresh().innerText()
    })
    await shot(page, 'stale-refresh-fresh')
  })
})

test('calls-disconnected', async ({ page }) => {
  test.setTimeout(180_000)
  await observe(page, 'calls-disconnected', async (rec, errs) => {
    // Refuse the project event socket (ws://.../ws/projects/<id>/events) before the app starts.
    let ws_attempts = 0
    await page.routeWebSocket(/\/ws\/projects\/[^/]+\/events/, (ws) => { ws_attempts += 1; void ws.close({ code: 1011, reason: 'blocked' }) })
    await page.setViewportSize({ width: 1440, height: 900 })
    await openGoal(page)
    await selectCalls(page, errs)
    const msg = 'Live call capture is unavailable while disconnected; calls made during that time will not appear.'
    rec.disconnected_text_present = await page.getByText(msg, { exact: true }).waitFor({ timeout: 60_000 }).then(() => true, () => false)
    rec.connectivity_text = await connectivityText(page, errs)
    rec.shows_connected = rec.connectivity_text === 'Connected' || rec.connectivity_text === 'Live connection'
    rec.ws_attempts = ws_attempts
    await shot(page, 'calls-disconnected-1440')
  })
})

test('status-plan-review', async ({ page }) => {
  test.setTimeout(180_000)
  await observe(page, 'status-plan-review', async (rec, errs) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await openGoal(page)
    await core(page, rec, errs)
    Object.assign(rec, await pendingQuestion(page, await fetchWorkflow(page, errs), errs))
    const buttons = await safe(errs, 'buttons', () => page.getByRole('button', { name: /start implementation/i }).count())
    const nextText = (rec.facts?.next_action_text ?? []).join(' ')
    rec.start_buttons_visible = await safe(errs, 'start buttons', async () => {
      const all = page.getByRole('button', { name: /^(start implementation|start work)$/i })
      let n = 0
      for (let i = 0; i < await all.count(); i++) if (await all.nth(i).isVisible()) n++
      return n
    })
    rec.start_implementation_offered = (buttons ?? 0) > 0 || /start implementation/i.test(nextText)
    rec.layout = {}
    // Human plan authority: the API guidance label must be on screen at both widths (observed, judged in Python).
    const human = process.env.LIVE_ACCEPTANCE_PLAN_AUTHORITY === 'human'
    const apiLabel: string | null = (await fetchWorkflow(page, errs))?.guidance?.label ?? null
    rec.api_guidance_label = apiLabel
    rec.guidance_control = {}
    for (const [w, h] of [[1440, 900], [375, 812]]) {
      rec.layout[w] = await layoutProbe(page, w, h, 'status-plan-review', out)
      if (!human || !apiLabel) continue
      // The real next-action cue: the "Next:" element, preferably inside the sticky control bar.
      const inBar = page.locator(BAR).locator('p, span').filter({ hasText: /^Next:/ }).first()
      const anywhere = page.locator('p, span').filter({ hasText: /^Next:/ }).first()
      const inBarFound = (await inBar.count()) > 0
      const loc = inBarFound ? inBar : anywhere
      const box = await safe(errs, `guidance box ${w}`, () => loc.boundingBox({ timeout: 10_000 }))
      rec.guidance_control[w] = {
        in_bar: inBarFound,
        visible: (await safe(errs, `guidance visible ${w}`, () => loc.isVisible({ timeout: 10_000 }))) === true,
        text: await safe(errs, `guidance text ${w}`, async () => (await loc.innerText({ timeout: 10_000 })).trim()),
        // Not scrolled: the bar is sticky, so presence is judged where the page loads (scrolled=false).
        scrolled: false,
        in_viewport: !!box && box.x >= 0 && box.y >= 0 && box.x + box.width <= w && box.y + box.height <= h,
      }
    }
  })
})
