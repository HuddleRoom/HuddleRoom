import { mkdirSync } from 'node:fs'
import { join } from 'node:path'
import type { Page } from '@playwright/test'

// Shared probes for the G5/G6 live steps.  Probes only observe: they never assert, and locator
// failures are pushed to `errs` (surfaced as `ui_errors` in the step JSON) instead of thrown.
export type Errs = string[]

export const BAR = '[data-testid="orchestration-sticky-bar"]' // goal control bar (OrchestrationPage)
export const SHELL_SCROLL = '.huddleroom-shell-content' // Shell scroll container (shell.css)
export const AUTH_BTN = /^(Continue baseline|Authorize (remaining )?baseline|Start baseline)$/
const GOAL_URL = () => `/api/v1/projects/${process.env.LIVE_ACCEPTANCE_PROJECT_ID}/orchestration/goals/${process.env.LIVE_ACCEPTANCE_GOAL_ID}`

export async function safe<T>(errs: Errs, label: string, fn: () => Promise<T>): Promise<T | null> {
  try { return await fn() } catch (e) { errs.push(`${label}: ${String(e).split('\n')[0]}`); return null }
}

export async function getJson(page: Page, path: string, errs: Errs): Promise<any> {
  return safe(errs, `GET ${path}`, async () => {
    const res = await page.request.get(path)
    if (!res.ok()) throw new Error(`HTTP ${res.status()}`)
    return res.json()
  })
}

/** supervision.status_line.workflow as the API reports it right now. */
export async function fetchWorkflow(page: Page, errs: Errs) {
  const body = await getJson(page, GOAL_URL(), errs)
  return body?.supervision?.status_line?.workflow ?? null
}

/** Visible buttons of interest (name = aria-label or text). */
export async function controls(page: Page, errs: Errs = []) {
  const names = await safe(errs, 'controls', () => page.evaluate(() => Array.from(document.querySelectorAll('button'))
    .filter((b) => { const r = b.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(b).visibility !== 'hidden' })
    .map((b) => (b.getAttribute('aria-label') ?? b.innerText).trim())))
  const wanted = /^(Pause goal|Resume goal|Start work|Continue baseline|Authorize (remaining )?baseline|Start baseline|continue.*)$/i
  return { visible_buttons: (names ?? []).filter((n) => wanted.test(n)) }
}

export async function statusFacts(page: Page, errs: Errs = []) {
  const facts = await safe(errs, 'statusFacts', () => page.evaluate(({ bar }) => {
    const body = document.body.innerText
    const barEl = document.querySelector<HTMLElement>(bar)
    const op = Array.from(document.querySelectorAll('section')).find((s) => s.getAttribute('aria-labelledby') === 'current-operation-title')
    const next = Array.from(document.querySelectorAll<HTMLElement>('p, span')).map((e) => e.innerText?.trim() ?? '')
      .filter((t) => /^Next:/.test(t))
    return {
      goal_badge_text: barEl?.innerText.match(/Goal:[^\n]*/)?.[0] ?? null,
      current_operation_text: op?.innerText ?? null,
      next_action_text: [...new Set(next)],
      paused_copy_present: body.includes('Work already started is not cancelled and may still finish.'),
      saved_not_applied_present: body.includes('Saved · not applied — Not in effect yet.'),
      answer_in_needs_you_present: body.includes('Answer in Needs You'),
    }
  }, { bar: BAR }))
  return facts
}

/** Pending owner decision: API question/title and whether the page text carries it. */
export async function pendingQuestion(page: Page, workflow: any, errs: Errs) {
  const id: string | null = workflow?.pending_decision_id ?? null
  const raw = id ? await getJson(page, `${GOAL_URL()}/decisions/${id}`, errs) : null
  const d = raw?.decision ?? raw
  const question: string | null = d?.question ?? null
  const title: string | null = d?.title ?? null
  const body = ((await safe(errs, 'body', () => page.evaluate(() => document.body.innerText))) ?? '').replace(/\s+/g, ' ')
  const has = (t: string | null) => !!t && body.includes(t.replace(/\s+/g, ' ').trim())
  return { pending_decision_id: id, question, title, question_visible: has(question), title_visible: has(title) }
}

/** D1-D4 at one viewport.  Reloads, screenshots `<step>-<width>.png`, then runs the probes. */
export async function layoutProbe(page: Page, width: number, height: number, step: string, out: string) {
  const errs: Errs = []
  const res: Record<string, unknown> = { width, height, ui_errors: errs }
  mkdirSync(out, { recursive: true })
  await safe(errs, 'viewport+reload', async () => { await page.setViewportSize({ width, height }); await page.reload() })
  await safe(errs, 'bar wait', () => page.locator(BAR).waitFor({ timeout: 60_000 }))
  await page.waitForLoadState('networkidle').catch(() => {})
  await safe(errs, 'shot', () => page.screenshot({ path: join(out, `${step}-${width}.png`) }))
  await safe(errs, 'shot full', () => page.screenshot({ path: join(out, `${step}-${width}-full.png`), fullPage: true }))

  // D1: status badge + lifecycle buttons inside the viewport; horizontal overflow.
  res.d1 = await safe(errs, 'd1', () => page.evaluate(({ bar, scroll }) => {
    const rectOf = (e: Element | null) => { if (!e) return null; const r = e.getBoundingClientRect(); return { top: r.top, bottom: r.bottom, left: r.left, right: r.right, width: r.width, height: r.height } }
    const inVp = (r: any) => !!r && r.top >= 0 && r.left >= 0 && r.bottom <= window.innerHeight && r.right <= window.innerWidth
    const barEl = document.querySelector(bar)
    const leaf = Array.from(barEl?.querySelectorAll('*') ?? []).filter((e) => /Goal:/.test(e.textContent ?? ''))
      .sort((a, b) => (a.textContent ?? '').length - (b.textContent ?? '').length)[0] ?? null
    const boxes: Record<string, any> = {}
    const badge = rectOf(leaf)
    boxes['Goal badge'] = badge && { ...badge, inViewport: inVp(badge) }
    for (const b of Array.from(barEl?.querySelectorAll('button') ?? [])) {
      const n = (b.getAttribute('aria-label') ?? (b as HTMLElement).innerText).trim()
      const r = rectOf(b)
      if (['Pause goal', 'Resume goal', 'Start work'].includes(n) && r && r.width > 0) boxes[n] = { ...r, inViewport: inVp(r) }
    }
    const sc = document.querySelector(scroll)
    return {
      boxes,
      doc_overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
      shell_overflow: !!sc && sc.scrollWidth > sc.clientWidth,
    }
  }, { bar: BAR, scroll: SHELL_SCROLL }))

  // D2: sticky bar after scrolling the shell container to the bottom.
  res.d2 = await safe(errs, 'd2', async () => {
    const position = await page.evaluate((bar) => getComputedStyle(document.querySelector(bar)!).position, BAR)
    if (width < 1024) return { position }
    const r = await page.evaluate(({ bar, scroll }) => {
      const sc = document.querySelector(scroll)!
      sc.scrollTop = sc.scrollHeight
      const b = document.querySelector(bar)!.getBoundingClientRect()
      return { scrollable: sc.scrollHeight > sc.clientHeight, rect: { top: b.top, bottom: b.bottom }, in_vp: b.top >= 0 && b.bottom <= window.innerHeight }
    }, { bar: BAR, scroll: SHELL_SCROLL })
    await safe(errs, 'shot scrolled', () => page.screenshot({ path: join(out, `${step}-${width}-scrolled.png`) }))
    return { position, scrollable: r.scrollable, bar_rect_after_scroll: r.rect, bar_in_viewport_after_scroll: r.in_vp }
  })

  const toTop = () => page.evaluate((scroll) => { document.querySelector(scroll)!.scrollTop = 0; (document.activeElement as HTMLElement | null)?.blur() }, SHELL_SCROLL)

  // D3: keyboard-only objective disclosure.
  res.d3 = await safe(errs, 'd3', async () => {
    await toTop()
    const active = () => page.evaluate(() => (document.activeElement as HTMLElement | null)?.innerText?.trim() ?? '')
    let tabs = 0
    let found = false
    for (; tabs < 60 && !found; tabs++) { await page.keyboard.press('Tab'); found = (await active()) === 'Show full objective' }
    if (!found) return { found: false, tabs_needed: null, expanded_text: null, toggle_text_after_open: null, collapsed_after_second_enter: null }
    await page.keyboard.press('Enter')
    const open = await page.evaluate(() => {
      const d = document.activeElement?.closest('details')
      return { text: d?.querySelector('p')?.textContent ?? null, toggle: (document.activeElement as HTMLElement).innerText.trim() }
    })
    await page.keyboard.press('Enter')
    const collapsed = await page.evaluate(() => !document.activeElement?.closest('details')?.open)
    return { found: true, tabs_needed: tabs, expanded_text: open.text, toggle_text_after_open: open.toggle, collapsed_after_second_enter: collapsed }
  })

  // D4: 25 Tabs from the first control in the bar; flag focus that lands under the sticky bar.
  if (width === 1440) {
    res.d4 = await safe(errs, 'd4', async () => {
      await toTop()
      const ok = await page.evaluate((bar) => {
        const el = document.querySelector(bar)?.querySelector<HTMLElement>('button, a[href], input, select, [tabindex]:not([tabindex="-1"])')
        el?.focus(); return !!el
      }, BAR)
      const entries: any[] = []
      let wrapped = false
      let seenInside = false
      for (let i = 0; i < 25 && ok; i++) {
        await page.keyboard.press('Tab')
        const e = await page.evaluate(({ bar, scroll }) => {
          const a = document.activeElement as HTMLElement | null
          const barEl = document.querySelector(bar)!
          const b = barEl.getBoundingClientRect()
          const r = a?.getBoundingClientRect()
          return { tag: a?.tagName ?? null, name: (a?.getAttribute('aria-label') ?? a?.innerText ?? '').trim().slice(0, 80),
            inside_bar: !!a && barEl.contains(a), in_shell_content: !!a && !!document.querySelector(scroll)?.contains(a),
            top: r?.top ?? null, left: r?.left ?? null, right: r?.right ?? null,
            bar_bottom: b.bottom, bar_left: b.left, bar_right: b.right }
        }, { bar: BAR, scroll: SHELL_SCROLL })
        if (e.in_shell_content) seenInside = true
        else if (seenInside) { wrapped = true; entries.push(e); break }
        entries.push(e)
      }
      // Violation = a real element in the scrolling content, overlapping the bar horizontally, tucked under it.
      const violations = entries.filter((e) => e.tag !== 'BODY' && e.in_shell_content && !e.inside_bar && e.top !== null
        && e.left < e.bar_right && e.right > e.bar_left && e.top < e.bar_bottom)
      return { first_focus_found: ok, wrapped, entries, violations }
    })
  }
  return res
}
