import type { Page, Route } from '@playwright/test'

export type UrlPattern = string | RegExp | ((url: URL) => boolean)

export interface FaultHandle {
  /** Number of matching requests intercepted. */
  readonly hits: number
  /** Release anything held, wait for in-flight handlers, remove the route. Idempotent. */
  release: () => Promise<void>
}

export interface ReloadFaultHandle extends FaultHandle {
  /** Wait for the held request, reload the page, then release. Throws after timeoutMs (default 10s). */
  reload: (timeoutMs?: number) => Promise<void>
}

interface Ctl {
  hits: number
  committed: number
  released: boolean
  /** Resolves true if it really waited, false if already released. */
  hold: (extraWakeMs?: number) => Promise<boolean>
}

const safe = async (fn: () => Promise<unknown>) => { try { await fn() } catch { /* route/page already gone */ } }

async function install(
  page: Page,
  pattern: UrlPattern,
  handler: (route: Route, ctl: Ctl) => Promise<void>,
): Promise<FaultHandle & { ctl: Ctl }> {
  const waiters = new Set<() => void>()
  const inflight = new Set<Promise<void>>()
  const ctl: Ctl = {
    hits: 0,
    committed: 0,
    released: false,
    hold: async (extraWakeMs) => {
      if (ctl.released) return false
      await new Promise<void>((resolve) => {
        waiters.add(resolve)
        if (extraWakeMs !== undefined) setTimeout(resolve, extraWakeMs)
      })
      return true
    },
  }
  const fn = (route: Route) => {
    if (ctl.released) return safe(() => route.fallback())
    ctl.hits += 1
    const p = safe(() => handler(route, ctl)).finally(() => inflight.delete(p))
    inflight.add(p)
    return p
  }
  await page.route(pattern, fn)
  let releasing: Promise<void> | null = null
  const release = () => (releasing ??= (async () => {
    ctl.released = true
    waiters.forEach((w) => w())
    waiters.clear()
    await Promise.allSettled([...inflight])
    await safe(() => page.unroute(pattern, fn))
  })())
  page.on('close', () => { void release() })
  return { get hits() { return ctl.hits }, release, ctl }
}

const expose = ({ hits, release }: FaultHandle): FaultHandle => ({ get hits() { return hits }, release })

/** Server commits the request, the client sees a connection reset. */
export const dropAfterCommit = async (page: Page, pattern: UrlPattern): Promise<FaultHandle> => {
  const h = await install(page, pattern, async (route, ctl) => {
    await route.fetch()
    ctl.committed += 1
    await route.abort('connectionreset')
  })
  return { get hits() { return h.hits }, release: h.release }
}

/** Server commits the request; the client never gets a response until release(), which aborts it. */
export const neverFulfil = async (page: Page, pattern: UrlPattern): Promise<FaultHandle> => {
  const h = await install(page, pattern, async (route, ctl) => {
    await route.fetch()
    ctl.committed += 1
    await ctl.hold()
    await route.abort('failed')
  })
  return { get hits() { return h.hits }, release: h.release }
}

/** Delay the request by ms (or until release()), then let it through normally. */
export const holdThenRelease = async (page: Page, pattern: UrlPattern, ms: number): Promise<FaultHandle> => {
  const h = await install(page, pattern, async (route, ctl) => {
    await ctl.hold(ms)
    await route.fallback()
  })
  return { get hits() { return h.hits }, release: h.release }
}

/** Answer the first `times` matching requests with a 500, then pass through. */
export const failWith500 = async (page: Page, pattern: UrlPattern, times: number): Promise<FaultHandle> => {
  const h = await install(page, pattern, async (route, ctl) => {
    if (ctl.hits <= times) {
      await route.fulfill({ status: 500, contentType: 'application/json', body: '{"detail":"injected failure"}' })
    } else {
      await route.fallback()
    }
  })
  return { get hits() { return h.hits }, release: h.release }
}

async function reloadHandle(
  page: Page,
  h: FaultHandle & { ctl: Ctl },
  ready: (ctl: Ctl) => boolean,
): Promise<ReloadFaultHandle> {
  return {
    get hits() { return h.hits },
    release: h.release,
    reload: async (timeoutMs = 10_000) => {
      const deadline = Date.now() + timeoutMs
      while (!ready(h.ctl)) {
        if (Date.now() > deadline) throw new Error(`reload fault: request not held within ${timeoutMs}ms`)
        await new Promise((r) => setTimeout(r, 20))
      }
      await page.reload()
      await h.release()
    },
  }
}

/**
 * Default reload fault: the server commits the request first, the response is held, then the page
 * reloads (the response is lost), then everything is released.
 */
export async function reloadMidRequest(page: Page, pattern: UrlPattern): Promise<ReloadFaultHandle> {
  const h = await install(page, pattern, async (route, ctl) => {
    const response = await route.fetch()
    ctl.committed += 1
    await ctl.hold()
    await route.fulfill({ response })
  })
  return reloadHandle(page, h, (c) => c.committed > 0)
}

/** Reload while the request is held before it reaches the server; on release it is aborted unsent. */
export async function reloadBeforeServer(page: Page, pattern: UrlPattern): Promise<ReloadFaultHandle> {
  const h = await install(page, pattern, async (route, ctl) => {
    await ctl.hold()
    await route.abort('aborted')
  })
  return reloadHandle(page, h, (c) => c.hits > 0)
}
