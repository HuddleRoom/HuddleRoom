import { createServer, type Server } from 'node:http'
import type { AddressInfo } from 'node:net'
import { expect, test, type Page } from '@playwright/test'
import { dropAfterCommit, reloadBeforeServer, failWith500, holdThenRelease, neverFulfil, reloadMidRequest } from '../helpers/faults'
import { neverObserves, observesWithin } from '../helpers/poll'
import { goalStatusBadge } from '../helpers/selectors'
import { captureReceipts } from '../helpers/receipts'

// route.fetch() needs a real origin, so serve the static page + fake API from a throwaway local
// node server (not HuddleRoom).
const server = { commits: 0 }
let ORIGIN = ''
let http: Server

test.beforeAll(async () => {
  http = createServer((req, res) => {
    const path = (req.url ?? '').split('?')[0]
    const json = (o: unknown) => { res.setHeader('content-type', 'application/json'); res.end(JSON.stringify(o)) }
    if (path === '/') {
      res.setHeader('content-type', 'text/html')
      return void res.end(`<button id=go>go</button><p id=out></p><script>
        document.getElementById('go').onclick = async () => {
          try { const r = await fetch('/api/goals/g1/pause-control', {method:'POST', body: JSON.stringify({a:1})});
            document.getElementById('out').textContent = 'status ' + r.status }
          catch (e) { document.getElementById('out').textContent = 'error' } }
      </script>`)
    }
    if (req.method === 'POST') server.commits += 1
    if (path.endsWith('/pause-control')) return json({ action_id: 'a1' })
    if (path.endsWith('/answer')) return json({ action_id: 'x', decision_ids: ['d1'], application_state: 'applied' })
    res.statusCode = 404
    res.end()
  })
  await new Promise<void>((r) => http.listen(0, '127.0.0.1', r))
  ORIGIN = `http://127.0.0.1:${(http.address() as AddressInfo).port}`
})
test.afterAll(() => new Promise((r) => { http.close(r); http.closeAllConnections() }))

async function setup(page: Page) {
  server.commits = 0
  await page.goto(`${ORIGIN}/`)
  return server
}

test('dropAfterCommit: server hit, client error', async ({ page }) => {
  const server = await setup(page)
  const f = await dropAfterCommit(page, '**/pause-control')
  await page.click('#go')
  await expect(page.locator('#out')).toHaveText('error')
  expect(server.commits).toBe(1)
  expect(f.hits).toBe(1)
  await f.release()
})

test('neverFulfil: server hit, no response until release', async ({ page }) => {
  const server = await setup(page)
  const f = await neverFulfil(page, '**/pause-control')
  await page.click('#go')
  await expect.poll(() => server.commits).toBe(1)
  expect(await neverObserves(async () => (await page.locator('#out').textContent()) !== '', 400)).toBe(true)
  await f.release()
  await expect(page.locator('#out')).toHaveText('error')
})

test('holdThenRelease delays at least ms', async ({ page }) => {
  await setup(page)
  const f = await holdThenRelease(page, '**/pause-control', 500)
  const t0 = Date.now()
  await page.click('#go')
  await expect(page.locator('#out')).toHaveText('status 200')
  expect(Date.now() - t0).toBeGreaterThanOrEqual(480)
  expect(f.hits).toBe(1)
  await f.release()
})

test('failWith500 fails N times then passes through', async ({ page }) => {
  await setup(page)
  const f = await failWith500(page, '**/pause-control', 2)
  for (const expected of ['status 500', 'status 500', 'status 200']) {
    await page.click('#go')
    await expect(page.locator('#out')).toHaveText(expected)
    await page.evaluate(() => { document.getElementById('out')!.textContent = '' })
  }
  expect(f.hits).toBe(3)
  await f.release()
})

test('reloadMidRequest: server commits exactly once, response lost', async ({ page }) => {
  const server = await setup(page)
  const f = await reloadMidRequest(page, '**/pause-control')
  await page.click('#go')
  await f.reload()
  expect(f.hits).toBe(1)
  await expect(page.locator('#out')).toHaveText('')
  expect(server.commits).toBe(1)
})

test('reloadBeforeServer: server never sees the request', async ({ page }) => {
  const server = await setup(page)
  const f = await reloadBeforeServer(page, '**/pause-control')
  await page.click('#go')
  await f.reload()
  expect(f.hits).toBe(1)
  expect(server.commits).toBe(0)
})

test('receipts capture pause and answer calls', async ({ page }) => {
  await setup(page)
  const r = captureReceipts(page)
  await page.click('#go')
  await expect(page.locator('#out')).toHaveText('status 200')
  await page.evaluate(() => fetch('/api/goals/g1/decisions/d9/answer', { method: 'POST', body: '{"x":1}' }))
  await expect.poll(() => r.lastAnswerReceipt()).not.toBeNull()
  expect(r.postCount('answer-post')).toBe(1)
  expect(r.postCount('pause-post')).toBe(1)
  expect(r.lastAnswerReceipt()).toEqual({ action_id: 'x', decision_ids: ['d1'], application_state: 'applied', urlDecisionId: 'd9' })
})

test('observesWithin / neverObserves', async ({ page }) => {
  await setup(page)
  const t0 = Date.now()
  expect(await observesWithin(() => Date.now() - t0 > 200, 1000)).toBe(true)
  expect(await neverObserves(() => false, 300)).toBe(true)
  expect(await neverObserves(() => true, 300)).toBe(false)
})

test('goalStatusBadge matches the dotted badge span and only its own label', async ({ page }) => {
  await page.setContent('<header data-testid="orchestration-goal-header"><span><span aria-hidden="true">●</span>Paused</span></header>')
  const hdr = page.getByTestId('orchestration-goal-header')
  await expect(goalStatusBadge(hdr, 'Paused')).toHaveCount(1)
  await expect(goalStatusBadge(hdr, 'Running')).toHaveCount(0)
  await expect(goalStatusBadge(hdr, 'Active')).toHaveCount(0)
})
