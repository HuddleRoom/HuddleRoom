import React, { act } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { descendants, getButton, mountWithTestDom, textOf } from '../../../tests/support/dom'
import { GoalAnnouncerProvider, useGoalAnnouncer } from './goalAnnouncer'

function Controls() {
  const { announceQueue, announceError } = useGoalAnnouncer()
  return <><button onClick={() => { announceQueue('New question arrived.'); announceQueue('Another question answered.'); }}>Both human updates</button>
    <button onClick={() => announceQueue('Repeated message.')}>Repeat</button>
    <button onClick={() => { announceQueue('Polite update.'); announceError('Separate error.'); }}>Separate channels</button></>
}
const region = (view: Awaited<ReturnType<typeof mountWithTestDom>>, role: string) => descendants(view.container).find((node) => node.getAttribute('role') === role)!

describe('shared GoalAnnouncer delivery', () => {
  it('batches simultaneous human question and answer messages without overwriting either', async () => {
    const view = await mountWithTestDom(() => <GoalAnnouncerProvider><Controls /></GoalAnnouncerProvider>, act)
    try {
      await act(async () => getButton(view.container, 'Both human updates').click())
      expect(textOf(region(view, 'status'))).toBe('New question arrived. Another question answered.')
      expect(descendants(view.container).filter((node) => node.getAttribute('aria-live') === 'polite')).toHaveLength(1)
    } finally { view.cleanup() }
  })
  it('retains repeated-message parity and separates polite updates from assertive errors', async () => {
    const view = await mountWithTestDom(() => <GoalAnnouncerProvider><Controls /></GoalAnnouncerProvider>, act)
    try {
      await act(async () => getButton(view.container, 'Repeat').click())
      const first = region(view, 'status').textContent
      await act(async () => getButton(view.container, 'Repeat').click())
      expect(textOf(region(view, 'status'))).toBe('Repeated message.')
      expect(region(view, 'status').textContent).not.toBe(first)
      await act(async () => getButton(view.container, 'Separate channels').click())
      expect(textOf(region(view, 'status'))).toBe('Polite update.')
      expect(textOf(region(view, 'alert'))).toBe('Separate error.')
    } finally { view.cleanup() }
  })
  it('cleans queued messages on goal-key replacement and unmount without stale delivery', async () => {
    let queue: (text: string) => void = () => undefined
    function Capture() { queue = useGoalAnnouncer().announceQueue; return null }
    const render = (key: string) => <GoalAnnouncerProvider key={key}><Capture /></GoalAnnouncerProvider>
    const view = await mountWithTestDom(() => render('goal-1'), act)
    let cleaned = false
    try {
      const pending: VoidFunction[] = []
      const schedule = vi.spyOn(globalThis, 'queueMicrotask').mockImplementation((callback) => { pending.push(callback) })
      queue('Old goal update.')
      schedule.mockRestore()
      await view.rerender(() => render('goal-2'))
      await act(async () => { for (const callback of pending) callback() })
      expect(textOf(region(view, 'status'))).toBe('')
      const afterUnmount: VoidFunction[] = []
      const later = vi.spyOn(globalThis, 'queueMicrotask').mockImplementation((callback) => { afterUnmount.push(callback) })
      queue('Disposed update.')
      later.mockRestore()
      view.cleanup(); cleaned = true
      const errors = vi.spyOn(console, 'error').mockImplementation(() => undefined)
      try { for (const callback of afterUnmount) callback(); expect(errors).not.toHaveBeenCalled() }
      finally { errors.mockRestore() }
    } finally { if (!cleaned) view.cleanup(); vi.restoreAllMocks() }
  })
})
