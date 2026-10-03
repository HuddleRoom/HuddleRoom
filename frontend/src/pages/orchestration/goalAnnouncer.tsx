import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from 'react'

type Announcement = { text: string; seq: number }

interface GoalAnnouncer {
  // True only inside GoalAnnouncerProvider — lets callers (e.g. ErrorRecord)
  // detect whether the shared page-level regions exist, so they can fall
  // back to their own local role="alert" when they don't.
  active: boolean
  announceQueue: (msg: string) => void
  announceError: (msg: string) => void
}

// ponytail: inactive no-op default lets useGoalAnnouncer() be called safely
// from components (e.g. ErrorRecord) that render outside GoalAnnouncerProvider.
const noop = () => {}
const GoalAnnouncerContext = createContext<GoalAnnouncer>({ active: false, announceQueue: noop, announceError: noop })

export function useGoalAnnouncer() {
  return useContext(GoalAnnouncerContext)
}

// The goal-detail page's one polite + one assertive live region. Every
// descendant announces through announceQueue/announceError instead of
// rendering its own aria-live region, so screen-reader users hear exactly
// two channels instead of ~29 scattered ones.
export function GoalAnnouncerProvider({ children }: { children: ReactNode }) {
  const [queueMsg, setQueueMsg] = useState<Announcement>({ text: '', seq: 0 })
  const [errorMsg, setErrorMsg] = useState<Announcement>({ text: '', seq: 0 })
  const announceQueue = useCallback((text: string) => setQueueMsg((prev) => ({ text, seq: prev.seq + 1 })), [])
  const announceError = useCallback((text: string) => setErrorMsg((prev) => ({ text, seq: prev.seq + 1 })), [])
  const value = useMemo(() => ({ active: true, announceQueue, announceError }), [announceQueue, announceError])
  return <GoalAnnouncerContext.Provider value={value}>
    {children}
    {/* Zero-width space toggled by seq parity forces a DOM text change (and
        so a re-announcement) even when the same message repeats. */}
    <div role="status" aria-live="polite" aria-atomic="true" className="sr-only">{queueMsg.text}{queueMsg.seq % 2 === 1 ? '​' : ''}</div>
    <div role="alert" aria-live="assertive" aria-atomic="true" className="sr-only">{errorMsg.text}{errorMsg.seq % 2 === 1 ? '​' : ''}</div>
  </GoalAnnouncerContext.Provider>
}
