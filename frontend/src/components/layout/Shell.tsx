import React, { useEffect, useRef, useState } from 'react'
import { Link, Outlet } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Toaster } from 'sonner'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'
import { ProjectRail } from './ProjectRail'
import { PageFrame } from './PageFrame'
import { useWSQuerySync } from '@/hooks/useWSQuerySync'
import { useAuthStore } from '@/stores/auth'
import { isAuthDisabled } from '@/lib/api-client'
import { apiFetch } from '@/lib/api-client'
import type { Project } from '@/lib/types'
import { useUIStore } from '@/stores/ui'
import { useWSStore } from '@/stores/ws'
import { useAgentResponseStore } from '@/stores/agent-response'
import { UI_COLORS } from '@/components/common/uiPrimitives'
import { Banner } from '@/components/common/Banner'
import './shell.css'

export function Shell({ configLoading = false }: { configLoading?: boolean } = {}) {
  const menuButtonRef = useRef<HTMLButtonElement>(null)
  const stripRef = useRef<HTMLButtonElement>(null)
  const [overlayOpen, setOverlayOpen] = useState(false)

  const activeProjectId = useUIStore((s) => s.activeProjectId)
  const sidebarCollapsed = useUIStore((s) => s.sidebarCollapsed)
  const sidebarOpen = useUIStore((s) => s.sidebarOpen)
  const closeSidebar = useUIStore((s) => s.closeSidebar)

  const handleCloseSidebar = React.useCallback(() => {
    closeSidebar()
    menuButtonRef.current?.focus()
  }, [closeSidebar])
  const connect = useWSStore((s) => s.connect)
  const disconnect = useWSStore((s) => s.disconnect)
  const connected = useWSStore((s) => s.connected)
  const retrying = useWSStore((s) => s.retrying)
  const connecting = useWSStore((s) => s.connecting)
  const gaveUp = useWSStore((s) => s.gaveUp)

  const token = useAuthStore((s) => s.token)
  const hasConnectedOnce = useWSStore((s) => s.hasConnectedOnce)
  const projectsQuery = useQuery({
    queryKey: ['projects'],
    queryFn: () => apiFetch<{ items: Project[] }>('/api/v1/projects').then((result) => result.items),
  })
  const activeProject = projectsQuery.data?.find((project) => project.id === activeProjectId)

  useEffect(() => {
    if (activeProjectId && activeProject) {
      useAgentResponseStore.getState().setProject(activeProjectId, activeProject.name)
      connect(activeProjectId, token)
    } else {
      useAgentResponseStore.getState().setProject(null)
      disconnect()
    }
  }, [activeProjectId, activeProject?.name, token]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    return () => { useWSStore.getState().disconnect() }
  }, [])

  useWSQuerySync()

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape' && !e.defaultPrevented) {
        if (overlayOpen) {
          setOverlayOpen(false)
          stripRef.current?.focus()
        } else if (sidebarOpen) {
          handleCloseSidebar()
        }
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [sidebarOpen, overlayOpen, handleCloseSidebar])

  const [showConnectingBanner, setShowConnectingBanner] = useState(false)
  useEffect(() => {
    if (!connecting || hasConnectedOnce) { setShowConnectingBanner(false); return }
    const timer = setTimeout(() => setShowConnectingBanner(true), 2000)
    return () => clearTimeout(timer)
  }, [connecting, hasConnectedOnce])

  const isReconnecting = !connected && hasConnectedOnce
  const [showReconnectingBanner, setShowReconnectingBanner] = useState(false)
  useEffect(() => {
    if (!isReconnecting || !activeProjectId || (!token && !isAuthDisabled())) { setShowReconnectingBanner(false); return }
    const timer = setTimeout(() => setShowReconnectingBanner(true), 2000)
    return () => clearTimeout(timer)
  }, [isReconnecting, activeProjectId, token])

  const hasAuth = !!token || isAuthDisabled()
  const showConnectingNotice = showConnectingBanner && connecting && !!activeProjectId && hasAuth
  const showPausedNotice = ((showReconnectingBanner && hasAuth) || (retrying && hasAuth)) && !!activeProjectId
  const showLiveNotice = connected && !!activeProjectId && hasAuth && !connecting && !retrying
  const needsWorkspace = projectsQuery.isSuccess && !!activeProjectId && !!activeProject && (activeProject.workspace_path?.trim() === '' || activeProject.workspace_path == null)

  const paneNotice = showConnectingNotice
    ? {
        tone: 'info' as const,
        title: 'Connecting live updates',
        message: 'This pane is waiting for the project stream. Content is not live yet.',
      }
    : showPausedNotice
      ? {
          tone: 'warning' as const,
          title: 'Live updates paused',
          message: 'Reconnecting automatically. Content in this pane may be stale until the stream reconnects.',
        }
      : null

  const lifecycleAnnouncement = useAgentResponseStore((s) => s.lifecycleAnnouncement)

  // Clear lifecycle announcement after it's been announced to prevent re-announcement on connectivity changes
  useEffect(() => {
    if (lifecycleAnnouncement) {
      const timer = setTimeout(() => {
        useAgentResponseStore.getState().clearLifecycleAnnouncement()
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [lifecycleAnnouncement])

  const connectivityAnnouncement = showConnectingNotice
    ? 'Live updates are connecting. This pane is not receiving live project data yet.'
    : showPausedNotice
      ? 'Live updates are paused while the connection retries automatically. Content in this pane may be stale until reconnection succeeds.'
      : showLiveNotice
        ? 'Live updates are active. This pane is receiving current project data.'
        : ''

  const liveRegionText = [connectivityAnnouncement, lifecycleAnnouncement]
    .filter(Boolean)
    .join(' ')

  return (
    <>
    <div
      className="huddleroom-shell"
      style={{
        '--huddleroom-text': UI_COLORS.textPrimary,
        '--huddleroom-bg': UI_COLORS.appBg,
        '--huddleroom-accent': UI_COLORS.primary,
        '--huddleroom-sidebar-width': sidebarCollapsed ? '4rem' : '15rem',
        backgroundColor: UI_COLORS.appBg,
      } as React.CSSProperties}
    >
      {/* Single persistent atomic polite live region combining connectivity + project + lifecycle */}
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {liveRegionText}
      </div>

      {sidebarOpen && <button type="button" className="huddleroom-sidebar-scrim" aria-label="Close sidebar menu" onClick={handleCloseSidebar} />}
      <Sidebar />
      <div className="huddleroom-shell-main">
        <TopBar menuButtonRef={menuButtonRef} />
        <div className="huddleroom-shell-body">
          <main className="huddleroom-shell-content">
            {configLoading ? (
              <PageFrame>
                <Banner variant="info" title="Connecting to HuddleRoom…" />
              </PageFrame>
            ) : (
              <PageFrame>
                {gaveUp && !!activeProjectId ? (
                  <Banner
                    variant="offline"
                    title="Live updates stopped"
                    message="This pane could not reconnect after several attempts. Content may be stale until you reconnect."
                    action={{ label: 'Reconnect', onClick: () => useWSStore.getState().manualReconnect() }}
                  />
                ) : paneNotice && (
                  <Banner variant={paneNotice.tone} title={paneNotice.title} message={paneNotice.message} />
                )}
                {needsWorkspace && (
                  <Banner
                    variant="warning"
                    title="Server directory required"
                    message={<>Execution is not enabled for this project until a server directory is configured. <Link to="/settings">Open project settings</Link></>}
                  />
                )}
                <React.Suspense fallback={<div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 300, color: UI_COLORS.textMuted, fontSize: 13 }}>Loading…</div>}>
                  <Outlet />
                </React.Suspense>
              </PageFrame>
            )}
          </main>

          {/* Rail (Ask|Calls tabs): persistent at >=1280px (always visible), strip+overlay below 1280px */}
          <div className="agent-activity-panel-container">
            {activeProjectId && <ProjectRail projectId={activeProjectId} />}
          </div>
        </div>

        {/* Activity Strip and Overlay for medium breakpoint */}
        <div className="agent-activity-strip-container">
          <button
            ref={stripRef}
            className="agent-activity-strip"
            aria-label="Activity"
            onClick={() => setOverlayOpen(!overlayOpen)}
            title="Activity"
          >
            Activity
          </button>
        </div>

        {overlayOpen && (
          <div className="agent-activity-overlay">
            {activeProjectId && <ProjectRail projectId={activeProjectId} />}
          </div>
        )}
      </div>
    </div>
      <Toaster
        theme="light"
        toastOptions={{
          style: {
            background: UI_COLORS.surface,
            border: `1px solid ${UI_COLORS.border}`,
            color: UI_COLORS.textPrimary,
            fontFamily: "'Inter', system-ui, sans-serif",
            fontSize: '13px',
          },
        }}
      />
    </>
  )
}
