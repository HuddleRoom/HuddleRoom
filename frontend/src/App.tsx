import React from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { ErrorBoundary } from '@/components/common/ErrorBoundary'
import { Shell } from '@/components/layout/Shell'
import { isAuthDisabled, loadRuntimeConfig } from '@/lib/api-client'
import { useAuthStore } from '@/stores/auth'
import { UI_COLORS } from '@/components/common/uiPrimitives'

const AgentsPage = React.lazy(() => import('@/pages/agents/AgentsPage').then(m => ({ default: m.AgentsPage })))
const DashboardPage = React.lazy(() => import('@/pages/dashboard/DashboardPage').then(m => ({ default: m.DashboardPage })))
const HooksPage = React.lazy(() => import('@/pages/hooks/HooksPage').then(m => ({ default: m.HooksPage })))
const KnowledgePage = React.lazy(() => import('@/pages/knowledge/KnowledgePage').then(m => ({ default: m.KnowledgePage })))
const LoginPage = React.lazy(() => import('@/pages/login/LoginPage').then(m => ({ default: m.LoginPage })))
const MeetingsPage = React.lazy(() => import('@/pages/meetings/MeetingsPage').then(m => ({ default: m.MeetingsPage })))
const NewMeetingPage = React.lazy(() => import('@/pages/meetings/NewMeetingPage').then(m => ({ default: m.NewMeetingPage })))
const MemoryPage = React.lazy(() => import('@/pages/memory/MemoryPage').then(m => ({ default: m.MemoryPage })))
const OptimizationsPage = React.lazy(() => import('@/pages/optimizations/OptimizationsPage').then(m => ({ default: m.OptimizationsPage })))
const OrchestrationPage = React.lazy(() => import('@/pages/orchestration/OrchestrationPage').then(m => ({ default: m.OrchestrationPage })))
const ProtocolsPage = React.lazy(() => import('@/pages/protocols/ProtocolsPage').then(m => ({ default: m.ProtocolsPage })))
const RulesPage = React.lazy(() => import('@/pages/rules/RulesPage').then(m => ({ default: m.RulesPage })))
const SettingsPage = React.lazy(() => import('@/pages/settings/SettingsPage').then(m => ({ default: m.SettingsPage })))
const TasksPage = React.lazy(() => import('@/pages/tasks/TasksPage').then(m => ({ default: m.TasksPage })))

export const PAGE_LOADING_FALLBACK = (
  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', color: UI_COLORS.textMuted, fontSize: 13 }}>
    Loading…
  </div>
)

export function RequireAuth() {
  const token = useAuthStore((s) => s.token)
  const [configResolved, setConfigResolved] = React.useState(false)

  React.useEffect(() => {
    let cancelled = false
    // loadRuntimeConfig() is memoized; safe to call again here even if
    // main.tsx already kicked it off.
    loadRuntimeConfig().then(() => {
      if (!cancelled) setConfigResolved(true)
    })
    return () => { cancelled = true }
  }, [])

  if (!configResolved) {
    return <Shell configLoading />
  }

  if (!isAuthDisabled() && !token) {
    return <Navigate to="/login" replace />
  }

  return <Shell />
}

export function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter basename="/dashboard">
        <Routes>
          <Route path="/login" element={<React.Suspense fallback={PAGE_LOADING_FALLBACK}><LoginPage /></React.Suspense>} />
          <Route path="/" element={<RequireAuth />}>
            <Route index element={<DashboardPage />} />
            <Route path="orchestration" element={<OrchestrationPage />} />
            <Route path="orchestration/:goalId" element={<OrchestrationPage />} />
            <Route path="tasks" element={<TasksPage />} />
            <Route path="agents" element={<AgentsPage />} />
            <Route path="agents/:agentId" element={<AgentsPage />} />
            <Route path="meetings" element={<MeetingsPage />} />
            <Route path="meetings/new" element={<NewMeetingPage />} />
            <Route path="meetings/:meetingId" element={<MeetingsPage />} />
            <Route path="protocols" element={<ProtocolsPage />} />
            <Route path="protocols/:protocolId" element={<ProtocolsPage />} />
            <Route path="knowledge" element={<KnowledgePage />} />
            <Route path="memory" element={<MemoryPage />} />
            <Route path="rules" element={<RulesPage />} />
            <Route path="hooks" element={<HooksPage />} />
            <Route path="optimizations" element={<OptimizationsPage />} />
            <Route path="settings" element={<SettingsPage />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
