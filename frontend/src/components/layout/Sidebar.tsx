import { useEffect, useRef, useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import {
  Bot,
  BookOpen,
  Brain,
  CheckSquare,
  ChevronLeft,
  ChevronRight,
  Filter,
  GitBranch,
  LayoutDashboard,
  Settings,
  Target,
  TrendingUp,
  Users,
  Zap,
} from 'lucide-react'
import { UI_COLORS, UI_FONT_FAMILY } from '@/components/common/uiPrimitives'
import { useUIStore } from '@/stores/ui'

const NAV_OPERATE = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/orchestration', icon: Target, label: 'Orchestration' },
  { to: '/tasks', icon: CheckSquare, label: 'Tasks' },
  { to: '/agents', icon: Bot, label: 'Agents' },
  { to: '/graphs', icon: GitBranch, label: 'Graphs' },
  { to: '/meetings', icon: Users, label: 'Meetings' },
] as const

const NAV_CONFIGURE = [
  { to: '/knowledge', icon: BookOpen, label: 'Knowledge' },
  { to: '/memory', icon: Brain, label: 'Memory' },
  { to: '/rules', icon: Filter, label: 'Rules' },
  { to: '/hooks', icon: Zap, label: 'Hooks' },
  { to: '/optimizations', icon: TrendingUp, label: 'Optimizations' },
  { to: '/settings', icon: Settings, label: 'Settings' },
] as const

export function Sidebar() {
  const collapsed = useUIStore((s) => s.sidebarCollapsed)
  const sidebarOpen = useUIStore((s) => s.sidebarOpen)
  const closeSidebar = useUIStore((s) => s.closeSidebar)
  const toggle = useUIStore((s) => s.toggleSidebar)
  const location = useLocation()
  const [isMobileViewport, setIsMobileViewport] = useState(() =>
    typeof window !== 'undefined' ? window.matchMedia('(max-width: 767px)').matches : false,
  )

  useEffect(() => {
    if (typeof window === 'undefined') {
      return undefined
    }

    const mediaQuery = window.matchMedia('(max-width: 767px)')
    const syncViewport = () => {
      setIsMobileViewport(mediaQuery.matches)
    }

    syncViewport()
    mediaQuery.addEventListener('change', syncViewport)

    return () => {
      mediaQuery.removeEventListener('change', syncViewport)
    }
  }, [])

  const asideRef = useRef<HTMLElement>(null)

  // Focus trap for mobile sidebar
  useEffect(() => {
    if (!sidebarOpen || !isMobileViewport || !asideRef.current) return

    const aside = asideRef.current
    const focusables = Array.from(aside.querySelectorAll('a[href], button:not([disabled])'))
    if (focusables.length === 0) return

    ;(focusables[0] as HTMLElement).focus()

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Tab') return
      const first = focusables[0] as HTMLElement
      const last = focusables[focusables.length - 1] as HTMLElement
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }

    aside.addEventListener('keydown', handleKeyDown)
    return () => aside.removeEventListener('keydown', handleKeyDown)
  }, [sidebarOpen, isMobileViewport])

  const isSidebarInert = isMobileViewport && !sidebarOpen
  const border = UI_COLORS.sidebarSurface

  const renderNavItem = ({ to, icon: Icon, label }: { to: string; icon: any; label: string }) => {
    const isActive = to === '/'
      ? location.pathname === '/'
      : location.pathname.startsWith(to)

    return (
      <NavLink
        key={to}
        to={to}
        aria-label={label}
        aria-current={isActive ? 'page' : undefined}
        title={collapsed ? label : undefined}
        onClick={() => { if (sidebarOpen) closeSidebar() }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: collapsed ? '8px 20px' : '8px 16px',
          fontSize: 13,
          fontFamily: UI_FONT_FAMILY,
          textDecoration: 'none',
          color: isActive ? UI_COLORS.sidebarActive : UI_COLORS.sidebarText,
          backgroundColor: isActive ? UI_COLORS.sidebarActiveBg : 'transparent',
          borderLeft: isActive ? `2px solid ${UI_COLORS.primary}` : '2px solid transparent',
          transition: 'color 120ms cubic-bezier(0.4, 0, 0.2, 1), border-color 120ms cubic-bezier(0.4, 0, 0.2, 1), background-color 120ms cubic-bezier(0.4, 0, 0.2, 1)',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
        }}
        onMouseEnter={(e) => {
          const el = e.currentTarget
          if (!isActive) {
            el.style.backgroundColor = UI_COLORS.sidebarSurface
            el.style.color = UI_COLORS.sidebarHoverText
          }
        }}
        onMouseLeave={(e) => {
          const el = e.currentTarget
          if (!isActive) {
            el.style.backgroundColor = 'transparent'
            el.style.color = UI_COLORS.sidebarText
          }
        }}
      >
        <Icon size={16} style={{ flexShrink: 0 }} />
        {!collapsed && (
          <span
            style={{
              minWidth: 0,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
          >
            {label}
          </span>
        )}
      </NavLink>
    )
  }

  return (
    <aside
      ref={asideRef}
      id="huddleroom-sidebar"
      className={`huddleroom-sidebar ${sidebarOpen ? 'huddleroom-sidebar-open' : ''}`}
      inert={isSidebarInert || undefined}
      style={{
        width: '100%',
        backgroundColor: UI_COLORS.sidebarBg,
        borderRight: `1px solid ${border}`,
        transition: 'transform 200ms cubic-bezier(0.4, 0, 0.2, 1)',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          height: 48,
          display: 'flex',
          alignItems: 'center',
          padding: collapsed ? '0 20px' : '0 16px',
          borderBottom: `1px solid ${border}`,
          flexShrink: 0,
        }}
      >
        {collapsed ? (
          <img src="/dashboard/favicon.png" alt="HuddleRoom" style={{ height: 24, width: 24 }} />
        ) : (
          <img src="/dashboard/huddleroom-logo-light.png" alt="HuddleRoom" style={{ height: 28, width: 'auto' }} />
        )}
      </div>

      <nav aria-label="Main navigation" style={{ flex: 1, padding: '8px 0', overflowY: 'auto', overflowX: 'hidden' }}>
        {!collapsed && (
          <div aria-hidden="true" style={{ padding: '8px 16px 4px', fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' as const, color: UI_COLORS.sidebarGroupLabel, fontFamily: UI_FONT_FAMILY }}>
            Operate
          </div>
        )}
        {NAV_OPERATE.map(renderNavItem)}
        {/* Divider */}
        <div
          aria-hidden="true"
          style={{
            margin: '6px 16px',
            borderTop: `1px solid ${UI_COLORS.sidebarSurface}`,
            display: collapsed ? 'none' : 'block',
          }}
        />
        {!collapsed && (
          <div aria-hidden="true" style={{ padding: '8px 16px 4px', fontSize: 11, fontWeight: 600, letterSpacing: '0.1em', textTransform: 'uppercase' as const, color: UI_COLORS.sidebarGroupLabel, fontFamily: UI_FONT_FAMILY }}>
            Configure
          </div>
        )}
        {NAV_CONFIGURE.map(renderNavItem)}
      </nav>

      <button
        className="huddleroom-sidebar-collapse"
        onClick={toggle}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 44,
          background: 'none',
          border: 'none',
          borderTop: `1px solid ${border}`,
          color: UI_COLORS.sidebarMuted,
          cursor: 'pointer',
          transition: 'color 120ms cubic-bezier(0.4, 0, 0.2, 1), background-color 120ms cubic-bezier(0.4, 0, 0.2, 1)',
        }}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
      </button>
    </aside>
  )
}
