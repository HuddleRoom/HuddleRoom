import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'

export const UI_STORAGE_KEY = 'huddleroom-ui'
export const LEGACY_UI_STORAGE_KEY = 'rally-ui'

export function migrateLegacyUIStorage(storage: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>): string | null {
  const current = storage.getItem(UI_STORAGE_KEY)
  if (current !== null) return current
  const legacy = storage.getItem(LEGACY_UI_STORAGE_KEY)
  if (legacy !== null) {
    storage.setItem(UI_STORAGE_KEY, legacy)
    storage.removeItem(LEGACY_UI_STORAGE_KEY)
  }
  return legacy
}

interface UIState {
  sidebarCollapsed: boolean
  sidebarOpen: boolean
  activeProjectId: string | null
  createProjectOpen: boolean
  toggleSidebar: () => void
  openSidebar: () => void
  closeSidebar: () => void
  setSidebarCollapsed: (v: boolean) => void
  setActiveProject: (id: string | null) => void
  setCreateProjectOpen: (v: boolean) => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      sidebarOpen: false,
      activeProjectId: null,
      createProjectOpen: false,
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      openSidebar: () => set({ sidebarOpen: true }),
      closeSidebar: () => set({ sidebarOpen: false }),
      setSidebarCollapsed: (v) => set({ sidebarCollapsed: v }),
      setActiveProject: (id) => set({ activeProjectId: id }),
      setCreateProjectOpen: (v) => set({ createProjectOpen: v }),
    }),
    {
      name: UI_STORAGE_KEY,
      storage: createJSONStorage(() => ({
        getItem: (name) => name === UI_STORAGE_KEY
          ? migrateLegacyUIStorage(localStorage)
          : localStorage.getItem(name),
        setItem: (name, value) => localStorage.setItem(name, value),
        removeItem: (name) => localStorage.removeItem(name),
      })),
      version: 1,
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
        activeProjectId: state.activeProjectId,
      }) as unknown as UIState,
      migrate: (persistedState: unknown, version: number) => {
        if (version === 0) {
          const state = persistedState as Record<string, unknown>
          return {
            sidebarCollapsed: typeof state.sidebarCollapsed === 'boolean' ? state.sidebarCollapsed : false,
            sidebarOpen: false,
            activeProjectId: typeof state.activeProjectId === 'string' ? state.activeProjectId : null,
          }
        }
        const s = persistedState as Record<string, unknown>
        return {
          sidebarCollapsed: typeof s?.sidebarCollapsed === 'boolean' ? s.sidebarCollapsed : false,
          sidebarOpen: false,
          activeProjectId: typeof s?.activeProjectId === 'string' ? s.activeProjectId : null,
        }
      },
    },
  ),
)
