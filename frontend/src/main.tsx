import React from 'react'
import ReactDOM from 'react-dom/client'

// ResizeObserver fires this benign warning when Monaco/ReactFlow resize cycles
// overlap within a single frame — no actual functionality is affected.
// Capture-phase listener runs before Vite's overlay listener and stops propagation.
window.addEventListener('error', (e) => {
  if (e.message?.includes('ResizeObserver loop')) {
    e.stopImmediatePropagation()
    e.preventDefault()
  }
}, true)
const _origError = window.onerror
window.onerror = (msg, ...rest) => {
  if (typeof msg === 'string' && msg.includes('ResizeObserver loop')) return true
  return _origError ? _origError.call(window, msg, ...rest) : false
}
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import { loadRuntimeConfig } from './lib/api-client'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 30_000, retry: 1 },
  },
})

// Fire-and-forget: the promise is memoized, RequireAuth also calls this and awaits it.
loadRuntimeConfig()

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
)
