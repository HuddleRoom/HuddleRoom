import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useUIStore } from '@/stores/ui'
import {
  useProjectMemory,
  useSearchProjectMemory,
  useDeleteProjectMemory,
  useGlobalMemory,
  useDeleteGlobalMemory,
} from '@/api/memory'
import { apiFetch } from '@/lib/api-client'
import { toast } from 'sonner'
import { Search, Trash2, Filter } from 'lucide-react'
import type { MemoryItem, User } from '@/lib/types'
import { Button, Input, QueryState, UI_COLORS, PageHeader, ConfirmDialog } from '@/components/common/uiPrimitives'
import { Tabs, TabPanel } from '@/components/common/Tabs'
import { EmptyState } from '@/components/common/EmptyState'
import { Tag } from '@/components/common/Tag'
import { STATUS_COLORS } from '@/lib/statusColors'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'
import { dateHeading } from '@/lib/time'

const MONO = 'ui-monospace, Menlo, Monaco, monospace'

export function MemoryPage() {
  useDocumentTitle('Memory')
  const activeProjectId = useUIStore((s) => s.activeProjectId)
  const [tab, setTab] = useState<'project' | 'global'>('project')
  const [agentFilter, setAgentFilter] = useState('')
  const [sharedFilter, setSharedFilter] = useState<boolean | undefined>()
  const [searchQuery, setSearchQuery] = useState('')
  const [searchMode, setSearchMode] = useState(false)
  const [searchResults, setSearchResults] = useState<MemoryItem[]>([])
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  // Fetch current user for admin check
  const { data: user } = useQuery({
    queryKey: ['me'],
    queryFn: () => apiFetch<User>('/api/v1/auth/me'),
  })

  // Project memory queries
  const projectQuery = useProjectMemory(
    activeProjectId && tab === 'project' ? activeProjectId : null,
    { agent_id: agentFilter || undefined, shared: sharedFilter }
  )

  const searchProjectMutation = useSearchProjectMemory(activeProjectId)
  const deleteProjectMutation = useDeleteProjectMemory(activeProjectId)

  // Global memory queries
  const globalQuery = useGlobalMemory()
  const deleteGlobalMutation = useDeleteGlobalMemory()

  const handleProjectSearch = async () => {
    if (!searchQuery.trim()) {
      setSearchMode(false)
      return
    }
    try {
      const results = await searchProjectMutation.mutateAsync({ query: searchQuery })
      setSearchResults(results)
      setSearchMode(true)
    } catch (error) {
      toast.error('Search failed')
    }
  }

  const handleDelete = async () => {
    if (!deleteConfirm) return
    try {
      if (tab === 'project') {
        await deleteProjectMutation.mutateAsync(deleteConfirm)
      } else {
        await deleteGlobalMutation.mutateAsync(deleteConfirm)
      }
      toast.success('Memory deleted')
      setDeleteConfirm(null)
    } catch (error) {
      toast.error('Delete failed')
    }
  }

  const isAdmin = user?.role === 'admin'

  return (
    <div>
      <PageHeader title="Memory" />

      {/* Tabs */}
      <Tabs
        idPrefix="memory"
        activeId={tab}
        onChange={(id) => {
          setTab(id as 'project' | 'global')
          setSearchMode(false)
          setDeleteConfirm(null)
        }}
        className="mb-5"
        tabs={[
          { id: 'project', label: 'Project' },
          { id: 'global', label: 'Global' },
        ]}
      />

      {/* Project Tab */}
      <TabPanel tabId="project" activeId={tab} idPrefix="memory">
        <div>
          {/* Filter bar */}
          <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            <div style={{ flex: '0 0 auto' }}>
              <Input
                label="Agent ID"
                placeholder="Filter by agent ID"
                value={agentFilter}
                onChange={(e) => setAgentFilter(e.target.value)}
                style={{ width: 200 }}
              />
            </div>
            <div style={{ flex: '0 0 auto' }}>
              <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer', minHeight: 44, paddingTop: 10, paddingBottom: 10 }}>
                <input
                  type="checkbox"
                  aria-label="Show shared memory only"
                  checked={sharedFilter === true}
                  onChange={(e) => setSharedFilter(e.target.checked ? true : undefined)}
                  style={{ width: 16, height: 16, cursor: 'pointer' }}
                />
                <span style={{ color: UI_COLORS.textMuted, fontSize: 11, marginBottom: 0 }}>Shared only</span>
              </label>
            </div>
            <div style={{ flex: '0 0 auto' }}>
              <Input
                label="Search"
                placeholder="Search memories..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                disabled={!activeProjectId}
                style={{ width: 200, opacity: activeProjectId ? 1 : 0.4 }}
              />
            </div>
            <Button
              variant="primary"
              size="sm"
              onClick={handleProjectSearch}
              style={{ opacity: activeProjectId ? 1 : 0.4, display: 'flex', alignItems: 'center', gap: 5 }}
              disabled={!activeProjectId}
            >
              <Search size={14} />
              Search
            </Button>
          </div>

          {/* Note */}
          <p className="text-xs text-huddleroom-text-muted mb-4 mt-0">
            Memories are written by agents. This view is read-only.
          </p>

          {/* Memory list */}
          {!activeProjectId ? (
            <p style={{ color: UI_COLORS.textMuted, fontSize: 12, fontFamily: MONO }}>Select a project to view memories.</p>
          ) : (
            <QueryState
              query={{
                isLoading: projectQuery.isLoading || (searchMode && searchProjectMutation.isPending),
                isError: projectQuery.isError || (searchMode && searchProjectMutation.isError),
                data: searchMode ? { items: searchResults } : projectQuery.data,
                refetch: searchMode ? undefined : projectQuery.refetch,
              }}
              skeleton="list"
              skeletonCount={4}
              errorLabel="Failed to load memories"
              emptyLabel={searchMode ? 'No search results' : 'No memory entries'}
              emptyDetail={searchMode ? undefined : 'Agent memory appears here as sessions accumulate context.'}
            >
              {(data) => (
                (data.items ?? []).length === 0 ? (
                  <EmptyState
                    title={searchMode ? 'No search results' : 'No memory entries'}
                    body={searchMode ? 'Try a different search term.' : 'Agent memory appears here as sessions accumulate context.'}
                  />
                ) : (
                <>
                  {(data.items ?? []).map((item) => (
                    <div
                      key={item.id}
                      data-testid={`memory-item-${item.id}`}
                      style={{
                        background: UI_COLORS.surface,
                        border: `1px solid ${UI_COLORS.border}`,
                        borderRadius: 4,
                        padding: 12,
                        marginBottom: 12,
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'flex-start',
                      }}
                    >
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <p style={{ color: UI_COLORS.textPrimary, fontSize: 12, fontFamily: "'Inter', system-ui, sans-serif", marginTop: 0, marginBottom: 8, wordBreak: 'break-word' }}>
                          {item.content.slice(0, 120)}
                          {item.content.length > 120 ? '...' : ''}
                        </p>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
                          {item.agent_id && (
                            <span style={{ color: STATUS_COLORS.blue, fontSize: 11, fontFamily: MONO }}>
                              agent: {item.agent_id.slice(0, 8)}
                            </span>
                          )}
                          {item.shared && (
                            <span
                              style={{
                                background: UI_COLORS.border,
                                color: STATUS_COLORS.blue,
                                borderRadius: 3,
                                padding: '1px 6px',
                                fontSize: 11,
                                fontFamily: MONO,
                              }}
                            >
                              SHARED
                            </span>
                          )}
                          {item.tags?.map((tag, idx) => (
                            <Tag key={idx}>{tag}</Tag>
                          ))}
                          <span style={{ color: item.scope === 'global' ? STATUS_COLORS.neutral : STATUS_COLORS.blue, fontSize: 11, fontFamily: MONO }}>
                            {item.scope}
                          </span>
                        </div>
                        <p style={{ color: UI_COLORS.textMuted, fontSize: 11, fontFamily: MONO, marginTop: 0, marginBottom: 0 }}>
                          {dateHeading(item.created_at)}
                        </p>
                      </div>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => setDeleteConfirm(item.id)}
                        style={{ display: 'flex', alignItems: 'center', gap: 5 }}
                      >
                        <Trash2 size={14} />
                        Delete
                      </Button>
                    </div>
                  ))}
                </>
                )
              )}
            </QueryState>
          )}
        </div>
      </TabPanel>

      {/* Global Tab */}
      <TabPanel tabId="global" activeId={tab} idPrefix="memory">
        <div>
          {!isAdmin && (
            <p className="text-xs text-huddleroom-text-muted mb-4 mt-0">
              Global memories are read-only for non-admin users.
            </p>
          )}

          <QueryState
            query={{
              isLoading: globalQuery.isLoading,
              isError: globalQuery.isError,
              data: globalQuery.data,
              refetch: globalQuery.refetch,
            }}
            skeleton="list"
            skeletonCount={4}
            errorLabel="Failed to load memories"
            emptyLabel="No memory entries"
            emptyDetail="Agent memory appears here as sessions accumulate context."
          >
            {(data) => (
              (data.items ?? []).length === 0 ? (
                <EmptyState
                  title="No memory entries"
                  body="Agent memory appears here as sessions accumulate context."
                />
              ) : (
              <>
                {(data.items ?? []).map((item) => (
                  <div
                    key={item.id}
                    data-testid={`memory-item-${item.id}`}
                    style={{
                      background: UI_COLORS.surface,
                      border: `1px solid ${UI_COLORS.border}`,
                      borderRadius: 4,
                      padding: 12,
                      marginBottom: 12,
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'flex-start',
                    }}
                  >
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <p style={{ color: UI_COLORS.textPrimary, fontSize: 12, fontFamily: "'Inter', system-ui, sans-serif", marginTop: 0, marginBottom: 8, wordBreak: 'break-word' }}>
                        {item.content.slice(0, 120)}
                        {item.content.length > 120 ? '...' : ''}
                      </p>
                      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
                        {item.agent_id && (
                          <span style={{ color: STATUS_COLORS.blue, fontSize: 11, fontFamily: MONO }}>
                            agent: {item.agent_id.slice(0, 8)}
                          </span>
                        )}
                        {item.shared && (
                          <span
                            style={{
                              background: UI_COLORS.border,
                              color: STATUS_COLORS.blue,
                              borderRadius: 3,
                              padding: '1px 6px',
                              fontSize: 11,
                              fontFamily: MONO,
                            }}
                          >
                            SHARED
                          </span>
                        )}
                        {item.tags?.map((tag, idx) => (
                          <Tag key={idx}>{tag}</Tag>
                        ))}
                        <span style={{ color: item.scope === 'global' ? STATUS_COLORS.neutral : STATUS_COLORS.blue, fontSize: 11, fontFamily: MONO }}>
                          {item.scope}
                        </span>
                      </div>
                      <p style={{ color: UI_COLORS.textMuted, fontSize: 11, fontFamily: MONO, marginTop: 0, marginBottom: 0 }}>
                        {dateHeading(item.created_at)}
                      </p>
                    </div>
                    {isAdmin && (
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => setDeleteConfirm(item.id)}
                        style={{ display: 'flex', alignItems: 'center', gap: 5 }}
                      >
                        <Trash2 size={14} />
                        Delete
                      </Button>
                    )}
                  </div>
                ))}
              </>
              )
            )}
          </QueryState>
        </div>
      </TabPanel>

      <ConfirmDialog
        open={deleteConfirm !== null}
        onOpenChange={(open) => { if (!open) setDeleteConfirm(null) }}
        title="Delete this memory?"
        consequence="This permanently removes the memory entry and cannot be undone."
        confirmLabel="Delete"
        onConfirm={handleDelete}
        isPending={deleteProjectMutation.isPending || deleteGlobalMutation.isPending}
      />
    </div>
  )
}
