import React, { useState } from 'react'
import * as RadixDialog from '@radix-ui/react-dialog'
import { toast } from 'sonner'
import { Dialog } from '@/components/common/Dialog'
import { Plus, Search, X, Trash2, Eye } from 'lucide-react'
import {
  UI_COLORS,
  UI_FONT_FAMILY,
  Button,
  Card,
  Input,
  Textarea,
  Select,
  QueryState,
  PageHeader,
  ConfirmDialog,
  SectionLabel,
} from '@/components/common/uiPrimitives'
import { EmptyState } from '@/components/common/EmptyState'
import { Tag } from '@/components/common/Tag'
import { STATUS_COLORS } from '@/lib/statusColors'
import { useUIStore } from '@/stores/ui'
import {
  useKnowledge,
  useSearchKnowledge,
  useCreateKnowledge,
  useDeleteKnowledge,
} from '@/api/knowledge'
import type { KnowledgeItem, KnowledgeSearchResult, ContentType } from '@/lib/types'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'
import { dateHeading } from '@/lib/time'

// ─── Badge for content type ────────────────────────────────────────────────────

function ContentTypeBadge({ type }: { type: ContentType }) {
  const colors: Record<ContentType, string> = {
    text: UI_COLORS.textMuted,
    markdown: STATUS_COLORS.green,
    code: STATUS_COLORS.amber,
    json: STATUS_COLORS.neutral,
  }

  return (
    <span
      style={{
        display: 'inline-block',
        padding: '2px 6px',
        fontSize: 11,
        fontFamily: UI_FONT_FAMILY,
        color: colors[type],
        border: `1px solid ${colors[type]}`,
        borderRadius: 2,
        textTransform: 'uppercase',
        letterSpacing: '0.5px',
      }}
    >
      {type}
    </span>
  )
}

// ─── Create Item Modal ─────────────────────────────────────────────────────────

function CreateModal({
  open,
  onClose,
  projectId,
}: {
  open: boolean
  onClose: () => void
  projectId: string | null
}) {
  const [content, setContent] = useState('')
  const [contentType, setContentType] = useState<ContentType>('text')
  const [title, setTitle] = useState('')
  const [tags, setTags] = useState('')
  const [provenance, setProvenance] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const mutation = useCreateKnowledge(projectId)

  const handleCreate = async () => {
    if (!content.trim()) {
      toast.error('Content is required')
      return
    }

    setIsLoading(true)
    try {
      const tagsArray = tags
        .split(',')
        .map((t) => t.trim())
        .filter((t) => t.length > 0)

      await mutation.mutateAsync({
        content,
        content_type: contentType,
        title: title.trim() || undefined,
        tags: tagsArray,
        provenance: provenance.trim() || undefined,
      })

      toast.success('Knowledge item created')
      setContent('')
      setTitle('')
      setTags('')
      setProvenance('')
      setContentType('text')
      onClose()
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to create knowledge item'
      toast.error(message)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => { if (!v) onClose() }}
      title="New knowledge item"
      description="Create or edit a knowledge entry."
      size="md"
      footer={{
        primaryLabel: isLoading ? 'Creating...' : 'Create',
        primaryType: 'submit',
        formId: 'create-knowledge-form',
        isPending: isLoading,
      }}
    >
      <form id="create-knowledge-form" onSubmit={(e) => { e.preventDefault(); handleCreate() }} style={{ display: 'flex', flexDirection: 'column', gap: 0, fontFamily: UI_FONT_FAMILY }}>
        <Textarea
          label="Content *"
          value={content}
          onChange={(e) => setContent(e.target.value)}
          className="min-h-30"
          placeholder="Enter knowledge content..."
        />

        <Select
          label="Content Type *"
          value={contentType}
          onChange={(e) => setContentType(e.target.value as ContentType)}
        >
          <option value="text">Text</option>
          <option value="markdown">Markdown</option>
          <option value="code">Code</option>
          <option value="json">JSON</option>
        </Select>

        <Input
          label="Title (optional)"
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Optional title"
        />

        <Input
          label="Tags (optional)"
          type="text"
          value={tags}
          onChange={(e) => setTags(e.target.value)}
          placeholder="comma-separated tags"
        />

        <Input
          label="Provenance (optional)"
          type="text"
          value={provenance}
          onChange={(e) => setProvenance(e.target.value)}
          placeholder="Source or reference"
        />
      </form>
    </Dialog>
  )
}

// ─── Detail Panel ──────────────────────────────────────────────────────────────

function DetailPanel({
  item,
  onClose,
}: {
  item: KnowledgeItem | KnowledgeSearchResult | null
  onClose: () => void
}) {
  const createdDate =
    item && 'created_at' in item
      ? dateHeading(item.created_at)
      : null

  return (
    <RadixDialog.Root open={item !== null} onOpenChange={(open) => { if (!open) onClose() }}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.4)',
            zIndex: 40,
          }}
        />
        <RadixDialog.Content
          data-testid="knowledge-detail"
          aria-describedby="detail-panel-desc"
          style={{
            position: 'fixed',
            top: 0,
            right: 0,
            bottom: 0,
            width: 480,
            maxWidth: '90vw',
            background: UI_COLORS.surface,
            borderLeft: `1px solid ${UI_COLORS.border}`,
            overflowY: 'auto',
            zIndex: 50,
            padding: 24,
            fontFamily: UI_FONT_FAMILY,
          }}
        >
          <RadixDialog.Title
            style={{
              color: UI_COLORS.primary,
              fontSize: 16,
              fontWeight: 400,
              marginBottom: 20,
              overflowWrap: 'anywhere',
            }}
          >
            {item?.title || 'Knowledge item'}
          </RadixDialog.Title>
          <RadixDialog.Description id="detail-panel-desc" style={{ display: 'none' }}>
            Knowledge item details
          </RadixDialog.Description>

          {item && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <SectionLabel>Content type</SectionLabel>
                <ContentTypeBadge type={item.content_type} />
              </div>

              <div>
                <SectionLabel>Content</SectionLabel>
                <Card padding="sm" style={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', fontSize: 13, color: UI_COLORS.textPrimary, maxHeight: 300, overflowY: 'auto' }}>
                  {item.content}
                </Card>
              </div>

              {item.tags && item.tags.length > 0 && (
                <div>
                  <SectionLabel>Tags</SectionLabel>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {item.tags.map((tag) => (
                      <Tag key={tag}>{tag}</Tag>
                    ))}
                  </div>
                </div>
              )}

              {'provenance' in item && item.provenance && (
                <div>
                  <SectionLabel>Provenance</SectionLabel>
                  <Card padding="sm" style={{ fontSize: 12, color: UI_COLORS.textPrimary, whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>
                    {item.provenance}
                  </Card>
                </div>
              )}

              {'relevance_score' in item && (
                <div>
                  <SectionLabel>Relevance</SectionLabel>
                  <div className="text-huddleroom-text-secondary" style={{ fontSize: 12, fontFamily: UI_FONT_FAMILY }}>
                    {Math.round(item.relevance_score * 100)}%
                  </div>
                </div>
              )}
              {createdDate && (
                <div>
                  <SectionLabel>Created</SectionLabel>
                  <div
                    style={{
                      fontSize: 12,
                      color: UI_COLORS.textPrimary,
                      fontFamily: UI_FONT_FAMILY,
                    }}
                  >
                    {createdDate}
                  </div>
                </div>
              )}

              <Button
                variant="secondary"
                onClick={onClose}
              >
                Close
              </Button>
            </div>
          )}

          <RadixDialog.Close asChild>
            <button
              style={{
                position: 'absolute',
                top: 12,
                right: 12,
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                color: UI_COLORS.textMuted,
                fontSize: 18,
                padding: 4,
                display: 'flex',
                alignItems: 'center',
              }}
            >
              ×
            </button>
          </RadixDialog.Close>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  )
}

// ─── Main KnowledgePage Component ──────────────────────────────────────────────

export function KnowledgePage() {
  useDocumentTitle('Knowledge')
  const { activeProjectId } = useUIStore()

  const [searchMode, setSearchMode] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<KnowledgeSearchResult[]>([])
  const [detailItem, setDetailItem] = useState<KnowledgeItem | KnowledgeSearchResult | null>(null)
  const [showCreate, setShowCreate] = useState(false)
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null)

  const knowledgeQuery = useKnowledge(activeProjectId)
  const searchMutation = useSearchKnowledge(activeProjectId)
  const deleteMutation = useDeleteKnowledge(activeProjectId)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!activeProjectId) {
      return
    }

    if (!searchQuery.trim()) {
      setSearchMode(false)
      setSearchResults([])
      return
    }

    try {
      const results = await searchMutation.mutateAsync(searchQuery)
      setSearchResults(results)
      setSearchMode(true)
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Search failed'
      toast.error(message)
    }
  }

  const handleClearSearch = () => {
    setSearchMode(false)
    setSearchQuery('')
    setSearchResults([])
  }

  const handleDeleteClick = (id: string) => {
    setDeleteConfirm(id)
  }

  const handleDeleteConfirm = async () => {
    if (!deleteConfirm) return

    try {
      await deleteMutation.mutateAsync(deleteConfirm)
      toast.success('Knowledge item deleted')
      setDeleteConfirm(null)
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to delete item'
      toast.error(message)
    }
  }

  const items = searchMode ? searchResults : knowledgeQuery.data?.items ?? []
  const isLoading = knowledgeQuery.isLoading || searchMutation.isPending


  const getItemTitle = (item: KnowledgeItem | KnowledgeSearchResult) => {
    if (item.title) return item.title
    return item.content.substring(0, 60) + (item.content.length > 60 ? '...' : '')
  }

  return (
    <div style={{ fontFamily: UI_FONT_FAMILY }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 20,
        }}
      >
        <PageHeader title="Knowledge" />
        <Button
          variant="primary"
          onClick={() => activeProjectId && setShowCreate(true)}
          disabled={!activeProjectId}
        >
          <Plus size={14} />
          New item
        </Button>
      </div>

      {/* Search bar */}
      <form onSubmit={handleSearch} style={{ marginBottom: 20 }}>
        <div
          style={{
            display: 'flex',
            gap: 8,
            alignItems: 'flex-start',
          }}
        >
          <input
            type="text"
            aria-label="Search knowledge"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-1 rounded-md px-2 py-1.5 text-sm transition-colors duration-[120ms]"
            style={{
              backgroundColor: UI_COLORS.depth,
              color: UI_COLORS.textPrimary,
              border: `1px solid ${UI_COLORS.border}`,
              outline: 'none',
            }}
            onFocus={(e) => e.currentTarget.style.borderColor = UI_COLORS.primary}
            onBlur={(e) => e.currentTarget.style.borderColor = UI_COLORS.border}
            placeholder="Search knowledge..."
          />
          <Button
            type="submit"
            variant="secondary"
          >
            <Search size={14} />
            Search
          </Button>
          {searchMode && (
            <Button
              type="button"
              variant="ghost"
              onClick={handleClearSearch}
            >
              Clear
            </Button>
          )}
        </div>
      </form>

      {/* Items list */}
      <QueryState
        query={{
          isLoading: knowledgeQuery.isLoading || (searchMode && searchMutation.isPending),
          isError: knowledgeQuery.isError || (searchMode && searchMutation.isError),
          data: searchMode ? { items } : knowledgeQuery.data,
          refetch: searchMode ? undefined : knowledgeQuery.refetch,
        }}
        skeleton="table"
        skeletonCount={5}
        errorLabel="Failed to load knowledge"
        emptyLabel={searchMode ? 'No results for your search' : 'No knowledge items'}
        emptyDetail={searchMode ? undefined : 'Add documents or notes to give agents domain context.'}
      >
        {() => (
          items.length === 0 ? (
            <EmptyState
              title={searchMode ? 'No results for your search' : 'No knowledge items'}
              body={searchMode ? 'Try a different search term.' : 'Add documents or notes to give agents domain context.'}
              action={searchMode ? undefined : { label: 'New item', onClick: () => activeProjectId && setShowCreate(true) }}
            />
          ) : (
          <div style={{ overflowX: 'auto' }}>
            <table
              style={{
                width: '100%',
                borderCollapse: 'collapse',
                fontSize: 12,
                fontFamily: UI_FONT_FAMILY,
              }}
            >
              <thead>
                <tr
                  style={{
                    borderBottom: `1px solid ${UI_COLORS.border}`,
                  }}
                >
                  <th
                    style={{
                      padding: '10px 8px',
                      textAlign: 'left',
                      color: UI_COLORS.textMuted,
                      fontWeight: 400,
                      textTransform: 'uppercase',
                      fontSize: 10,
                      letterSpacing: '0.5px',
                    }}
                  >
                    Title
                  </th>
                  <th
                    style={{
                      padding: '10px 8px',
                      textAlign: 'left',
                      color: UI_COLORS.textMuted,
                      fontWeight: 400,
                      textTransform: 'uppercase',
                      fontSize: 10,
                      letterSpacing: '0.5px',
                      width: 100,
                    }}
                  >
                    Type
                  </th>
                  <th
                    style={{
                      padding: '10px 8px',
                      textAlign: 'left',
                      color: UI_COLORS.textMuted,
                      fontWeight: 400,
                      textTransform: 'uppercase',
                      fontSize: 10,
                      letterSpacing: '0.5px',
                      width: 150,
                    }}
                  >
                    Tags
                  </th>
                  <th
                    style={{
                      padding: '10px 8px',
                      textAlign: 'left',
                      color: UI_COLORS.textMuted,
                      fontWeight: 400,
                      textTransform: 'uppercase',
                      fontSize: 10,
                      letterSpacing: '0.5px',
                      width: 80,
                    }}
                  >
                    Created
                  </th>
                  <th
                    style={{
                      padding: '10px 8px',
                      textAlign: 'center',
                      color: UI_COLORS.textMuted,
                      fontWeight: 400,
                      textTransform: 'uppercase',
                      fontSize: 10,
                      letterSpacing: '0.5px',
                      width: 120,
                    }}
                  >
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr
                    key={item.id}
                    data-testid={`knowledge-item-${item.id}`}
                    style={{
                      borderBottom: `1px solid ${UI_COLORS.border}`,
                    }}
                  >
                    <td
                      style={{
                        padding: '10px 8px',
                        color: UI_COLORS.textPrimary,
                        maxWidth: 0,
                        overflowWrap: 'anywhere',
                      }}
                    >
                      {getItemTitle(item)}
                    </td>
                    <td style={{ padding: '10px 8px' }}>
                      <ContentTypeBadge type={item.content_type} />
                    </td>
                    <td
                      style={{
                        padding: '10px 8px',
                        color: UI_COLORS.textPrimary,
                        fontSize: 11,
                        overflowWrap: 'anywhere',
                      }}
                    >
                      {(item.tags ?? []).join(', ') || '—'}
                    </td>
                    <td
                      style={{
                        padding: '10px 8px',
                        color: UI_COLORS.textPrimary,
                        fontSize: 11,
                      }}
                    >
                      {'created_at' in item
                        ? dateHeading(item.created_at)
                        : `${Math.round((item as KnowledgeSearchResult).relevance_score * 100)}%`}
                    </td>
                    <td
                      style={{
                        padding: '10px 8px',
                        display: 'flex',
                        gap: 8,
                        justifyContent: 'center',
                      }}
                    >
                      <Button
                        variant="secondary"
                        size="sm"
                        aria-label={`View knowledge ${getItemTitle(item)}`}
                        onClick={() => setDetailItem(item)}
                      >
                        <Eye size={12} />
                        View
                      </Button>
                      <Button
                        variant="secondary"
                        size="sm"
                        aria-label={`Delete knowledge ${getItemTitle(item)}`}
                        onClick={() => handleDeleteClick(item.id)}
                      >
                        <Trash2 size={12} />
                        Delete
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          )
        )}
      </QueryState>

      {/* Detail panel */}
      <DetailPanel
        item={detailItem}
        onClose={() => setDetailItem(null)}
      />

      {/* Delete confirmation dialog */}
      <ConfirmDialog
        open={deleteConfirm !== null}
        onOpenChange={(open) => { if (!open) setDeleteConfirm(null) }}
        title="Delete this item?"
        consequence="This permanently removes the knowledge item and cannot be undone."
        confirmLabel="Delete"
        onConfirm={handleDeleteConfirm}
        isPending={deleteMutation?.isPending}
      />

      {/* Create modal */}
      <CreateModal
        open={showCreate}
        onClose={() => setShowCreate(false)}
        projectId={activeProjectId}
      />
    </div>
  )
}
