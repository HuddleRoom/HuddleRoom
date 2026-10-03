import React, { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useApiKeys, useCreateApiKey, useRevokeApiKey, useCurrentUser } from '@/api/auth'
import { useResetProject } from '@/api/projects'
import { ApiError, apiFetch } from '@/lib/api-client'
import { useUIStore } from '@/stores/ui'
import { toast } from 'sonner'
import { Plus, Trash2, Save } from 'lucide-react'
import { LazyMonacoEditor as Editor } from '@/lib/lazyMonaco'
import { Button, Field, Section, Input, SectionLabel, PageHeader, QueryState, ConfirmDialog } from '@/components/common/uiPrimitives'
import { Dialog } from '@/components/common/Dialog'
import type { ApiKey, User, Project } from '@/lib/types'
import { useDocumentTitle } from '@/hooks/useDocumentTitle'
import { dateHeading } from '@/lib/time'

interface CreateKeyModalProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

function CreateKeyModal({ open, onOpenChange }: CreateKeyModalProps) {
  const [label, setLabel] = useState('')
  const [agentId, setAgentId] = useState('')
  const [projectId, setProjectId] = useState('')
  const [expiresAt, setExpiresAt] = useState('')
  const [newKey, setNewKey] = useState<string | null>(null)
  const { activeProjectId } = useUIStore()
  const createMutation = useCreateApiKey()

  useEffect(() => {
    if (activeProjectId) {
      setProjectId(activeProjectId)
    }
  }, [activeProjectId])

  const handleCreate = async () => {
    try {
      const result = await createMutation.mutateAsync({
        label: label || undefined,
        agent_id: agentId || undefined,
        project_id: projectId || undefined,
        expires_at: expiresAt || undefined,
      })
      setNewKey(result.key)
      setLabel('')
      setAgentId('')
      setProjectId(activeProjectId || '')
      setExpiresAt('')
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Could not create API key — check the fields and try again.')
    }
  }

  const handleDone = () => {
    setNewKey(null)
    onOpenChange(false)
  }

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title="New API key"
      description="Create a new API key for programmatic access."
      size="sm"
      footer={
        newKey
          ? { cancelLabel: 'Close', primaryLabel: 'Done', onPrimary: handleDone }
          : {
              primaryLabel: 'Create key',
              primaryType: 'submit',
              formId: 'create-key-form',
              isPending: createMutation.isPending,
            }
      }
    >
      {newKey ? (
        <div>
          <div
            className="border border-huddleroom-status-green rounded-[4px] bg-huddleroom-depth"
            style={{
              padding: 12,
              marginTop: 12,
              marginBottom: 16,
            }}
          >
            <div
              className="text-huddleroom-text-muted text-[11px] uppercase"
              style={{
                marginBottom: 6,
              }}
            >
              API Key — save this now, it won't be shown again
            </div>
            <code
              className="text-huddleroom-status-green text-[13px]"
              style={{
                wordBreak: 'break-all',
              }}
            >
              {newKey}
            </code>
          </div>
        </div>
      ) : (
        <form id="create-key-form" onSubmit={(event) => { event.preventDefault(); handleCreate() }}>
          <Input
            label="Label (Optional)"
            type="text"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            placeholder="e.g., Production Server"
          />

          <Input
            label="Agent ID (Optional)"
            type="text"
            value={agentId}
            onChange={(e) => setAgentId(e.target.value)}
            placeholder="Leave empty for global access"
          />

          <Input
            label="Project ID (Optional)"
            type="text"
            value={projectId}
            onChange={(e) => setProjectId(e.target.value)}
            placeholder="Leave empty for global access"
          />

          <Input
            label="Expires At (Optional)"
            type="date"
            value={expiresAt}
            onChange={(e) => setExpiresAt(e.target.value)}
          />
        </form>
      )}
    </Dialog>
  )
}

function ApiKeysSection() {
  const [showCreateKey, setShowCreateKey] = useState(false)
  const [revokeConfirm, setRevokeConfirm] = useState<string | null>(null)
  const apiKeysQuery = useApiKeys()
  const revokeMutation = useRevokeApiKey()

  const handleRevoke = async () => {
    if (!revokeConfirm) return
    try {
      await revokeMutation.mutateAsync(revokeConfirm)
      toast.success('API key revoked')
      setRevokeConfirm(null)
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Could not revoke API key.')
    }
  }

  const apiKeys = apiKeysQuery.data || []

  return (
    <Section title="API keys">
      <div style={{ padding: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <Button variant="primary" data-testid="new-api-key-button" onClick={() => setShowCreateKey(true)}>
            <Plus size={14} />
            New API key
          </Button>
        </div>

        <QueryState
          query={{
            isLoading: apiKeysQuery.isLoading,
            isError: apiKeysQuery.isError,
            data: { items: apiKeys },
            refetch: apiKeysQuery.refetch,
          }}
          skeleton="table"
          skeletonCount={3}
          errorLabel="Failed to load API keys"
          emptyLabel="No API keys"
          emptyDetail="Create one to allow programmatic access."
          isEmpty={(d) => (d.items ?? []).length === 0}
        >
          {(data) => (
            <div style={{ overflowX: 'auto' }}>
              <table
                className="text-xs"
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                }}
              >
              <thead>
                <tr className="border-b border-huddleroom-border">
                  <th className="text-huddleroom-text-muted" style={{ textAlign: 'left', padding: '8px' }}>Label</th>
                  <th className="text-huddleroom-text-muted" style={{ textAlign: 'left', padding: '8px' }}>Prefix</th>
                  <th className="text-huddleroom-text-muted" style={{ textAlign: 'left', padding: '8px' }}>Scope</th>
                  <th className="text-huddleroom-text-muted" style={{ textAlign: 'left', padding: '8px' }}>Created</th>
                  <th className="text-huddleroom-text-muted" style={{ textAlign: 'center', padding: '8px' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((key) => {
                  const scope =
                    key.agent_id && key.agent_id.length > 0
                      ? `agent: ${key.agent_id.substring(0, 8)}`
                      : key.project_id && key.project_id.length > 0
                        ? `project: ${key.project_id.substring(0, 8)}`
                        : 'global'

                  return (
                    <tr key={key.id} className="border-b border-huddleroom-border">
                      <td className="text-huddleroom-text-primary" style={{ padding: '8px' }}>
                        {key.label || '(no label)'}
                      </td>
                      <td style={{ padding: '8px' }}>
                        <code className="text-huddleroom-status-blue">
                          {key.prefix}...
                        </code>
                      </td>
                      <td className="text-huddleroom-text-primary" style={{ padding: '8px' }}>{scope}</td>
                      <td className="text-huddleroom-text-primary" style={{ padding: '8px' }}>
                        {dateHeading(key.created_at)}
                      </td>
                      <td style={{ padding: '8px', textAlign: 'center' }}>
                        <Button
                          variant="danger"
                          onClick={() => setRevokeConfirm(key.id)}
                        >
                          <Trash2 size={14} />
                          Revoke
                        </Button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          )}
        </QueryState>

        <ConfirmDialog
          open={revokeConfirm !== null}
          onOpenChange={(open) => { if (!open) setRevokeConfirm(null) }}
          title="Revoke this API key?"
          consequence="The key will stop working immediately. Any integrations using it will break."
          confirmLabel="Revoke key"
          onConfirm={handleRevoke}
          isPending={revokeMutation.isPending}
        />

        <CreateKeyModal open={showCreateKey} onOpenChange={setShowCreateKey} />
      </div>
    </Section>
  )
}

function ProjectConfigSection() {
  const { activeProjectId } = useUIStore()
  const qc = useQueryClient()
  const [configJson, setConfigJson] = useState<string>('{}')
  const [archiveConfirm, setArchiveConfirm] = useState(false)
  const [jsonError, setJsonError] = useState<string | null>(null)
  const [workspacePath, setWorkspacePath] = useState('')
  const [workspaceError, setWorkspaceError] = useState<string | null>(null)
  const [workspaceStatus, setWorkspaceStatus] = useState('')
  const workspaceInput = useRef<HTMLInputElement>(null)
  const [resetConfirmName, setResetConfirmName] = useState('')
  const [resetError, setResetError] = useState<string | null>(null)
  const [resetStatus, setResetStatus] = useState('')
  const resetInput = useRef<HTMLInputElement>(null)
  const resetMutation = useResetProject(activeProjectId ?? '')

  const projectQuery = useQuery({
    queryKey: ['project', activeProjectId],
    queryFn: () => apiFetch<Project>(`/api/v1/projects/${activeProjectId}`),
    enabled: !!activeProjectId,
  })

  // Validate JSON whenever configJson changes
  const validateJson = (json: string): boolean => {
    try {
      JSON.parse(json)
      setJsonError(null)
      return true
    } catch (e) {
      const message = e instanceof Error ? e.message : 'Invalid JSON'
      setJsonError(message)
      return false
    }
  }

  const saveConfigMutation = useMutation({
    mutationFn: async () => {
      let config: any
      try {
        config = JSON.parse(configJson)
      } catch {
        throw new Error('Configuration is not valid JSON')
      }
      return apiFetch<Project>(`/api/v1/projects/${activeProjectId}`, {
        method: 'PUT',
        body: JSON.stringify({ name: projectQuery.data!.name, config }),
      })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['project', activeProjectId] })
      toast.success('Config saved')
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Could not save config — verify the JSON and your permissions.')
    },
  })

  const saveWorkspaceMutation = useMutation({
    mutationFn: () => apiFetch<Project>(`/api/v1/projects/${activeProjectId}`, {
      method: 'PUT',
      body: JSON.stringify({ workspace_path: workspacePath.trim() }),
    }),
    onSuccess: (project) => {
      setWorkspacePath(project.workspace_path ?? '')
      setWorkspaceError(null)
      setWorkspaceStatus('Server directory saved')
      qc.invalidateQueries({ queryKey: ['project', activeProjectId] })
      qc.invalidateQueries({ queryKey: ['projects'] })
      toast.success('Server directory saved')
    },
    onError: (error) => {
      setWorkspaceStatus('')
      const detail = error instanceof ApiError ? error.detail : undefined
      const validation = Array.isArray(detail) ? detail.find((item) =>
        typeof item === 'object' && item !== null && Array.isArray((item as { loc?: unknown }).loc)
          && (item as { loc: unknown[] }).loc.includes('workspace_path')) as { msg?: unknown } | undefined : undefined
      setWorkspaceError(
        error instanceof ApiError && error.status === 409
          ? 'Cannot change the server directory while project work is active.'
          : typeof validation?.msg === 'string'
            ? validation.msg
            : 'Could not save the server directory. Try again.',
      )
      requestAnimationFrame(() => workspaceInput.current?.focus())
    },
  })

  const archiveMutation = useMutation({
    mutationFn: async () => {
      await apiFetch<void>(`/api/v1/projects/${activeProjectId}`, {
        method: 'DELETE',
      })
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['project', activeProjectId] })
      toast.success('Project archived')
      setArchiveConfirm(false)
    },
    onError: (error) => {
      toast.error(error instanceof Error ? error.message : 'Could not archive project — try again.')
    },
  })

  useEffect(() => {
    if (projectQuery.data) {
      setConfigJson(JSON.stringify(projectQuery.data.config, null, 2))
      setWorkspacePath(projectQuery.data.workspace_path ?? '')
    }
  }, [projectQuery.data])

  const handleSaveWorkspace = (event: React.FormEvent) => {
    event.preventDefault()
    setWorkspaceStatus('')
    if (!workspacePath.trim()) {
      setWorkspaceError('Enter a server directory.')
      workspaceInput.current?.focus()
      return
    }
    saveWorkspaceMutation.mutate()
  }

  const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? '' : 's'}`

  const handleResetSubmit = (event: React.FormEvent) => {
    event.preventDefault()
    setResetStatus('')
    if (resetConfirmName !== projectQuery.data?.name) return // button is already disabled; defensive
    resetMutation.mutate({ confirm_name: resetConfirmName }, {
      onSuccess: (result) => {
        const total = Object.values(result.deletions).reduce((sum, n) => sum + n, 0)
        setResetConfirmName('')
        setResetError(null)
        setResetStatus(
          `Reset complete — cancelled ${plural(result.cancelled_sessions, 'session')}, ` +
          `${plural(result.cancelled_meeting_tasks, 'meeting task')}, and removed ${plural(total, 'other operational record')}. ` +
          `Configuration and workspace files were left untouched.`
        )
        toast.success('Project reset — operational data cleared')
      },
      onError: (error) => {
        setResetStatus('')
        setResetError(
          error instanceof ApiError && error.status === 400
            ? 'Confirmation did not match the project name. Check the exact spelling, case, and spaces, then try again.'
            : error instanceof ApiError && error.status === 409
              ? 'This project can’t be reset — it may have been archived. Reload and check the project status.'
              : error instanceof Error ? error.message : 'Could not reset the project. Try again.'
        )
        requestAnimationFrame(() => resetInput.current?.focus())
      },
    })
  }

  if (!activeProjectId) {
    return (
      <Section title="Project Config">
        <div className="text-huddleroom-text-muted text-[13px]" style={{ padding: '16px' }}>
          Select a project to manage its configuration.
        </div>
      </Section>
    )
  }

  return (
    <Section title="Project config">
      <div style={{ padding: '16px' }}>
        <QueryState
          query={{
            isLoading: projectQuery.isLoading,
            isError: projectQuery.isError,
            data: projectQuery.data,
            refetch: projectQuery.refetch,
          }}
          skeleton="list"
          skeletonCount={2}
          errorLabel="Failed to load project"
          emptyLabel="Project not found"
        >
          {(data) => (
            <>
              <Field label="Project Name">
                <div className="text-huddleroom-text-primary" style={{ padding: '6px 8px' }}>
                  {data.name}
                </div>
              </Field>

              <form onSubmit={handleSaveWorkspace} style={{ marginBottom: 14 }}>
                <Input
                  ref={workspaceInput}
                  id="project-workspace-path"
                  label="Server directory"
                  value={workspacePath}
                  onChange={(event) => {
                    setWorkspacePath(event.target.value)
                    setWorkspaceError(null)
                    setWorkspaceStatus('')
                  }}
                  error={workspaceError ?? undefined}
                  aria-invalid={!!workspaceError}
                  aria-describedby="project-workspace-help"
                  className="font-mono"
                />
                <p id="project-workspace-help" className="text-huddleroom-text-muted text-xs" style={{ margin: '-8px 0 10px' }}>
                  Absolute directory on the HuddleRoom server. It must exist and be readable, writable, and searchable.
                </p>
                <Button
                  variant="primary"
                  type="submit"
                  data-testid="settings-save-workspace"
                  disabled={saveWorkspaceMutation.isPending || !workspacePath.trim() || workspacePath.trim() === (data.workspace_path ?? '')}
                >
                  <Save size={14} />
                  {saveWorkspaceMutation.isPending ? 'Saving…' : 'Save server directory'}
                </Button>
                <p className="sr-only" role="status" aria-live="polite">
                  {workspaceStatus}
                </p>
              </form>

              <div style={{ marginBottom: 14 }}>
                <SectionLabel>Configuration (JSON)</SectionLabel>
                <div data-testid="settings-config-editor" className={`border rounded-md overflow-hidden ${jsonError ? 'border-huddleroom-danger' : 'border-huddleroom-border'}`}>
                  <React.Suspense fallback={<div className="rounded-md bg-huddleroom-depth flex items-center justify-center" style={{ height: '360px' }}><span className="text-huddleroom-text-muted text-[11px]">Loading editor…</span></div>}>
                    <Editor
                      height="360px"
                      language="json"
                      theme="light"
                      value={configJson}
                      onChange={(value) => {
                        setConfigJson(value || '{}')
                        validateJson(value || '{}')
                      }}
                      options={{
                        minimap: { enabled: false },
                        fontSize: 12,
                        scrollBeyondLastLine: false,
                      }}
                    />
                  </React.Suspense>
                </div>
                {jsonError && (
                  <p className="text-huddleroom-danger text-xs" style={{ marginTop: 6 }}>
                    Invalid JSON: {jsonError}
                  </p>
                )}
              </div>

              <div style={{ marginBottom: 16 }}>
                <Button
                  variant="primary"
                  data-testid="settings-save-config"
                  onClick={() => saveConfigMutation.mutate()}
                  disabled={saveConfigMutation.isPending || jsonError !== null}
                >
                  <Save size={14} />
                  Save config
                </Button>
                <div className="mt-4 border-t border-huddleroom-border pt-4">
                  <Button
                    variant="danger"
                    data-testid="settings-archive-project"
                    onClick={() => setArchiveConfirm(true)}
                  >
                    <Trash2 size={14} />
                    Archive project
                  </Button>
                </div>
              </div>

              {archiveConfirm && (
                <div
                  className="border border-huddleroom-danger rounded-md text-huddleroom-danger text-xs bg-huddleroom-danger/5"
                  style={{
                    padding: 12,
                    marginBottom: 16,
                  }}
                >
                  <div style={{ marginBottom: 8 }}>
                    Are you sure? This action cannot be undone.
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <Button
                      variant="secondary"
                      onClick={() => setArchiveConfirm(false)}
                    >
                      Cancel
                    </Button>
                    <Button
                      variant="danger"
                      onClick={() => archiveMutation.mutate()}
                      disabled={archiveMutation.isPending}
                    >
                      Confirm archive
                    </Button>
                  </div>
                </div>
              )}

              <div className="border-t border-huddleroom-border" style={{ marginTop: 20, paddingTop: 20 }}>
                <div className="text-[11px] font-semibold uppercase text-huddleroom-danger tracking-[0.06em]" style={{ marginBottom: 10 }}>
                  Danger zone
                </div>
                <div data-testid="settings-danger-zone" className="border border-huddleroom-danger rounded-md bg-huddleroom-danger/5" style={{ padding: 16 }}>
                  <div className="text-[13px] font-semibold text-huddleroom-text-primary" style={{ marginBottom: 6 }}>Reset project data</div>
                  <p className="text-xs text-huddleroom-text-secondary" style={{ lineHeight: 1.5, marginBottom: 8 }}>
                    This permanently deletes all operational history for this project — sessions, tasks, meetings, orchestration goals and decisions, memory, channels, artifacts, and the event log — and resets hook counters to zero.
                  </p>
                  <p className="text-xs text-huddleroom-text-secondary" style={{ lineHeight: 1.5, marginBottom: 10 }}>
                    It does <strong>not</strong> touch the project name, description, config, API keys, users, agents, protocol and routing rules, or any file in the workspace directory — including <code className="font-mono">.huddleroom</code>.
                  </p>
                  <p className="text-xs font-semibold text-huddleroom-danger" style={{ marginBottom: 12 }}>This cannot be undone.</p>
                  <div className="text-xs text-huddleroom-text-secondary" style={{ marginBottom: 8 }}>
                    Type the project name to confirm:{' '}
                    <code className="font-mono font-semibold text-huddleroom-text-primary bg-huddleroom-surface border border-huddleroom-border rounded-[4px]" style={{ padding: '2px 6px' }}>{data.name}</code>
                  </div>
                  <form onSubmit={handleResetSubmit}>
                    <Input
                      ref={resetInput}
                      id="project-reset-confirm"
                      label="Confirm project name"
                      placeholder="Exact project name"
                      value={resetConfirmName}
                      onChange={(e) => { setResetConfirmName(e.target.value); setResetError(null); setResetStatus('') }}
                      error={resetError ?? undefined}
                      aria-invalid={!!resetError}
                      aria-describedby="project-reset-help"
                      disabled={resetMutation.isPending}
                      autoComplete="off"
                      spellCheck={false}
                      data-testid="settings-reset-confirm-input"
                      className="font-mono"
                    />
                    <p id="project-reset-help" className="text-huddleroom-text-muted text-xs" style={{ margin: '-8px 0 10px' }}>
                      Case-sensitive and whitespace-sensitive — must match exactly.
                    </p>
                    <Button variant="danger" type="submit" data-testid="settings-reset-submit" disabled={resetMutation.isPending || resetConfirmName !== data.name}>
                      <Trash2 size={14} />
                      {resetMutation.isPending ? 'Resetting…' : 'Reset project data'}
                    </Button>
                    {/* input is disabled while pending because this action is destructive and irreversible — do not "align" it with the workspace field */}
                    <p role="status" aria-live="polite" data-testid="settings-reset-status" className={`text-xs ${resetStatus ? 'text-huddleroom-status-green bg-huddleroom-depth border border-huddleroom-status-green rounded-[4px]' : ''}`} style={{ marginTop: 10, ...(resetStatus ? { padding: '8px 10px' } : {}) }}>
                      {resetStatus}
                    </p>
                  </form>
                </div>
              </div>
            </>
          )}
        </QueryState>
      </div>
    </Section>
  )
}

function UserProfileSection() {
  const userQuery = useCurrentUser()

  return (
    <Section title="User profile">
      <div style={{ padding: '16px' }}>
        <QueryState
          query={{
            isLoading: userQuery.isLoading,
            isError: userQuery.isError,
            data: userQuery.data,
            refetch: userQuery.refetch,
          }}
          skeleton="list"
          skeletonCount={2}
          errorLabel="Failed to load user profile"
          emptyLabel="User profile not found"
        >
          {(user) => (
            <>
              <Field label="Email">
                <div className="text-huddleroom-text-primary" style={{ padding: '6px 8px' }}>
                  {user.email}
                </div>
              </Field>

              <Field label="Display Name">
                <div className="text-huddleroom-text-primary" style={{ padding: '6px 8px' }}>
                  {user.display_name || '—'}
                </div>
              </Field>

              <Field label="Role">
                <div className="text-huddleroom-text-primary" style={{ padding: '6px 8px' }}>
                  {user.role}
                </div>
              </Field>

              <div className="text-huddleroom-text-muted text-xs" style={{ marginTop: 16 }}>
                Password changes are not available yet.
              </div>
            </>
          )}
        </QueryState>
      </div>
    </Section>
  )
}

export function SettingsPage() {
  useDocumentTitle('Settings')
  const { activeProjectId } = useUIStore()
  return (
    <div>
      <div style={{ maxWidth: 900, margin: '0 auto' }}>
        <PageHeader title="Settings" className="mb-6" />

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <ApiKeysSection />
          <ProjectConfigSection key={activeProjectId ?? 'none'} />
          <UserProfileSection />
        </div>
      </div>
    </div>
  )
}
