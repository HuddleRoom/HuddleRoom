export type ControlExpectation =
  | { kind: 'role'; role: 'button' | 'link' | 'combobox' | 'textbox' | 'checkbox' | 'heading' | 'tab' | 'radio'; name: string | RegExp; enabled?: boolean }
  | { kind: 'testId'; testId: string }
  | { kind: 'text'; text: string | RegExp }

export interface RouteControlMap {
  path: string
  label: string
  controls: ControlExpectation[]
}

export const CONTROL_MAP: RouteControlMap[] = [
  {
    path: '/dashboard',
    label: 'Dashboard',
    controls: [
      { kind: 'role', role: 'combobox', name: 'Project switcher' },
      { kind: 'role', role: 'button', name: 'View ready tasks' },
      // Dashboard StatTiles render as links (to="/agents" etc, count-linked per
      // design spec Section C) — name is "{label} {count}", so match by prefix.
      { kind: 'role', role: 'link', name: /^Sessions/ },
      { kind: 'role', role: 'link', name: /^Meetings/ },
      { kind: 'role', role: 'link', name: /^Protocols/ },
      { kind: 'role', role: 'link', name: /^Ready tasks/ },
      { kind: 'role', role: 'button', name: 'Collapse sidebar' },
      { kind: 'role', role: 'button', name: 'Log out' },
    ],
  },
  {
    path: '/dashboard/orchestration',
    label: 'Orchestration',
    controls: [
      { kind: 'role', role: 'heading', name: /Orchestration/i },
      { kind: 'text', text: /No orchestration goals|success criterion/ },
    ],
  },
  {
    path: '/dashboard/tasks',
    label: 'Tasks',
    controls: [
      { kind: 'role', role: 'button', name: 'New task' },
      { kind: 'role', role: 'combobox', name: 'Filter tasks by agent' },
      { kind: 'role', role: 'combobox', name: 'Filter tasks by status' },
      { kind: 'text', text: /Task board|Create task/ },
    ],
  },
  {
    path: '/dashboard/agents',
    label: 'Agents',
    controls: [
      { kind: 'role', role: 'button', name: /new agent/i },
      { kind: 'role', role: 'button', name: /show inactive|hide inactive/i },
      { kind: 'role', role: 'button', name: /Edit agent huddleroom-ui-architect/ },
      { kind: 'role', role: 'button', name: /Deactivate agent huddleroom-ui-architect/ },
    ],
  },
  {
    path: '/dashboard/meetings',
    label: 'Meetings',
    controls: [
      { kind: 'role', role: 'button', name: /new meeting/i },
      { kind: 'role', role: 'radio', name: 'All' },
      { kind: 'role', role: 'radio', name: 'Active' },
      { kind: 'role', role: 'radio', name: 'Concluded' },
    ],
  },
  {
    path: '/dashboard/protocols',
    label: 'Protocols',
    controls: [
      { kind: 'role', role: 'button', name: /new protocol/i },
      { kind: 'role', role: 'textbox', name: 'Search protocols' },
      { kind: 'role', role: 'checkbox', name: /include inactive/i },
      { kind: 'text', text: 'ui_seed_protocol' },
    ],
  },
  {
    path: '/dashboard/knowledge',
    label: 'Knowledge',
    controls: [
      { kind: 'role', role: 'button', name: 'New item' },
      { kind: 'role', role: 'textbox', name: /Search knowledge/i },
      { kind: 'role', role: 'button', name: 'Search' },
    ],
  },
  {
    path: '/dashboard/memory',
    label: 'Memory',
    controls: [
      { kind: 'role', role: 'tab', name: 'Project' },
      { kind: 'role', role: 'tab', name: 'Global' },
      { kind: 'role', role: 'textbox', name: 'Agent ID' },
      { kind: 'role', role: 'checkbox', name: 'Show shared memory only' },
      { kind: 'role', role: 'textbox', name: 'Search' },
      { kind: 'role', role: 'button', name: 'Search' },
    ],
  },
  {
    path: '/dashboard/rules',
    label: 'Rules',
    controls: [
      { kind: 'role', role: 'button', name: 'New rule' },
      { kind: 'role', role: 'button', name: /Edit rule: UI seed routing rule/ },
      { kind: 'role', role: 'button', name: /Delete rule: UI seed routing rule/ },
      { kind: 'role', role: 'checkbox', name: /Disable rule|Enable rule/ },
    ],
  },
  {
    path: '/dashboard/hooks',
    label: 'Hooks',
    controls: [
      { kind: 'role', role: 'button', name: 'New hook' },
      { kind: 'role', role: 'button', name: /Disable hook ui_seed_hook|Activate hook ui_seed_hook|Shadow hook ui_seed_hook/ },
    ],
  },
  {
    path: '/dashboard/optimizations',
    label: 'Optimizations',
    controls: [
      { kind: 'role', role: 'button', name: /Submit for Approval|Approve|Reject|Activate/ },
      { kind: 'role', role: 'button', name: /Delete optimization/ },
      { kind: 'role', role: 'tab', name: 'Proposed' },
    ],
  },
  {
    path: '/dashboard/settings',
    label: 'Settings',
    controls: [
      { kind: 'testId', testId: 'new-api-key-button' },
      { kind: 'testId', testId: 'settings-save-config' },
      { kind: 'testId', testId: 'settings-archive-project' },
    ],
  },
]
