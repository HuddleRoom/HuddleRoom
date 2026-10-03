import ReactFlow, { Position, type Edge, type Node } from 'reactflow'
import 'reactflow/dist/style.css'
import dagre from 'dagre'
import { UI_COLORS } from '@/components/common/uiPrimitives'
import { STATUS_COLORS } from '@/lib/statusColors'

const NODE_WIDTH = 160
const NODE_HEIGHT = 44

function layout(nodes: Node[], edges: Edge[]) {
  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: 'TB', ranksep: 36, nodesep: 24 })
  g.setDefaultEdgeLabel(() => ({}))
  for (const node of nodes) g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  for (const edge of edges) g.setEdge(edge.source, edge.target)
  dagre.layout(g)
  return nodes.map((node) => {
    const pos = g.node(node.id) ?? { x: 0, y: 0 }
    return { ...node, position: { x: pos.x - NODE_WIDTH / 2, y: pos.y - NODE_HEIGHT / 2 } }
  })
}

function flatNode(id: string, label: string, sublabel: string | undefined, borderColor: string): Node {
  return {
    id,
    data: { label: <div style={{ textAlign: 'center' }}>
      <div style={{ fontWeight: 700, fontSize: 12, color: borderColor }}>{label}</div>
      {sublabel && <div style={{ fontSize: 11, color: UI_COLORS.textSecondary, marginTop: 2 }}>{sublabel}</div>}
    </div> },
    position: { x: 0, y: 0 },
    sourcePosition: Position.Bottom,
    targetPosition: Position.Top,
    style: {
      background: UI_COLORS.surface,
      border: `1px solid ${borderColor}`,
      borderRadius: 3,
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: 4,
    },
  }
}

export interface HierarchyGraphProps {
  manager: string
  roles: readonly { role: string; agent: string }[]
  missing: readonly string[]
  weakFits: readonly string[]
}

// Team-hierarchy DAG (manager → roles → agents), operator-console styling:
// flat nodes, 1px borders, status-colored headers, no decorations. Static —
// no pan/zoom/drag — fitView is enough since the diagram never needs
// exploring. The text HierarchyCard rendered alongside this is both the
// fallback (when the guard in ProcessFocus keeps this component unmounted)
// and the a11y alternative, so this diagram is decorative.
// ponytail: no useMemo — roles/missing/weakFits are fresh array references
// from the caller's parseHierarchy() on every render anyway (memo deps would
// never hit), and laying out a handful of nodes is cheap for a step-detail-
// scoped diagram. Add memoization back if this ever renders large graphs.
export function HierarchyGraph({ manager, roles, missing, weakFits }: HierarchyGraphProps) {
  const missingSet = new Set(missing)
  const weakSet = new Set(weakFits)
  const managerId = 'manager'
  const roleNodes: Node[] = []
  const agentNodes = new Map<string, Node>()
  const edgeList: Edge[] = []

  for (const { role, agent } of roles) {
    const roleId = `role:${role}`
    const roleColor = missingSet.has(role) ? STATUS_COLORS.red : weakSet.has(role) ? STATUS_COLORS.amber : STATUS_COLORS.green
    roleNodes.push(flatNode(roleId, role, undefined, roleColor))
    edgeList.push({ id: `${managerId}->${roleId}`, source: managerId, target: roleId, style: { stroke: UI_COLORS.borderStrong, strokeWidth: 1 } })

    const agentId = `agent:${agent}`
    if (!agentNodes.has(agentId)) agentNodes.set(agentId, flatNode(agentId, agent, undefined, STATUS_COLORS.blue))
    edgeList.push({ id: `${roleId}->${agentId}`, source: roleId, target: agentId, style: { stroke: UI_COLORS.borderStrong, strokeWidth: 1 } })
  }

  const allNodes = [flatNode(managerId, manager, 'Manager', STATUS_COLORS.blue), ...roleNodes, ...agentNodes.values()]
  const nodes = layout(allNodes, edgeList)

  return (
    <div
      aria-hidden="true"
      className="mt-4 overflow-hidden rounded border border-huddleroom-border bg-huddleroom-depth"
      style={{ height: 220 }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edgeList}
        fitView
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        // The container is aria-hidden (decorative — HierarchyCard's text is
        // the real a11y alternative), but reactflow defaults nodes/edges to
        // keyboard-focusable regardless, which would let keyboard users tab
        // into AT-invisible nodes. disableKeyboardA11y additionally drops the
        // ARIA roles/keyboard handlers reactflow would otherwise attach
        // (review round 2, finding I3).
        nodesFocusable={false}
        edgesFocusable={false}
        disableKeyboardA11y
        elementsSelectable={false}
        panOnDrag={false}
        panOnScroll={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        preventScrolling={false}
      />
    </div>
  )
}
