import { useMemo } from 'react'
import ReactFlow, { Background, Controls, Node, Edge, MarkerType } from 'reactflow'
import 'reactflow/dist/style.css'
import dagre from 'dagre'
import { UI_COLORS } from '@/components/common/uiPrimitives'
import { STATUS_COLORS } from '@/lib/statusColors'
import type { ParsedFlow } from './GraphsPage'

const MONO = 'var(--huddleroom-font-mono)'
const SANS = 'var(--huddleroom-font-sans)'

// ─── Build react-flow diagram ─────────────────────────────────────────────────

function buildFlowDiagram(
  parsed: ParsedFlow,
  currentNode?: string
): { nodes: Node[]; edges: Edge[] } {
  const nodeIds = Array.from(parsed.nodes.keys())
  const parsedEdges = parsed.edges

  if (nodeIds.length === 0) {
    return { nodes: [], edges: [] }
  }

  // Create nodes and edges for dagre layout
  const g = new dagre.graphlib.Graph()
  g.setGraph({ rankdir: 'LR', ranksep: 60, nodesep: 40 })
  g.setDefaultEdgeLabel(() => ({}))

  nodeIds.forEach((s) => {
    g.setNode(s, { width: 120, height: 40 })
  })

  parsedEdges.forEach((t) => {
    g.setEdge(t.from, t.to)
  })

  dagre.layout(g)

  // Build react-flow nodes
  const nodes: Node[] = nodeIds.map((s) => {
    const pos = g.node(s) ?? { x: 80, y: 80 }
    const nodeInfo = parsed.nodes.get(s)!
    const isCurrent = s === currentNode
    const isInitial = nodeInfo.isInitial
    const isTerminal = nodeInfo.isTerminal

    let bgColor: string = UI_COLORS.surface
    let borderColor: string = UI_COLORS.border
    let textColor: string = UI_COLORS.textPrimary
    let borderWidth = 1

    if (isInitial) {
      borderColor = STATUS_COLORS.green
      textColor = STATUS_COLORS.green
      borderWidth = 2
    } else if (isTerminal) {
      borderColor = UI_COLORS.danger
      textColor = UI_COLORS.danger
      borderWidth = 2
    } else if (isCurrent) {
      borderColor = STATUS_COLORS.blue
      textColor = STATUS_COLORS.blue
      borderWidth = 2
    }

    return {
      id: s,
      data: { label: s },
      position: { x: pos.x, y: pos.y },
      style: {
        background: bgColor,
        border: `${borderWidth}px solid ${borderColor}`,
        borderRadius: 6,
        fontFamily: MONO,
        fontSize: 12,
        color: textColor,
        padding: '6px 10px',
        width: 'auto',
        height: 'auto',
        minWidth: 120,
        textAlign: 'center',
      },
    }
  })

  // Build react-flow edges
  // Pre-build running index map to handle multiple edges between same nodes
  const edgeRunningIndex = new Map<string, number>()
  const edges: Edge[] = parsedEdges.map((t) => {
    const key = JSON.stringify([t.from, t.to])
    const edgeIndex = edgeRunningIndex.get(key) ?? 0
    edgeRunningIndex.set(key, edgeIndex + 1)

    return {
      id: `${key}-${edgeIndex}`,
      source: t.from,
      target: t.to,
      label: t.event,
      animated: false,
      style: { stroke: UI_COLORS.textMuted, strokeWidth: 1 },
      labelStyle: { fontSize: 11, color: UI_COLORS.textMuted, fontFamily: MONO },
      markerEnd: { type: MarkerType.ArrowClosed },
    } as Edge
  })

  return { nodes, edges }
}

// ─── GraphDiagram ────────────────────────────────────────────────────────────

export function GraphDiagram({ parsed, currentNode }: { parsed: ParsedFlow; currentNode?: string }) {
  const { nodes, edges } = useMemo(
    () => buildFlowDiagram(parsed, currentNode),
    [parsed, currentNode]
  )

  if (nodes.length === 0) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%',
        color: UI_COLORS.textMuted, fontSize: 12, fontFamily: SANS,
      }}>
        no graph diagram available
      </div>
    )
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      fitView
      proOptions={{ hideAttribution: true }}
    >
      <Background color={UI_COLORS.borderStrong} gap={16} />
      <Controls />
    </ReactFlow>
  )
}
