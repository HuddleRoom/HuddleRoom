import type { NowLineSegment } from './humanize'

// Narrative "now" line (Z2) — one always-present sentence directly under the
// stepper. Prose only (step names are humanized, not raw system tokens, so
// there's no mono segment to carry — see nowLine()'s "next" fix).
export function NowLine({ segments }: { segments: readonly NowLineSegment[] }) {
  return (
    <p className="mt-3 border-t border-huddleroom-border pt-3 text-sm text-huddleroom-text-secondary">
      {segments.map((segment, index) => <span key={index}>{segment.text}</span>)}
    </p>
  )
}
