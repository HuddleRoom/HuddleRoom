// DESIGN.md `title` token: ink, 14px/600, sentence case, no tracking.
// Typographic values only — color is separate (see ZONE_TITLE_CLASS below)
// because the Details tabs (Z5) are the one consumer that needs the token's
// size/weight but NOT its static ink color: a tab's color is a selection
// signal (primary blue when active, muted otherwise, per the Blue Signal
// Rule), not a static structural label.
export const TITLE_TOKEN_TYPE_CLASS = 'text-sm font-semibold'

// Full token (typography + ink structural-label color) applied to exactly
// the goal-detail page's zone headings: Orchestrator steps, Needs you, Step
// detail, Needs attention. Z5 (Details) has no separate visible zone
// heading of its own — the tab strip is that zone's visible surface — so
// it carries TITLE_TOKEN_TYPE_CLASS directly (see DetailsTabs.tsx) instead
// of this ink variant.
export const ZONE_TITLE_CLASS = `${TITLE_TOKEN_TYPE_CLASS} text-huddleroom-text-primary`
