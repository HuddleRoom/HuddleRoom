# Rally Frontend Design System

## Stack

React 19 · TypeScript · Vite · Tailwind CSS (utility layer only) · Radix UI (primitives) · Zustand · TanStack Query

## Design register

Product UI / dashboard tool. Design serves the product — clarity and function over expression.

---

## Color

### Brand palette — `UI_COLORS` (uiPrimitives.tsx)

| Token | Value | Usage |
|---|---|---|
| `primary` | `#2563EB` | Primary actions, focus rings, links |
| `primaryHover` | `#1D4ED8` | Primary button hover |
| `primaryForeground` | `#FFFFFF` | Text on primary bg |
| `teal` | `#0D9488` | Secondary brand accent |
| `surface` | `#FFFFFF` | Card / panel surface |
| `depth` | `#F1F5F9` | Recessed inputs, inactive tabs |
| `border` | `#E2E8F0` | Default borders |
| `borderStrong` | `#CBD5E1` | Emphasized / hover borders |
| `textPrimary` | `#0F172A` | Headings, high-emphasis body |
| `textSecondary` | `#475569` | Supporting body text |
| `textMuted` | `#64748B` | Labels, placeholders (4.5:1 AA) |
| `danger` | `#DC2626` | Destructive actions, error states |
| `overlay` | `rgba(0,0,0,0.7)` | Modal backdrop |
| `sidebarBg` | `#0F172A` | Sidebar container |
| `sidebarSurface` | `#1E293B` | Sidebar item hover / active bg |
| `sidebarText` | `#CBD5E1` | Sidebar default text |
| `sidebarMuted` | `#64748B` | Sidebar secondary labels |
| `sidebarActive` | `#60A5FA` | Active nav item text |
| `sidebarActiveBg` | `rgba(37,99,235,0.2)` | Active nav item background |
| `sidebarHoverText` | `#F8FAFC` | Sidebar item hover text |
| `sidebarHoverBorder` | `#94A3B8` | Sidebar item hover left border |

### Status + priority colors — `statusColors.ts` (single source of truth)

All status and priority colors are AA-darkened to meet WCAG 4.5:1 against white.  
**Do not hardcode status colors inline. Import from `@/lib/statusColors`.**

```ts
import { STATUS_COLORS, PRIORITY_COLORS, TASK_COLUMN_COLOR } from '@/lib/statusColors'
```

| Export | Keys | Sample value |
|---|---|---|
| `STATUS_COLORS` | `blue amber green red purple teal neutral` | blue=`#0369A1` |
| `PRIORITY_COLORS` | `low medium high critical` | high=`#B45309` |
| `TASK_COLUMN_COLOR` | `backlog ready in_progress blocked done failed cancelled` | done=`#15803D` |

### CSS custom properties

Status colors are also available as `--status-{blue,amber,green,red,purple,teal,neutral}` in `:root` for stylesheet use (see `index.css`).

### Contrast rules

- Body / label text: ≥ 4.5:1 against background
- Placeholder text: use `textMuted` (`#64748B`, 4.5:1) — never `#94A3B8` (2.3:1, fails AA)
- Gray text on colored bg: use a darkened shade of the bg hue, not a generic gray

---

## Z-index scale

Semantic scale in two places — always use the constant or custom property, never a raw integer.

### TypeScript — `Z` export (uiPrimitives.tsx)

```ts
import { Z } from '@/components/common/uiPrimitives'

// Z.dropdown      = 10
// Z.sticky        = 20
// Z.modalBackdrop = 40
// Z.modal         = 50
// Z.toast         = 60
// Z.tooltip       = 70
```

### CSS custom properties (index.css)

```css
--z-dropdown: 10;
--z-sticky: 20;
--z-modal-backdrop: 40;
--z-modal: 50;
--z-toast: 60;
--z-tooltip: 70;
```

---

## Typography

- Font: `'Inter', system-ui, sans-serif` (body); `ui-monospace, Menlo, Monaco, monospace` (code)
- Base: 14px / 1.5 line-height
- Headings: weight 600; `text-wrap: balance`
- Max body line length: 65–75ch
- Antialiasing: `-webkit-font-smoothing: antialiased`

---

## Spacing & layout

- App shell: 48px top bar + collapsible sidebar + scrollable main content area
- Page padding: handled by shell; pages do not set min-height or full-bleed backgrounds
- Cards: `rally-card` class (`bg-white border border-rally-border rounded-md`, hover border/shadow transition)
- Nested cards: never

---

## Components (uiPrimitives.tsx)

### `Button`

```tsx
<Button variant="primary|secondary|ghost|danger|teal" size="sm|md|lg" />
```

- All interactive states (hover, focus-visible, disabled) are built in
- Use `Button` everywhere — do not write raw `<button>` with inline styles

### `Input`, `Textarea`, `Select`

- Use `focusInput` / `blurInput` helpers for controlled focus styling
- Placeholder: always `textMuted` (#64748B) for AA compliance

### Style exports

| Export | Usage |
|---|---|
| `primaryButtonStyle` | Direct style prop when Button wrapper isn't suitable |
| `secondaryButtonStyle` | — |
| `dangerButtonStyle` | — |
| `ghostIconButtonStyle` | Icon-only ghost button |
| `inputStyle` | Text input / textarea |
| `selectStyle` | `<select>` element |

---

## Scrollbars

Thin (6px), track `#F1F5F9`, thumb `#CBD5E1` → hover `#94A3B8`. Defined globally in `index.css`.

---

## Motion

All transitions use `cubic-bezier(0.4, 0, 0.2, 1)` ease-out, 120–300ms.  
Every animation has a `@media (prefers-reduced-motion: reduce)` override (`transition-duration: 0ms`).

---

## Accessibility

- Focus rings: `outline: 2px solid #2563EB; outline-offset: 2px` (global in `index.css`)
- `aria-live="polite"` on async status regions (TopBar connection dot, project loading)
- Empty states: always include a labelled action button, not just descriptive text
- Modals: overlay at `Z.modalBackdrop`, content at `Z.modal`; include `Dialog.Description`

---

## Skeleton / loading states

`.skeleton` class: `background: #E2E8F0`, `animation: skeleton-pulse 1.5s ease-in-out infinite`.  
Pair with `aria-busy="true"` and `aria-live="polite"` on the container.

---

## Status class utilities (index.css)

`.status-{pending,backlog,ready,in_progress,running,blocked,done,completed,active,failed,cancelled,scheduled,preparing,concluding,concluded,proposed,shadow,disabled,paused}` — apply to any element to get the correct status color via `color: var(--status-*)`.
