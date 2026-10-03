import type { Config } from 'tailwindcss'

export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        huddleroom: {
          base:      '#FFFFFF',      // page background
          'bg-2':    '#F5F6F8',      // panel header rows / bands
          surface:   '#FFFFFF',      // card/panel surface
          depth:     '#F1F5F9',      // inputs, secondary panels
          border:    '#E2E8F0',      // all borders
          'border-strong': '#CBD5E1', // stronger borders
          hover:     '#1D4ED8',      // primary hover
          'row-hover': '#F8FAFC',    // table row hover
          primary:   '#1a5cff',      // primary blue
          danger:    '#DC2626',      // danger text/border (distinct from status-red #B91C1C)
          'surface-blue': '#EFF6FF', // info banner/notice surface
          faint:     '#94A3B8',      // muted text
          'text-primary':   '#1f2226',
          'text-secondary': '#33373d',
          'text-muted':     '#5d6470',
          // Sidebar (stays dark)
          'sidebar-bg':     '#0F172A',
          'sidebar-surface':'#1E293B',
          'sidebar-border': '#1E293B',
          'sidebar-text':   '#CBD5E1',
          'sidebar-muted':  '#64748B',
          // Status
          'status-blue':   '#0369A1',
          'status-amber':  '#B45309',
          'status-green':  '#15803D',
          'status-red':    '#B91C1C',
          // Callout
          'warning-bg':     '#FEF3C7',
          'warning-border': '#F59E0B',
          'warning-text':   '#92400E',
          'info-border':    '#93C5FD',
        },
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
      },
      fontFamily: {
        sans: ["'Inter'", 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'Menlo', 'Monaco', "'Cascadia Code'", "'Segoe UI Mono'", "'Roboto Mono'", 'monospace'],
      },
      borderRadius: {
        lg: 'var(--radius)',              // 12px panels
        md: 'calc(var(--radius) - 4px)',  // 8px controls
        sm: 'calc(var(--radius) - 4px)',  // 8px controls
      },
      fontSize: {
        'micro': ['11px', { lineHeight: '1.3' }],
        'label': ['12px', { lineHeight: '1.4' }],
        'body':  ['14px', { lineHeight: '1.5' }],
        'title': ['16px', { lineHeight: '1.3' }],
        'display': ['18px', { lineHeight: '1.2' }],
      },
    },
  },
  plugins: [],
} satisfies Config
