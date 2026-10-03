import React, { useState } from 'react'
import type { OnMount } from '@monaco-editor/react'
import { LazyMonacoEditor } from '@/lib/lazyMonaco'
import { UI_COLORS } from '@/components/common/uiPrimitives'

interface TextareaThenMonacoProps {
  value: string
  onChange: (value: string) => void
  language: string
  height: string | number
  options?: Record<string, unknown>
}

const MONO = 'ui-monospace, Menlo, Monaco, "Cascadia Code", monospace'

export function TextareaThenMonaco({ value, onChange, language, height, options }: TextareaThenMonacoProps) {
  const [upgraded, setUpgraded] = useState(false)

  const textareaStyle: React.CSSProperties = {
    width: '100%',
    height,
    fontFamily: MONO,
    fontSize: 12,
    background: UI_COLORS.depth,
    color: UI_COLORS.textPrimary,
    border: `1px solid ${UI_COLORS.border}`,
    borderRadius: 3,
    padding: 8,
    resize: 'vertical',
    boxSizing: 'border-box',
  }

  const makePlainTextarea = (onFocus?: () => void, autoFocus?: boolean) => (
    <textarea
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onFocus={onFocus}
      style={textareaStyle}
      spellCheck={false}
      autoFocus={autoFocus}
    />
  )

  if (!upgraded) {
    return makePlainTextarea(() => setUpgraded(true))
  }

  const handleMount: OnMount = (editor) => {
    editor.focus()
  }

  return (
    <React.Suspense fallback={makePlainTextarea(undefined, true)}>
      <LazyMonacoEditor
        value={value}
        onChange={(v) => onChange(v ?? '')}
        language={language}
        height={height}
        options={options}
        onMount={handleMount}
      />
    </React.Suspense>
  )
}
