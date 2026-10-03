import React from 'react'

export function PageFrame({ children }: { children: React.ReactNode }) {
  return <div className="flex min-w-0 flex-1 flex-col gap-3 p-4 min-[900px]:p-6">{children}</div>
}
