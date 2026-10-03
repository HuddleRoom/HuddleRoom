import { useEffect } from 'react'

export function useDocumentTitle(pageName: string) {
  useEffect(() => {
    document.title = `${pageName} · HuddleRoom`
  }, [pageName])
}
