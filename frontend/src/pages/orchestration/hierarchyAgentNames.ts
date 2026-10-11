import type { OrchestrationAgentReviewRecord } from '@/lib/types'

export function hierarchyAgentResolver(reviews: readonly OrchestrationAgentReviewRecord[]) {
  const names = new Map(reviews.flatMap((review) => {
    if (!review.agent_id) return []
    const name = typeof review.definition_snapshot.name === 'string' && review.definition_snapshot.name.trim()
      ? review.definition_snapshot.name
      : review.approved_for_work_functions.find((item): item is string => typeof item === 'string' && !!item.trim())
    return name ? [[review.agent_id, name] as const] : []
  }))
  return (value: unknown): string | null => {
    if (typeof value !== 'string' || !value.trim()) return null
    const isUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value)
    return names.get(value) ?? (isUuid ? null : value)
  }
}
