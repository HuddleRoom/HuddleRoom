import { useTaskCount } from './tasks'
import { useGraphRunCount } from './graphs'
import { useSessionCount } from './sessions'
import { useOrchestrationGoals } from '@/api/orchestration'

export function useDashboardTriage(projectId: string | null) {
  const blockedTasksQuery = useTaskCount(projectId, 'blocked')
  const failedTasksQuery = useTaskCount(projectId, 'failed')
  const failedGraphRunsQuery = useGraphRunCount(projectId, 'failed')
  const failedSessionsQuery = useSessionCount(projectId, 'failed')
  const goalsQuery = useOrchestrationGoals(projectId)

  const blockedTasksCount = blockedTasksQuery.data?.count
  const failedTasksCount = failedTasksQuery.data?.count
  const failedGraphRunsCount = failedGraphRunsQuery.data?.count
  const failedSessionsCount = failedSessionsQuery.data?.count

  // LM retries and unacknowledged warnings are already folded into needs_you_count by the backend — do not add separate fetches.
  // ponytail: only fetched pages counted, may undercount past page 1; acceptable for a dashboard triage rollup.
  const goals = goalsQuery.data?.pages.flatMap((p) => p.items) ?? []
  const needsYouGoalsList = goals.filter((g) => g.needs_you_count > 0)
  const blockedGoalsList = goals.filter((g) => g.status === 'blocked')

  const hasAttention =
    (blockedTasksCount ?? 0) > 0 ||
    (failedTasksCount ?? 0) > 0 ||
    (failedGraphRunsCount ?? 0) > 0 ||
    (failedSessionsCount ?? 0) > 0 ||
    needsYouGoalsList.length > 0 ||
    blockedGoalsList.length > 0

  const isLoading =
    blockedTasksQuery.isLoading ||
    failedTasksQuery.isLoading ||
    failedGraphRunsQuery.isLoading ||
    failedSessionsQuery.isLoading ||
    goalsQuery.isLoading

  const isError =
    blockedTasksQuery.isError ||
    failedTasksQuery.isError ||
    failedGraphRunsQuery.isError ||
    failedSessionsQuery.isError ||
    goalsQuery.isError

  return {
    blockedTasks: {
      count: blockedTasksCount,
      isLoading: blockedTasksQuery.isLoading,
      isError: blockedTasksQuery.isError,
    },
    failedTasks: {
      count: failedTasksCount,
      isLoading: failedTasksQuery.isLoading,
      isError: failedTasksQuery.isError,
    },
    failedGraphRuns: {
      count: failedGraphRunsCount,
      isLoading: failedGraphRunsQuery.isLoading,
      isError: failedGraphRunsQuery.isError,
    },
    failedSessions: {
      count: failedSessionsCount,
      isLoading: failedSessionsQuery.isLoading,
      isError: failedSessionsQuery.isError,
    },
    needsYouGoals: {
      count: goalsQuery.data ? needsYouGoalsList.length : undefined,
      isLoading: goalsQuery.isLoading,
      isError: goalsQuery.isError,
      goals: needsYouGoalsList,
    },
    blockedGoals: {
      count: goalsQuery.data ? blockedGoalsList.length : undefined,
      isLoading: goalsQuery.isLoading,
      isError: goalsQuery.isError,
      goals: blockedGoalsList,
    },
    hasAttention,
    isLoading,
    isError,
  }
}
