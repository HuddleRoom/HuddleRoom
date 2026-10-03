import type { FullResult, Reporter, TestCase, TestResult } from '@playwright/test/reporter'
import { readRunState, writeRunState } from './run-state'

function markFailed(): void {
  try {
    const state = readRunState()
    if (!state.failed) {
      writeRunState({ ...state, failed: true })
    }
  } catch {
    // Global setup owns the run-state file. If it does not exist yet, there is nothing to mark.
  }
}

class FailureReporter implements Reporter {
  onTestEnd(_test: TestCase, result: TestResult): void {
    if (result.status === 'failed' || result.status === 'timedOut' || result.status === 'interrupted') {
      markFailed()
    }
  }

  onEnd(result: FullResult): void {
    if (result.status !== 'passed') {
      markFailed()
    }
  }
}

export default FailureReporter
