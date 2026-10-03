# HuddleRoom Dashboard Live UI Tests

These Playwright tests run Chromium against the built HuddleRoom dashboard served by a real `huddleroom serve` process.

## Command

From the repository root:

```bash
npm --prefix frontend run test:e2e:live
```

The command builds `frontend/dist`, copies it into `huddleroom/static/dashboard`, creates an isolated SQLite database under a run-specific temp directory, runs Alembic migrations, seeds deterministic data, starts HuddleRoom, runs Chromium tests, and stops the server.

## Environment

- `HUDDLEROOM_UI_TEST_PORT`: override the default port `39123`
- `HUDDLEROOM_UI_TEST_KEEP_DB=1`: keep the SQLite DB after teardown
- `HUDDLEROOM_UI_TEST_VIDEO=1`: retain failure videos locally

The suite runs in the supported local, unauthenticated mode. The seed helper creates the records required by existing UI scenarios.

## Artifacts

- Playwright traces/screenshots/videos: `frontend/test-results/live-ui/artifacts`
- Playwright HTML report: `frontend/playwright-report`
- HuddleRoom server log: printed by teardown and stored in the run temp directory
- Isolated DB: removed on success, preserved when `HUDDLEROOM_UI_TEST_KEEP_DB=1` or when the suite fails

## Coverage Model

Workflow specs cover navigation, dashboard quick actions, tasks, agents, meetings, protocols, knowledge, memory, rules, hooks, optimizations, and settings.

`support/control-map.ts` is the explicit inventory of visible, meaningful controls per primary route. Update it in the same change that adds, removes, or renames user-facing controls.
