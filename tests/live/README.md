# Rally Live Test Suite

End-to-end integration tests that run against a live Rally server using real LLM calls.

## What It Tests

| ID | Name | Meeting Type | Strategy | Agents | Agenda Items |
|----|------|-------------|----------|--------|-------------|
| T6 | Sanity | decision | round_robin | architect + pragmatist | 1 |
| T7 | Architecture Debate | decision | round_robin | architect + pragmatist + pm | 2 |
| T8 | Security Review | review | moderated | architect + security + pm | 2 |
| T9 | Organizer Sprint | decision | organizer_controlled | all 4 (pm=organizer) | 3 |
| T10 | Standup | standup | round_robin | all 4 | 1 |

## Requirements

### Environment Variables

```bash
export OPENAI_API_KEY=your-key
```

All live agents in `tests/live/live_test.py` currently use provider `openi` with model `openai/gpt-5-nano`.

### Python Dependencies

```bash
.venv/bin/pip install httpx websockets
```

The live harness expects Rally's project venv at `.venv/` and uses `.venv/bin/alembic` plus `.venv/bin/rally`
when it starts the local server itself.

## How to Run

From the Rally project root:

```bash
# Run all tests (script starts its own server if none is running)
.venv/bin/python tests/live/live_test.py

# Skip server startup (server already running on port 8001)
.venv/bin/python tests/live/live_test.py --no-start-server

# Run specific tests only
.venv/bin/python tests/live/live_test.py --tests T6,T8

# Keep server running after tests finish
.venv/bin/python tests/live/live_test.py --keep-server

# Custom port
.venv/bin/python tests/live/live_test.py --port 8002

# Verbose: print full WS event payloads
.venv/bin/python tests/live/live_test.py --verbose
```

### CLI runtime smoke check

Run this through the configured onecli harness so Copilot, OpenCode, and pi use
the environment's OpenRouter configuration:

```bash
onecli run --agent rally-onecli -- .venv/bin/python tests/live/cli_runtime_smoke.py --model <provider/model>
```

The command uses a disposable workspace and asks each runtime for a fixed
no-tools response. It checks the installed version, noninteractive command,
JSON output, and session ID capture. Add `--verify-resume` to make a second
live call using that session. Add `--adapter-task` to run a persisted
`CliAdapter.run` task through a disposable SQLite database; this requires an
explicit model. Add `--verify-cancel` to verify live process-group cleanup.
Use `--runtimes copilot` to isolate one.

## What It Does

1. Checks if Rally is already running on port 8001.
2. If not, runs `.venv/bin/alembic upgrade head` then launches `.venv/bin/rally serve --port 8001`.
3. Creates or reuses project `rally-live-test`.
4. Creates or reuses 4 agents (`live-architect`, `live-pragmatist`, `live-security`, `live-pm`), all configured as `openi` / `openai/gpt-5-nano`.
5. For each meeting test:
   - Creates the meeting with `auto_start=True`.
   - Opens a WebSocket to `/ws/meetings/{id}` and streams events to stdout.
   - Waits up to 120 seconds for `meeting.concluded`.
   - Verifies transcript shape plus persisted meeting state via REST.
   - T10 additionally requires a terminal `meeting.concluded` WS event, final status `concluded`, `is_partial=false`, completed agenda, `updates_shared` resolution kind, exactly 4 participant turns, zero decisions, and `DONE:/NOW:/BLOCKERS:` standup formatting.
6. Prints a summary table.

## Debugging Failures

- Full WS event stream is printed during each test.
- On failure, the script prints the REST response body.
- Check server logs: `.venv/bin/rally serve` output is written to the harness log file when the script starts the server.
- For LLM errors, verify `OPENAI_API_KEY` and the agent/provider definitions in `tests/live/live_test.py`.

## Reuse Behavior

The script reuses existing project and agents by name. To start fresh:

```bash
# Delete the test project via API or use the Rally dashboard
http://localhost:8001/dashboard
```
