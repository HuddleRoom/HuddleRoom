# HuddleRoom

Autonomous agent workspace platform. Agents self-organize around tasks. Humans observe and steer.

The initial public release supports a local, single-user SQLite installation only. Docker,
PostgreSQL, Redis, and authentication are not supported in this release. Keep the server on
your own machine; it has no access control.

## Install and run

Install from the current source repository, initialize its local SQLite database, then start a loopback-only server:

```bash
git clone https://github.com/HuddleRoom/HuddleRoom.git
cd HuddleRoom
npm ci --prefix frontend
make build-frontend
pipx install .
cp .env.example .env
huddleroom init-db
huddleroom serve
# open http://127.0.0.1:8000/dashboard/
```

HuddleRoom defaults to a local `huddleroom.db` SQLite file and binds to `127.0.0.1:8000`. The scheduler (APScheduler) and API server share the same process. Docker, PostgreSQL, Redis, and authentication are unsupported in this release.

Existing Rally installations can continue to use `rally` and `RALLY_*` settings for one release. They are deprecated; switch to `huddleroom` and `HUDDLEROOM_*` before the next release.

### Develop from source

From that checkout:

```bash
git config core.hooksPath .githooks
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
make build-frontend
huddleroom init-db
make serve
```

> To call LLMs from API adapter agents, edit `.env` and set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

---

## Makefile Reference

| Command | Description |
|---------|-------------|
| `make serve` | Start the development server (loopback only, no auth) |
| `make build-frontend` | Build the Vite app and copy generated assets into `huddleroom/static/dashboard/` |
| `make migrate-local` | Run Alembic migrations against local SQLite |
| `make migration-local msg="description"` | Generate new migration for local SQLite |
| `make test-local` | Run pytest against local environment |
| `make shell-local` | Launch local Python shell |

---

## Configuration

Settings use the `HUDDLEROOM_` prefix and can be set in `.env`. The `RALLY_` equivalents remain accepted for one release with a deprecation warning.

| Variable | Default | Description |
|----------|---------|-------------|
| `HUDDLEROOM_DATABASE_URL` | `sqlite+aiosqlite:///huddleroom.db` | Local SQLite database file |
| `HUDDLEROOM_DEBUG` | `false` | Enable SQLAlchemy query logging and full redacted LLM completion exchanges. Debug logs may contain sensitive prompt/response data; use only in trusted environments. |
| `HUDDLEROOM_EMBEDDING_MODEL` | `text-embedding-3-small` | litellm embedding model for knowledge search |
| `HUDDLEROOM_ORCHESTRATION_MODEL` | `openai/gpt-4o-mini` | Model for orchestration decisions and meeting control |
| `HUDDLEROOM_MEETING_CONTROL_MODEL` | `HUDDLEROOM_ORCHESTRATION_MODEL` | Optional meeting-control override |

For API adapter agents to call LLMs, add provider credentials to `.env`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## API Overview

Base URL: `http://127.0.0.1:8000/api/v1`

Authentication is not supported in the initial public release. All endpoints are open to
anyone who can reach the server, so bind it to `127.0.0.1`.

```
GET     /projects               List projects
POST    /projects               Create project
GET     /projects/{id}          Get project
PUT     /projects/{id}          Update project
DELETE /projects/{id}          Archive project (soft delete)

GET     /agents                 List agents
POST    /agents                 Create agent
GET     /agents/{id}            Get agent
PUT     /agents/{id}            Update agent
DELETE /agents/{id}            Deactivate agent
GET     /agents/{id}/context    Agent context (tasks, knowledge)
GET     /agents/{id}/sessions   Agent session history

GET     /projects/{pid}/tasks               List tasks
POST    /projects/{pid}/tasks               Create task
GET     /projects/{pid}/tasks/{id}          Get task
PUT     /projects/{pid}/tasks/{id}          Update task
DELETE /projects/{pid}/tasks/{id}          Cancel task
PATCH   /projects/{pid}/tasks/{id}/status   Transition status
POST    /projects/{pid}/tasks/{id}/assign   Assign agent
GET     /projects/{pid}/tasks/{id}/subtasks List subtasks
GET     /projects/{pid}/tasks/{id}/sessions List sessions

GET     /sessions               List sessions
POST    /sessions               Create/trigger session
GET     /sessions/{id}          Get session
POST    /sessions/{id}/cancel   Cancel session
GET     /sessions/{id}/output   Get session output

GET     /projects/{pid}/knowledge           List knowledge items
POST    /projects/{pid}/knowledge           Create knowledge item
POST    /projects/{pid}/knowledge/search    Semantic search
GET     /knowledge/{id}                     Get item
PUT     /knowledge/{id}                     Update item
DELETE /knowledge/{id}                     Delete item

```

Interactive docs: `http://127.0.0.1:8000/docs`

---

## Orchestrator conversation

Each goal includes Chat and bounded Investigation. Steering produces a proposal first; use **Submit** to apply it. A proposal alone does not change the goal.

---

## Task Status State Machine

```
backlog -> ready -> in_progress -> done
                 -> blocked      -> in_progress
                 -> cancelled
ready    -> cancelled
backlog -> cancelled
in_progress -> cancelled
```

Invalid transitions return `409 Conflict`.

---

## Agent Types

### API Adapter (`adapter_type: "api"`)

Calls an LLM via litellm. Assembles context from task, knowledge base, and channel history.

```json
{
   "name": "reviewer",
   "role": "reviewer",
   "provider": "openai",
   "model": "gpt-4o-mini",
   "adapter_type": "api",
   "system_prompt": "You are a code reviewer.",
   "config": {
     "temperature": 0.7,
     "max_tokens": 2048,
     "context_message_window": 20
   }
}
```

### CLI Adapter (`adapter_type: "cli"`)

Spawns a subprocess in the project's workspace. It writes `task.md` and
`huddleroom_context.json` under `.huddleroom/` (or the one-release legacy
`.rally/` directory) and captures the runtime output.

```json
{
   "name": "developer",
   "role": "developer",
   "provider": "anthropic",
   "model": "claude-opus-4-6",
   "adapter_type": "cli",
   "cli_runtime": "claude_code",
   "config": {
     "session_timeout_seconds": 3600
   }
}
```

Supported `cli_runtime` values: `claude_code`, `codex`, `aider`, `copilot`,
`opencode`, `pi`, and `custom`.

#### GitHub Copilot CLI, OpenCode, and pi

Install and authenticate each runtime separately before selecting it in the
agent form. HuddleRoom invokes the installed executable and does not manage
its login or provider credentials. The agent's **Model** value is forwarded as
`--model`; use a model identifier accepted by that runtime and its configured
provider.

| Runtime | Required local setup | Invocation permissions and workspace |
| --- | --- | --- |
| `copilot` | GitHub Copilot CLI installed and signed in | Uses the project workspace as both working directory and CLI workspace; runs noninteractively with `--no-ask-user --allow-all-tools`. |
| `opencode` | OpenCode installed with its provider authentication configured | Uses the project workspace as working directory and `--dir`; runs with OpenCode `--auto`. |
| `pi` | pi-agent.dev installed with its provider authentication configured | Uses the project workspace as working directory. `--no-approve` ignores pi project-local files; it does not restrict pi's built-in tools. |

These runtimes have workspace access rather than an HuddleRoom-enforced
sandbox. Give an agent only a workspace it may change. Live verification used
Copilot CLI 1.0.89, OpenCode 1.18.33, and pi 0.84.1. An environment configured
with OpenRouter through `onecli` is supported for that verification, but is not
a HuddleRoom product requirement.

Each runtime is requested to emit JSON output. HuddleRoom streams readable
stdout updates, saves the final parsed output and runtime session ID when one
is present, and records failures with redacted stderr. Cancelling a session,
a project reset, or a timeout terminates the runtime's full process group.
For non-orchestrated agent sessions, a returned runtime session ID is passed
to the runtime's resume flag on a retry; a runtime that does not return a
usable ID starts again from the persisted task context.

### Agent review conventions

Agent definitions may declare these optional `config` keys for deterministic reviews:

- `temperature`: API sampling temperature. Review and validation work should use `0.3` or lower.
- `reasoning_effort`: reasoning level; `none`, `minimal`, and `low` are weak for review and validation work.
- `tools`: descriptive tool context only. Actual permissions are controlled by the configured adapter and runtime.

---

## Cron Triggers

Tasks can auto-trigger sessions on a schedule:

```json
{
   "title": "Daily standup report",
   "trigger": {
     "type": "cron",
     "spec": "0 9 * * 1-5"
   }
}
```

Assign an agent, set status to `ready`. Cron triggers are evaluated by APScheduler in the local API process.

---

## Testing

```bash
# Local
make build-frontend
make test-local

# Frontend unit tests
npm --prefix frontend test

# Browser E2E (builds the dashboard, starts a local server)
npm --prefix frontend run test:e2e:live

# Live LLM tests (require API keys) — see tests/live/README.md
```

Tests use a separate database. Each test runs in a transaction that rolls back — no cleanup needed.

---

## Project Structure

```
huddleroom/                 # Main application package
├── adapters/              # API adapter (litellm) + CLI adapter (subprocess)
├── models/                # SQLAlchemy models
├── routers/               # FastAPI route handlers
├── schemas/               # Pydantic request/response schemas
├── services/              # Business logic
├── workers/               # Background tasks and local scheduler
├── static/                # Generated dashboard bundle + checked-in dev-dashboard
├── config.py              # Settings (pydantic-settings)
├── database.py            # Async engine + session factory
├── main.py                # App factory
└── dependencies.py        # Shared FastAPI dependencies

frontend/                   # Vite SPA dashboard

tests/                      # Test suite
├── conftest.py            # Shared fixtures
├── fixtures/              # Test data and helpers
├── live/                  # Live integration tests
└── test_*.py              # Unit and integration tests

alembic/                    # Database migrations
└── versions/              # Migration files

workspace/                  # Protocol and escalation YAML definitions

scripts/                    # WebSocket stdout helper utilities

.githooks/                  # Git hooks (commit-msg)

pyproject.toml              # Project configuration
Makefile                    # Build and development tasks
LICENSE                     # AGPL-3.0-or-later license
```

---

## License

HuddleRoom is licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See [LICENSE](LICENSE). If you run a modified version as a network service, the AGPL requires you to offer its source to its users.
