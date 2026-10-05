# HuddleRoom

Autonomous agent workspace platform. Agents self-organize around tasks. Humans observe and steer.

The initial public release supports a local, single-user SQLite installation only. Docker,
PostgreSQL, Redis, and authentication are not supported in this release. Keep the server on
your own machine; it has no access control.

## Install and run

Install a published release with [pipx](https://pipx.pypa.io/). pipx creates an
isolated Python environment and exposes the `huddleroom` command; there is no
separate pipx package format or registry. The package includes the dashboard and
database migrations, so this path needs neither Git nor Node. These commands apply
after the first PyPI release is published; until then, install a built local wheel.

```bash
pipx install huddleroom
huddleroom setup       # optional, recommended for guided configuration
huddleroom serve
# open http://127.0.0.1:8000/dashboard/
```

`serve` applies pending migrations before it starts and never prompts. It is valid
to start without a provider key when using authenticated CLI agents; API-backed
agents, orchestration, meeting control, and embeddings need their configured
provider credential when those operations run. `huddleroom init-db` remains an
idempotent diagnostic command for running the same migrations manually.

New installations store their database, workspace, and optional configuration in
`~/.huddleroom`. An existing `./huddleroom.db`, legacy `./rally.db`, or
`./workspace` is retained when present. HuddleRoom binds to `127.0.0.1:8000` by
default. The scheduler (APScheduler) and API server share one process. Docker,
PostgreSQL, Redis, and authentication are unsupported in this release.

### Upgrade and uninstall

```bash
pipx upgrade huddleroom
huddleroom serve

pipx uninstall huddleroom
```

Uninstalling the application retains `~/.huddleroom`, including its database and
configuration. Remove that data only when it is no longer needed:

```bash
rm -rf ~/.huddleroom
```

Legacy `RALLY_*` settings remain accepted for one release. They are deprecated; switch to `HUDDLEROOM_*` before the next release. The installed command is `huddleroom`.

### Develop from source

The published-package flow above is for users. Contributors who need a checkout
can instead use this source setup:

```bash
git config core.hooksPath .githooks
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
# Optional: configure ~/.huddleroom/config.toml or .env; see Configuration below.
make build-frontend
make serve
```

> To call LLMs from API adapter agents, set a provider credential in `~/.huddleroom/config.toml`, `.env`, or the process environment.

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

Run `huddleroom setup` for guided configuration, or configure HuddleRoom directly.
Setup writes only its own keys in `~/.huddleroom/config.toml`, preserves unrelated
TOML/comments, and asks for direct provider credentials with hidden input. It is
safe to rerun. Non-secret setup options are available for automation; provider
secrets are deliberately not accepted as command arguments.

Direct mode never requires or probes OneCLI. HuddleRoom automatically reads
`~/.huddleroom/config.toml` when it exists. Copy `config.toml.example` there to
start with every supported option when working from a source checkout. Pipx users
can create the file directly; it uses flat, lowercase application keys such as
`database_url`, while provider names remain uppercase, such as `OPENAI_API_KEY`.

```bash
mkdir -p ~/.huddleroom
cp config.toml.example ~/.huddleroom/config.toml
```

The working directory's `.env` remains supported. Its application settings use the
`HUDDLEROOM_` prefix; legacy `RALLY_` names remain accepted for one release with a
deprecation warning. Values resolve in this order: explicit `Settings(...)` values,
process environment, `.env`, `~/.huddleroom/config.toml`, then built-in defaults.
Provider credentials can be in either file or the process environment and use the
same process environment, `.env`, TOML precedence.

| Variable | Default | Description |
|----------|---------|-------------|
| `HUDDLEROOM_DATABASE_URL` | `sqlite+aiosqlite:///$HOME/.huddleroom/huddleroom.db` | Local SQLite database file; an existing CWD database wins when no value is supplied |
| `HUDDLEROOM_DEBUG` | `false` | Enable SQLAlchemy query logging and full redacted LLM completion exchanges. Debug logs may contain sensitive prompt/response data; use only in trusted environments. |
| `HUDDLEROOM_EMBEDDING_MODEL` | `text-embedding-3-small` | litellm embedding model for knowledge search |
| `HUDDLEROOM_ORCHESTRATION_MODEL` | `openai/gpt-4o-mini` | Model for orchestration decisions and meeting control |
| `HUDDLEROOM_MEETING_CONTROL_MODEL` | `HUDDLEROOM_ORCHESTRATION_MODEL` | Optional meeting-control override |

For API adapter agents to call LLMs, add provider credentials to either configuration file:

```bash
# ~/.huddleroom/config.toml
OPENAI_API_KEY = "sk-..."
ANTHROPIC_API_KEY = "sk-ant-..."

# or .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### OneCLI mode

OneCLI is optional. Select it in `huddleroom setup` only when a supported OneCLI
CLI, management service, and gateway are already available. Setup verifies
`/v1/health`, `/v1/agents`, `/v1/agents/{id}/effective-credentials`,
`/v1/agents/{id}/grants`, secret metadata at `/v1/secrets`, and gateway `/healthz`;
it then lets you select or create a gateway agent. The selected `onecli_agent` is a
OneCLI gateway identifier, not a HuddleRoom database agent.

The supported contract uses `onecli agents credentials`, `onecli agents grants
list`, `onecli agents grants attach-secret`, and `onecli run --agent --gateway`.
It reads only secret metadata and effective access, preserves existing grants, and
rechecks effective access after an additive grant. Automatic onboarding is limited
to verified OpenAI API-key (`api.openai.com`, Bearer) and Anthropic API-key
(`api.anthropic.com`, `x-api-key`) recipes. OAuth, OpenRouter, Gemini, Ollama,
and unknown credential types remain manual configuration paths.

OneCLI secrets are not stored in HuddleRoom TOML, logs, or command arguments.
The HuddleRoom CLI agent executable owns its own authentication and can run
without a provider API key. API-backed agents and control-plane operations still
need an effective provider credential through the selected gateway. OneCLI mode
fails closed if the CLI/service/gateway, selected agent, or required modern schema
is unavailable; it does not fall back to direct credentials.

The verified development CLI is OneCLI 2.11.0, but support is capability/schema
based rather than a claimed server-version floor. A management server that lacks
the required effective-credentials or grants endpoints is intentionally rejected
until it is upgraded.

### Troubleshooting

- If `huddleroom` is not found after `pipx install`, ensure pipx's binary directory
  is on `PATH` (`pipx ensurepath`, then open a new shell).
- If an API-backed operation reports missing credentials, add its provider key in
  the process environment, `.env`, or `~/.huddleroom/config.toml`, or configure
  effective access through the selected OneCLI gateway. Startup itself does not
  require a provider key.
- If setup or serve reports an unwritable data path, choose writable
  `--database-path`/`--workspace-dir` values in setup or fix permissions for
  `~/.huddleroom`; do not delete an existing database to recover.
- If migration fails, `serve` stops before opening the server. Correct the reported
  configuration/path issue and rerun `huddleroom init-db` or `huddleroom serve`.
- If OneCLI setup reports an unsupported CLI or management response, upgrade the
  CLI/service together to the supported capability/schema contract. HuddleRoom
  will not replace grants or retrieve secrets to work around that error.

### Maintainer release runbook

Publishing is intentionally not configured here. Before an approved release,
verify the PyPI `huddleroom` project name and ownership immediately before use,
register a PyPI trusted publisher for the exact repository, workflow, and
environment, choose a new unpublished package version, and create the matching
release tag. The package name/ownership check is still outstanding because its
PyPI lookup was deferred; do not assume the name is available.

From the release checkout, build the dashboard and distribution once, then inspect
and test those exact wheel and sdist artifacts:

```bash
rtk make build-frontend
rtk .venv/bin/python -m build
rtk .venv/bin/python -m pytest tests/test_distribution.py -q --tb=short
```

After the trusted-publisher setup has separately been approved, GitHub Actions
should run the focused distribution checks and publish those exact built artifacts
through PyPI OIDC. It must not rebuild the frontend on a user's machine. Finally,
verify `pipx install huddleroom` and `pipx upgrade huddleroom` against PyPI. Never
reuse an already published version.

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

The distribution acceptance test also requires a separately installed native [pipx](https://pipx.pypa.io/stable/installation/); it is not an application dependency.

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
