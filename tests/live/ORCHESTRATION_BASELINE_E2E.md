# Controlled orchestration baseline E2E

Use this from a fresh agent session to inspect and judge HuddleRoom's orchestration output. A terminal process status is not a pass.

## Controlled run

Stop any other HuddleRoom server using the same database, then run one phase:

```bash
uv run python tests/live/orchestration_baseline_e2e.py --controlled-server --step
```

The runner starts a loopback HuddleRoom child with debug enabled, captures its stdout/stderr, executes or inspects one phase, then always stops that child. It does not seed, reset, or configure the database.

`--step`/`--run` advance six phases in order: `goal_definition`, `manager_selection`, `agent_definition_review`, `team_hierarchy`, `effectiveness_review`, `goal_closeout`. Scope to one with `--phase <name>`. The last two are post-execution lifecycle processes: on a goal that has not executed they print a `SKIP:` line and move on — `effectiveness_review idle` when no trigger fired, `goal_closeout not ready` when closeout preconditions are unmet, or a completed closeout reports `no_executable_work: true` with `full_closeout: false`. A SKIP is a normal outcome, not an error, and writes no judgment packet. When they do produce output they can park at `waiting_decision` (a disposition/sign-off decision); the runner judges that parked output like any other phase rather than treating it as required user input.

Read the packet it prints at `tests/live/logs/orchestration-baseline-e2e/judgment-packet.json`. It contains the phase purpose and criteria, redacted request/response trace, terminal conversation emitted during this phase, process inputs/outputs, warnings, and any pending user decision. Supporting artifacts are `http-trace.jsonl`, `server.log`, `events.jsonl`, and `verdicts.jsonl` in the same directory.

## Judgment loop

Record only a clear, evidence-based judgment:

```bash
uv run python tests/live/orchestration_baseline_e2e.py --controlled-server --record pass --reason "State the packet evidence"
uv run python tests/live/orchestration_baseline_e2e.py --controlled-server --record minor --reason "Describe the concrete minor issue"
uv run python tests/live/orchestration_baseline_e2e.py --controlled-server --record needs_human_judgment --reason "Describe what is unclear"
```

`minor` creates an SPR. `needs_human_judgment` stops. On the next `--step`, HuddleRoom state is fetched again; a prior pass/minor is reused only when the same current process and evidence still match.

## User input

When the packet says `USER INPUT REQUIRED`, stop. It shows the exact question, options, and context. Do not automate a browser or guess an authority, security, or ambiguous product decision.

Only after the user explicitly delegates a low-risk clarification present in that packet may the agent answer it:

```bash
uv run python tests/live/orchestration_baseline_e2e.py --controlled-server --answer-delegated DECISION_ID:OPTION --reason "User explicitly delegated this low-risk clarification"
```

Then run `--step` again. The runner rejects stale packets, non-delegated decisions, and options not in the packet. Redact secrets from reasons and terminal conversation; never edit artifacts to force a resume.
