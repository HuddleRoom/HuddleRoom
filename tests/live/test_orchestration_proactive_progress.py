"""Bounded LIVE-model stage evaluation of the proactive-progress contract.

Run explicitly (default runs exclude the ``live`` marker and skip without the env flag)::

    HUDDLEROOM_LIVE_EVAL=1 .venv/bin/python -m pytest tests/live/test_orchestration_proactive_progress.py \
        -q --tb=short -m live [-k fresh]

What is live: ONLY the orchestration/supervision model calls (request_llm_decision -> adapter ->
orchestration_completion, and the scheduler's real OrchestrationSupervisionAnalyzer judge).
Everything else is local: fresh temp SQLite DB per scenario, baseline processes completed with the
conftest Safe* analyzers (setup, not the stage under test), TaskService.run replaced by a recorder,
any subprocess other than the orchestration ``codex`` backend blocked, no free-running scheduler.

Per-scenario caps (fail closed): 8 ticks, 3 actions/tick, 12 provider calls, 10 min wall, 150k tokens
(NON-CACHED = uncached input + output; provider-reported, chars/4 estimate of prompt+output when the backend reports none).
Reports: tests/live/logs/orchestration-proactive-progress/<scenario>-<utc>.json (logs/ is gitignored).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pwd
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

if os.environ.get("HUDDLEROOM_LIVE_EVAL") != "1":
    pytest.skip("live orchestration eval needs HUDDLEROOM_LIVE_EVAL=1", allow_module_level=True)

import huddleroom  # noqa: E402
from huddleroom.config import settings  # noqa: E402
from huddleroom.models.agent import Agent  # noqa: E402
from huddleroom.models.artifact import Artifact  # noqa: E402,F401
from huddleroom.models.base import Base  # noqa: E402
from huddleroom.models.graph import Graph  # noqa: E402
from huddleroom.models.meeting import Meeting, MeetingActionItem  # noqa: E402
from huddleroom.models.orchestration import (  # noqa: E402
    OrchestrationAction, OrchestrationDecision, OrchestrationEvidence, OrchestrationGate,
    OrchestrationRun, OrchestrationWait,
)
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision, OrchestrationWarning  # noqa: E402
from huddleroom.models.project import Project  # noqa: E402
from huddleroom.models.session import Session  # noqa: E402
from huddleroom.models.task import Task  # noqa: E402
from huddleroom.models.user import User  # noqa: E402
from huddleroom.security import hash_password  # noqa: E402
from huddleroom.services import orchestration_completion as completion_module  # noqa: E402
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService  # noqa: E402
from huddleroom.services.orchestration_decision_validator import ALLOWED_ACTION_SCHEMAS  # noqa: E402
from huddleroom.services.orchestration_service import OrchestrationService  # noqa: E402
from huddleroom.services.orchestration_supervision_scheduler import OrchestrationSupervisionScheduler  # noqa: E402
from huddleroom.services.secret_redaction import redact_secrets  # noqa: E402
from huddleroom.services.task_service import TaskService  # noqa: E402
from tests.conftest import _seed_anon_actors  # noqa: E402
from tests.test_orchestration_runtime_e2e import _agent, _authorized_run, _seed_accepted_plan  # noqa: E402

pytestmark = [pytest.mark.live, pytest.mark.asyncio]

CHECKOUT = Path(__file__).resolve().parents[2]
LOG_DIR = Path(__file__).resolve().parent / "logs" / "orchestration-proactive-progress"
MAX_TICKS, MAX_ACTIONS_PER_TICK, MAX_CALLS, MAX_WALL_SECONDS = 8, 3, 12, 600
# 150k cap applies to NON-CACHED tokens; cached/raw totals are reported separately.
MAX_TOKENS = int(os.environ.get("HUDDLEROOM_LIVE_EVAL_MAX_TOKENS", "150000"))
SUBSTANTIVE = frozenset({
    "request_plan", "accept_plan", "create_delegation_task", "request_verification", "retry_task",
    "reassign_task", "schedule_meeting", "start_graph", "request_plan_revision", "request_roadmap_replan",
    "suggest_agent",
})
_ENV_KEYS = {  # only these .env keys are ever read; nothing else from the file is touched
    "HUDDLEROOM_ORCHESTRATION_BACKEND": "orchestration_backend",
    "HUDDLEROOM_ORCHESTRATION_CLI_MODEL": "orchestration_cli_model",
    "HUDDLEROOM_ORCHESTRATION_EFFORT": "orchestration_effort",
}


class CapExceeded(AssertionError):
    pass


class Budget:
    def __init__(self):
        self.calls, self.tokens, self.estimated = 0, 0, False  # tokens = NON-CACHED (cap basis)
        self.raw_total, self.cached, self.uncached_flagged, self.last_codex = 0, 0, [], None
        self.started = time.monotonic()
        self.armed, self.fail_mode, self.cap_hit = False, False, None
        self.records, self.denied_subprocess, self.task_runs, self.stay_failed = [], [], [], set()


# ── redaction / summaries ────────────────────────────────────────────────────────────────────
def _prompt_summary(request: dict) -> dict:
    out = {"model": request.get("model"), "messages": []}
    for message in request.get("messages", []):
        content = message.get("content") or ""
        item = {"role": message.get("role"), "chars": len(content), "sha1": hashlib.sha1(content.encode()).hexdigest()[:12]}
        if message.get("role") == "user":
            try:
                parsed = json.loads(content)
                item["context_keys"] = {k: len(json.dumps(v, default=str)) for k, v in sorted(parsed.items())} if isinstance(parsed, dict) else "non-object"
            except ValueError:
                item["head"] = redact_secrets(content[:200])
        else:
            item["head"] = redact_secrets(content[:200])
        out["messages"].append(item)
    return out


def _response_text(response) -> str:
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return repr(response)[:2000]


# ── environment / harness ────────────────────────────────────────────────────────────────────
def _dotenv_provider() -> dict:
    values = {}
    for line in (CHECKOUT / ".env").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() in _ENV_KEYS:
            values[_ENV_KEYS[key.strip()]] = value.strip().strip("'\"")
    return values


def _patch_session_factory(monkeypatch, factory):
    for name, module in list(sys.modules.items()):
        if name.startswith("huddleroom") and module is not None and hasattr(module, "AsyncSessionLocal"):
            monkeypatch.setattr(module, "AsyncSessionLocal", factory)


async def _preflight(factory):
    """Running interpreter imports THIS checkout and the tick-owned-transaction marker behaves."""
    assert Path(huddleroom.__file__).resolve().is_relative_to(CHECKOUT), f"huddleroom imported from {huddleroom.__file__}"
    service, marker = OrchestrationService(), "orchestration_tick_owns_transaction"
    async with factory() as db:
        goal_id = uuid.uuid4()
        async with service._lock_goal_for_baseline_transition(db, goal_id, tick_owns_transaction=True):
            assert db.info.get(marker) is True, "tick-owned-transaction marker not set while tick owns the transaction"
        assert marker not in db.info, "tick-owned-transaction marker leaked after lock exit"
        async with service._lock_goal_for_baseline_transition(db, goal_id):
            assert marker not in db.info, "tick-owned-transaction marker set although tick does not own the transaction"


@pytest_asyncio.fixture
async def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", pwd.getpwuid(os.getuid()).pw_dir)  # tests/conftest points HOME at a temp dir; codex auth lives in the real one
    provider = _dotenv_provider()
    if provider.get("orchestration_backend") not in {"codex", "claude", "api"}:
        pytest.skip("HuddleRoom/.env does not configure HUDDLEROOM_ORCHESTRATION_BACKEND")
    for field, value in provider.items():
        monkeypatch.setattr(settings, field, value)
    monkeypatch.setattr(settings, "orchestration_max_actions_per_tick", MAX_ACTIONS_PER_TICK)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(settings, "workspace_dir", str(workspace))

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'live.db'}", connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine.sync_engine, "connect")
    def _pragmas(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _seed_anon_actors(conn)
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    _patch_session_factory(monkeypatch, factory)

    budget = Budget()
    real_completion = completion_module.orchestration_completion

    async def counting_completion(**request):
        if not budget.armed:
            raise CapExceeded("provider call outside an armed window (seeding must not call the model)")
        elapsed = time.monotonic() - budget.started
        for name, hit in (("calls", budget.calls >= MAX_CALLS), ("tokens", budget.tokens >= MAX_TOKENS), ("wall", elapsed >= MAX_WALL_SECONDS)):
            if hit:
                budget.cap_hit = budget.cap_hit or name
                raise CapExceeded(f"cap reached: {name}")
        budget.calls += 1
        record = {"call": budget.calls, "prompt": _prompt_summary(request), "started_s": round(elapsed, 1)}
        budget.records.append(record)
        if budget.fail_mode:
            record["error"] = "simulated provider failure (no live call)"
            raise completion_module.OrchestrationBackendError(completion_module.OrchestrationBackendErrorKind.TIMEOUT, "simulated provider timeout")
        try:
            response = await asyncio.wait_for(real_completion(**request), timeout=max(1.0, MAX_WALL_SECONDS - elapsed))
        except Exception as exc:
            record["error"] = redact_secrets(f"{type(exc).__name__}: {exc}")[:500]
            raise
        text = _response_text(response)
        usage = response.get("usage") if isinstance(response, dict) else None
        detail, budget.last_codex = budget.last_codex, None
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
            total = usage["total_tokens"]
            if detail is not None:  # codex: input_tokens includes cached input
                cached, used = detail["cached"], detail["input"] - detail["cached"] + detail["output"]
            else:  # cached info unavailable: fail-safe, count the full total and flag it
                cached, used = 0, total
                budget.uncached_flagged.append(budget.calls)
        else:
            total = used = (sum(m["chars"] for m in record["prompt"]["messages"]) + len(text)) // 4
            cached, budget.estimated = 0, True
        budget.raw_total += total
        budget.cached += cached
        budget.tokens += used
        if budget.tokens > MAX_TOKENS:  # post-call: the call that crosses the cap is still a hard failure
            budget.cap_hit = budget.cap_hit or "tokens"
        record.update(tokens_total=total, tokens_cached=cached, cached_info_missing=budget.calls in budget.uncached_flagged)
        record.update(tokens=used, output=redact_secrets(text)[:4000])
        return response

    monkeypatch.setattr(completion_module, "orchestration_completion", counting_completion)
    real_codex_usage = completion_module._codex_usage

    def capturing_codex_usage(raw):  # production drops cached_input_tokens; record it here, return unchanged
        for line in reversed(raw.splitlines()):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            u = ev.get("usage") if isinstance(ev, dict) and ev.get("type") == "turn.completed" else None
            if isinstance(u, dict) and isinstance(u.get("input_tokens"), int) and isinstance(u.get("output_tokens"), int) \
                    and isinstance(u.get("cached_input_tokens"), int):
                budget.last_codex = {"input": u["input_tokens"], "output": u["output_tokens"], "cached": u["cached_input_tokens"]}
                break
        return real_codex_usage(raw)

    monkeypatch.setattr(completion_module, "_codex_usage", capturing_codex_usage)

    async def no_run(self, db, project_id, task_id, *args, **kwargs):  # never launch a real agent CLI
        budget.task_runs.append(str(task_id))
        task = await self.get_or_404(db, project_id, task_id)
        if task.id not in budget.stay_failed:  # a run that fails again immediately leaves the task failed
            task.status = "in_progress"  # what a started run does to the task; no session, no CLI
        await db.flush()
        return task, uuid.uuid4()

    monkeypatch.setattr(TaskService, "run", no_run)

    real_exec = asyncio.create_subprocess_exec

    async def guarded_exec(*args, **kwargs):
        exe, rest = os.path.basename(str(args[0])), [str(a) for a in args[1:]]
        ok = exe == "codex" and (rest[:2] == ["login", "status"] or (rest[:1] == ["exec"] and "read-only" in rest))
        if not ok:
            budget.denied_subprocess.append([exe, *rest[:2]])
            raise AssertionError(f"live harness blocked subprocess: {exe} {rest[:2]}")
        return await real_exec(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", guarded_exec)

    await _preflight(factory)
    yield SimpleNamespace(factory=factory, budget=budget, tmp=tmp_path, provider=provider)
    await engine.dispose()


# ── durable-state capture ────────────────────────────────────────────────────────────────────
def _j(value):
    return json.loads(json.dumps(value, default=str))


async def snapshot(db: AsyncSession, run_id) -> dict:
    run = await db.get(OrchestrationRun, run_id, populate_existing=True)
    async def rows(model, *where):
        return list((await db.scalars(select(model).where(*where).order_by(model.id))).all())

    actions = await db.scalars(select(OrchestrationAction).where(OrchestrationAction.run_id == run_id).order_by(OrchestrationAction.created_at, OrchestrationAction.id))
    decisions = await db.scalars(select(OrchestrationDecision).where(OrchestrationDecision.run_id == run_id).order_by(OrchestrationDecision.created_at, OrchestrationDecision.id))
    waits = await rows(OrchestrationWait, OrchestrationWait.run_id == run_id)
    authority = await rows(OrchestrationAuthorityDecision, OrchestrationAuthorityDecision.goal_id == run.goal_id)
    goal = run.goal_id
    tasks = await rows(Task)  # fresh DB per scenario: every task belongs to this scenario
    warnings = await rows(OrchestrationWarning, OrchestrationWarning.goal_id == goal)
    return {
        "run": {"status": run.status, "phase": run.phase, "blockers": _j(run.active_blockers or []), "plan_status": (run.plan_state or {}).get("status"),
                "supervision": {k: (run.supervision_state or {}).get(k) for k in ("judgment_failures", "judgment_in_flight", "judgment_dirty", "judgment_due_at", "needs_judgment", "unchanged", "no_progress_asks")}},
        "actions": [{"id": str(a.id), "type": a.action_type, "status": a.status, "key": a.idempotency_key[-60:], "target": str(a.target_id) if a.target_id else None,
                     "request": _j(a.request), "contract": _j(a.dispatch_contract), "error": a.error} for a in actions],
        "decisions": [{"id": str(d.id), "action": (d.parsed_decision or {}).get("action_type"), "status": d.validator_status, "rejection": d.rejection_reason,
                       "parsed": _j(d.parsed_decision)} for d in decisions],
        "waits": [{"id": str(w.id), "key": w.wait_key[-60:], "status": w.status, "owner": _j(w.owner), "awaited_event": _j(w.awaited_event),
                   "due_recheck_at": str(w.due_recheck_at), "fallback": _j(w.fallback)} for w in waits],
        "authority": [{"id": str(a.id), "key": a.decision_key, "status": a.status, "question": a.question, "selected": a.selected_option} for a in authority],
        "tasks": [{"id": str(t.id), "title": t.title, "status": t.status} for t in tasks],
        "meetings": [{"id": str(m.id), "title": m.title, "status": m.status, "source_task_id": str(m.source_task_id) if m.source_task_id else None}
                     for m in await rows(Meeting)],
        "graphs": [str(g.id) for g in await rows(Graph)],
        "warnings": [{"type": w.warning_type, "active": w.active} for w in warnings],
    }


def _sig(snap: dict):
    s = lambda key, *fields: tuple(tuple(i[f] for f in fields) for i in snap[key])  # noqa: E731
    return (snap["run"]["status"], snap["run"]["phase"], s("actions", "id", "status"), s("waits", "id", "status"),
            s("tasks", "id", "status"), s("authority", "id", "status"), s("meetings", "id", "status"), len(snap["graphs"]))


def _wait_gaps(wait: dict) -> list[str]:
    gaps = []
    if not (wait["awaited_event"] or {}).get("event_type"):
        gaps.append("dependency/awaited event")
    if not ((wait["owner"] or {}).get("type") and (wait["owner"] or {}).get("id")):
        gaps.append("owner")
    fallback = wait["fallback"] or {}
    if not str(fallback.get("expected_result") or "").strip():
        gaps.append("expected evidence")
    if fallback.get("action_type") not in {"continue", "attention"}:
        gaps.append("fallback")
    if not wait["due_recheck_at"] or wait["due_recheck_at"] == "None":
        gaps.append("recheck")
    return gaps


def check_tick(before: dict, after: dict, *, tick_result=None, provider_failing=False) -> tuple[list[str], str]:
    """Global invariant: substantive action, bounded wait, or actionable human blocker; no non-dispatchable names."""
    violations = []
    seen = {d["id"] for d in before["decisions"]}
    for decision in (d for d in after["decisions"] if d["id"] not in seen):
        if provider_failing and decision["action"] == "invalid_llm_output":
            continue  # adapter's own failure marker under a simulated provider outage; the invariant below must still hold
        if decision["action"] not in ALLOWED_ACTION_SCHEMAS:
            violations.append(f"non-dispatchable decision action {decision['action']!r}: {decision['rejection']}")
        elif decision["status"] != "accepted":
            violations.append(f"decision {decision['action']} not accepted: {decision['rejection']}")
    old_actions = {a["id"] for a in before["actions"]}
    new_actions = [a for a in after["actions"] if a["id"] not in old_actions]
    old_tasks = {t["id"]: t["status"] for t in before["tasks"]}
    substantive = any(a["type"] in SUBSTANTIVE and a["status"] == "completed" for a in new_actions) or any(
        old_tasks.get(t["id"]) != t["status"] for t in after["tasks"])
    open_waits = [w for w in after["waits"] if w["status"] == "open"]
    for wait in open_waits:
        if gaps := _wait_gaps(wait):
            violations.append(f"open wait {wait['key']} missing {gaps}")
    bounded_wait = bool(open_waits) and not any(_wait_gaps(w) for w in open_waits)
    blocker = (
        any(a["status"] == "pending" and a["question"].strip() for a in after["authority"])
        or any(a["type"] == "ask_human" and a["status"] == "completed" and str(a["request"].get("question", "")).strip() for a in new_actions)
        or any(b.get("reason") for b in after["run"]["blockers"] if isinstance(b, dict))
    )
    terminal = after["run"]["status"] in {"completed", "cancelled", "failed"}
    # A live durable source (active meeting/graph) or a registered wait is itself the named dependency.
    outcome = ((tick_result or {}).get("authorized_execution") or {}).get("outcome")
    durable = outcome in {"durable_source_active", "waiting"} and not any(_wait_gaps(w) for w in open_waits)
    if not (substantive or bounded_wait or blocker or terminal or durable):
        violations.append("no substantive action, bounded wait, or actionable human blocker after tick")
    kind = "substantive" if substantive else "bounded_wait" if bounded_wait else "human_blocker" if blocker else "durable_hold" if durable else "terminal" if terminal else "NONE"
    return violations, kind


def _duplicates(snap: dict) -> list[str]:
    out = []
    done = [(a["type"], json.dumps(a["request"], sort_keys=True), a["target"]) for a in snap["actions"] if a["status"] == "completed" and a["type"] in SUBSTANTIVE]
    out += [f"duplicate completed action {d[0]}" for d in {x for x in done if done.count(x) > 1}]
    keys = [a["key"] for a in snap["authority"] if a["status"] == "pending"]
    out += [f"duplicate pending authority decision {k}" for k in set(keys) if keys.count(k) > 1]
    live = [(m["title"], m["source_task_id"]) for m in snap["meetings"] if m["status"] in {"scheduled", "preparing", "active", "concluding"}]
    out += [f"duplicate live meeting {m[0]}" for m in {x for x in live if live.count(x) > 1}]
    return out


# ── driver ───────────────────────────────────────────────────────────────────────────────────
async def _force_judge_due(env, run_id):
    async with env.factory() as db:
        run = await db.get(OrchestrationRun, run_id)
        state = dict(run.supervision_state or {})
        state.update(judgment_dirty=True, judgment_due_at=(datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat())
        run.supervision_state = state
        await db.commit()


async def _supervision_keys(env, run_id):
    async with env.factory() as db:
        run = await db.get(OrchestrationRun, run_id, populate_existing=True)
        st = run.supervision_state or {}
        return {k: st.get(k) for k in ("unchanged", "no_progress_asks", "needs_judgment", "judgment_dirty", "judgment_in_flight")}


async def answer_via_runtime_path(env, decision_id):
    """Answer exactly like POST .../decisions/{id}/answer does for runtime decisions (answer_runtime_question)."""
    async with env.factory() as s:
        user = await s.scalar(select(User).where(User.email == "live-eval-owner@example.com"))
        if user is None:
            user = User(email="live-eval-owner@example.com", hashed_password=hash_password("pw-not-used"), display_name="Live Eval Owner", role="member")
            s.add(user)
            await s.flush()
        decision = await s.get(OrchestrationAuthorityDecision, decision_id)
        option = next(o["key"] if isinstance(o, dict) else o for o in decision.options)  # free text is not part of this path
        result = await OrchestrationAuthorityDecisionService().answer_runtime_question(
            s, decision, option, actor_user_id=user.id, contract_version=decision.contract_version)
        await s.commit()
        return {"decision_id": str(decision.id), "selected_option": option, "continuation_applied": result.continuation_applied,
                "question": decision.question[:300], "note": "neutral: no preference; proceed with best judgment (runtime path offers option keys only)"}


async def auto_answer_model_questions(env, run_id):
    """Answer pending decisions created by MODEL actions; leave the system no-progress ask pending."""
    answered = []
    async with env.factory() as s:
        pending = (await s.scalars(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.run_id == run_id, OrchestrationAuthorityDecision.status == "pending",
            OrchestrationAuthorityDecision.runtime_identity.is_not(None)))).all()
        system_ids = set((await s.scalars(select(OrchestrationAction.target_id).where(
            OrchestrationAction.run_id == run_id, OrchestrationAction.action_type == "ask_human",
            OrchestrationAction.idempotency_key.like("%:ask_human:no_progress:%")))).all())
        ids = [d.id for d in pending if d.id not in system_ids]
    for decision_id in ids:
        answered.append(await answer_via_runtime_path(env, decision_id))
    return answered


async def drive(env, run_id, *, judge=False, max_ticks=MAX_TICKS - 1, fail_judge=False, refail=None, stop_when=None, sleep=0.0, expire_model_waits=False, auto_answer=False, tolerate_409=False):
    """Explicit ticks only (never the scheduler loop). Returns the per-tick report list."""
    budget, service, ticks = env.budget, OrchestrationService(), []
    budget.armed = True
    async with env.factory() as db:
        previous = await snapshot(db, run_id)
    stable, completed_naturally = 0, False
    for index in range(1, max_ticks + 1):
        if index > 1 and sleep:
            await asyncio.sleep(sleep)  # real elapsed time for the time-gated no-progress counter
        calls0, tokens0 = budget.calls, budget.tokens
        entry = {"tick": index, "calls": [], "error": None, "judge_result": None}
        try:
            async with env.factory() as db:
                result = await service.tick(db, run_id)
                entry["tick_result"] = _j({k: result.get(k) for k in ("status", "run_completed", "authorized_execution", "action_ids") if k in result})
            entry["supervision_after_tick"] = await _supervision_keys(env, run_id)
            if judge:
                await _force_judge_due(env, run_id)
                async with env.factory() as db:
                    entry["judge_result"] = await OrchestrationSupervisionScheduler().evaluate_run(db, run_id)
            entry["supervision_after_judge"] = await _supervision_keys(env, run_id)
        except Exception as exc:  # noqa: BLE001 - recorded, then asserted by the test
            entry["error"] = redact_secrets(f"{type(exc).__name__}: {exc}")[:800]
        async with env.factory() as db:
            # Stand-in for the (not running) task runner: released backlog tasks start. No session, no CLI.
            for task in (await db.scalars(select(Task).where(Task.status == "backlog"))).all():
                task.status = "in_progress"
            if expire_model_waits:  # the model's own wait elapsed / its wake event fired between ticks
                for wait in (await db.scalars(select(OrchestrationWait).where(OrchestrationWait.run_id == run_id, OrchestrationWait.status == "open"))).all():
                    if (wait.owner or {}).get("type") not in {"session", "task", "authority_decision", "child_run"}:
                        wait.status, wait.cleared_at = "cleared", datetime.now(timezone.utc)
            for task_id in refail or ():  # runner keeps failing this task again (no new session)
                (await db.get(Task, task_id)).status = "failed"
            await db.commit()
            current = await snapshot(db, run_id)
        entry["calls"] = [r for r in budget.records if r["call"] > calls0]
        entry["provider_calls"], entry["tokens"] = budget.calls - calls0, budget.tokens - tokens0
        entry["violations"], entry["progress_kind"] = check_tick(previous, current, tick_result=entry.get("tick_result"), provider_failing=budget.fail_mode)
        entry["state"] = current
        entry["auto_answers"] = await auto_answer_model_questions(env, run_id) if auto_answer and not (stop_when and stop_when(current)) else []
        ticks.append(entry)
        if entry["error"] and tolerate_409 and entry["error"].startswith("HTTPException: 409"):
            entry["tolerated_error"], entry["error"] = entry["error"], None  # a scheduler would retry on its next sweep; recorded as a finding
        if entry["error"] or budget.cap_hit:
            break
        stable = stable + 1 if _sig(current) == _sig(previous) else 0
        previous = current
        if stop_when is not None:
            if stop_when(current):
                completed_naturally = True
                break
            continue
        if stable >= 1 or current["run"]["status"] in {"completed", "cancelled", "failed"}:
            completed_naturally = True
            break
    return ticks, completed_naturally


async def replay_last_tick(env, run_id):
    """Re-run one more tick: no duplicate meeting/graph/authority request/substantive action may appear."""
    budget, calls0 = env.budget, env.budget.calls
    async with env.factory() as db:
        before = await snapshot(db, run_id)
    error = None
    try:
        async with env.factory() as db:
            await OrchestrationService().tick(db, run_id)
    except Exception as exc:  # noqa: BLE001
        error = redact_secrets(f"{type(exc).__name__}: {exc}")[:800]
    async with env.factory() as db:
        after = await snapshot(db, run_id)
    problems = _duplicates(after)
    if budget.calls == calls0:  # same context -> must be a pure replay
        for key in ("meetings", "graphs", "authority", "tasks"):
            if len(after[key]) != len(before[key]):
                problems.append(f"replay without model call changed {key}: {len(before[key])} -> {len(after[key])}")
    return {"error": error, "model_calls": budget.calls - calls0, "problems": problems, "state": after}


# ── seeding ──────────────────────────────────────────────────────────────────────────────────
async def seed_base(env, *, with_plan_items=None):
    async with env.factory() as s:
        project = Project(name="Live eval project", description="Isolated live evaluation", workspace_path=str(env.tmp / "workspace"), config={})
        planner, dev, reviewer, writer = (_agent("planner", ["planning"]), _agent("dev", ["implementation"]),
                                          _agent("reviewer", ["validation", "review"]), _agent("writer", ["summarization"]))
        s.add_all([project, planner, dev, reviewer, writer])
        await s.flush()
        service, goal, run = await _authorized_run(s, project)
        if with_plan_items:
            await _seed_accepted_plan(s, project, service, run, planner, plan_items=with_plan_items(dev))
        await s.commit()
        return SimpleNamespace(project_id=project.id, goal_id=goal.id, run_id=run.id, planner=planner.id, dev=dev.id, reviewer=reviewer.id,
                               extra={})


def _items(dev):
    return [
        {"id": f"item-{n}", "work_function": "implementation", "scope": f"Implement slice {n} of the accepted plan.",
         "deliverable": "A code change plus test output.", "agent_id": str(dev.id)} for n in (1, 2)]


def _run_task(project_id, run_id, title, status, assigned_to, **meta):
    return Task(project_id=project_id, title=title, status=status, assigned_to=assigned_to,
                metadata_={"orchestration": {"run_id": str(run_id), **meta}})


async def seed_fresh(env):
    return await seed_base(env)


async def seed_meeting_plus_work(env):
    seed = await seed_base(env, with_plan_items=_items)
    async with env.factory() as s:
        source = _run_task(seed.project_id, seed.run_id, "Design review prep", "in_progress", seed.dev)
        s.add(source)
        await s.flush()
        s.add(Meeting(project_id=seed.project_id, title="Design sync", meeting_type="decision", status="active",
                      source_task_id=source.id, participant_agent_ids=[str(seed.dev), str(seed.reviewer)],
                      trigger_reason=f"Orchestration run {seed.run_id}"))
        await s.commit()
        seed.extra["source_task_id"] = source.id
    return seed


async def seed_recoverable_blocker(env):
    seed = await seed_base(env, with_plan_items=_items)
    async with env.factory() as s:
        failed = _run_task(seed.project_id, seed.run_id, "Implement slice 1", "failed", seed.dev, plan_item_id="item-1")
        s.add(failed)
        await s.flush()
        s.add(Session(project_id=seed.project_id, task_id=failed.id, agent_id=seed.dev, adapter_type="api", status="failed",
                      output="Transient failure: package index timed out; nothing was changed.", metadata_={}, origin="auto"))
        await s.commit()
        seed.extra["failed_task_id"] = failed.id
    return seed


async def _pending_decision(env, seed, key="scope:benchmark-dataset"):
    async with env.factory() as s:
        asked = OrchestrationAction(run_id=seed.run_id, idempotency_key=f"live-eval:ask_human:{uuid.uuid4()}", action_type="ask_human",
                                    request={"action_type": "ask_human", "question": "May the benchmark use the local staging dataset?"}, status="completed")
        s.add(asked)  # answered-decision follow-ups only track decisions raised by an ask_human action
        await s.flush()
        decision = await OrchestrationAuthorityDecisionService().create_pending(
            s, seed.goal_id, decision_key=key, title="Benchmark dataset", authority="human", run_id=seed.run_id, related_action_id=asked.id,
            question="May the benchmark use the local staging dataset instead of production exports?",
            options=[{"key": "approve", "label": "Use staging"}, {"key": "deny", "label": "Do not use staging"}])
        await s.commit()
        return decision.id


async def seed_authority_decision(env):
    seed = await seed_base(env)
    seed.extra["decision_id"] = await _pending_decision(env, seed)
    return seed


async def seed_answered_decision(env):
    seed = await seed_base(env)
    decision_id = await _pending_decision(env, seed)
    async with env.factory() as s:
        user = User(email=f"live-{uuid.uuid4().hex[:8]}@example.com", hashed_password=hash_password("pw-not-used"), display_name="Live Eval Owner", role="member")
        s.add(user)
        await s.flush()
        decision = await s.get(OrchestrationAuthorityDecision, decision_id)
        await OrchestrationAuthorityDecisionService().answer_decision(s, decision, selected_option="approve", reason="Staging data is fine.", decided_by_user_id=user.id)
        s.add(_run_task(seed.project_id, seed.run_id, "Unrelated housekeeping", "done", seed.dev))  # newer, unrelated activity
        await s.commit()
    seed.extra["decision_id"] = decision_id
    return seed


async def seed_no_progress_wait(env):
    seed = await seed_base(env, with_plan_items=_items)
    async with env.factory() as s:
        s.add(_run_task(seed.project_id, seed.run_id, "Implement slice 1 (in flight)", "in_progress", seed.dev, plan_item_id="item-1"))
        await s.commit()
    return seed


async def seed_analyzer_failure(env):
    return await seed_base(env, with_plan_items=_items)  # no in-flight source -> local liveness says "continue" -> judge needed


async def seed_closeout_commitment(env):
    """Mirrors tests/test_orchestration_goal_closeout.py::completion_ready_goal, plus an unfinished meeting commitment."""
    seed = await seed_base(env)
    async with env.factory() as s:
        run = await s.get(OrchestrationRun, seed.run_id)
        gate = OrchestrationGate(run_id=run.id, success_criterion_key="plan_item:done", gate_type="work_completed",
                                 required_evidence={"success_criterion_keys": ["done"]}, status="accepted")
        final_gate = OrchestrationGate(run_id=run.id, success_criterion_key="final_summary", gate_type="final_summary_accepted",
                                       required_evidence={}, status="accepted")
        s.add_all([gate, final_gate])
        await s.flush()
        evidence = OrchestrationEvidence(run_id=run.id, gate_id=gate.id, source_type="verification", source_id=seed.goal_id, verdict="accepted", evidence_metadata={})
        summary_task = _run_task(seed.project_id, run.id, "Summarize closeout", "done", seed.dev,
                                 gate_id=str(final_gate.id), work_function="summarization", final_summary=True)
        source = _run_task(seed.project_id, run.id, "Source work", "done", seed.dev)
        s.add_all([evidence, summary_task, source])
        await s.flush()
        session = Session(project_id=seed.project_id, task_id=summary_task.id, agent_id=seed.dev, adapter_type="api", status="completed", metadata_={}, origin="auto",
                          output=json.dumps({"summary": "Work is complete.", "criteria": [{"criterion_key": "done", "evidence_ids": [str(evidence.id)]}], "unresolved_gaps": []}))
        s.add(session)
        await s.flush()
        s.add(OrchestrationEvidence(run_id=run.id, gate_id=final_gate.id, source_type="session", source_id=session.id, producer_agent_id=seed.dev, verdict="accepted", evidence_metadata={}))
        meeting = Meeting(project_id=seed.project_id, title="Wrap-up sync", meeting_type="standup", source_task_id=source.id, status="completed")
        s.add(meeting)
        await s.flush()
        item = MeetingActionItem(meeting_id=meeting.id, description="Write the rollback note agreed in the wrap-up sync")
        s.add(item)
        await s.commit()
        seed.extra["action_item_id"] = item.id
    return seed


# ── scenario assertions (beyond the global per-tick invariant) ───────────────────────────────
def _types(ticks):
    return [a["type"] for a in ticks[-1]["state"]["actions"]] if ticks else []


def x_fresh(seed, ticks, replay):
    kinds = [t["progress_kind"] for t in ticks]
    return [] if kinds and kinds[0] in {"substantive", "human_blocker"} else [f"fresh goal first tick must start work or ask the owner, got {kinds[:1]}"]


def x_meeting(seed, ticks, replay):
    last = ticks[-1]["state"]
    released = [t for t in last["tasks"] if t["title"] != "Design review prep" and t["id"] != str(seed.extra["source_task_id"])]
    return [] if len(released) > 1 or any(a["type"] in SUBSTANTIVE and a["status"] == "completed" for a in last["actions"][2:]) else \
        ["independent work (plan item 2) was not advanced while the meeting was active"]


def x_blocker(seed, ticks, replay):
    last = ticks[-1]["state"]
    recovered = any(a["type"] in {"retry_task", "reassign_task"} and a["status"] == "completed" for a in last["actions"])
    asked = any(a["status"] == "pending" for a in last["authority"]) or any(a["type"] == "ask_human" for a in last["actions"])
    return [] if recovered or asked else ["recoverable failed task neither retried/reassigned nor escalated with an exact question"]


def x_authority(seed, ticks, replay):
    pending = [a for a in ticks[-1]["state"]["authority"] if a["status"] == "pending"]
    return [] if len(pending) == 1 else [f"expected exactly one pending authority decision, found {len(pending)}"]


def x_answered(seed, ticks, replay):
    did = str(seed.extra["decision_id"])
    applied = any(did in json.dumps(a["contract"]) or a["request"].get("applies_decision_id") == did
                  for t in ticks for a in t["state"]["actions"] if a["status"] == "completed" and a["type"] != "noop")
    return [] if applied else ["answered decision was never applied by a completed continuation action (applies_decision_id)"]


def x_no_progress(seed, ticks, replay):
    keys = [json.dumps([w["owner"], w["awaited_event"]], sort_keys=True) for w in ticks[-1]["state"]["waits"] if w["status"] == "open"]
    dupes = len(keys) - len(set(keys))
    return [] if not dupes else [f"{dupes} duplicate open waits on the same owner/event (repeated same wait)"]


def x_analyzer(seed, ticks, replay):
    out, sup = [], ticks[-1]["state"]["run"]["supervision"]
    if sup.get("judgment_in_flight"):
        out.append("judgment flight bit stranded after provider failure")
    if not sup.get("judgment_failures"):
        out.append("provider failure was not accounted (judgment_failures == 0); judge may not have been reached")
    if not sup.get("judgment_dirty"):
        out.append("failed judgment did not stay dirty for retry")
    if (sup.get("judgment_failures") or 0) >= 3 and not any(b.get("kind") == "supervision_judgment_failures" for b in ticks[-1]["state"]["run"]["blockers"]):
        out.append("repeated analyzer failure did not escalate to a human-visible blocker")
    return out


def x_closeout(seed, ticks, replay):
    last, item = ticks[-1]["state"], str(seed.extra["action_item_id"])
    out = ["run completed despite an unfinished meeting commitment"] if last["run"]["status"] == "completed" else []
    handled = any(a["type"] in SUBSTANTIVE and a["status"] == "completed" for a in last["actions"]) or any(item in json.dumps(w) for w in last["waits"]) \
        or any(a["status"] == "pending" for a in last["authority"]) or any(a["type"] == "ask_human" for a in last["actions"]) \
        or any(item in json.dumps(b) for b in last["run"]["blockers"])
    return out if handled else out + ["unfinished commitment was neither worked, waited on (naming it), nor escalated"]


async def seed_model_authority(env):
    """Only the owner can supply the missing input; nothing is pre-seeded as pending."""
    seed = await seed_base(env)
    async with env.factory() as s:
        from huddleroom.models.orchestration import OrchestrationGoal
        goal = await s.get(OrchestrationGoal, seed.goal_id)
        goal.objective = ("Publish the new pricing page using the price tier the owner selects. The owner has NOT yet chosen between "
                          "Tier A and Tier B, no document in the project states the choice, and no plan or work can be started "
                          "until the owner decides. Only the owner can make this choice.")
        await s.commit()
    return seed


async def _seed_exhausted_failed_task(env, seed, title="Implement slice 1"):
    """Failed task whose deterministic retry already happened (same idempotency key), so only the model can act."""
    async with env.factory() as s:
        failed = _run_task(seed.project_id, seed.run_id, title, "failed", seed.dev, plan_item_id="item-1")
        s.add(failed)
        await s.flush()
        s.add(Session(project_id=seed.project_id, task_id=failed.id, agent_id=seed.dev, adapter_type="api", status="failed",
                      output="Provider quota exhausted while running the task.", metadata_={}, origin="auto"))
        suffix = await OrchestrationService()._task_recovery_attempt_suffix(s, seed.run_id, failed.id)
        s.add(OrchestrationAction(run_id=seed.run_id, idempotency_key=f"run:{seed.run_id}:kind:retry_task:task:{failed.id}{suffix}",
                                  action_type="retry_task", request={"action_type": "retry_task", "task_id": str(failed.id)}, status="completed"))
        await s.commit()
        return failed.id


async def seed_model_recovery(env):
    seed = await seed_base(env, with_plan_items=_items)
    seed.extra["failed_task_id"] = await _seed_exhausted_failed_task(env, seed)
    return seed


async def seed_no_progress_ask(env):
    seed = await seed_meeting_plus_work(env)  # accepted plan + active meeting = proactive branch
    seed.extra["failed_task_id"] = await _seed_exhausted_failed_task(env, seed)
    return seed


def x_model_authority(seed, ticks, replay):
    last, out = ticks[-1]["state"], []
    pending = [a for a in last["authority"] if a["status"] == "pending"]
    ask = [a for a in last["actions"] if a["type"] == "ask_human" and a["status"] == "completed"]
    model_asked = any(d["action"] == "ask_human" for t in ticks for d in t["state"]["decisions"])
    if not pending:
        return [f"model never produced an owner question (actions: {sorted({a['type'] for a in last['actions']})})"]
    if not ask or not model_asked:
        out.append("pending decision exists but was not created by a model-chosen ask_human action")
    question = pending[0]["question"].strip()
    if len(question) < 25 or "?" not in question:
        out.append(f"owner question is not specific/exact: {question!r}")
    if len(pending) != 1:
        out.append(f"expected one pending decision, found {len(pending)}")
    return out


def x_model_recovery(seed, ticks, replay):
    last, tid = ticks[-1]["state"], str(seed.extra["failed_task_id"])
    model_actions = [d["action"] for t in ticks for d in t["state"]["decisions"]]
    recovered = any(d["action"] in {"reassign_task", "retry_task", "ask_human"} and (tid in json.dumps(d["parsed"]) or d["action"] == "ask_human")
                    for t in ticks for d in t["state"]["decisions"])
    bounded = any(d["action"] == "noop" for t in ticks for d in t["state"]["decisions"]) and not any(_wait_gaps(w) for w in last["waits"] if w["status"] == "open")
    if not model_actions:
        return ["the model was never consulted: deterministic recovery pre-empted it (cannot evaluate model recovery)"]
    return [] if recovered or bounded else [f"model decisions {model_actions} neither recovered the failed task nor waited with evidence"]


def x_no_progress_ask(seed, ticks, replay):
    last, out = ticks[-1]["state"], []
    sup = last["run"]["supervision"]
    pending = [a for a in last["authority"] if a["status"] == "pending" and "no progress" in a["question"]]
    if not pending:
        return [f"no-progress ask not reached within caps: unchanged={sup.get('unchanged')} asks={sup.get('no_progress_asks')}"]
    if "failed" not in pending[0]["question"]:
        out.append(f"ask does not list the stuck follow-up: {pending[0]['question'][:300]!r}")
    if last["run"]["supervision"].get("no_progress_asks") != 1:
        out.append(f"no_progress_asks should be 1, got {sup.get('no_progress_asks')}")
    return out


SCENARIOS = {
    "fresh": (seed_fresh, x_fresh, {}),
    "meeting_plus_work": (seed_meeting_plus_work, x_meeting, {"judge": True}),
    "recoverable_blocker": (seed_recoverable_blocker, x_blocker, {}),
    "authority_decision": (seed_authority_decision, x_authority, {}),
    "answered_decision": (seed_answered_decision, x_answered, {}),
    "no_progress_wait": (seed_no_progress_wait, x_no_progress, {"judge": True, "wake_max": 1}),
    "analyzer_failure": (seed_analyzer_failure, x_analyzer, {"judge": True, "fail_judge": True, "max_ticks": 2}),
    "closeout_commitment": (seed_closeout_commitment, x_closeout, {}),
}
# Model-driven stages (team-lead verdict): run only with -k model_stage.
MODEL_STAGES = {
    "authority_request_by_model": (seed_model_authority, x_model_authority, {"max_ticks": 4}),
    "recoverable_blocker_by_model": (seed_model_recovery, x_model_recovery, {"max_ticks": 4}),
    "no_progress_ask": (seed_no_progress_ask, x_no_progress_ask, {"max_ticks": MAX_TICKS - 1, "wake_max": 1, "sleep": 1.3}),
}


def _write_report(name, env, seed, ticks, replay, failures):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    budget = env.budget
    report = {
        "scenario": name, "utc": datetime.now(timezone.utc).isoformat(), "provider": env.provider, "passed": not failures, "failures": failures,
        "caps": {"ticks": MAX_TICKS, "actions_per_tick": MAX_ACTIONS_PER_TICK, "provider_calls": MAX_CALLS, "wall_seconds": MAX_WALL_SECONDS, "tokens": MAX_TOKENS},
        "usage": {"provider_calls": budget.calls, "tokens_uncached_cap_basis": budget.tokens, "tokens_total_incl_cached": budget.raw_total,
                  "tokens_cached": budget.cached, "calls_without_cached_info": budget.uncached_flagged, "tokens_estimated_chars_div_4": budget.estimated,
                  "wall_seconds": round(time.monotonic() - budget.started, 1), "ticks": len(ticks) + (1 if replay else 0), "cap_hit": budget.cap_hit,
                  "real_task_runs_prevented": budget.task_runs, "denied_subprocess": budget.denied_subprocess},
        "provider_calls": budget.records, "ticks": ticks, "replay": replay,
        "seed": {"goal_id": str(seed.goal_id), "run_id": str(seed.run_id)} if seed else None,
    }
    path = LOG_DIR / f"{name}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.json"
    path.write_text(redact_secrets(json.dumps(report, indent=2, default=str)), encoding="utf-8")
    print(f"\n[live-eval] {name}: calls={budget.calls} uncached={budget.tokens} total={budget.raw_total} cached={budget.cached}{'(est)' if budget.estimated else ''} unflagged_missing={budget.uncached_flagged} report={path}")
    return path


# ── tests ────────────────────────────────────────────────────────────────────────────────────
async def test_preflight_running_source(env):
    """Fixture already ran the checkout + tick-owned-transaction marker preflight; this makes it an explicit result."""
    assert Path(huddleroom.__file__).resolve().is_relative_to(CHECKOUT)


@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_scenario_seed_is_valid(env, name):
    """No model call: each seed builds on a fresh DB and leaves an authorized, tickable run."""
    seed = await SCENARIOS[name][0](env)
    async with env.factory() as db:
        snap = await snapshot(db, seed.run_id)
    assert (snap["run"]["status"], snap["run"]["phase"]) == ("running", "authorized"), snap["run"]
    assert env.budget.calls == 0


@pytest.mark.parametrize("name", list(SCENARIOS))
async def test_live_scenario(env, name, monkeypatch):
    seeder, extra_check, options = SCENARIOS[name]
    if options.get("wake_max"):
        monkeypatch.setattr(settings, "orchestration_wake_max_seconds", options["wake_max"])
    seed = await seeder(env)
    budget = env.budget
    budget.started, budget.armed = time.monotonic(), True
    budget.fail_mode = bool(options.get("fail_judge"))
    ticks, replay, failures = [], None, []
    try:
        ticks, settled = await drive(env, seed.run_id, judge=options.get("judge", False), max_ticks=options.get("max_ticks", MAX_TICKS - 1))
        for t in ticks:
            failures += [f"tick {t['tick']}: {v}" for v in t["violations"]]
            if t["error"]:
                failures.append(f"tick {t['tick']} raised: {t['error']}")
        if budget.cap_hit:
            failures.append(f"CAP REACHED (fail closed): {budget.cap_hit}")
        elif not settled and not options.get("max_ticks"):
            failures.append(f"did not settle within {MAX_TICKS - 1} ticks (+1 replay): cap reached, fail closed")
        if budget.denied_subprocess:
            failures.append(f"blocked unsafe subprocess attempts: {budget.denied_subprocess}")
        if ticks and not any(t["error"] for t in ticks):
            if not options.get("fail_judge"):
                budget.fail_mode = False
                replay = await replay_last_tick(env, seed.run_id)
                failures += [f"replay: {p}" for p in replay["problems"]]
                if replay["error"]:
                    failures.append(f"replay raised: {replay['error']}")
            failures += extra_check(seed, ticks, replay)
    finally:
        _write_report(name, env, seed, ticks, replay, failures)
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("name", list(MODEL_STAGES))
async def test_model_stage(env, name, monkeypatch):
    seeder, extra_check, options = MODEL_STAGES[name]
    if options.get("wake_max"):
        monkeypatch.setattr(settings, "orchestration_wake_max_seconds", options["wake_max"])
    seed = await seeder(env)
    budget = env.budget
    budget.started, budget.armed = time.monotonic(), True
    if "failed_task_id" in seed.extra:
        budget.stay_failed.add(seed.extra["failed_task_id"])  # the runner keeps failing this task
    ticks, replay, failures, post = [], None, [], None
    try:
        kwargs = {"max_ticks": options["max_ticks"], "judge": True}
        if name in {"no_progress_ask", "recoverable_blocker_by_model"}:
            kwargs["refail"] = [seed.extra["failed_task_id"]]
        if name == "no_progress_ask":
            kwargs.update(sleep=options["sleep"], expire_model_waits=True, auto_answer=True, tolerate_409=True,
                          stop_when=lambda st: any(a["status"] == "pending" and "no progress" in a["question"] for a in st["authority"]))
        elif name == "authority_request_by_model":
            kwargs["stop_when"] = lambda st: any(a["status"] == "pending" for a in st["authority"])
        ticks, _ = await drive(env, seed.run_id, **kwargs)
        for t in ticks:
            failures += [f"tick {t['tick']}: {v}" for v in t["violations"]]
            if t["error"]:
                failures.append(f"tick {t['tick']} raised: {t['error']}")
        if budget.cap_hit:
            failures.append(f"CAP REACHED (fail closed): {budget.cap_hit}")
        if budget.denied_subprocess:
            failures.append(f"blocked unsafe subprocess attempts: {budget.denied_subprocess}")
        if ticks and not any(t["error"] for t in ticks):
            replay = await replay_last_tick(env, seed.run_id)
            failures += [f"replay: {p}" for p in replay["problems"]]
            if name == "authority_request_by_model" and replay["state"]["authority"] and len([a for a in replay["state"]["authority"] if a["status"] == "pending"]) != 1:
                failures.append("replay changed the number of pending owner questions")
            failures += extra_check(seed, ticks, replay)
            if name == "no_progress_ask" and any(a["status"] == "pending" and "no progress" in a["question"] for a in ticks[-1]["state"]["authority"]):
                post = await _answer_and_confirm_restart(env, seed)
                failures += post["problems"]
    finally:
        path = _write_report(name, env, seed, ticks, replay, failures)
        if post:
            path.write_text(path.read_text().rstrip().rstrip("}") + f', "answer_phase": {json.dumps(post, default=str)}}}')
    assert not failures, "\n".join(failures)


async def _answer_and_confirm_restart(env, seed):
    """Answer the system no-progress ask via the real runtime path, tick again, confirm the counter restarted."""
    problems = []
    async with env.factory() as s:
        decision_id = (await s.scalars(select(OrchestrationAuthorityDecision.id).where(
            OrchestrationAuthorityDecision.run_id == seed.run_id, OrchestrationAuthorityDecision.status == "pending",
            OrchestrationAuthorityDecision.question.like("%no progress%")))).first()
    answer = await answer_via_runtime_path(env, decision_id)
    await asyncio.sleep(1.3)
    ticks, _ = await drive(env, seed.run_id, judge=True, max_ticks=1, refail=[seed.extra["failed_task_id"]], stop_when=lambda st: False,
                           expire_model_waits=True)
    async with env.factory() as s:
        run = await s.get(OrchestrationRun, seed.run_id, populate_existing=True)
        state = dict(run.supervision_state or {})
    unchanged = state.get("unchanged") or {}
    snap = ticks[-1]["state"] if ticks else {"authority": []}
    if any(a["status"] == "pending" and "no progress" in a["question"] for a in snap["authority"]):
        problems.append("answered no-progress ask was immediately re-asked")
    if int(unchanged.get("n") or 0) > 1 or unchanged.get("ask_decision_id"):
        problems.append(f"counter did not restart after the answer: unchanged={unchanged}")
    if state.get("no_progress_asks") != 1:
        problems.append(f"next ask generation should be :1 (no_progress_asks == 1), got {state.get('no_progress_asks')}")
    return {"problems": problems, "answer": answer, "unchanged_after_answer": unchanged, "no_progress_asks": state.get("no_progress_asks"),
            "tick_error": ticks[-1]["error"] if ticks else None, "provider_calls_in_restart_tick": ticks[-1]["provider_calls"] if ticks else None}
