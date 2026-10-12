"""SPR 180/184: global roster mutation coordinator, lock order and collateral invalidation.

This module also hosts the shared lock-trace harness reused by the cases in
test_baseline_processes_locking.py, test_orchestration_baseline_runtime.py,
test_orchestration_agent_definition_review.py, test_orchestration_team_hierarchy.py and
test_orchestration_debug.py. names are resolved lazily (``roster_api``) so collecting
this module never breaks the pre-existing cases that import the harness.

Interface the harness relies on (huddleroom.services.orchestration_roster_mutation):
- ``AgentRosterWriteLock`` with an ``acquire(self)`` async context manager (process-wide).
- ``AgentRosterMutationCoordinator()`` (no-arg) with ``apply`` / ``mutation_guard`` per the brief.
- ``PreparedRosterMutation`` / ``LockedRosterMutation`` frozen dataclasses.
- ``RosterMutationResult`` with ``value``, ``affected_goal_ids``, ``goal_fingerprints``.
- the coordinator resolves the module-global ``preview_roster_delta`` at call time and acquires
  goal locks through ``OrchestrationService._lock_single_goal`` (directly or via
  ``_lock_goal_for_baseline_transition``).
"""
from __future__ import annotations

import asyncio
import dataclasses
import importlib
import json
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import pytest_asyncio
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Session as SyncSession

from huddleroom.models.agent import Agent
from huddleroom.models.base import Base
from huddleroom.models.orchestration import OrchestrationAction, OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import (
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
    OrchestrationWarning,
)
from huddleroom.models.project import Project
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.services.orchestration_agent_definition_analyzer import (
    AgentDefinitionSemanticAnalyzer,
    SemanticAgentAssessment,
)
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_service import OrchestrationService

ROSTER_MODULE = "huddleroom.services.orchestration_roster_mutation"
TASK7_NAMES = (
    "AgentRosterWriteLock",
    "AgentRosterMutationCoordinator",
    "PreparedRosterMutation",
    "LockedRosterMutation",
    "RosterMutationResult",
)
PROPOSED_DESCRIPTION = "Developer role responsible for implementation."
PROPOSED_PERSONA = "Implement scoped changes and report verification evidence."
RACE_TIMEOUT = 20


def roster_api():
    module = importlib.import_module(ROSTER_MODULE)
    missing = [name for name in TASK7_NAMES if not hasattr(module, name)]
    assert not missing, f"interface missing from {ROSTER_MODULE}: {missing}"
    return module


def stale_error(outcome) -> str | None:
    """A stale apply may return a result with ``error`` or raise an exception carrying ``error``."""
    return getattr(outcome, "error", None)


async def outcome_of(awaitable):
    try:
        return await awaitable
    except Exception as exc:  # noqa: BLE001 -- the stale contract allows raise-or-return
        return exc


def assert_orm_free(value, path="prepared"):
    assert not isinstance(value, Base), f"{path} carries ORM instance {type(value).__name__}"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            assert_orm_free(getattr(value, field.name), f"{path}.{field.name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_orm_free(key, f"{path}[key]")
            assert_orm_free(item, f"{path}[{key!r}]")
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            assert_orm_free(item, f"{path}[{index}]")


# --------------------------------------------------------------------------- lock trace


class LockTracer:
    """Per-task lock/write/commit trace. ``goals``/``roster`` are the state *after* each event."""

    def __init__(self):
        self.events: list[SimpleNamespace] = []
        self.prepared: list = []
        self._goals: dict = {}
        self._roster: Counter = Counter()
        self.after_roster_release: list = []  # one-shot async callbacks
        self.after_preview: list = []  # async hooks(plan) run after every traced preview

    def start(self):
        """Drop events from test seeding so only the operation under test is traced."""
        self.events.clear()
        self.prepared.clear()

    def record(self, kind, value=None):
        task = asyncio.current_task()
        held = self._goals.get(task, Counter())
        self.events.append(SimpleNamespace(
            kind=kind, value=value, task=task,
            goals=frozenset(goal for goal, depth in held.items() if depth > 0),
            roster=self._roster[task] > 0,
        ))

    def enter_goal(self, goal_id):
        held = self._goals.setdefault(asyncio.current_task(), Counter())
        held[goal_id] += 1
        self.record("goal:acquire" if held[goal_id] == 1 else "goal:reenter", goal_id)

    def exit_goal(self, goal_id):
        held = self._goals[asyncio.current_task()]
        held[goal_id] -= 1
        self.record("goal:release" if held[goal_id] == 0 else "goal:reexit", goal_id)

    def enter_roster(self):
        self._roster[asyncio.current_task()] += 1
        self.record("roster:acquire")

    def exit_roster(self):
        self._roster[asyncio.current_task()] -= 1
        self.record("roster:release")

    def of(self, kind, task=None):
        return [item for item in self.events if item.kind == kind and (task is None or item.task is task)]

    def index(self, predicate, start=0):
        return next((i for i in range(start, len(self.events)) if predicate(self.events[i])), None)

    def handoff_trace(self, origin_goal_id) -> list[str]:
        """Normalize to the brief's ``["goal:reserve", "goal:release", "roster:acquire", *goal locks]``."""
        out, state = [], "before"
        for item in self.events:
            if state == "before":
                if item.kind == "prepare":
                    out.append(
                        "goal:reserve" if origin_goal_id in item.goals and not item.roster
                        else f"prepare(goals={sorted(map(str, item.goals))}, roster={item.roster})"
                    )
                    state = "reserved"
                continue
            if item.kind == "agent:write":
                break
            if state == "reserved":
                if item.kind == "goal:release" and not item.goals:
                    out.append("goal:release")
                    state = "released"
                elif item.kind == "roster:acquire":
                    out.append("roster:acquire-under-goal-lock")
                    state = "released"
                continue
            if item.kind == "roster:acquire":
                out.append("roster:acquire" if not item.goals else "roster:acquire-under-goal-lock")
            elif item.kind == "goal:acquire":
                out.append(f"goal:{item.value}" if item.roster else f"goal-without-roster:{item.value}")
            elif item.kind == "roster:release":
                out.append("roster:release-before-write")
        return out


@pytest.fixture
def lock_tracer(monkeypatch):
    rm = roster_api()
    tracer = LockTracer()

    original_goal_lock = OrchestrationService._lock_single_goal

    @asynccontextmanager
    async def traced_goal_lock(self, db, goal_id, **kwargs):
        async with original_goal_lock(self, db, goal_id, **kwargs):
            tracer.enter_goal(goal_id)
            try:
                yield
            finally:
                tracer.exit_goal(goal_id)

    monkeypatch.setattr(OrchestrationService, "_lock_single_goal", traced_goal_lock)

    original_roster_lock = rm.AgentRosterWriteLock.acquire

    @asynccontextmanager
    async def traced_roster_lock(self, *args, **kwargs):
        async with original_roster_lock(self, *args, **kwargs):
            tracer.enter_roster()
            try:
                yield
            finally:
                tracer.exit_roster()
        while tracer.after_roster_release:
            await tracer.after_roster_release.pop(0)()

    monkeypatch.setattr(rm.AgentRosterWriteLock, "acquire", traced_roster_lock)

    original_init = rm.PreparedRosterMutation.__init__

    def traced_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        tracer.prepared.append(self)
        tracer.record("prepare")

    monkeypatch.setattr(rm.PreparedRosterMutation, "__init__", traced_init)

    original_preview = rm.preview_roster_delta

    async def traced_preview(db, delta, *, origin_goal_id):
        plan = await original_preview(db, delta, origin_goal_id=origin_goal_id)
        tracer.record("preview", tuple(plan.affected_goal_ids))
        for hook in list(tracer.after_preview):
            await hook(plan)
        return plan

    monkeypatch.setattr(rm, "preview_roster_delta", traced_preview)

    def before_flush(session, _flush_context, _instances):
        for obj in list(session.new) + list(session.dirty) + list(session.deleted):
            if isinstance(obj, Agent) and (obj not in session.dirty or session.is_modified(obj)):
                tracer.record("agent:write", (obj.id, obj.name))

    def after_commit(session):
        tracer.record("commit", id(session))

    event.listen(SyncSession, "before_flush", before_flush)
    event.listen(SyncSession, "after_commit", after_commit)
    try:
        yield tracer
    finally:
        event.remove(SyncSession, "before_flush", before_flush)
        event.remove(SyncSession, "after_commit", after_commit)


def assert_roster_first_publication(tracer: LockTracer, affected_goal_ids, *, task=None):
    """Every agent write and its publishing commit happen under roster + every affected goal lock."""
    affected = set(affected_goal_ids)
    writes = tracer.of("agent:write", task)
    assert writes, "no agent row write observed"
    for write in writes:
        assert write.roster, f"uncoordinated agent write {write.value}"
        assert affected <= write.goals, (
            f"agent write {write.value} missing goal locks {sorted(map(str, affected - write.goals))}"
        )
    first_write = tracer.events.index(writes[0])
    commit_at = tracer.index(lambda e: e.kind == "commit" and (task is None or e.task is task), first_write)
    assert commit_at is not None, "agent write was never committed"
    commit = tracer.events[commit_at]
    assert commit.roster and affected <= commit.goals, "agent write published after a guard was released"
    roster_acquires = [e for e in tracer.events[:first_write] if e.kind == "roster:acquire"
                       and (task is None or e.task is task)]
    assert roster_acquires and not roster_acquires[-1].goals, "roster lock acquired while holding a goal lock"
    lock_start = tracer.events.index(roster_acquires[-1])
    acquired = [e.value for e in tracer.events[lock_start:first_write]
                if e.kind == "goal:acquire" and (task is None or e.task is task)]
    assert acquired == sorted(affected, key=str), "affected goal locks not acquired in sorted UUID order"


def assert_entry_handoff(tracer: LockTracer, origin_goal_id, affected_goal_ids=None):
    affected = sorted(set(affected_goal_ids or [origin_goal_id]), key=str)
    assert tracer.handoff_trace(origin_goal_id) == [
        "goal:reserve", "goal:release", "roster:acquire", *[f"goal:{goal}" for goal in affected],
    ]
    assert tracer.prepared, "no PreparedRosterMutation was reserved"
    prepared = tracer.prepared[0]
    assert type(prepared).__dataclass_params__.frozen
    assert_orm_free(prepared)
    prepare_at = tracer.index(lambda e: e.kind == "prepare")
    release_at = tracer.index(lambda e: e.kind == "goal:release" and not e.goals, prepare_at)
    assert tracer.index(lambda e: e.kind == "commit", prepare_at) < release_at, (
        "reservation was not committed before the origin goal lock was released"
    )
    assert_roster_first_publication(tracer, affected)
    first_write = tracer.events.index(tracer.of("agent:write")[0])
    roster_release = tracer.index(lambda e: e.kind == "roster:release", first_write)
    assert roster_release is not None
    assert tracer.index(
        lambda e: e.kind == "goal:acquire" and e.value == origin_goal_id and not e.roster, roster_release
    ) is not None, "process did not resume under the origin goal lock after the roster apply"


# --------------------------------------------------------------------------- seeding


@pytest_asyncio.fixture
async def sessions(test_engine):
    return async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def approve_semantic_reviews(monkeypatch):
    async def approve(self, request, candidate_work_functions, *, project_id=None):
        return SemanticAgentAssessment("approved", (), "Definition is actionable.", tuple(candidate_work_functions))

    monkeypatch.setattr(AgentDefinitionSemanticAnalyzer, "review_request", approve)


class OneProposalAnalyzer:
    def __init__(self):
        self.calls = 0

    @staticmethod
    def build_request(agent_snapshot, goal_snapshot, candidate_work_functions, project=None):
        return {"agent": agent_snapshot, "goal": goal_snapshot, "candidate_work_functions": candidate_work_functions}

    async def review_request(self, request, candidate_work_functions, *, project_id=None):
        self.calls += 1
        if self.calls == 1:
            return SemanticAgentAssessment(
                "improvement_proposed", ("Persona does not assign implementation ownership.",),
                "Makes the definition actionable.", ("implementation",), PROPOSED_DESCRIPTION, PROPOSED_PERSONA,
            )
        return SemanticAgentAssessment("approved", (), "Definition is actionable.", ("implementation",))

    async def review(self, agent_snapshot, goal_snapshot, candidate_work_functions, *, project_id=None):
        return await self.review_request({}, candidate_work_functions, project_id=project_id)


def make_agent(**overrides) -> Agent:
    values = {
        "id": uuid.uuid4(), "name": f"agent-{uuid.uuid4().hex[:8]}", "role": "developer",
        "description": "Implements scoped product changes.",
        "system_prompt": "Implement the requested change and verify it.",
        "provider": "openai", "model": "gpt-4o-mini", "adapter_type": "api", "cli_runtime": None,
        "capabilities": ["implementation"], "config": {}, "is_active": True,
    }
    values.update(overrides)
    return Agent(**values)


async def seed_project_user(s, tmp_path) -> tuple[Project, User]:
    workspace = tmp_path / f"workspace-{uuid.uuid4().hex[:6]}"
    workspace.mkdir()
    project = Project(name="Roster mutation", workspace_path=str(workspace.resolve()), config={})
    user = User(email=f"roster-{uuid.uuid4()}@example.com", hashed_password="x", display_name="Roster", role="member")
    s.add_all([project, user])
    await s.flush()
    return project, user


async def seed_goal_run(s, project, *, goal_id=None, weight="standard", phase="baseline",
                        authority_model="no_manager", manager_user_id=None, user=None,
                        objective="Implement the API", with_run=True):
    goal = OrchestrationGoal(
        id=goal_id or uuid.uuid4(), project_id=project.id, objective=objective,
        success_criteria=[{"key": "done", "description": "change works"}], constraints={},
        weight=weight, status="active", authority_model=authority_model,
        manager_user_id=manager_user_id, created_by_user_id=user.id if user else None,
    )
    s.add(goal)
    await s.flush()
    run = None
    if with_run:
        run = OrchestrationRun(goal_id=goal.id, status="running", phase=phase, baseline_authorized=True)
        s.add(run)
        await s.flush()
    return goal, run


async def complete(s, goal, run, process_type, outputs=None):
    service = OrchestrationProcessService()
    process = await service.start_process(s, goal.id, process_type=process_type,
                                          trigger_reason="test prerequisite", run_id=run.id)
    await service.complete_process(s, process, outputs=outputs)
    return process


async def assign(s, project_id, run_id, agent, work_function):
    task = Task(project_id=project_id, title="work", assigned_to=agent.id,
                metadata_={"orchestration": {"run_id": str(run_id), "work_function": work_function}})
    s.add(task)
    await s.flush()
    return task


async def seed_definition_proposal(sessions, tmp_path, *, answer=False, running=False):
    """One goal parked on one agent-definition improvement proposal (optionally answered/resumed)."""
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService

    async with sessions() as s:
        project, user = await seed_project_user(s, tmp_path)
        goal, run = await seed_goal_run(s, project, objective="Ship the reviewed change")
        agent = make_agent(name=f"implementation-agent-{uuid.uuid4().hex[:6]}", description="Helps with code.", system_prompt="Be useful.",
                           adapter_type="cli", cli_runtime="codex")
        s.add(agent)
        await s.flush()
        await assign(s, project.id, run.id, agent, "implementation")
        await complete(s, goal, run, "goal_definition")
        await complete(s, goal, run, "manager_selection")
        summary = await AgentDefinitionReviewProcess(analyzer=OneProposalAnalyzer()).advance(s, goal, run)
        assert summary["status"] == "waiting_decision"
        decisions = await OrchestrationAuthorityDecisionService().list_decisions(s, goal.id, status="pending")
        assert len(decisions) == 1
        decision = decisions[0]
        if answer:
            await OrchestrationAuthorityDecisionService().answer_decision(
                s, decision, selected_option="approve", reason="Clearer.", decided_by_user_id=user.id)
        if running:
            current = await OrchestrationProcessService().get_current(s, goal.id, "agent_definition_review")
            await OrchestrationProcessService().resume_process(s, current)
        await s.commit()
        return SimpleNamespace(project_id=project.id, goal_id=goal.id, run_id=run.id, agent_id=agent.id,
                               decision_id=decision.id, user_id=user.id)


def proposed_agent_definition(name="optional-specialist"):
    return {
        "proposal_id": name,
        "definition": {
            "name": name, "role": "specialist", "description": "Supports specialized implementation work.",
            "provider": "openai", "model": "gpt-4o-mini",
            "system_prompt": "Handle assigned specialist work and report concrete results.",
            "adapter_type": "api", "cli_runtime": None, "capabilities": ["implementation"], "config": {},
        },
    }


def install_hierarchy_analyzer(monkeypatch, *, proposal_name="optional-specialist"):
    from huddleroom.services.orchestration_team_hierarchy_analyzer import TeamHierarchyAnalysis, TeamHierarchyAnalyzer

    calls = []

    async def review(_self, payload, project=None, *, project_id=None):
        calls.append(payload)
        agents = {capability: agent["id"] for agent in payload["agents"] for capability in agent["capabilities"]}
        assignments = tuple({"work_function": work, "agent_ref": agents[work]}
                            for work in payload["required_work_functions"])
        refs = dict.fromkeys(item["agent_ref"] for item in assignments)
        return TeamHierarchyAnalysis(
            proposed_agents=(proposed_agent_definition(proposal_name),) if len(calls) == 1 else (),
            assignments=assignments,
            reporting_lines=tuple({"agent_ref": ref, "reports_to": "manager"} for ref in refs),
            documented_gaps=(), rationale="Covers the available work.",
            self_review="Checked definitions, references, coverage, reporting, and independence.",
        )

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    return calls


async def seed_hierarchy_proposal(sessions, tmp_path, monkeypatch, *, answer=None, running=False):
    """One substantial goal parked on one team-hierarchy agent proposal (optionally answered/resumed)."""
    from huddleroom.services.orchestration_agent_review_service import OrchestrationAgentReviewService
    from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
    from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess

    install_hierarchy_analyzer(monkeypatch)
    async with sessions() as s:
        project, user = await seed_project_user(s, tmp_path)
        goal, run = await seed_goal_run(s, project, weight="substantial", authority_model="human_manager",
                                        manager_user_id=user.id, user=user)
        suffix = uuid.uuid4().hex[:6]
        agents = {
            "planning": make_agent(name=f"planner-{suffix}", role="planner", capabilities=["planning"]),
            "implementation": make_agent(name=f"implementer-{suffix}", capabilities=["implementation"]),
            "summarization": make_agent(name=f"summarizer-{suffix}", role="writer", capabilities=["summarization"]),
            "review": make_agent(name=f"reviewer-{suffix}", role="reviewer", capabilities=["review"]),
            "validation": make_agent(name=f"validator-{suffix}", role="validator", capabilities=["validation"]),
        }
        s.add_all(agents.values())
        await s.flush()
        reviews = [await OrchestrationAgentReviewService().create_review(
            s, goal.id, agent_id=agent.id, fit_summary="Reviewed for hierarchy proposal.",
            proposed_work_functions=[function], approved_for_work_functions=[function], run_id=run.id,
        ) for function, agent in agents.items()]
        await complete(s, goal, run, "goal_definition")
        await complete(s, goal, run, "manager_selection")
        await complete(s, goal, run, "agent_definition_review",
                       outputs={"review_ids": [str(review.id) for review in reviews]})
        summary = await TeamHierarchyProcess().advance(s, goal, run)
        assert summary["status"] == "waiting_decision", summary
        decisions = await OrchestrationAuthorityDecisionService().list_decisions(s, goal.id, status="pending")
        assert [item.decision_key for item in decisions] == ["team_hierarchy:agent:optional-specialist"]
        decision = decisions[0]
        if answer is not None:
            await OrchestrationAuthorityDecisionService().answer_decision(
                s, decision, selected_option=answer, reason="Needed.", decided_by_user_id=user.id)
        if running:
            current = await OrchestrationProcessService().get_current(s, goal.id, "team_hierarchy")
            await OrchestrationProcessService().resume_process(s, current)
        await s.commit()
        return SimpleNamespace(project_id=project.id, goal_id=goal.id, run_id=run.id, decision_id=decision.id,
                               user_id=user.id, proposal_name="optional-specialist")


def stub_prerequisite_processes(monkeypatch, *, definition_review=False, hierarchy=None):
    """Tick reaches the process under test without provider calls."""
    from huddleroom.services.orchestration_agent_definition_review import AgentDefinitionReviewProcess
    from huddleroom.services.orchestration_goal_definition import GoalDefinitionProcess
    from huddleroom.services.orchestration_manager_selection import ManagerSelectionProcess
    from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess

    async def completed(self, db, goal, run, **_kwargs):
        return {"status": "completed"}

    monkeypatch.setattr(GoalDefinitionProcess, "advance", completed)
    monkeypatch.setattr(ManagerSelectionProcess, "advance", completed)
    if definition_review:
        monkeypatch.setattr(AgentDefinitionReviewProcess, "advance", completed)
    if hierarchy is not None:
        async def parked(self, db, goal, run, **_kwargs):
            return {"status": hierarchy}

        monkeypatch.setattr(TeamHierarchyProcess, "advance", parked)


async def act_definition_batch(sessions, seeded):
    from huddleroom.routers.orchestration_decisions import answer_agent_definition_review_batch
    from huddleroom.schemas.orchestration import OrchestrationAgentDefinitionReviewBatchAnswerRequest

    async with sessions() as s:
        user = await s.get(User, seeded.user_id)
        response = await answer_agent_definition_review_batch(
            seeded.project_id, seeded.goal_id,
            OrchestrationAgentDefinitionReviewBatchAnswerRequest(
                answers=[{"decision_id": str(seeded.decision_id), "selected_option": "approve"}]),
            user, s,
        )
        await s.commit()  # get_db commits after the route returns
        return response


async def act_hierarchy_answer(sessions, seeded, option="approve"):
    from huddleroom.routers.orchestration_decisions import answer_decision
    from huddleroom.schemas.orchestration import OrchestrationDecisionAnswerRequest

    body = {"selected_option": option}
    if option == "edit":
        body["edited_agent"] = {**proposed_agent_definition()["definition"], "name": f"edited-{seeded.proposal_name}"}
    async with sessions() as s:
        user = await s.get(User, seeded.user_id)
        response = await answer_decision(
            seeded.project_id, seeded.goal_id, seeded.decision_id,
            OrchestrationDecisionAnswerRequest(**body), user, s,
        )
        await s.commit()
        return response


async def act_tick(sessions, seeded):
    async with sessions() as s:
        return await OrchestrationService().tick(s, seeded.run_id)


async def act_debug_step(sessions, seeded, process_type):
    from huddleroom.services.orchestration_debug_service import OrchestrationDebugService

    async with sessions() as s:
        return await OrchestrationDebugService().step(s, seeded.project_id, seeded.goal_id, process_type)


async def outcome_or_conflict(awaitable):
    """Resume refusal may surface as a 409 / interrupted result; anything else propagates."""
    from fastapi import HTTPException

    try:
        return await awaitable
    except HTTPException as exc:
        assert exc.status_code == 409, exc.detail
        return exc


def interfere_after_roster_release(tracer, sessions, seeded, kind):
    """Commit an out-of-band change between the roster apply and the origin-goal resume."""
    from huddleroom.services import orchestration_baseline_runtime as runtime

    async def interfere():
        if kind == "reset_fence":
            runtime._resetting[seeded.goal_id] = runtime._resetting.get(seeded.goal_id, 0) + 1
            operation = runtime._active.get(seeded.goal_id)
            if operation is not None:
                operation.interrupted.set()
            return
        async with sessions() as s:
            if kind == "run_replaced":
                await s.execute(update(OrchestrationRun).where(OrchestrationRun.id == seeded.run_id)
                                .values(status="cancelled"))
                s.add(OrchestrationRun(goal_id=seeded.goal_id, status="running", phase="baseline"))
            elif kind == "run_paused":
                await s.execute(update(OrchestrationRun).where(OrchestrationRun.id == seeded.run_id)
                                .values(status="paused"))
            elif kind == "approval_revoked":
                await s.execute(update(OrchestrationAuthorityDecision)
                                .where(OrchestrationAuthorityDecision.id == seeded.decision_id)
                                .values(status="cancelled"))
            else:
                raise AssertionError(kind)
            await s.commit()

    tracer.after_roster_release.append(interfere)


def clear_reset_fence(goal_id):
    from huddleroom.services import orchestration_baseline_runtime as runtime

    runtime._resetting.pop(goal_id, None)


def single_agent_write(tracer, *, agent_id=None, name=None):
    writes = [e for e in tracer.of("agent:write")
              if (agent_id is None or e.value[0] == agent_id) and (name is None or e.value[1] == name)]
    assert len(writes) == 1, f"expected the roster effect to apply exactly once, saw {len(writes)} writes"
    return writes[0]


async def current_process(sessions, goal_id, process_type):
    async with sessions() as s:
        return await OrchestrationProcessService().get_current(s, goal_id, process_type)


async def fresh_agent(sessions, agent_id):
    async with sessions() as s:
        return await s.get(Agent, agent_id)


async def agents_named(sessions, name):
    async with sessions() as s:
        return list((await s.execute(select(Agent).where(Agent.name == name))).scalars())


async def decisions_for(sessions, goal_id):
    async with sessions() as s:
        return list((await s.execute(select(OrchestrationAuthorityDecision).where(
            OrchestrationAuthorityDecision.goal_id == goal_id))).scalars())


def canonical(agent: Agent):
    from huddleroom.services.orchestration_roster_mutation import _canonical_agent

    return _canonical_agent(agent)


def new_agent_state(name="recovery-planner", **overrides):
    from huddleroom.services.orchestration_roster_mutation import CanonicalAgentState

    values = dict(id=uuid.uuid4(), name=name, role="planner", description="Plans scoped work.",
                  provider="openai", model="gpt-4o-mini", system_prompt="Plan the work and report the plan.",
                  adapter_type="api", cli_runtime=None, capabilities=("planning",), config={}, is_active=True)
    values.update(overrides)
    return CanonicalAgentState(**values)


async def create_from_delta(db, delta, plan):
    for state in delta.creates:
        db.add(Agent(id=state.id, name=state.name, role=state.role, description=state.description,
                     provider=state.provider, model=state.model, system_prompt=state.system_prompt,
                     adapter_type=state.adapter_type, cli_runtime=state.cli_runtime,
                     capabilities=list(state.capabilities), config=dict(state.config), is_active=state.is_active))
    for state in delta.updates:
        agent = await db.get(Agent, state.id)
        agent.description, agent.system_prompt = state.description, state.system_prompt
    await db.flush()
    return [state.id for state in delta.creates + delta.updates]


async def seed_roster_goals(sessions, tmp_path, *, goal_ids, phase="baseline", with_run=None):
    """Several standard goals sharing one roster; each active run assigns the shared implementer."""
    with_run = with_run or {}
    async with sessions() as s:
        project, user = await seed_project_user(s, tmp_path)
        implementer = make_agent(name=f"implementer-{uuid.uuid4().hex[:6]}")
        s.add(implementer)
        await s.flush()
        goals = []
        for goal_id in goal_ids:
            goal, run = await seed_goal_run(s, project, goal_id=goal_id, phase=phase,
                                            with_run=with_run.get(goal_id, True))
            if run is not None:
                await assign(s, project.id, run.id, implementer, "implementation")
            goals.append((goal, run))
        await s.commit()
        return SimpleNamespace(project_id=project.id, user_id=user.id, agent_id=implementer.id,
                               goal_ids=[goal.id for goal, _ in goals],
                               run_ids={goal.id: run.id for goal, run in goals if run is not None})


# Deliberately out of sorted order so creation order != lock order.
GOAL_HIGH = uuid.UUID("f0000000-0000-4000-8000-000000000001")
GOAL_LOW = uuid.UUID("10000000-0000-4000-8000-000000000002")
GOAL_MID = uuid.UUID("80000000-0000-4000-8000-000000000003")


# --------------------------------------------------------------------------- interface


def test_task7_interfaces_are_frozen_and_orm_free():
    rm = roster_api()
    for name in ("PreparedRosterMutation", "LockedRosterMutation"):
        cls = getattr(rm, name)
        assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen, name
    prepared = {field.name for field in dataclasses.fields(rm.PreparedRosterMutation)}
    assert {"delta", "expected_plan"} <= prepared
    locked = {field.name for field in dataclasses.fields(rm.LockedRosterMutation)}
    assert {"delta", "plan"} <= locked
    coordinator = rm.AgentRosterMutationCoordinator()
    assert callable(coordinator.apply) and callable(coordinator.mutation_guard)


def test_coordinator_module_does_not_import_agent_service():
    source = importlib.util.find_spec(ROSTER_MODULE).origin
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    roster_api()
    assert "agent_service" not in text and "AgentService" not in text


# --------------------------------------------------------------------------- coordinator


@pytest.mark.asyncio
async def test_apply_locks_roster_then_preview_seeded_sorted_goals_and_returns_result(
    sessions, tmp_path, lock_tracer
):
    rm = roster_api()
    seeded = await seed_roster_goals(sessions, tmp_path, goal_ids=[GOAL_HIGH, GOAL_LOW, GOAL_MID])
    delta = rm.ValidatedRosterDelta(creates=(new_agent_state(),))
    async with sessions() as s:
        plan = await rm.preview_roster_delta(s, delta, origin_goal_id=GOAL_HIGH)
        await s.commit()
    assert set(plan.affected_goal_ids) == {GOAL_HIGH, GOAL_LOW, GOAL_MID}
    lock_tracer.start()

    calls = []

    async def mutation(db, locked_delta, locked_plan):
        calls.append((locked_delta, locked_plan))
        return await create_from_delta(db, locked_delta, locked_plan)

    async with sessions() as s:
        result = await rm.AgentRosterMutationCoordinator().apply(
            s, delta, plan, mutation, origin_goal_id=GOAL_HIGH, provenance={"source": "test"})

    assert len(calls) == 1 and calls[0] == (delta, plan)
    assert result.value == [delta.creates[0].id]
    assert tuple(result.affected_goal_ids) == tuple(sorted(plan.affected_goal_ids, key=str))
    assert {item.goal_id for item in result.goal_fingerprints} >= {GOAL_LOW, GOAL_MID}
    previews = lock_tracer.of("preview")
    assert previews and not previews[0].goals and previews[0].roster, "initial preview must run under roster only"
    assert previews[0].value == plan.affected_goal_ids
    assert_roster_first_publication(lock_tracer, plan.affected_goal_ids)
    assert (await fresh_agent(sessions, delta.creates[0].id)) is not None


@pytest.mark.asyncio
async def test_stale_identity_map_is_reloaded_before_plan_comparison(sessions, tmp_path, lock_tracer):
    rm = roster_api()
    seeded = await seed_roster_goals(sessions, tmp_path, goal_ids=[GOAL_LOW])
    async with sessions() as a:
        agent = await a.get(Agent, seeded.agent_id)  # stays in A's identity map (expire_on_commit=False)
        delta = rm.ValidatedRosterDelta(updates=(dataclasses.replace(canonical(agent), description="A edit"),))
        plan = await rm.preview_roster_delta(a, delta, origin_goal_id=None)
        await a.commit()
        async with sessions() as b:
            await b.execute(update(Agent).where(Agent.id == seeded.agent_id).values(description="B edit"))
            await b.commit()
        spy = []

        async def mutation(db, locked_delta, locked_plan):
            spy.append(1)
            return await create_from_delta(db, locked_delta, locked_plan)

        outcome = await outcome_of(rm.AgentRosterMutationCoordinator().apply(
            a, delta, plan, mutation, origin_goal_id=None, provenance={}))
    assert stale_error(outcome) == "state_changed"
    assert spy == []
    assert (await fresh_agent(sessions, seeded.agent_id)).description == "B edit"


@pytest.mark.asyncio
async def test_roster_lock_is_never_acquired_while_holding_a_goal_lock(sessions, tmp_path, lock_tracer):
    rm = roster_api()
    seeded = await seed_roster_goals(sessions, tmp_path, goal_ids=[GOAL_LOW])
    delta = rm.ValidatedRosterDelta(creates=(new_agent_state(),))
    async with sessions() as s:
        plan = await rm.preview_roster_delta(s, delta, origin_goal_id=GOAL_LOW)
        await s.commit()
    spy = []

    async def mutation(db, locked_delta, locked_plan):
        spy.append(1)
        return await create_from_delta(db, locked_delta, locked_plan)

    async with sessions() as s:
        async with OrchestrationService()._lock_goal_for_baseline_transition(s, GOAL_LOW):
            outcome = await asyncio.wait_for(outcome_of(rm.AgentRosterMutationCoordinator().apply(
                s, delta, plan, mutation, origin_goal_id=GOAL_LOW, provenance={})), RACE_TIMEOUT)
    assert isinstance(outcome, Exception), "apply under a held goal lock must fail before effects"
    assert spy == []
    assert not [e for e in lock_tracer.of("roster:acquire") if e.goals]
    assert await fresh_agent(sessions, delta.creates[0].id) is None


# --------------------------------------------------------------------------- agent routes


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update", "delete"])
async def test_agent_routes_publish_through_roster_coordinator(sessions, tmp_path, lock_tracer, operation):
    from huddleroom.routers import agents as agents_router
    from huddleroom.schemas.agent import AgentCreate, AgentUpdate

    rm = roster_api()
    seeded = await seed_roster_goals(sessions, tmp_path, goal_ids=[GOAL_HIGH, GOAL_LOW])
    async with sessions() as s:
        user = await s.get(User, seeded.user_id)
        lock_tracer.start()
        if operation == "create":
            await agents_router.create_agent(AgentCreate(
                name="route-created", role="planner", provider="openai", model="gpt-4o-mini",
                adapter_type="api", capabilities=["planning"]), user, s)
        elif operation == "update":
            await agents_router.update_agent(seeded.agent_id, AgentUpdate(description="Route edit."), user, s)
        else:
            await agents_router.deactivate_agent(seeded.agent_id, user, s)
        await s.commit()  # what get_db does after the route returns
    previews = lock_tracer.of("preview")
    assert previews and previews[0].roster and not previews[0].goals
    assert set(previews[0].value) == {GOAL_HIGH, GOAL_LOW}
    assert_roster_first_publication(lock_tracer, previews[0].value)
    if operation == "update":
        assert (await fresh_agent(sessions, seeded.agent_id)).description == "Route edit."
    elif operation == "delete":
        assert (await fresh_agent(sessions, seeded.agent_id)).is_active is False
    else:
        assert len(await agents_named(sessions, "route-created")) == 1


@pytest.mark.asyncio
async def test_concurrent_route_and_recovery_mutations_serialize_without_inversion(
    sessions, tmp_path, lock_tracer
):
    from huddleroom.routers import agents as agents_router
    from huddleroom.schemas.agent import AgentCreate, AgentUpdate

    rm = roster_api()
    seeded = await seed_roster_goals(sessions, tmp_path, goal_ids=[GOAL_HIGH, GOAL_LOW])
    delta = rm.ValidatedRosterDelta(creates=(new_agent_state(name="recovery-created"),))
    async with sessions() as s:
        plan = await rm.preview_roster_delta(s, delta, origin_goal_id=GOAL_HIGH)
        user = await s.get(User, seeded.user_id)
        await s.commit()
    lock_tracer.start()

    async def route_update():
        async with sessions() as s:
            await agents_router.update_agent(seeded.agent_id, AgentUpdate(description="Concurrent."), user, s)
            await s.commit()

    async def route_create():
        async with sessions() as s:
            await agents_router.create_agent(AgentCreate(
                name="concurrent-created", role="planner", provider="openai", model="gpt-4o-mini",
                adapter_type="api", capabilities=["planning"]), user, s)
            await s.commit()

    async def recovery_apply():
        async with sessions() as s:
            return await rm.AgentRosterMutationCoordinator().apply(
                s, delta, plan, create_from_delta, origin_goal_id=GOAL_HIGH, provenance={"source": "recovery"})

    outcomes = await asyncio.wait_for(
        asyncio.gather(route_update(), route_create(), recovery_apply(), return_exceptions=True), RACE_TIMEOUT * 2)
    # The recovery plan was sealed before the routes ran: it applies first or fails stale; never deadlocks.
    assert not isinstance(outcomes[0], Exception) and not isinstance(outcomes[1], Exception), outcomes
    assert not isinstance(outcomes[2], Exception) or stale_error(outcomes[2]) == "state_changed", outcomes
    assert (await fresh_agent(sessions, seeded.agent_id)).description == "Concurrent."
    assert not [e for e in lock_tracer.of("roster:acquire") if e.goals], "roster acquired under a goal lock"
    assert all(e.roster for e in lock_tracer.of("agent:write")), "uncoordinated global agent write"
    holders = []
    for item in lock_tracer.events:  # the process-wide roster lock is never held by two tasks at once
        if item.kind == "roster:acquire":
            assert not holders, "two tasks held the roster write lock concurrently"
            holders.append(item.task)
        elif item.kind == "roster:release":
            holders.remove(item.task)


# --------------------------------------------------------------------------- collateral


@pytest.mark.asyncio
async def test_roster_change_invalidates_other_goal_with_existing_fingerprints(sessions, tmp_path, lock_tracer):
    rm = roster_api()
    async with sessions() as s:
        project, user = await seed_project_user(s, tmp_path)
        implementer = make_agent(name="shared-implementer")
        s.add(implementer)
        await s.flush()
        pairs = []
        for goal_id in (GOAL_HIGH, GOAL_LOW):
            goal, run = await seed_goal_run(s, project, goal_id=goal_id, phase="authorized")
            await assign(s, project.id, run.id, implementer, "implementation")
            for process_type in ("goal_definition", "manager_selection"):
                await complete(s, goal, run, process_type)
            await complete(s, goal, run, "agent_definition_review", outputs={"review_ids": [], "eligibility": {}})
            await complete(s, goal, run, "team_hierarchy", outputs={"approval": "approve"})
            pairs.append((goal, run))
        (goal_a, run_a), (goal_b, run_b) = pairs
        proposal = OrchestrationAction(run_id=run_a.id, idempotency_key=f"recovery:{uuid.uuid4()}:proposal",
                                       action_type="recovery_proposal", request={}, status="reserved")
        s.add(proposal)
        await s.commit()
        assert await OrchestrationService()._baseline_processes_ready_for_goal(s, goal_b.id, allow_heal=False)
        await s.commit()
        old_b = {ptype: (await OrchestrationProcessService().get_current(s, goal_b.id, ptype)).id
                 for ptype in ("agent_definition_review", "team_hierarchy")}
        delta = rm.ValidatedRosterDelta(creates=(new_agent_state(),))
        plan = await rm.preview_roster_delta(s, delta, origin_goal_id=goal_a.id)
        await s.commit()
    assert goal_b.id in plan.affected_goal_ids, "precondition: adding an agent changes goal B's fingerprints"
    lock_tracer.start()

    async with sessions() as s:
        result = await rm.AgentRosterMutationCoordinator().apply(
            s, delta, plan, create_from_delta, origin_goal_id=goal_a.id,
            provenance={"proposal_action_id": str(proposal.id), "source": "approved_recovery"})

    assert goal_b.id in result.affected_goal_ids
    change_b = next(item for item in result.goal_fingerprints if item.goal_id == goal_b.id)
    assert change_b.before_hierarchy_fingerprint != change_b.after_hierarchy_fingerprint
    goal_b_hierarchy = await current_process(sessions, goal_b.id, "team_hierarchy")
    goal_b_review = await current_process(sessions, goal_b.id, "agent_definition_review")
    assert goal_b_hierarchy.status == rm.INVALIDATED_PROCESS_STATUS == "running"
    assert goal_b_review.status == rm.INVALIDATED_PROCESS_STATUS
    assert old_b["team_hierarchy"] != goal_b_hierarchy.id and old_b["agent_definition_review"] != goal_b_review.id
    async with sessions() as s:
        superseded = await s.get(OrchestrationProcessRun, old_b["team_hierarchy"])
        assert superseded.superseded_by_id == goal_b_hierarchy.id
        warnings = list((await s.execute(select(OrchestrationWarning).where(
            OrchestrationWarning.goal_id == goal_b.id, OrchestrationWarning.active.is_(True)))).scalars())
        goal_b_warning = next(item for item in warnings if item.related_action_id is not None)
        assert goal_b_warning.related_action_id == proposal.id
        assert not await OrchestrationService()._baseline_processes_ready_for_goal(s, goal_b.id, allow_heal=False)
        # The invalidation committed atomically with the agent row.
        assert await s.get(Agent, delta.creates[0].id) is not None
    first_write = lock_tracer.events.index(lock_tracer.of("agent:write")[0])
    commit_at = lock_tracer.index(lambda e: e.kind == "commit", first_write)
    assert lock_tracer.index(lambda e: e.kind == "roster:release", first_write) > commit_at


# --------------------------------------------------------------------------- follow-up


def _set_reset_fence_on_preview(tracer, goal_id, call_number):
    """Raise the reset fence during the Nth traced preview (1 = prepare, 2 = coordinator)."""
    from huddleroom.services import orchestration_baseline_runtime as runtime

    calls = []

    async def hook(_plan):
        calls.append(1)
        if len(calls) == call_number:
            runtime._resetting[goal_id] = runtime._resetting.get(goal_id, 0) + 1

    tracer.after_preview.append(hook)


@pytest.mark.asyncio
async def test_reset_between_reserve_and_apply_writes_no_agent(
    sessions, tmp_path, approve_semantic_reviews, lock_tracer
):
    seeded = await seed_definition_proposal(sessions, tmp_path)
    before = (await fresh_agent(sessions, seeded.agent_id)).description
    lock_tracer.start()
    _set_reset_fence_on_preview(lock_tracer, seeded.goal_id, 2)  # under the roster lock, before goal locks
    try:
        outcome = await outcome_or_conflict(act_definition_batch(sessions, seeded))
    finally:
        clear_reset_fence(seeded.goal_id)

    assert getattr(outcome, "status_code", None) == 409
    assert not lock_tracer.of("agent:write"), "a global agent change landed for a discarded baseline"
    assert (await fresh_agent(sessions, seeded.agent_id)).description == before
    assert (await current_process(sessions, seeded.goal_id, "agent_definition_review")).status != "completed"


@pytest.mark.asyncio
async def test_reset_racing_reservation_commit_discards_reservation(
    sessions, tmp_path, approve_semantic_reviews, lock_tracer
):
    seeded = await seed_definition_proposal(sessions, tmp_path)
    before = (await fresh_agent(sessions, seeded.agent_id)).description
    lock_tracer.start()
    _set_reset_fence_on_preview(lock_tracer, seeded.goal_id, 1)  # during prepare, under the goal lock
    try:
        outcome = await outcome_or_conflict(act_definition_batch(sessions, seeded))
    finally:
        clear_reset_fence(seeded.goal_id)

    assert getattr(outcome, "status_code", None) == 409
    assert not lock_tracer.of("roster:acquire") and not lock_tracer.of("agent:write")
    assert (await fresh_agent(sessions, seeded.agent_id)).description == before
    decision = next(item for item in await decisions_for(sessions, seeded.goal_id) if item.id == seeded.decision_id)
    assert decision.status == "pending", "the reservation was committed despite the reset"


@pytest.mark.asyncio
async def test_debug_retry_failed_hands_off_roster_effect(sessions, tmp_path, monkeypatch, lock_tracer):
    from huddleroom.services import orchestration_debug_service as debug_module
    from huddleroom.services.orchestration_team_hierarchy import TeamHierarchyProcess

    seeded = await seed_hierarchy_proposal(sessions, tmp_path, monkeypatch, answer="approve", running=True)
    async with sessions() as s:
        current = await OrchestrationProcessService().get_current(s, seeded.goal_id, "team_hierarchy")
        current.outputs = {**current.outputs, "_lm_retry": {"kind": "team_hierarchy", "version": 1, "request": {}}}
        await s.commit()
    monkeypatch.setattr(debug_module, "valid_lm_retry_checkpoint", lambda _checkpoint, _process_type: True)

    async def retried(self, db, goal, run, current):
        current.outputs = {key: value for key, value in current.outputs.items() if key != "_lm_retry"}
        await db.flush()
        return await self.advance(db, goal, run)

    monkeypatch.setattr(TeamHierarchyProcess, "retry_failed", retried)
    lock_tracer.start()

    async with sessions() as s:
        result = await debug_module.OrchestrationDebugService().retry_failed(
            s, seeded.project_id, seeded.goal_id, "team_hierarchy")

    assert_entry_handoff(lock_tracer, seeded.goal_id)
    single_agent_write(lock_tracer, name=seeded.proposal_name)
    assert len(await agents_named(sessions, seeded.proposal_name)) == 1
    assert result["process_type"] == "team_hierarchy"
    assert any(item.decision_key == "team_hierarchy:approval" and item.status == "pending"
               for item in await decisions_for(sessions, seeded.goal_id))


@pytest.mark.asyncio
async def test_collateral_invalidated_goal_rebuilds_on_next_tick(
    sessions, tmp_path, approve_semantic_reviews, lock_tracer, monkeypatch
):
    from huddleroom.services.orchestration_team_hierarchy_analyzer import TeamHierarchyAnalysis, TeamHierarchyAnalyzer

    rm = roster_api()

    async def review(_self, payload, project=None, *, project_id=None):
        capable = {capability: agent["id"] for agent in payload["agents"] for capability in agent["capabilities"]}
        required = payload["required_work_functions"]
        assignments = tuple({"work_function": work, "agent_ref": capable[work]} for work in required if work in capable)
        return TeamHierarchyAnalysis(
            proposed_agents=(), assignments=assignments,
            reporting_lines=tuple({"agent_ref": ref, "reports_to": "manager"}
                                  for ref in dict.fromkeys(item["agent_ref"] for item in assignments)),
            documented_gaps=tuple(work for work in required if work not in capable),
            rationale="Covers the available work.",
            self_review="Checked definitions, references, coverage, reporting, and independence.",
        )

    monkeypatch.setattr(TeamHierarchyAnalyzer, "review", review)
    async with sessions() as s:
        project, _user = await seed_project_user(s, tmp_path)
        implementer = make_agent(name="shared-implementer", adapter_type="cli", cli_runtime="codex")
        s.add(implementer)
        await s.flush()
        pairs = []
        for goal_id in (GOAL_HIGH, GOAL_LOW):
            goal, run = await seed_goal_run(s, project, goal_id=goal_id, phase="authorized")
            await assign(s, project.id, run.id, implementer, "implementation")
            for process_type in ("goal_definition", "manager_selection"):
                await complete(s, goal, run, process_type)
            await complete(s, goal, run, "agent_definition_review", outputs={"review_ids": [], "eligibility": {}})
            await complete(s, goal, run, "team_hierarchy", outputs={"approval": "approve"})
            pairs.append((goal, run))
        (goal_a, run_a), (goal_b, run_b) = pairs
        proposal = OrchestrationAction(run_id=run_a.id, idempotency_key=f"recovery:{uuid.uuid4()}:proposal",
                                       action_type="recovery_proposal", request={}, status="reserved")
        s.add(proposal)
        await s.commit()
    delta = rm.ValidatedRosterDelta(creates=(new_agent_state(),))
    async with sessions() as s:
        plan = await rm.preview_roster_delta(s, delta, origin_goal_id=goal_a.id)
        await s.commit()
    async with sessions() as s:
        await rm.AgentRosterMutationCoordinator().apply(
            s, delta, plan, create_from_delta, origin_goal_id=goal_a.id,
            provenance={"proposal_action_id": str(proposal.id)})
    invalidated = await current_process(sessions, goal_b.id, "team_hierarchy")
    assert invalidated.status == rm.INVALIDATED_PROCESS_STATUS and "recovery_invalidation" in invalidated.outputs

    stub_prerequisite_processes(monkeypatch)
    result = await asyncio.wait_for(act_tick(sessions, SimpleNamespace(run_id=run_b.id)), RACE_TIMEOUT)

    review_row = await current_process(sessions, goal_b.id, "agent_definition_review")
    hierarchy_row = await current_process(sessions, goal_b.id, "team_hierarchy")
    assert result["agent_definition_review_process"]["status"] == "completed"
    assert review_row.status == "completed" and review_row.outputs["gates"]["agent_definitions_reviewed"] is True
    assert "recovery_invalidation" not in review_row.outputs
    assert hierarchy_row.status in {"waiting_decision", "completed"}, "goal B stalled on the invalidation row"
    assert isinstance(hierarchy_row.outputs.get("analysis"), dict)
