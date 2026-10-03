import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

FIXED_TS = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def orch_goal(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGoal

    goal = OrchestrationGoal(
        project_id=test_project.id,
        objective="Ship the widget",
        success_criteria=[{"key": "works", "description": "widget works"}],
        constraints={"deadline": "2026-08-01"},
    )
    db_session.add(goal)
    await db_session.flush()
    return goal


@pytest_asyncio.fixture
async def orch_run(db_session, orch_goal):
    from huddleroom.models.orchestration import OrchestrationRun

    run = OrchestrationRun(goal_id=orch_goal.id)
    db_session.add(run)
    await db_session.flush()
    return run


@pytest.mark.asyncio
async def test_list_queries_break_timestamp_ties_by_id(db_session, orch_goal):
    from huddleroom.models.orchestration_process import OrchestrationWarning
    from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

    for _ in range(4):
        db_session.add(
            OrchestrationWarning(
                goal_id=orch_goal.id,
                warning_type="tie_check",
                severity="warning",
                message="same timestamp",
                created_at=FIXED_TS,
            )
        )
    await db_session.flush()

    service = OrchestrationWarningService()
    first = [w.id for w in await service.list_warnings(db_session, orch_goal.id)]
    second = [w.id for w in await service.list_warnings(db_session, orch_goal.id)]

    assert first == second
    assert first == sorted(first, key=str)


@pytest.mark.asyncio
@pytest.mark.parametrize("entity_type", ["process_run", "decision", "section"])
async def test_query_breaks_timestamp_ties_by_id(db_session, test_project, orch_goal, entity_type):
    """Verify that list queries maintain consistent ordering when timestamps match."""
    from huddleroom.models.orchestration_memory import OrchestrationMemorySection
    from huddleroom.models.orchestration_process import (
        OrchestrationAuthorityDecision,
        OrchestrationProcessRun,
    )
    from huddleroom.services.orchestration_authority_service import (
        OrchestrationAuthorityDecisionService,
    )
    from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
    from huddleroom.services.orchestration_process_service import OrchestrationProcessService

    if entity_type == "process_run":
        # Distinct process_types: the partial unique index forbids more than one
        # non-superseded run per (goal_id, process_type), so ties on
        # (started_at, created_at) that must fall back to id are exercised across
        # types (list_process_runs orders by timestamp then id regardless of type).
        for process_type in (
            "goal_definition",
            "manager_selection",
            "team_hierarchy",
            "agent_definition_review",
        ):
            db_session.add(
                OrchestrationProcessRun(
                    goal_id=orch_goal.id,
                    process_type=process_type,
                    trigger_reason="tie_check",
                    status="running",
                    started_at=FIXED_TS,
                    created_at=FIXED_TS,
                )
            )
        await db_session.flush()
        service = OrchestrationProcessService()
        first = [r.id for r in await service.list_process_runs(db_session, orch_goal.id)]
        second = [r.id for r in await service.list_process_runs(db_session, orch_goal.id)]
    elif entity_type == "decision":
        for i in range(4):
            db_session.add(
                OrchestrationAuthorityDecision(
                    goal_id=orch_goal.id,
                    decision_key=f"tie-{i}",
                    title=f"tie-{i}",
                    status="answered",
                    authority="human",
                    question="pick one",
                    options=["a", "b"],
                    selected_option="a",
                    asked_at=FIXED_TS,
                    decided_at=FIXED_TS,
                    created_at=FIXED_TS,
                )
            )
        await db_session.flush()
        service = OrchestrationAuthorityDecisionService()
        first = [d.id for d in await service.list_decisions(db_session, orch_goal.id, status="answered")]
        second = [d.id for d in await service.list_decisions(db_session, orch_goal.id, status="answered")]
    else:  # section
        for i in range(4):
            db_session.add(
                OrchestrationMemorySection(
                    project_id=test_project.id,
                    goal_id=orch_goal.id,
                    section_key=f"tie_key_{i}",
                    title=f"tie-{i}",
                    body="body",
                    toc_order=0,
                    created_by="orchestrator",
                    created_at=FIXED_TS,
                )
            )
        await db_session.flush()
        service = OrchestrationMemoryService()
        first = [s.id for s in await service.list_sections(db_session, test_project.id, orch_goal.id)]
        second = [s.id for s in await service.list_sections(db_session, test_project.id, orch_goal.id)]

    assert first == second
    assert first == sorted(first, key=str)


EXPECTED_KEY_ORDER = [
    "objective",
    "objective_notes",
    "goal_status",
    "goal_weight",
    "run_status",
    "current_process",
    "manager",
    "hierarchy",
    "constraints",
    "active_warnings",
    "recent_decisions",
    "open_blockers",
    "skipped_processes",
    "introduction",
    "always_loaded",
    "toc",
]


@pytest.mark.asyncio
async def test_preface_skeleton_fixed_key_order(db_session, orch_goal, orch_run):
    from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder

    preface = await OrchestrationMemoryPrefaceBuilder().build(db_session, orch_goal, orch_run)

    assert list(preface.keys()) == EXPECTED_KEY_ORDER
    assert preface["objective"] == "Ship the widget"
    assert preface["goal_status"] == "active"
    assert preface["goal_weight"] == "standard"
    assert preface["run_status"] == "running"
    assert preface["current_process"] is None
    assert preface["manager"] is None
    assert preface["hierarchy"] is None
    assert '"deadline"' in preface["constraints"]
    assert preface["active_warnings"] == []
    assert preface["recent_decisions"] == []
    assert preface["open_blockers"] == []
    assert preface["skipped_processes"] == []
    assert preface["introduction"] is None
    assert preface["always_loaded"] == []
    assert preface["toc"] == []


@pytest.mark.asyncio
async def test_preface_without_run(db_session, orch_goal):
    from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder

    preface = await OrchestrationMemoryPrefaceBuilder().build(db_session, orch_goal, None)

    assert preface["run_status"] is None
    assert preface["open_blockers"] == []


@pytest.mark.asyncio
async def test_preface_truncates_long_objective(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationGoal
    from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder

    goal = OrchestrationGoal(project_id=test_project.id, objective="x" * 1000)
    db_session.add(goal)
    await db_session.flush()

    preface = await OrchestrationMemoryPrefaceBuilder().build(db_session, goal, None)

    assert len(preface["objective"]) == 300
    assert preface["objective"].endswith("...")


@pytest.mark.asyncio
async def test_preface_is_deterministic(db_session, orch_goal, orch_run):
    import json

    from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder

    builder = OrchestrationMemoryPrefaceBuilder()
    first = await builder.build(db_session, orch_goal, orch_run)
    second = await builder.build(db_session, orch_goal, orch_run)

    assert first == second
    assert json.dumps(first, default=str) == json.dumps(second, default=str)


@pytest_asyncio.fixture
async def full_state(db_session, orch_goal, orch_run):
    """Goal with process runs, warnings, decisions, blockers, memory sections."""
    from datetime import timedelta

    from huddleroom.models.orchestration_memory import OrchestrationMemorySection
    from huddleroom.models.orchestration_process import (
        OrchestrationAuthorityDecision,
        OrchestrationProcessRun,
        OrchestrationWarning,
    )

    # Insert the current (non-superseded) run first so it is the sole row with
    # superseded_by_id IS NULL for this (goal_id, process_type); the superseded
    # run is then inserted already pointing at it. Inserting both with a NULL
    # superseded_by_id before wiring the link would violate the partial unique
    # index (and superseded_by_id's FK forbids pointing at a not-yet-inserted
    # row), so the link must be set before the second flush.
    current = OrchestrationProcessRun(
        goal_id=orch_goal.id,
        process_type="goal_definition",
        trigger_reason="rerun",
        status="running",
        started_at=FIXED_TS - timedelta(hours=1),
    )
    db_session.add(current)
    await db_session.flush()
    superseded = OrchestrationProcessRun(
        goal_id=orch_goal.id,
        process_type="goal_definition",
        trigger_reason="first attempt",
        status="completed",
        started_at=FIXED_TS - timedelta(hours=3),
        superseded_by_id=current.id,
    )
    db_session.add(superseded)
    await db_session.flush()
    skipped = OrchestrationProcessRun(
        goal_id=orch_goal.id,
        process_type="team_hierarchy",
        trigger_reason="kickoff",
        status="skipped",
        skipped_by="human:u1",
        started_at=FIXED_TS - timedelta(hours=2),
    )
    db_session.add(skipped)

    for i in range(12):
        db_session.add(
            OrchestrationWarning(
                goal_id=orch_goal.id,
                warning_type=f"type_{i:02d}",
                severity="warning",
                message=f"warning-{i:02d}",
                created_at=FIXED_TS + timedelta(minutes=i),
            )
        )
    db_session.add(
        OrchestrationWarning(
            goal_id=orch_goal.id,
            warning_type="resolved_type",
            severity="warning",
            message="already resolved",
            active=False,
            created_at=FIXED_TS + timedelta(minutes=30),
        )
    )

    for i in range(7):
        db_session.add(
            OrchestrationAuthorityDecision(
                goal_id=orch_goal.id,
                decision_key=f"answered-{i}",
                title=f"decision-{i}",
                status="answered",
                authority="human",
                question="pick one",
                options=["a", "b"],
                selected_option="a",
                asked_at=FIXED_TS + timedelta(minutes=i),
                decided_at=FIXED_TS + timedelta(minutes=10 + i),
            )
        )
    db_session.add(
        OrchestrationAuthorityDecision(
            goal_id=orch_goal.id,
            decision_key="still-pending",
            title="pending decision",
            status="pending",
            authority="human",
            question="waiting",
            options=["a", "b"],
            asked_at=FIXED_TS,
        )
    )

    sections = [
        OrchestrationMemorySection(
            project_id=orch_goal.project_id,
            goal_id=orch_goal.id,
            section_key="introduction",
            title="Introduction",
            body="intro body " * 100,
            always_load=True,
            toc_order=0,
            created_by="orchestrator",
        ),
        OrchestrationMemorySection(
            project_id=orch_goal.project_id,
            goal_id=orch_goal.id,
            section_key="manager_authority",
            title="Manager",
            body="long manager rationale " * 50,
            summary="Alice is manager",
            toc_order=1,
            created_by="orchestrator",
        ),
        OrchestrationMemorySection(
            project_id=orch_goal.project_id,
            goal_id=orch_goal.id,
            section_key="team_hierarchy",
            title="Hierarchy",
            body="lead: bob; reviewers: carol " * 40,
            toc_order=2,
            created_by="orchestrator",
        ),
        OrchestrationMemorySection(
            project_id=orch_goal.project_id,
            goal_id=orch_goal.id,
            section_key="operating_assumptions",
            title="Operating Assumptions",
            body="assumption details " * 50,
            summary="two assumptions active",
            always_load=True,
            toc_order=3,
            created_by="orchestrator",
        ),
        OrchestrationMemorySection(
            project_id=orch_goal.project_id,
            goal_id=orch_goal.id,
            section_key="recovery_history",
            title="Recovery History",
            body="RECOVERY_BODY_MARKER " * 500,
            toc_order=4,
            created_by="orchestrator",
        ),
    ]
    db_session.add_all(sections)

    orch_run.active_blockers = [
        {"kind": "gate", "gate_id": "g1", "message": "gate failed"},
        {"kind": "task", "task_id": "t1", "message": "task stuck"},
    ]
    await db_session.flush()
    return orch_goal


@pytest.mark.asyncio
async def test_list_section_metadata_includes_created_at_excludes_body(db_session, full_state):
    """Merged test: verifies list_section_metadata() includes created_at and excludes body.

    Combines two regression tests:
    1. Fix 3: created_at is included in metadata select
    2. Excludes body field to avoid loading large section bodies
    """
    from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService

    rows = await OrchestrationMemoryService().list_section_metadata(
        db_session, full_state.project_id, full_state.id
    )

    assert len(rows) == 5

    # Check section order (from original test_list_section_metadata_excludes_body)
    assert [r.section_key for r in rows] == [
        "introduction",
        "manager_authority",
        "team_hierarchy",
        "operating_assumptions",
        "recovery_history",
    ]

    # Check that body is NOT loaded (verifies select excludes body)
    for row in rows:
        assert not hasattr(row, "body") or row.body is None

    # Check that created_at IS present for all rows (verifies select includes created_at)
    for row in rows:
        assert hasattr(row, "created_at")
        assert row.created_at is not None
        # Verify created_at is a reasonable datetime (handle both aware and naive)
        row_dt = row.created_at
        now_dt = datetime.now(timezone.utc)
        if row_dt.tzinfo is not None:
            row_dt = row_dt.astimezone(timezone.utc).replace(tzinfo=None)
        if now_dt.tzinfo is not None:
            now_dt = now_dt.astimezone(timezone.utc).replace(tzinfo=None)
        assert row_dt <= now_dt


@pytest.mark.asyncio
async def test_preface_full_content(db_session, full_state, orch_run):
    import json

    from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder

    # Explicit large budget: this test pins content, not budget behavior. The
    # full_state preface sits near the 3000-char default; without this, Task 4's
    # eviction would trim the toc and break the assertions below.
    preface = await OrchestrationMemoryPrefaceBuilder().build(
        db_session, full_state, orch_run, budget_chars=10_000
    )

    assert preface["current_process"] == {"process_type": "goal_definition", "status": "running"}
    assert preface["skipped_processes"] == [
        {"process_type": "team_hierarchy", "skipped_by": "human:u1"}
    ]

    assert len(preface["active_warnings"]) == 10
    messages = [w["message"] for w in preface["active_warnings"]]
    assert messages[0] == "warning-02"  # oldest two evicted, most recent N kept
    assert messages[-1] == "warning-11"
    assert "already resolved" not in messages
    assert preface["active_warnings"][0]["acknowledged"] is False

    assert len(preface["recent_decisions"]) == 5
    titles = [d["title"] for d in preface["recent_decisions"]]
    assert titles == ["decision-2", "decision-3", "decision-4", "decision-5", "decision-6"]
    assert all(d["authority"] == "human" for d in preface["recent_decisions"])
    assert "pending decision" not in titles

    assert preface["manager"] == "Alice is manager"
    assert preface["hierarchy"].startswith("lead: bob")
    assert len(preface["hierarchy"]) <= 300
    assert preface["introduction"].startswith("intro body")
    assert len(preface["introduction"]) <= 500

    assert preface["always_loaded"] == [
        {"section_key": "operating_assumptions", "summary": "two assumptions active"}
    ]
    assert [t["section_key"] for t in preface["toc"]] == [
        "introduction",
        "manager_authority",
        "team_hierarchy",
        "operating_assumptions",
        "recovery_history",
    ]

    assert len(preface["open_blockers"]) == 2
    assert "gate failed" in preface["open_blockers"][0]

    # spec Phase 4 check: large memory sections omitted by default
    assert "RECOVERY_BODY_MARKER" not in json.dumps(preface, default=str)


@pytest.mark.asyncio
async def test_recent_decisions_excludes_team_lead_and_agent_authority(db_session, orch_goal, orch_run):
    """spec 5.3: 'recent human or manager decisions' — team_lead/agent decisions
    must not consume the max_decisions cap or displace a human/manager one."""
    from datetime import timedelta

    from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
    from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder

    for i, authority in enumerate(["human", "manager", "team_lead", "agent"]):
        db_session.add(
            OrchestrationAuthorityDecision(
                goal_id=orch_goal.id,
                decision_key=f"mixed-{i}",
                title=f"mixed-{authority}",
                status="answered",
                authority=authority,
                question="pick one",
                options=["a", "b"],
                selected_option="a",
                asked_at=FIXED_TS + timedelta(minutes=i),
                # team_lead/agent decisions are the most recent by decided_at;
                # if they weren't filtered before the cap, they'd displace
                # the human/manager ones below under max_decisions=2.
                decided_at=FIXED_TS + timedelta(minutes=10 + i),
            )
        )
    await db_session.flush()

    preface = await OrchestrationMemoryPrefaceBuilder().build(
        db_session, orch_goal, orch_run, max_decisions=2
    )

    titles = [d["title"] for d in preface["recent_decisions"]]
    assert titles == ["mixed-human", "mixed-manager"]
    assert all(d["authority"] in ("human", "manager") for d in preface["recent_decisions"])


@pytest.mark.parametrize("kwarg", ["max_warnings", "max_decisions"])
@pytest.mark.parametrize("value", [0, -1])
@pytest.mark.asyncio
async def test_max_caps_reject_non_positive_values(db_session, orch_goal, orch_run, kwarg, value):
    """0 would silently mean 'include everything' via items[-0:] == items[0:]
    if unvalidated — this must raise instead of defeating the cap."""
    from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder

    with pytest.raises(ValueError):
        await OrchestrationMemoryPrefaceBuilder().build(
            db_session, orch_goal, orch_run, **{kwarg: value}
        )


@pytest.mark.asyncio
async def test_preface_respects_budget(db_session, full_state, orch_run):
    from huddleroom.services.orchestration_memory_preface import (
        OrchestrationMemoryPrefaceBuilder,
        preface_size,
    )

    builder = OrchestrationMemoryPrefaceBuilder()
    default = await builder.build(db_session, full_state, orch_run)
    assert preface_size(default) <= 3000

    small = await builder.build(db_session, full_state, orch_run, budget_chars=800)
    assert preface_size(small) <= 800
    assert small["toc"] == []
    assert small["active_warnings"] == []
    assert small["recent_decisions"] == []
    # skeleton survives: identity fields are never evicted
    assert small["objective"] == "Ship the widget"
    assert small["run_status"] == "running"


@pytest.mark.asyncio
async def test_budget_evicts_toc_before_warnings(db_session, full_state, orch_run):
    from huddleroom.services.orchestration_memory_preface import (
        OrchestrationMemoryPrefaceBuilder,
        preface_size,
    )

    builder = OrchestrationMemoryPrefaceBuilder()
    default = await builder.build(db_session, full_state, orch_run)
    squeezed = await builder.build(
        db_session, full_state, orch_run, budget_chars=preface_size(default) - 1
    )

    assert preface_size(squeezed) <= preface_size(default) - 1
    assert len(squeezed["toc"]) < len(default["toc"])
    assert squeezed["active_warnings"] == default["active_warnings"]


@pytest.mark.asyncio
async def test_budget_enforcement_is_deterministic(db_session, full_state, orch_run):
    from huddleroom.services.orchestration_memory_preface import OrchestrationMemoryPrefaceBuilder

    builder = OrchestrationMemoryPrefaceBuilder()
    first = await builder.build(db_session, full_state, orch_run, budget_chars=1500)
    second = await builder.build(db_session, full_state, orch_run, budget_chars=1500)

    assert first == second


@pytest.mark.asyncio
async def test_budget_evicts_skipped_processes(db_session, full_state, orch_run):
    from huddleroom.services.orchestration_memory_preface import (
        OrchestrationMemoryPrefaceBuilder,
        preface_size,
    )

    builder = OrchestrationMemoryPrefaceBuilder()
    default = await builder.build(db_session, full_state, orch_run)
    assert default["skipped_processes"] != []

    # Squeeze past the point where lists/decisions/warnings/blockers are
    # already empty (per test_preface_respects_budget at budget_chars=800)
    # but before the scalar fallback fires, to prove skipped_processes is
    # itself evicted rather than kept unconditionally.
    tiny = await builder.build(db_session, full_state, orch_run, budget_chars=700)
    assert preface_size(tiny) <= 700
    assert tiny["skipped_processes"] == []
    assert tiny["objective"] == "Ship the widget"


@pytest.mark.asyncio
async def test_budget_below_floor_raises(db_session, full_state, orch_run):
    from huddleroom.services.orchestration_memory_preface import (
        PREFACE_MIN_CHARS,
        OrchestrationMemoryPrefaceBuilder,
    )

    builder = OrchestrationMemoryPrefaceBuilder()
    with pytest.raises(ValueError):
        await builder.build(db_session, full_state, orch_run, budget_chars=PREFACE_MIN_CHARS - 1)


@pytest.mark.asyncio
async def test_budget_at_floor_never_exceeds(db_session, full_state, orch_run):
    from huddleroom.services.orchestration_memory_preface import (
        PREFACE_MIN_CHARS,
        OrchestrationMemoryPrefaceBuilder,
        preface_size,
    )

    builder = OrchestrationMemoryPrefaceBuilder()
    preface = await builder.build(db_session, full_state, orch_run, budget_chars=PREFACE_MIN_CHARS)

    assert preface_size(preface) <= PREFACE_MIN_CHARS


@pytest.mark.asyncio
async def test_decision_context_includes_memory_preface(db_session, full_state, orch_run):
    import json

    from huddleroom.services.orchestration_service import OrchestrationService

    context = await OrchestrationService()._decision_context(db_session, full_state, orch_run)

    assert list(context["memory_preface"].keys()) == EXPECTED_KEY_ORDER
    assert context["memory_preface"]["objective"] == "Ship the widget"
    assert context["memory_preface"]["manager"] == "Alice is manager"
    # spec Phase 4 check: tick context stays compact — no full section bodies
    assert "RECOVERY_BODY_MARKER" not in json.dumps(context["memory_preface"], default=str)
