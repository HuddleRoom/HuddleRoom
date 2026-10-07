import argparse
import asyncio
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from huddleroom.models.agent import Agent
from huddleroom.models.event_log import EventLog
from huddleroom.models.hook import Hook
from huddleroom.models.knowledge_item import KnowledgeItem
from huddleroom.models.meeting import (
    Meeting,
    MeetingActionItem,
    MeetingAgendaItem,
    MeetingDecision,
    MeetingEvent,
    MeetingParticipantSignal,
    MeetingTurn,
)
from huddleroom.models.memory_item import MemoryItem
from huddleroom.models.optimization import CostMetric, Optimization, Pattern
from huddleroom.models.project import Project
from huddleroom.models.graph import Graph, GraphRun, GraphRunTimeout, GraphRunStep
from huddleroom.models.routing_rule import RoutingRule
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.models.user import User
from huddleroom.security import hash_password
import huddleroom.models  # noqa: F401

SEED_NAMES = {
    "project": "HuddleRoom UI Live Project",
    "user_email": "ui-live@example.test",
    "architect": "huddleroom-ui-architect",
    "reviewer": "huddleroom-ui-reviewer",
    "inactive_agent": "huddleroom-ui-inactive",
    "ready_task": "UI seed ready task",
    "backlog_task": "UI seed backlog task",
    "blocked_task": "UI seed blocked task",
    "done_task": "UI seed done task",
    "disposable_task": "UI disposable task",
    "active_meeting": "UI seed active meeting",
    "concluded_meeting": "UI seed concluded meeting",
    "active_graph": "ui_seed_graph",
    "inactive_graph": "ui_inactive_graph",
    "knowledge": "UI seed rollout policy",
    "disposable_knowledge": "UI disposable knowledge",
    "rule": "UI seed routing rule",
    "rule_disabled": "UI seed disabled rule",
    "hook": "ui_seed_hook",
    "hook_disabled": "ui_disabled_hook",
    "optimization": "UI seed optimization",
}

SEED_AGENT_NAMES = [
    SEED_NAMES["architect"],
    SEED_NAMES["reviewer"],
    SEED_NAMES["inactive_agent"],
]
SEED_MARKER = {
    "purpose": "live-ui-tests",
    "seed_version": 1,
}

SEEDED_MODELS = (
    Project,
    User,
    Agent,
    Task,
    Session,
    Meeting,
    MeetingAgendaItem,
    MeetingTurn,
    MeetingDecision,
    MeetingActionItem,
    MeetingEvent,
    MeetingParticipantSignal,
    Graph,
    GraphRun,
    GraphRunStep,
    GraphRunTimeout,
    KnowledgeItem,
    MemoryItem,
    RoutingRule,
    Hook,
    Pattern,
    Optimization,
    CostMetric,
    EventLog,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _resolve_sqlite_file_path(database_url: str) -> Path:
    raw_path = unquote(database_url.removeprefix("sqlite+aiosqlite:///"))
    if not raw_path:
        raise ValueError("database_url must include a sqlite file path")
    if raw_path.startswith("/"):
        return Path(raw_path).resolve(strict=False)
    return (Path.cwd() / raw_path).resolve(strict=False)


def _validate_database_url(database_url: str) -> None:
    if not database_url:
        raise ValueError("database_url is required")
    if not database_url.startswith("sqlite+aiosqlite:///"):
        raise ValueError("database_url must be a sqlite file URL")
    if database_url.endswith("/:memory:"):
        raise ValueError("database_url must be a sqlite file URL")
    protected = {name: (Path.cwd() / name).resolve(strict=False) for name in ("huddleroom.db", "rally.db")}
    for name, protected_path in protected.items():
        if _resolve_sqlite_file_path(database_url) == protected_path:
            raise ValueError(f"database_url must not target {name}")


def _validate_workspace_dir(workspace_dir: Path | None) -> Path:
    if workspace_dir is None:
        raise ValueError("workspace_dir is required")
    return Path(workspace_dir)


async def _count_rows(session, model) -> int:
    count = await session.scalar(select(func.count()).select_from(model))
    return int(count or 0)


async def _seed_project_ids(session) -> list:
    projects = (await session.execute(select(Project).where(Project.name == SEED_NAMES["project"]))).scalars().all()
    return [
        project.id
        for project in projects
        if isinstance(project.config, dict)
        and project.config.get("purpose") == "live-ui-tests"
        and project.config.get("seed_version") == 1
    ]


def _is_seeded_agent(agent: Agent) -> bool:
    return (
        isinstance(agent.config, dict)
        and agent.config.get("seed_metadata") == {**SEED_MARKER, "kind": "agent"}
    )


async def _seed_agent_ids(session) -> list:
    seed_agents = (
        await session.execute(select(Agent).where(Agent.name.in_(SEED_AGENT_NAMES)))
    ).scalars().all()
    agent_ids = []
    collisions = []
    for agent in seed_agents:
        if _is_seeded_agent(agent):
            agent_ids.append(agent.id)
            continue
        collisions.append(agent.name)

    if collisions:
        raise ValueError(
            "reserved seed agent name already exists without seed metadata: "
            + ", ".join(sorted(collisions))
        )

    return agent_ids


async def _delete_existing_seed_records(session) -> None:
    project_ids = await _seed_project_ids(session)
    agent_ids = await _seed_agent_ids(session)

    if project_ids:
        meeting_ids = (
            await session.execute(select(Meeting.id).where(Meeting.project_id.in_(project_ids)))
        ).scalars().all()
        graph_run_ids = (
            await session.execute(
                select(GraphRun.id).where(GraphRun.project_id.in_(project_ids))
            )
        ).scalars().all()

        if graph_run_ids:
            await session.execute(
                delete(GraphRunTimeout).where(GraphRunTimeout.graph_run_id.in_(graph_run_ids))
            )
            await session.execute(
                delete(GraphRunStep).where(
                    GraphRunStep.graph_run_id.in_(graph_run_ids)
                )
            )

        if meeting_ids:
            await session.execute(
                delete(MeetingParticipantSignal).where(
                    MeetingParticipantSignal.meeting_id.in_(meeting_ids)
                )
            )
            await session.execute(delete(MeetingEvent).where(MeetingEvent.meeting_id.in_(meeting_ids)))
            await session.execute(
                delete(MeetingActionItem).where(MeetingActionItem.meeting_id.in_(meeting_ids))
            )
            await session.execute(
                delete(MeetingDecision).where(MeetingDecision.meeting_id.in_(meeting_ids))
            )
            await session.execute(delete(MeetingTurn).where(MeetingTurn.meeting_id.in_(meeting_ids)))
            await session.execute(
                delete(MeetingAgendaItem).where(MeetingAgendaItem.meeting_id.in_(meeting_ids))
            )

        for model in (
            CostMetric,
            Optimization,
            Pattern,
            Hook,
            RoutingRule,
            MemoryItem,
            KnowledgeItem,
            GraphRun,
            Graph,
            Meeting,
            Session,
            Task,
            EventLog,
        ):
            await session.execute(delete(model).where(model.project_id.in_(project_ids)))

        await session.execute(delete(Project).where(Project.id.in_(project_ids)))

    if agent_ids:
        await session.execute(delete(MemoryItem).where(MemoryItem.agent_id.in_(agent_ids)))
        await session.execute(delete(Session).where(Session.agent_id.in_(agent_ids)))
        await session.execute(delete(Agent).where(Agent.id.in_(agent_ids)))

    await session.execute(delete(User).where(User.email == SEED_NAMES["user_email"]))
    await session.flush()


async def seed_database(database_url: str, workspace_dir: Path | None) -> dict:
    _validate_database_url(database_url)
    workspace_root = _validate_workspace_dir(workspace_dir)
    seed_workspace = workspace_root / "huddleroom-ui-live-project"
    seed_workspace.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    now = _utcnow()

    try:
        async with session_factory() as session:
            await _delete_existing_seed_records(session)

            user = User(
                email=SEED_NAMES["user_email"],
                hashed_password=hash_password("ui-live-password"),
                display_name="UI Live Tester",
                role="admin",
                is_active=True,
            )
            session.add(user)
            await session.flush()

            project = Project(
                name=SEED_NAMES["project"],
                description="Deterministic project for live UI tests",
                workspace_path=str(seed_workspace.resolve()),
                config={"purpose": "live-ui-tests", "seed_version": 1},
            )
            session.add(project)
            await session.flush()

            architect = Agent(
                name=SEED_NAMES["architect"],
                role="architect",
                provider="openai",
                model="gpt-4o-mini",
                adapter_type="api",
                description="Seed architect agent",
                capabilities=["design", "review"],
                config={"memory_enabled": True, "seed_metadata": {**SEED_MARKER, "kind": "agent"}},
                is_active=True,
            )
            reviewer = Agent(
                name=SEED_NAMES["reviewer"],
                role="reviewer",
                provider="openai",
                model="gpt-4o-mini",
                adapter_type="routine",
                description="Seed reviewer agent",
                capabilities=["review", "testing"],
                config={"memory_enabled": True, "seed_metadata": {**SEED_MARKER, "kind": "agent"}},
                is_active=True,
            )
            inactive_agent = Agent(
                name=SEED_NAMES["inactive_agent"],
                role="observer",
                provider="openai",
                model="gpt-4o-mini",
                adapter_type="api",
                description="Inactive seed agent",
                capabilities=[],
                config={"memory_enabled": False, "seed_metadata": {**SEED_MARKER, "kind": "agent"}},
                is_active=False,
            )
            session.add_all([architect, reviewer, inactive_agent])
            await session.flush()

            tasks = [
                Task(project_id=project.id, title=SEED_NAMES["ready_task"], description="Ready seed task", status="ready", priority=80, assigned_to=architect.id, created_by_user=user.id, metadata_={"seed": True}),
                Task(project_id=project.id, title=SEED_NAMES["backlog_task"], description="Backlog seed task", status="backlog", priority=50, assigned_to=reviewer.id, created_by_user=user.id, metadata_={"seed": True}),
                Task(project_id=project.id, title=SEED_NAMES["blocked_task"], description="Blocked seed task", status="blocked", priority=90, assigned_to=architect.id, created_by_user=user.id, metadata_={"seed": True}),
                Task(project_id=project.id, title=SEED_NAMES["done_task"], description="Completed seed task", status="done", priority=30, assigned_to=reviewer.id, created_by_user=user.id, completed_at=now - timedelta(minutes=3), metadata_={"seed": True}),
                Task(project_id=project.id, title=SEED_NAMES["disposable_task"], description="Disposable seed task", status="backlog", priority=40, assigned_to=architect.id, created_by_user=user.id, metadata_={"seed": True, "disposable": True}),
            ]
            session.add_all(tasks)
            await session.flush()

            sessions = [
                Session(project_id=project.id, task_id=tasks[0].id, agent_id=architect.id, adapter_type="api", status="completed", input_context={"source": "seed"}, output="done", origin="manual", started_at=now - timedelta(minutes=20), ended_at=now - timedelta(minutes=19)),
                Session(project_id=project.id, task_id=tasks[2].id, agent_id=reviewer.id, adapter_type="routine", status="running", input_context={"source": "seed"}, output=None, origin="meeting", started_at=now - timedelta(minutes=5)),
            ]
            session.add_all(sessions)
            await session.flush()

            active_meeting = Meeting(
                project_id=project.id,
                title=SEED_NAMES["active_meeting"],
                meeting_type="decision",
                status="active",
                turn_strategy="round_robin",
                deadlock_strategy="human_intervention",
                participant_agent_ids=[str(architect.id), str(reviewer.id)],
                participant_user_ids=[str(user.id)],
                organizer_agent_id=architect.id,
                max_duration_minutes=30,
                active_started_at=now - timedelta(minutes=10),
                auto_start=False,
            )
            concluded_meeting = Meeting(
                project_id=project.id,
                title=SEED_NAMES["concluded_meeting"],
                meeting_type="review",
                status="concluded",
                turn_strategy="moderated",
                deadlock_strategy="human_intervention",
                participant_agent_ids=[str(architect.id), str(reviewer.id)],
                participant_user_ids=[str(user.id)],
                max_duration_minutes=30,
                summary="Concluded seed meeting",
                concluded_at=now - timedelta(hours=1),
                auto_start=False,
            )
            session.add_all([active_meeting, concluded_meeting])
            await session.flush()

            agenda_active = MeetingAgendaItem(
                meeting_id=active_meeting.id,
                order=1,
                title="Decide UI rollout",
                description="Discuss rollout order",
                question="What ships first?",
                status="active",
                started_at=now - timedelta(minutes=9),
            )
            agenda_done = MeetingAgendaItem(
                meeting_id=concluded_meeting.id,
                order=1,
                title="Review seeded result",
                description="Review the completed workflow",
                question="Was the workflow acceptable?",
                status="done",
                resolved_at=now - timedelta(hours=1),
            )
            session.add_all([agenda_active, agenda_done])
            await session.flush()

            session.add_all(
                [
                    MeetingTurn(meeting_id=active_meeting.id, agenda_item_id=agenda_active.id, turn_number=1, round_number=1, speaker_agent_id=architect.id, content="Keep the suite deterministic.", references=[]),
                    MeetingTurn(meeting_id=active_meeting.id, agenda_item_id=agenda_active.id, turn_number=2, round_number=1, speaker_agent_id=reviewer.id, content="Disposable state keeps destructive flows safe.", references=[]),
                    MeetingTurn(meeting_id=concluded_meeting.id, agenda_item_id=agenda_done.id, turn_number=1, round_number=1, speaker_agent_id=architect.id, content="The seeded review is complete.", references=[]),
                    MeetingDecision(meeting_id=concluded_meeting.id, agenda_item_id=agenda_done.id, title="Use isolated DB", question="How should the suite isolate data?", chosen_option="Use a temporary SQLite database", rationale="It avoids mutating developer data.", alternatives_rejected=["Reuse huddleroom.db"], participants_agreed=[str(architect.id), str(reviewer.id)], dissent=[], decided_by="consensus", confidence=0.95),
                    MeetingActionItem(meeting_id=active_meeting.id, description="Verify live Playwright artifacts", assignee_agent_id=reviewer.id, priority=70, status="open"),
                    MeetingEvent(meeting_id=active_meeting.id, event_type="meeting.turn.added", payload={"seed": True}, actor_agent_id=architect.id),
                    MeetingEvent(meeting_id=active_meeting.id, event_type="meeting.signal.raised", payload={"seed": True}, actor_agent_id=reviewer.id),
                    MeetingEvent(meeting_id=concluded_meeting.id, event_type="meeting.concluded", payload={"seed": True}, actor_agent_id=architect.id),
                    MeetingParticipantSignal(meeting_id=active_meeting.id, agent_id=architect.id, signal_type="ready", message="Architect ready"),
                    MeetingParticipantSignal(meeting_id=active_meeting.id, agent_id=reviewer.id, signal_type="blocked", message="Need reviewer input"),
                ]
            )

            graph_definition = {
                "start_node": "draft",
                "nodes": {
                    "draft": {"edges": [{"name": "submit", "to": "review"}]},
                    "review": {"edges": [{"name": "approve", "to": "done"}]},
                    "done": {"terminal": True},
                },
            }
            active_graph = Graph(
                project_id=project.id,
                name=SEED_NAMES["active_graph"],
                version="1.0.0",
                description="Active seed graph",
                definition=graph_definition,
                triggers=[{"event_type": "task.created"}],
                is_active=True,
            )
            inactive_graph = Graph(
                project_id=project.id,
                name=SEED_NAMES["inactive_graph"],
                version="1.0.0",
                description="Inactive seed graph",
                definition=graph_definition,
                triggers=[{"event_type": "manual"}],
                is_active=False,
            )
            session.add_all([active_graph, inactive_graph])
            await session.flush()

            graph_runs = [
                GraphRun(project_id=project.id, graph_id=active_graph.id, linked_task_id=tasks[0].id, current_node="review", status="active", actor_assignments={"reviewer": str(reviewer.id)}, context={"seed": True}, started_at=now - timedelta(minutes=15), last_stepped_at=now - timedelta(minutes=10)),
                GraphRun(project_id=project.id, graph_id=active_graph.id, linked_task_id=tasks[1].id, current_node="draft", status="paused", actor_assignments={"architect": str(architect.id)}, context={"seed": True}, started_at=now - timedelta(minutes=30), last_stepped_at=now - timedelta(minutes=20)),
            ]
            session.add_all(graph_runs)
            await session.flush()

            session.add_all(
                [
                    GraphRunStep(graph_run_id=graph_runs[0].id, from_node="draft", to_node="review", edge_name="submit", trigger_reason="Seed step", actor_id=architect.id, actions_executed=["notify"], guard_context={"ok": True}),
                    GraphRunStep(graph_run_id=graph_runs[1].id, from_node="draft", to_node="draft", edge_name="pause", trigger_reason="Seed pause", actor_id=reviewer.id, actions_executed=["pause"], guard_context={"ok": True}),
                    GraphRunTimeout(graph_run_id=graph_runs[0].id, node_name="review", timeout_action="remind", expires_at=now + timedelta(hours=1), resolved=False),
                ]
            )

            session.add_all(
                [
                    KnowledgeItem(project_id=project.id, title=SEED_NAMES["knowledge"], content="Rollout policy: run live UI tests before release.", content_type="markdown", tags=["ui", "playwright"], provenance_type="human", created_by_user=user.id, metadata_={"seed": True}),
                    KnowledgeItem(project_id=project.id, title=SEED_NAMES["disposable_knowledge"], content="Disposable knowledge item for delete coverage.", content_type="text", tags=["ui", "delete"], provenance_type="human", created_by_user=user.id, metadata_={"seed": True, "disposable": True}),
                    MemoryItem(project_id=project.id, agent_id=architect.id, scope="project", content="Remember the UI rollout guardrail", tags=["ui"], shared=True),
                    MemoryItem(project_id=None, agent_id=reviewer.id, scope="global", content="Global reviewer memory for UI tests", tags=["global"], shared=False),
                    RoutingRule(project_id=project.id, name=SEED_NAMES["rule"], description="Route task events to seeded reviewer", priority=10, on_event="task.created", conditions={"status": "ready"}, actions={"assign_to": str(reviewer.id)}, enabled=True),
                    RoutingRule(project_id=project.id, name=SEED_NAMES["rule_disabled"], description="Disabled seed rule", priority=20, on_event="meeting.created", conditions={}, actions={"notify": True}, enabled=False),
                    Hook(project_id=project.id, name=SEED_NAMES["hook"], description="Active seeded hook", trigger_event="task.created", code="return { ok: true }", status="active", execution_count=3, error_count=0),
                    Hook(project_id=project.id, name=SEED_NAMES["hook_disabled"], description="Disabled seeded hook", trigger_event="meeting.created", code="return { disabled: true }", status="disabled", execution_count=0, error_count=0),
                ]
            )
            await session.flush()

            pattern = Pattern(
                project_id=project.id,
                pattern_type="repeated_failure",
                description="Repeated review bottleneck in seeded workflows",
                confidence=0.87,
                sample_size=12,
                context={"source": "seed"},
            )
            session.add(pattern)
            await session.flush()

            session.add_all(
                [
                    Optimization(project_id=project.id, pattern_id=pattern.id, type="hook", generated_code="// UI seed optimization\nreturn { optimized: true }", status="proposed", error_rate=0.01, fire_count=4),
                    Optimization(project_id=project.id, pattern_id=pattern.id, type="rule", generated_code='{"actions": {"assign_to": "reviewer"}}', status="requires_approval", error_rate=0.0, fire_count=2),
                    Optimization(project_id=project.id, pattern_id=pattern.id, type="shortcut", generated_code="shortcut:review-ready", status="active", error_rate=0.02, fire_count=7),
                    CostMetric(project_id=project.id, optimization_id=None, date=date.today(), llm_calls_saved=12, estimated_cost_saved_usd=1.25),
                    CostMetric(project_id=project.id, optimization_id=None, date=date.today() - timedelta(days=1), llm_calls_saved=9, estimated_cost_saved_usd=0.95),
                    CostMetric(project_id=project.id, optimization_id=None, date=date.today() - timedelta(days=2), llm_calls_saved=5, estimated_cost_saved_usd=0.55),
                    EventLog(project_id=project.id, event_type="task.created", dedup_key="ui-seed-task-created", payload={"task": SEED_NAMES["ready_task"]}, source="seed", emitted_at=now - timedelta(minutes=30)),
                    EventLog(project_id=project.id, event_type="meeting.started", dedup_key="ui-seed-meeting-started", payload={"meeting": SEED_NAMES["active_meeting"]}, source="seed", emitted_at=now - timedelta(minutes=20)),
                    EventLog(project_id=project.id, event_type="graph.run_advanced", dedup_key="ui-seed-graph-run-advanced", payload={"graph": SEED_NAMES["active_graph"]}, source="seed", emitted_at=now - timedelta(minutes=10)),
                ]
            )

            await session.commit()

            counts = {
                model.__tablename__: await _count_rows(session, model)
                for model in SEEDED_MODELS
            }
            return {
                "database_url": database_url,
                "project_name": project.name,
                "workspace_dir": str(seed_workspace.resolve()),
                "counts": counts,
            }
    finally:
        await engine.dispose()


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Seed deterministic live UI test data.")
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--workspace-dir", required=True)
    args = parser.parse_args()

    summary = await seed_database(
        database_url=args.database_url,
        workspace_dir=Path(args.workspace_dir),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())
