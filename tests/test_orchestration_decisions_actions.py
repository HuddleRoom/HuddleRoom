import asyncio
import os
import subprocess
import sys
import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError


def test_migration_017_creates_orchestration_decisions_actions_schema_contract(tmp_path):
    db_path = tmp_path / "migrated.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "RALLY_DATABASE_URL": db_url},
    )
    assert result.returncode == 0, result.stderr

    engine = create_engine(f"sqlite:///{db_path}")
    with engine.connect() as conn:
        inspector = sa.inspect(conn)
        assert {"orchestration_decisions", "orchestration_actions"} <= set(inspector.get_table_names())

        decision_columns = {column["name"] for column in inspector.get_columns("orchestration_decisions")}
        assert {
            "id",
            "run_id",
            "decision_type",
            "input_snapshot",
            "llm_output",
            "parsed_decision",
            "validator_status",
            "rejection_reason",
            "reason",
            "created_at",
            "updated_at",
        } <= decision_columns

        action_columns = {column["name"] for column in inspector.get_columns("orchestration_actions")}
        assert {
            "id",
            "run_id",
            "decision_id",
            "idempotency_key",
            "action_type",
            "request",
            "target_type",
            "target_id",
            "status",
            "error",
            "created_at",
            "updated_at",
        } <= action_columns

        decision_checks = {check["name"] for check in inspector.get_check_constraints("orchestration_decisions")}
        action_checks = {check["name"] for check in inspector.get_check_constraints("orchestration_actions")}
        assert "ck_orchestration_decisions_validator_status" in decision_checks
        assert "ck_orchestration_actions_status" in action_checks

        decision_indexes = {index["name"] for index in inspector.get_indexes("orchestration_decisions")}
        action_indexes = {index["name"] for index in inspector.get_indexes("orchestration_actions")}
        assert {
            "idx_orch_decisions_run_created",
            "idx_orch_decisions_validator_status",
        } <= decision_indexes
        assert {
            "idx_orch_actions_run_status",
        } <= action_indexes

        unique_action_constraints = {
            constraint["name"]: constraint["column_names"]
            for constraint in inspector.get_unique_constraints("orchestration_actions")
        }
        assert unique_action_constraints["uq_orch_actions_run_idempotency_key"] == [
            "run_id",
            "idempotency_key",
        ]


def test_orchestration_decisions_actions_model_timestamps_match_migration_timezone_contract():
    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision

    assert OrchestrationDecision.__table__.c.created_at.type.timezone is True
    assert OrchestrationDecision.__table__.c.updated_at.type.timezone is True
    assert OrchestrationAction.__table__.c.created_at.type.timezone is True
    assert OrchestrationAction.__table__.c.updated_at.type.timezone is True


@pytest.mark.asyncio
async def test_decision_and_action_status_constraints(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Ship an orchestration decision",
            success_criteria=[{"key": "ledger", "description": "Decision/action ledger exists."}],
        ),
        created_by_user_id=None,
    )

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                OrchestrationDecision(
                    run_id=run.id,
                    decision_type="coordination",
                    input_snapshot={},
                    llm_output=None,
                    parsed_decision={},
                    validator_status="not-real",
                )
            )
            await db_session.flush()

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            db_session.add(
                OrchestrationAction(
                    run_id=run.id,
                    idempotency_key="run:test:invalid-status",
                    action_type="noop",
                    request={},
                    status="not-real",
                )
            )
            await db_session.flush()


@pytest.mark.asyncio
async def test_action_reservation_is_idempotent_and_preserves_failed_error(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Replay guard",
            success_criteria=[{"key": "once", "description": "Same action key reserves once."}],
        ),
        created_by_user_id=None,
    )
    decision = OrchestrationDecision(
        run_id=run.id,
        decision_type="coordination",
        input_snapshot={"open_work": []},
        llm_output={"action_type": "noop"},
        parsed_decision={"action_type": "noop"},
        validator_status="accepted",
        reason="No work can advance yet.",
    )
    db_session.add(decision)
    await db_session.flush()

    first = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:abc:kind:noop:input:empty",
        action_type="noop",
        request={"action_type": "noop", "reason": "nothing ready"},
        decision_id=decision.id,
    )
    await service.mark_action_failed(db_session, first, "executor unavailable")

    second = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:abc:kind:noop:input:empty",
        action_type="noop",
        request={"action_type": "noop", "reason": "new request ignored"},
        decision_id=decision.id,
    )

    assert second.id == first.id
    assert second.status == "failed"
    assert second.error == "executor unavailable"
    assert second.request == {"action_type": "noop", "reason": "nothing ready"}
    assert second.decision_id == decision.id

    result = await db_session.execute(select(func.count()).select_from(OrchestrationAction))
    assert result.scalar_one() == 1


@pytest.mark.asyncio
async def test_action_reservation_replays_existing_action_after_run_completes(db_session, test_project):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Replay after completion",
            success_criteria=[{"key": "replay", "description": "Existing reservations replay after run completes."}],
        ),
        created_by_user_id=None,
    )
    first = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:terminal-replay:kind:noop:input:empty",
        action_type="noop",
        request={"action_type": "noop", "reason": "first"},
    )
    run.status = "completed"
    await db_session.flush()

    replayed = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:terminal-replay:kind:noop:input:empty",
        action_type="noop",
        request={"action_type": "noop", "reason": "ignored"},
    )

    assert replayed.id == first.id
    assert replayed.request == {"action_type": "noop", "reason": "first"}


@pytest.mark.asyncio
async def test_decision_updated_at_changes_when_validator_status_changes(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationDecision
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Decision audit timestamps",
            success_criteria=[{"key": "updated", "description": "Decision updates are timestamped."}],
        ),
        created_by_user_id=None,
    )
    decision = OrchestrationDecision(
        run_id=run.id,
        decision_type="coordination",
        input_snapshot={"open_work": []},
        llm_output={"action_type": "noop"},
        parsed_decision={"action_type": "noop"},
    )
    db_session.add(decision)
    await db_session.flush()
    created_at = decision.created_at
    first_updated_at = decision.updated_at

    await asyncio.sleep(0.001)
    decision.validator_status = "accepted"
    await db_session.flush()

    assert decision.created_at == created_at
    assert decision.updated_at > first_updated_at


@pytest.mark.asyncio
async def test_action_reservation_returns_existing_after_duplicate_race(db_session, test_project):
    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationDecision
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    class StaleSelectResult:
        def scalar_one_or_none(self):
            return None

    class StaleFirstSelectSession:
        def __init__(self, session):
            self.session = session
            self.stale_once = True

        def __getattr__(self, name):
            return getattr(self.session, name)

        async def execute(self, *args, **kwargs):
            statement = str(args[0])
            if self.stale_once and "orchestration_actions" in statement:
                self.stale_once = False
                return StaleSelectResult()
            return await self.session.execute(*args, **kwargs)

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Replay race guard",
            success_criteria=[{"key": "once", "description": "Concurrent reservation returns existing action."}],
        ),
        created_by_user_id=None,
    )
    existing = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:race:kind:noop:input:empty",
        action_type="noop",
        request={"action_type": "noop", "reason": "first"},
    )

    raced = await service.reserve_action(
        StaleFirstSelectSession(db_session),
        run_id=run.id,
        idempotency_key="run:race:kind:noop:input:empty",
        action_type="noop",
        request={"action_type": "noop", "reason": "raced"},
    )

    assert raced.id == existing.id
    assert raced.request == {"action_type": "noop", "reason": "first"}

    result = await db_session.execute(select(func.count()).select_from(OrchestrationAction))
    assert result.scalar_one() == 1

    db_session.add(
        OrchestrationDecision(
            run_id=run.id,
            decision_type="sentinel",
            input_snapshot={},
            llm_output=None,
            parsed_decision={},
        )
    )
    await db_session.flush()


@pytest.mark.asyncio
async def test_action_reservation_rejects_raced_replay_with_mismatched_action_type(db_session, test_project):
    from fastapi import HTTPException

    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    class StaleSelectResult:
        def scalar_one_or_none(self):
            return None

    class StaleFirstSelectSession:
        def __init__(self, session):
            self.session = session
            self.stale_once = True

        def __getattr__(self, name):
            return getattr(self.session, name)

        async def execute(self, *args, **kwargs):
            statement = str(args[0])
            if self.stale_once and "orchestration_actions" in statement:
                self.stale_once = False
                return StaleSelectResult()
            return await self.session.execute(*args, **kwargs)

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Replay race mismatch guard",
            success_criteria=[{"key": "conflict", "description": "Raced mismatched replays are rejected."}],
        ),
        created_by_user_id=None,
    )
    await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:race-mismatch:kind:noop:input:empty",
        action_type="noop",
        request={"action_type": "noop"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.reserve_action(
            StaleFirstSelectSession(db_session),
            run_id=run.id,
            idempotency_key="run:race-mismatch:kind:noop:input:empty",
            action_type="different-noop",
            request={"action_type": "different-noop"},
        )

    assert exc_info.value.status_code == 409
    result = await db_session.execute(select(func.count()).select_from(OrchestrationAction))
    assert result.scalar_one() == 1


@pytest.mark.asyncio
async def test_action_reservation_requires_active_run(db_session, test_project):
    from fastapi import HTTPException

    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Inactive run guard",
            success_criteria=[{"key": "inactive", "description": "Inactive runs reject reservations."}],
        ),
        created_by_user_id=None,
    )
    run.status = "completed"
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.reserve_action(
            db_session,
            run_id=run.id,
            idempotency_key="run:completed:kind:noop:input:empty",
            action_type="noop",
            request={"action_type": "noop"},
        )

    assert exc_info.value.status_code == 409
    result = await db_session.execute(select(func.count()).select_from(OrchestrationAction))
    assert result.scalar_one() == 0


@pytest.mark.asyncio
async def test_action_reservation_rejects_stale_inactive_run(db_session, test_project):
    from fastapi import HTTPException

    from huddleroom.models.orchestration import OrchestrationAction, OrchestrationRun
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Stale inactive run guard",
            success_criteria=[{"key": "stale", "description": "Stale workers cannot reserve actions."}],
        ),
        created_by_user_id=None,
    )
    await db_session.execute(
        sa.update(OrchestrationRun)
        .where(OrchestrationRun.id == run.id)
        .values(status="completed")
        .execution_options(synchronize_session=False)
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.reserve_action(
            db_session,
            run_id=run.id,
            idempotency_key="run:stale-completed:kind:noop:input:empty",
            action_type="noop",
            request={"action_type": "noop"},
        )

    assert exc_info.value.status_code == 409
    assert run.status == "running"
    result = await db_session.execute(select(func.count()).select_from(OrchestrationAction))
    assert result.scalar_one() == 0


@pytest.mark.asyncio
async def test_action_reservation_rejects_missing_run(db_session):
    from fastapi import HTTPException

    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()

    with pytest.raises(HTTPException) as exc_info:
        await service.reserve_action(
            db_session,
            run_id=uuid.uuid4(),
            idempotency_key="run:missing:kind:noop:input:empty",
            action_type="noop",
            request={"action_type": "noop"},
        )

    assert exc_info.value.status_code == 404
    result = await db_session.execute(select(func.count()).select_from(OrchestrationAction))
    assert result.scalar_one() == 0


@pytest.mark.asyncio
async def test_action_reservation_rejects_unknown_decision(db_session, test_project):
    from fastapi import HTTPException

    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Decision guard",
            success_criteria=[{"key": "decision", "description": "Unknown decisions reject reservations."}],
        ),
        created_by_user_id=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.reserve_action(
            db_session,
            run_id=run.id,
            idempotency_key="run:decision:kind:noop:input:empty",
            action_type="noop",
            request={"action_type": "noop"},
            decision_id=uuid.uuid4(),
        )

    assert exc_info.value.status_code == 404
    result = await db_session.execute(select(func.count()).select_from(OrchestrationAction))
    assert result.scalar_one() == 0


@pytest.mark.parametrize(
    ("idempotency_key", "objective"),
    [
        ("x" * 256, "Key length guard"),
        ("", "Key presence guard"),
    ],
    ids=["long", "empty"],
)
@pytest.mark.asyncio
async def test_action_reservation_rejects_invalid_idempotency_key(db_session, test_project, idempotency_key, objective):
    from fastapi import HTTPException

    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective=objective,
            success_criteria=[{"key": "key", "description": "Action key validation."}],
        ),
        created_by_user_id=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.reserve_action(
            db_session,
            run_id=run.id,
            idempotency_key=idempotency_key,
            action_type="noop",
            request={"action_type": "noop"},
        )

    assert exc_info.value.status_code == 400
    result = await db_session.execute(select(func.count()).select_from(OrchestrationAction))
    assert result.scalar_one() == 0


@pytest.mark.parametrize(
    ("action_type", "objective"),
    [
        ("x" * 101, "Action type length guard"),
        ("", "Action type presence guard"),
    ],
    ids=["long", "empty"],
)
@pytest.mark.asyncio
async def test_action_reservation_rejects_invalid_action_type(db_session, test_project, action_type, objective):
    from fastapi import HTTPException

    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective=objective,
            success_criteria=[{"key": "type", "description": "Action type validation."}],
        ),
        created_by_user_id=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.reserve_action(
            db_session,
            run_id=run.id,
            idempotency_key="run:action-type:kind:noop:input:empty",
            action_type=action_type,
            request={"action_type": "noop"},
        )

    assert exc_info.value.status_code == 400
    result = await db_session.execute(select(func.count()).select_from(OrchestrationAction))
    assert result.scalar_one() == 0


@pytest.mark.asyncio
async def test_action_reservation_rejects_replay_with_mismatched_action_type(db_session, test_project):
    from fastapi import HTTPException

    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Replay mismatch guard",
            success_criteria=[{"key": "conflict", "description": "Mismatched action replays are rejected."}],
        ),
        created_by_user_id=None,
    )
    await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:replay-mismatch:kind:noop:input:empty",
        action_type="noop",
        request={"action_type": "noop"},
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.reserve_action(
            db_session,
            run_id=run.id,
            idempotency_key="run:replay-mismatch:kind:noop:input:empty",
            action_type="different-noop",
            request={"action_type": "different-noop"},
        )

    assert exc_info.value.status_code == 409
    result = await db_session.execute(select(func.count()).select_from(OrchestrationAction))
    assert result.scalar_one() == 1


@pytest.mark.asyncio
async def test_mark_action_failed_does_not_overwrite_completed_action(db_session, test_project):
    from fastapi import HTTPException

    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Completed action guard",
            success_criteria=[{"key": "terminal", "description": "Completed actions stay completed."}],
        ),
        created_by_user_id=None,
    )
    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:completed:kind:noop:input:empty",
        action_type="noop",
        request={"action_type": "noop"},
    )
    action.status = "completed"
    await db_session.flush()

    with pytest.raises(HTTPException) as exc_info:
        await service.mark_action_failed(db_session, action, "late failure")

    assert exc_info.value.status_code == 409
    await db_session.refresh(action)
    assert action.status == "completed"
    assert action.error is None


@pytest.mark.asyncio
async def test_mark_action_failed_does_not_overwrite_stale_completed_action(db_session, test_project):
    from fastapi import HTTPException

    from huddleroom.models.orchestration import OrchestrationAction
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Stale completed action guard",
            success_criteria=[{"key": "terminal", "description": "Stale workers cannot fail completed actions."}],
        ),
        created_by_user_id=None,
    )
    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:stale-completed:kind:noop:input:empty",
        action_type="noop",
        request={"action_type": "noop"},
    )
    await db_session.execute(
        sa.update(OrchestrationAction)
        .where(OrchestrationAction.id == action.id)
        .values(status="completed")
        .execution_options(synchronize_session=False)
    )

    with pytest.raises(HTTPException) as exc_info:
        await service.mark_action_failed(db_session, action, "late failure")

    assert exc_info.value.status_code == 409
    assert action.status == "completed"
    assert action.error is None


@pytest.mark.asyncio
async def test_action_reservation_deep_copies_request(db_session, test_project):
    from huddleroom.schemas.orchestration import OrchestrationGoalCreate
    from huddleroom.services.orchestration_service import OrchestrationService

    service = OrchestrationService()
    _, run = await service.create_goal(
        db_session,
        project_id=test_project.id,
        data=OrchestrationGoalCreate(
            objective="Copy request",
            success_criteria=[{"key": "copy", "description": "Action request is immutable from caller edits."}],
        ),
        created_by_user_id=None,
    )
    request = {"payload": {"count": 1}}

    action = await service.reserve_action(
        db_session,
        run_id=run.id,
        idempotency_key="run:copy:kind:noop:input:one",
        action_type="noop",
        request=request,
    )
    request["payload"]["count"] = 2

    assert action.request == {"payload": {"count": 1}}
