"""SPR188/SPR232: detail GET presentation across Saved -> Pause -> Resume -> authorize -> applied."""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.dependencies import get_db
from huddleroom.schemas.orchestration import OrchestrationDecisionAnswerRequest
from huddleroom.services import orchestration_decision_application as application
from huddleroom.services.orchestration_service import OrchestrationService
from tests.test_orchestration_decision_application import _spr231_committed_ordinary
from tests.test_orchestration_roster_mutation import sessions  # noqa: F401

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures("safe_goal_analysis")]


async def test_saved_pause_resume_authorize_applied_presentation(sessions, test_engine, client, tmp_path):  # noqa: F811
    project, user, goal, run, _source, decision = await _spr231_committed_ordinary(sessions, tmp_path)
    async with sessions() as db:  # baseline not yet authorized (fixture authorizes by default)
        stored = await db.get(type(run), run.id)
        stored.baseline_authorized = False
        await db.commit()
    async with sessions() as db:
        saved = await application.answer_decision(db, project.id, goal.id, decision.id,
            OrchestrationDecisionAnswerRequest(selected_option="approve", reason="Reviewed"), user)
        action_id = str(saved.answer_receipt.action_id)

    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)

    async def fresh_db():
        async with factory() as s:
            yield s
    client._transport.app.dependency_overrides[get_db] = fresh_db

    async def workflow():
        resp = await client.get(f"/api/v1/projects/{project.id}/orchestration/goals/{goal.id}")
        assert resp.status_code == 200, resp.text
        body = resp.json()["supervision"]
        return body["condition"], body["status_line"]["workflow"]

    async with sessions() as db:
        await OrchestrationService().pause_goal(db, project.id, goal.id)
        await db.commit()
    # "Reload": a fresh HTTP read after pausing.
    condition, wf = await workflow()
    assert condition == "paused"
    assert wf["pending_decision_id"] is None
    assert wf["application"]["state"] == "pending" and wf["application"]["action_id"] == action_id
    assert wf["guidance"]["kind"] == "resume" and wf["guidance"]["label"] == "Resume goal"

    async with sessions() as db:
        await OrchestrationService().resume_goal(db, project.id, goal.id)
        await db.commit()
    condition, wf = await workflow()
    assert condition != "paused"
    assert (wf["guidance"]["kind"], wf["guidance"]["label"]) == ("authorize_baseline", "Authorize remaining baseline")
    assert wf["application"]["state"] == "pending"

    async with sessions() as db:
        await OrchestrationService().authorize_baseline(db, project.id, goal.id, actor=f"human:{user.id}")
        await db.commit()
    for _ in range(2):  # a second drain must not re-apply (application-once)
        async with sessions() as db:
            await application.drain_answer_applications(db, run.id)
            await db.commit()
    _, wf = await workflow()
    assert wf["application"] is None and wf["additional_application_count"] == 0
    assert wf["guidance"]["kind"] != "authorize_baseline"
