"""Meeting turns must not hold a DB write transaction across slow external calls,
and turn numbering / organizer fallback must be consistent."""
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from huddleroom.models.agent import Agent
from huddleroom.models.project import Project
from huddleroom.models.meeting import Meeting, MeetingAgendaItem, MeetingTurn
from huddleroom.services.meeting_runner import MeetingRunner


@pytest.fixture
def db_session(autobegin_db_session):
    """Override: these tests drive runner code that commits mid-turn."""
    return autobegin_db_session



async def _make_meeting(db, project, agents, **kw):
    meeting = Meeting(
        project_id=project.id,
        title="Lock test",
        meeting_type="decision",
        participant_agent_ids=[str(a.id) for a in agents],
        status="active",
        turn_strategy=kw.pop("turn_strategy", "round_robin"),
        **kw,
    )
    db.add(meeting)
    await db.flush()
    item = MeetingAgendaItem(
        meeting_id=meeting.id, order=1, title="Item", status="active", max_rounds=2
    )
    db.add(item)
    await db.flush()
    return meeting, item


def _agent(**kw):
    return Agent(
        name=f"a-{uuid.uuid4()}", role=kw.pop("role", "dev"), provider="openai",
        model="gpt-4o-mini", adapter_type=kw.pop("adapter_type", "api"),
        capabilities=[], config={}, **kw,
    )


@pytest.mark.asyncio
async def test_next_turn_number_is_max_based(db_session, test_project, test_agent):
    meeting, item = await _make_meeting(db_session, test_project, [test_agent])
    for n in (1, 5):
        db_session.add(MeetingTurn(
            meeting_id=meeting.id, agenda_item_id=item.id, turn_number=n, round_number=1,
            speaker_agent_id=test_agent.id, content="x", is_human_turn=False,
        ))
    await db_session.flush()
    assert await MeetingRunner()._next_turn_number(db_session, meeting.id) == 6
    assert await MeetingRunner()._next_turn_number(db_session, uuid.uuid4()) == 1


@pytest.mark.asyncio
async def test_human_turn_round_and_number(client, auth_headers, runnable_project, test_agent, db_session):
    meeting, item = await _make_meeting(db_session, runnable_project, [test_agent])
    item.current_round = 2
    db_session.add(MeetingTurn(
        meeting_id=meeting.id, agenda_item_id=item.id, turn_number=7, round_number=1,
        speaker_agent_id=test_agent.id, content="x", is_human_turn=False,
    ))
    await db_session.flush()
    resp = await client.post(
        f"/api/v1/meetings/{meeting.id}/human-turn", json={"content": "hi"}, headers=auth_headers
    )
    assert resp.status_code == 201
    assert resp.json()["turn_number"] == 8
    assert resp.json()["round_number"] == 3


@pytest.mark.asyncio
async def test_human_turn_insert_succeeds_during_cli_call(test_engine):
    """A second session can write while the agent turn is inside the external CLI call."""
    factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    cli_agent = _agent(adapter_type="cli")
    async with factory() as db:
        project = Project(name="Lock Project", description="d", config={})
        db.add_all([cli_agent, project])
        await db.flush()
        meeting, item = await _make_meeting(db, project, [cli_agent])
        await db.commit()

        written: dict = {}

        async def slow_cli(**_kw):
            async with factory() as other:
                other.add(MeetingTurn(
                    meeting_id=meeting.id, agenda_item_id=item.id, turn_number=1,
                    round_number=1, speaker_user_id=None, content="human", is_human_turn=True,
                ))
                await other.commit()
            written["ok"] = True
            return "agent reply", None, 5

        async def prompt_with_write(**_kw):
            meeting.title = "dirty"  # takes the write lock, like the trace event does in production
            await db.flush()
            return "p", None

        runner = MeetingRunner()
        with patch.object(runner._ctx, "build_cli_turn_prompt", side_effect=prompt_with_write), \
             patch("huddleroom.adapters.cli_adapter.CliAdapter.run_meeting_turn", side_effect=slow_cli):
            turn = await runner._execute_cli_agent_turn(db=db, meeting=meeting, agent=cli_agent)
        assert written.get("ok")
        # Recomputed after the call, so no duplicate with the concurrent human turn.
        assert turn.turn_number == 2


@pytest.mark.asyncio
async def test_commit_before_api_llm_call(db_session, test_project):
    agent = _agent()
    db_session.add(agent)
    await db_session.flush()
    meeting, _ = await _make_meeting(db_session, test_project, [agent])
    runner = MeetingRunner()
    commits: list[int] = []
    real_commit = db_session.commit

    async def tracking_commit():
        commits.append(1)
        await real_commit()

    async def fake_loop(**_kw):
        assert commits, "write transaction must be committed before the LLM call"
        return "reply"

    with patch.object(db_session, "commit", side_effect=tracking_commit), \
         patch("huddleroom.services.meeting_runner.run_tool_loop", side_effect=fake_loop), \
         patch.object(runner._ctx, "build_turn_prompt", new_callable=AsyncMock, return_value=[{"role": "user", "content": "q"}]):
        turn = await runner.execute_agent_turn(db=db_session, meeting=meeting, agent=agent)
    assert turn is not None


@pytest.mark.asyncio
async def test_organizer_prompt_lists_only_eligible(db_session, test_project):
    org, a, b = _agent(), _agent(), _agent()
    db_session.add_all([org, a, b])
    await db_session.flush()
    meeting, _ = await _make_meeting(
        db_session, test_project, [org, a, b],
        turn_strategy="organizer_controlled", organizer_agent_id=org.id,
    )
    runner = MeetingRunner()
    seen: dict = {}

    async def fake_repair(_fn, kwargs, *_a, **_k):
        seen["messages"] = kwargs["messages"]
        return {"next_speaker_id": str(a.id), "reason": "r"}

    with patch.object(runner._ctx, "format_transcript", new_callable=AsyncMock, return_value=""), \
         patch.object(runner._svc, "get_pending_signals", new_callable=AsyncMock, return_value=[]), \
         patch("huddleroom.services.meeting_runner.complete_with_repair", side_effect=fake_repair):
        await runner._organizer_agent_select(db_session, meeting, org, [str(a.id)])
    user_msg = seen["messages"][1]["content"]
    assert str(a.id) in user_msg
    assert str(b.id) not in user_msg and str(org.id) not in user_msg


@pytest.mark.asyncio
async def test_ineligible_organizer_pick_is_marked_overridden(db_session, test_project):
    org, a, b = _agent(), _agent(), _agent()
    db_session.add_all([org, a, b])
    await db_session.flush()
    meeting, item = await _make_meeting(
        db_session, test_project, [org, a, b],
        turn_strategy="organizer_controlled", organizer_agent_id=org.id,
    )
    runner = MeetingRunner()
    picked = {"next_speaker_id": str(a.id), "reason": "a is great", "messages": []}
    spoke = {str(org.id): True, str(a.id): True, str(b.id): False}
    with patch.object(runner, "_spoke_this_round", new_callable=AsyncMock, return_value=spoke), \
         patch.object(runner, "_organizer_agent_select", new_callable=AsyncMock, return_value=picked) as sel:
        speaker, selection = await runner._determine_next_speaker(db_session, meeting, item)
    assert speaker == str(b.id)
    assert sel.await_args.args[3] == [str(b.id)]
    assert selection["overridden"] is True
    assert selection["requested_speaker_id"] == str(a.id)
    assert selection["next_speaker_id"] == str(b.id)
    assert "not eligible" in selection["reason"]
