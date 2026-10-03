from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.database import AsyncSessionLocal
from huddleroom.models.event_log import EventLog
from huddleroom.models.meeting import Meeting, MeetingTurn
from huddleroom.services.event_bus import BusEvent, get_event_bus
from huddleroom.workers.consumers.ws_hub import get_registry

logger = logging.getLogger(__name__)

router = APIRouter()


async def _auth_ws_token(token: str | None, db: AsyncSession) -> bool:
    """Returns True if auth is disabled or token is valid."""
    if not settings.auth_enabled:
        return True
    if not token:
        return False
    # Try JWT
    try:
        from huddleroom.security import decode_access_token
        from huddleroom.models.user import User
        payload = decode_access_token(token)
        user_id_str = payload.get("sub")
        if user_id_str:
            result = await db.execute(select(User).where(User.id == uuid.UUID(user_id_str)))
            user = result.scalar_one_or_none()
            if user and user.is_active:
                return True
    except Exception as exc:
        logger.warning("WS auth check error: %s", exc)
    # Try API key
    try:
        from huddleroom.security import hash_api_key
        from huddleroom.models.api_key import ApiKey
        hashed = hash_api_key(token)
        result = await db.execute(select(ApiKey).where(ApiKey.hashed_key == hashed))
        api_key = result.scalar_one_or_none()
        if api_key:
            from datetime import timezone
            if not api_key.expires_at or api_key.expires_at > datetime.now(timezone.utc):
                return True
    except Exception as exc:
        logger.warning("WS auth check error: %s", exc)
    return False
async def _load_meeting_turn(
    db: AsyncSession,
    meeting_id: uuid.UUID,
    payload: dict,
) -> MeetingTurn | None:
    turn_id = payload.get("turn_id")
    if turn_id:
        try:
            turn = await db.get(MeetingTurn, uuid.UUID(str(turn_id)))
            if turn and turn.meeting_id == meeting_id:
                return turn
        except (ValueError, TypeError):
            logger.warning("Invalid turn_id in meeting websocket payload: %s", turn_id)

    turn_number = payload.get("turn_number")
    if turn_number is None:
        return None

    result = await db.execute(
        select(MeetingTurn)
        .where(
            MeetingTurn.meeting_id == meeting_id,
            MeetingTurn.turn_number == turn_number,
        )
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _build_meeting_message(
    db: AsyncSession,
    meeting: Meeting,
    event_type: str,
    payload: dict,
    source: str,
    emitted_at: datetime,
    event_id: uuid.UUID | None = None,
) -> dict | None:
    payload_meeting_id = payload.get("meeting_id")
    if str(payload_meeting_id) != str(meeting.id):
        return None

    message = {
        "project_id": str(meeting.project_id),
        "meeting_id": str(meeting.id),
        "event_type": event_type,
        "payload": payload,
        "source": source,
        "emitted_at": emitted_at.isoformat(),
    }
    if event_id is not None:
        message["id"] = str(event_id)

    if event_type in {"meeting.turn_complete", "meeting.human_turn"}:
        turn = await _load_meeting_turn(db=db, meeting_id=meeting.id, payload=payload)
        if turn is not None:
            message["turn"] = {
                "id": str(turn.id),
                "meeting_id": str(turn.meeting_id),
                "agenda_item_id": str(turn.agenda_item_id) if turn.agenda_item_id else None,
                "turn_number": turn.turn_number,
                "round_number": turn.round_number,
                "speaker_agent_id": str(turn.speaker_agent_id) if turn.speaker_agent_id else None,
                "speaker_user_id": str(turn.speaker_user_id) if turn.speaker_user_id else None,
                "content": turn.content,
                "references": turn.references,
                "is_human_turn": turn.is_human_turn,
                "is_override": turn.is_override,
                "moderator_note": turn.moderator_note,
                "token_count": turn.token_count,
                "model_used": turn.model_used,
                "latency_ms": turn.latency_ms,
                "prompt_messages": turn.prompt_messages,
                "raw_response": turn.raw_response,
                "organizer_selection": turn.organizer_selection,
                "reasoning_content": turn.reasoning_content,
                "created_at": turn.created_at.isoformat(),
            }

    return message


@router.websocket("/ws/projects/{project_id}/events")
async def ws_project_events(
    project_id: uuid.UUID,
    websocket: WebSocket,
    token: str | None = Query(default=None),
    event_types: str | None = Query(default=None),
    replay_since: str | None = Query(default=None),
) -> None:
    await websocket.accept()
    async with AsyncSessionLocal() as db:
        if not await _auth_ws_token(token, db):
            await websocket.close(code=4001)
            return
        conn_id = str(uuid.uuid4())
        type_filter = set(t.strip() for t in event_types.split(",")) if event_types else None

        # Replay missed events
        if replay_since:
            try:
                since_dt = datetime.fromisoformat(replay_since.replace(" ", "+"))
                result = await db.execute(
                    select(EventLog)
                    .where(EventLog.project_id == project_id, EventLog.emitted_at >= since_dt)
                    .order_by(EventLog.emitted_at)
                )
                for entry in result.scalars():
                    if type_filter and entry.event_type not in type_filter:
                        continue
                    await websocket.send_text(json.dumps({
                        "id": str(entry.id),
                        "project_id": str(entry.project_id),
                        "event_type": entry.event_type,
                        "payload": entry.payload,
                        "source": entry.source,
                        "emitted_at": entry.emitted_at.isoformat(),
                    }))
            except Exception as exc:
                logger.warning("WS replay failed: %s", exc)

    registry = get_registry()
    registry.add(project_id, conn_id, websocket, type_filter)

    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=8)
            except asyncio.TimeoutError:
                if not await registry.enqueue(
                    project_id, conn_id, json.dumps({"type": "ping"}), evictable=False
                ):
                    break
            except WebSocketDisconnect:
                break
            except Exception:
                break
    finally:
        await registry.remove(project_id, conn_id)


@router.websocket("/ws/meetings/{meeting_id}")
async def ws_meeting_events(
    meeting_id: uuid.UUID,
    websocket: WebSocket,
    token: str | None = Query(default=None),
    replay_since: str | None = Query(default=None),
) -> None:
    await websocket.accept()

    async with AsyncSessionLocal() as db:
        if not await _auth_ws_token(token, db):
            await websocket.close(code=4001)
            return

        meeting = await db.get(Meeting, meeting_id)
        if not meeting:
            await websocket.close(code=4404)
            return

        if replay_since:
            try:
                since_dt = datetime.fromisoformat(replay_since.replace(" ", "+"))
                result = await db.execute(
                    select(EventLog)
                    .where(
                        EventLog.project_id == meeting.project_id,
                        EventLog.emitted_at >= since_dt,
                    )
                    .order_by(EventLog.emitted_at)
                )
                for entry in result.scalars():
                    message = await _build_meeting_message(
                        db=db,
                        meeting=meeting,
                        event_type=entry.event_type,
                        payload=entry.payload,
                        source=entry.source,
                        emitted_at=entry.emitted_at,
                        event_id=entry.id,
                    )
                    if message is not None:
                        await websocket.send_text(json.dumps(message))
            except Exception as exc:
                logger.warning("Meeting WS replay failed: %s", exc)

        bus = get_event_bus()
        subscription = bus.subscribe(project_id=meeting.project_id)
        event_iter = subscription.__aiter__()

        try:
            pending_event_task = asyncio.create_task(anext(event_iter))
            pending_receive_task = asyncio.create_task(websocket.receive_text())
            while True:
                done, _ = await asyncio.wait(
                    {pending_event_task, pending_receive_task},
                    timeout=8,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    try:
                        await websocket.send_text(json.dumps({"type": "ping"}))
                    except Exception:
                        break
                    continue

                if pending_receive_task in done:
                    try:
                        pending_receive_task.result()
                    except WebSocketDisconnect:
                        break
                    except Exception:
                        break
                    pending_receive_task = asyncio.create_task(websocket.receive_text())

                if pending_event_task in done:
                    try:
                        event = pending_event_task.result()
                    except StopAsyncIteration:
                        break
                    except Exception as exc:
                        logger.warning("Meeting WS stream error: %s", exc)
                        break

                    message = await _build_meeting_message(
                        db=db,
                        meeting=meeting,
                        event_type=event.event_type,
                        payload=event.payload,
                        source=event.source,
                        emitted_at=event.emitted_at,
                        event_id=event.id,
                    )
                    if message is not None:
                        try:
                            await websocket.send_text(json.dumps(message))
                        except WebSocketDisconnect:
                            break
                        except Exception as exc:
                            logger.warning("Meeting WS send error: %s", exc)
                            break

                    pending_event_task = asyncio.create_task(anext(event_iter))
        finally:
            try:
                pending_event_task.cancel()
                pending_receive_task.cancel()
                await asyncio.gather(pending_event_task, pending_receive_task, return_exceptions=True)
            except Exception:
                pass
            await subscription.aclose()
