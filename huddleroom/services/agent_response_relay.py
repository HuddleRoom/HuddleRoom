"""Ephemeral agent-response and project-reset transport."""

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from weakref import WeakKeyDictionary
from uuid import UUID, uuid4

from huddleroom.config import settings
from huddleroom.services.agent_response_stream import AgentResponseEvent

logger = logging.getLogger(__name__)

_RESPONSE_CHANNEL = "rally:agent_response_events"
_METRICS_KEY = "rally:agent_response_metrics"
_RESET_CHANNEL = "rally:project_reset_events"
_redis_clients: WeakKeyDictionary = WeakKeyDictionary()
_hub_pubsubs: WeakKeyDictionary = WeakKeyDictionary()
_subscribers: list[Callable[[AgentResponseEvent], Awaitable[None] | None]] = []
_agent_response_metrics = {"started_total": 0, "terminal_total": 0}
_reset_markers: dict[UUID, str] = {}
_reset_waiters: dict[UUID, set[asyncio.Future[str]]] = {}
_reset_monitor_acks: dict[UUID, set[asyncio.Future[None]]] = {}
_reset_targets: dict[tuple[UUID, str], tuple[asyncio.Future[None], ...]] = {}
_RESET_LEASE_SECONDS = 60
_RESET_RENEW_SECONDS = 20
_RESET_GENERATION_TTL_SECONDS = 300


@dataclass(frozen=True)
class ResetRegistration:
    token: asyncio.Future[None] | str | None
    generation: str | None = None

    def done(self) -> bool:
        return bool(isinstance(self.token, asyncio.Future) and self.token.done())


ResetMonitor = ResetRegistration


def _redis_client():
    loop = asyncio.get_running_loop()
    client = _redis_clients.get(loop)
    if client is None:
        from redis.asyncio import Redis  # pylint: disable=import-error

        client = Redis.from_url(settings.redis_url or "redis://localhost:6379/0", decode_responses=True)
        _redis_clients[loop] = client
    return client


def _metric_field(event: AgentResponseEvent) -> str | None:
    if event.event_type == "agent_response.started":
        return "started_total"
    # A reset fence emits only terminal sequence 0; it never opened a counted call.
    if event.event_type == "agent_response.terminal" and event.sequence > 0:
        return "terminal_total"
    return None


def subscribe_agent_responses(
    callback: Callable[[AgentResponseEvent], Awaitable[None] | None],
) -> Callable[[], None]:
    _subscribers.append(callback)

    def unsubscribe() -> None:
        if callback in _subscribers:
            _subscribers.remove(callback)

    return unsubscribe


async def _notify_subscribers(event: AgentResponseEvent) -> None:
    for callback in tuple(_subscribers):
        try:
            result = callback(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.warning("Agent response subscriber failed", exc_info=True)


async def publish_agent_response(event: AgentResponseEvent) -> None:
    field = _metric_field(event)
    if settings.is_sqlite:
        await _notify_subscribers(event)
        if field:
            _agent_response_metrics[field] += 1
        return
    transaction = _redis_client().pipeline(transaction=True)
    transaction.publish(_RESPONSE_CHANNEL, event.model_dump_json())
    if field:
        transaction.hincrby(_METRICS_KEY, field, 1)
    await transaction.execute()


async def get_agent_response_metrics() -> dict[str, int | float]:
    if settings.is_sqlite:
        values = _agent_response_metrics.copy()
    else:
        raw = await _redis_client().hgetall(_METRICS_KEY)
        values = {key: int(raw.get(key, 0)) for key in _agent_response_metrics}
    started, terminal = values["started_total"], values["terminal_total"]
    return {
        **values,
        "omission_gap": max(started - terminal, 0),
        "completion_rate": 1.0 if not started else terminal / started,
    }


async def run_agent_response_hub(ready: asyncio.Event | None = None) -> None:
    if settings.is_sqlite:
        if ready is not None:
            ready.set()
        await asyncio.Future()
    pubsub = _redis_client().pubsub()
    _hub_pubsubs[asyncio.get_running_loop()] = pubsub
    try:
        await pubsub.subscribe(_RESPONSE_CHANNEL)
        if ready is not None:
            ready.set()
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            try:
                event = AgentResponseEvent.model_validate_json(message["data"])
            except Exception:
                logger.warning("Malformed agent response message")
                continue
            await _notify_subscribers(event)
    finally:
        await pubsub.unsubscribe(_RESPONSE_CHANNEL)
        await pubsub.aclose()
        _hub_pubsubs.pop(asyncio.get_running_loop(), None)


async def close_agent_response_relay() -> None:
    from huddleroom.services.agent_response_stream import close_agent_response_monitors

    loop = asyncio.get_running_loop()
    failure = None
    try:
        await close_agent_response_monitors()
    except BaseException as exc:  # cleanup must not strand loop-owned resources
        failure = exc
    # Calls may lazily create this loop's client while unregistering.
    for resource in (_hub_pubsubs.pop(loop, None), _redis_clients.pop(loop, None)):
        if resource is None:
            continue
        try:
            await resource.aclose()
        except BaseException as exc:
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure


async def publish_project_reset(project_id: UUID) -> str:
    generation = str(uuid4())
    if settings.is_sqlite:
        if previous := _reset_markers.get(project_id):
            _reset_targets.pop((project_id, previous), None)
        _reset_markers[project_id] = generation
        _reset_targets[project_id, generation] = tuple(_reset_monitor_acks.get(project_id, ()))
        for waiter in tuple(_reset_waiters.get(project_id, ())):
            if not waiter.done():
                waiter.set_result(generation)
        return generation
    client = _redis_client()
    await client.eval(
        "local previous=redis.call('GET', KEYS[1]); if previous then redis.call('DEL', ARGV[4]..previous, ARGV[5]..previous) end; "
        "local tokens=redis.call('SMEMBERS', KEYS[2]); local live={}; "
        "for _,token in ipairs(tokens) do if redis.call('EXISTS', ARGV[3]..token) == 1 then table.insert(live, token) else redis.call('SREM', KEYS[2], token) end end; "
        "redis.call('DEL', KEYS[3], KEYS[4]); if #live > 0 then redis.call('SADD', KEYS[3], unpack(live)) end; "
        "redis.call('EXPIRE', KEYS[3], tonumber(ARGV[6])); redis.call('SADD', KEYS[4], '__empty__'); redis.call('SREM', KEYS[4], '__empty__'); redis.call('EXPIRE', KEYS[4], tonumber(ARGV[6])); "
        "redis.call('SET', KEYS[1], ARGV[1]); redis.call('PUBLISH', KEYS[5], ARGV[2]); return #live",
        5,
        f"rally:project_reset:{project_id}",
        f"rally:project_reset_monitors:{project_id}",
        f"rally:project_reset_target:{project_id}:{generation}",
        f"rally:project_reset_acks:{project_id}:{generation}",
        _RESET_CHANNEL,
        generation,
        json.dumps({"project_id": str(project_id), "generation": generation}),
        f"rally:project_reset_monitor_lease:{project_id}:",
        f"rally:project_reset_target:{project_id}:",
        f"rally:project_reset_acks:{project_id}:",
        _RESET_GENERATION_TTL_SECONDS,
    )
    return generation


async def wait_for_project_reset(project_id: UUID) -> str:
    if settings.is_sqlite:
        if generation := _reset_markers.get(project_id):
            return generation
        waiter = asyncio.get_running_loop().create_future()
        waiters = _reset_waiters.setdefault(project_id, set())
        waiters.add(waiter)
        try:
            if generation := _reset_markers.get(project_id):
                return generation
            return await waiter
        finally:
            waiters.discard(waiter)
            if not waiters:
                _reset_waiters.pop(project_id, None)
    client = _redis_client()
    pubsub = client.pubsub()
    await pubsub.subscribe(_RESET_CHANNEL)
    try:
        if generation := await client.get(f"rally:project_reset:{project_id}"):
            return generation
        async for message in pubsub.listen():
            if message.get("type") == "message":
                generation = _parse_reset_message(message.get("data"), project_id)
                if generation and await client.get(f"rally:project_reset:{project_id}") == generation:
                    return generation
    finally:
        await pubsub.unsubscribe(_RESET_CHANNEL)
        await pubsub.aclose()


async def clear_project_reset(project_id: UUID, generation: str) -> None:
    if settings.is_sqlite:
        if _reset_markers.get(project_id) != generation:
            raise RuntimeError("project reset generation changed")
        _reset_markers.pop(project_id)
        _reset_targets.pop((project_id, generation), None)
        return
    result = await _redis_client().eval(
        "if redis.call('GET', KEYS[1]) == ARGV[1] then redis.call('DEL', KEYS[2], KEYS[3]); return redis.call('DEL', KEYS[1]) end return 0",
        3,
        f"rally:project_reset:{project_id}",
        f"rally:project_reset_target:{project_id}:{generation}",
        f"rally:project_reset_acks:{project_id}:{generation}",
        generation,
    )
    if result != 1:
        raise RuntimeError("project reset generation changed")


def _parse_reset_message(data: object, project_id: UUID) -> str | None:
    try:
        payload = json.loads(data) if isinstance(data, str) else data
        if not isinstance(payload, dict) or payload.get("project_id") != str(project_id):
            return None
        generation = payload.get("generation")
        if not isinstance(generation, str) or not generation:
            raise ValueError
        return generation
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.warning("Malformed project reset message")
        return None


async def _deliver_reset_message(data: object) -> None:
    """SQLite test hook mirroring the defensive Redis subscriber parser."""
    for project_id, waiters in tuple(_reset_waiters.items()):
        generation = _parse_reset_message(data, project_id)
        if generation:
            for waiter in tuple(waiters):
                if not waiter.done():
                    waiter.set_result(generation)


async def register_project_reset_monitor(project_id: UUID) -> ResetRegistration:
    acknowledgement = asyncio.get_running_loop().create_future()
    if not settings.is_sqlite:
        token = str(uuid4())
        generation = await _redis_client().eval(
            "local generation=redis.call('GET', KEYS[1]); if generation then return generation end; "
            "redis.call('SADD', KEYS[2], ARGV[1]); redis.call('SET', KEYS[3], '1', 'EX', tonumber(ARGV[2])); return false",
            3,
            f"rally:project_reset:{project_id}",
            f"rally:project_reset_monitors:{project_id}",
            f"rally:project_reset_monitor_lease:{project_id}:{token}",
            token,
            _RESET_LEASE_SECONDS,
        )
        return ResetRegistration(None if generation else token, str(generation) if generation else None)
    generation = _reset_markers.get(project_id)
    if generation:
        return ResetRegistration(None, generation)
    _reset_monitor_acks.setdefault(project_id, set()).add(acknowledgement)
    return ResetRegistration(acknowledgement)


async def renew_project_reset_monitor(project_id: UUID, monitor: ResetRegistration) -> str | None:
    if monitor.token is None:
        return monitor.generation
    if settings.is_sqlite:
        if monitor.token not in _reset_monitor_acks.get(project_id, set()):
            raise RuntimeError("stale project reset monitor")
        return _reset_markers.get(project_id)
    result = await _redis_client().eval(
        "local generation=redis.call('GET', KEYS[1]); if generation then return generation end; "
        "if redis.call('SISMEMBER', KEYS[2], ARGV[1]) == 0 or redis.call('EXISTS', KEYS[3]) == 0 then redis.call('SREM', KEYS[2], ARGV[1]); return '__stale__' end; redis.call('EXPIRE', KEYS[3], tonumber(ARGV[2])); return false",
        3, f"rally:project_reset:{project_id}", f"rally:project_reset_monitors:{project_id}",
        f"rally:project_reset_monitor_lease:{project_id}:{monitor.token}", monitor.token, _RESET_LEASE_SECONDS,
    )
    if result == "__stale__":
        raise RuntimeError("stale project reset monitor")
    return str(result) if result else None


async def unregister_project_reset_monitor(project_id: UUID, monitor: ResetRegistration) -> None:
    acknowledgement = monitor.token
    if acknowledgement is None:
        return
    if not settings.is_sqlite:
        await _redis_client().eval(
            "local generation=redis.call('GET', KEYS[3]); if generation then local target=ARGV[2]..generation; "
            "if redis.call('SISMEMBER', target, ARGV[1]) == 1 then local ack=ARGV[3]..generation; "
            "redis.call('SADD', ack, ARGV[1]); redis.call('EXPIRE', ack, tonumber(ARGV[4])) end end; "
            "redis.call('SREM', KEYS[1], ARGV[1]); redis.call('DEL', KEYS[2]); return generation",
            3,
            f"rally:project_reset_monitors:{project_id}",
            f"rally:project_reset_monitor_lease:{project_id}:{acknowledgement}",
            f"rally:project_reset:{project_id}",
            acknowledgement,
            f"rally:project_reset_target:{project_id}:",
            f"rally:project_reset_acks:{project_id}:",
            _RESET_GENERATION_TTL_SECONDS,
        )
        return
    for (target_project, _), acknowledgements in tuple(_reset_targets.items()):
        if target_project == project_id and acknowledgement in acknowledgements and not acknowledgement.done():
            acknowledgement.set_result(None)
    monitors = _reset_monitor_acks.get(project_id)
    if monitors:
        monitors.discard(acknowledgement)
        if not monitors:
            _reset_monitor_acks.pop(project_id, None)


async def wait_for_project_reset_monitors(
    project_id: UUID, generation: str, timeout: float = 0.5
) -> None:
    if settings.is_sqlite:
        acknowledgements = _reset_targets.get((project_id, generation), ())
        if acknowledgements:
            _, pending = await asyncio.wait(acknowledgements, timeout=timeout)
            if pending:
                raise TimeoutError("timed out waiting for reset acknowledgements")
        return
    client = _redis_client()
    target_key = f"rally:project_reset_target:{project_id}:{generation}"
    ack_key = f"rally:project_reset_acks:{project_id}:{generation}"
    if not await client.scard(target_key):
        return
    deadline = asyncio.get_running_loop().time() + timeout
    while await client.sdiff(target_key, ack_key):
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"timed out waiting for reset acknowledgements project={project_id} generation={generation}")
        await asyncio.sleep(0.01)
