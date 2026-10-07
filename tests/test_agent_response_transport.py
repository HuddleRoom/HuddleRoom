import asyncio
import json
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from huddleroom.models.event_log import EventLog
from huddleroom.services.agent_response_stream import AgentResponseEvent, AgentResponseInvocation, InvocationContext
from huddleroom.workers.consumers.ws_hub import ConnectionRegistry


class _FakeRedisBackend:
    """Small shared Redis model for the relay's four Lua transactions."""

    def __init__(self):
        self.now = 0.0
        self.strings, self.sets, self.hashes = {}, defaultdict(set), defaultdict(dict)
        self.expires, self.subscribers = {}, defaultdict(set)
        self.on_publish = None

    def advance(self, seconds):
        self.now += seconds

    def _alive(self, key):
        if self.expires.get(key, float("inf")) <= self.now:
            self.strings.pop(key, None)
            self.sets.pop(key, None)
            self.hashes.pop(key, None)
            self.expires.pop(key, None)
            return False
        return key in self.strings or key in self.sets or key in self.hashes

    def _delete(self, *keys):
        for key in keys:
            self.strings.pop(key, None)
            self.sets.pop(key, None)
            self.hashes.pop(key, None)
            self.expires.pop(key, None)

    def _expire(self, key, seconds):
        self.expires[key] = self.now + float(seconds)

    async def publish(self, channel, data):
        if self.on_publish:
            self.on_publish(channel, data)
        for pubsub in tuple(self.subscribers[channel]):
            await pubsub.queue.put({"type": "message", "channel": channel, "data": data})


class _FakePipeline:
    def __init__(self, client):
        self.client, self.operations = client, []

    def publish(self, channel, data):
        self.operations.append(("publish", channel, data))

    def hincrby(self, key, field, amount):
        self.operations.append(("hincrby", key, field, amount))

    async def execute(self):
        for operation in self.operations:
            if operation[0] == "publish":
                await self.client.backend.publish(*operation[1:])
            else:
                _, key, field, amount = operation
                self.client.backend.hashes[key][field] = int(self.client.backend.hashes[key].get(field, 0)) + amount


class _FakePubSub:
    def __init__(self, client):
        self.client, self.queue, self.channels, self.close_calls = client, asyncio.Queue(), set(), 0

    async def subscribe(self, *channels):
        for channel in channels:
            self.channels.add(channel)
            self.client.backend.subscribers[channel].add(self)

    async def unsubscribe(self, *channels):
        for channel in channels or tuple(self.channels):
            self.channels.discard(channel)
            self.client.backend.subscribers[channel].discard(self)

    async def listen(self):
        while True:
            message = await self.queue.get()
            if message is None:
                return
            yield message

    async def aclose(self):
        self.close_calls += 1
        await self.unsubscribe()
        await self.queue.put(None)


class _FakeRedisClient:
    def __init__(self, backend):
        self.backend, self.pubsubs, self.close_calls = backend, [], 0

    def pipeline(self, transaction=True):
        assert transaction is True
        return _FakePipeline(self)

    def pubsub(self):
        pubsub = _FakePubSub(self)
        self.pubsubs.append(pubsub)
        return pubsub

    async def get(self, key):
        return self.backend.strings.get(key) if self.backend._alive(key) else None

    async def hgetall(self, key):
        self.backend._alive(key)
        return self.backend.hashes[key].copy()

    async def scard(self, key):
        return len(self.backend.sets[key]) if self.backend._alive(key) else 0

    async def sdiff(self, left, right):
        self.backend._alive(left)
        self.backend._alive(right)
        return self.backend.sets[left] - self.backend.sets[right]

    async def eval(self, script, key_count, *values):
        keys, argv, store = values[:key_count], values[key_count:], self.backend
        if "local previous=redis.call('GET'" in script:
            assert "SMEMBERS" in script and "EXISTS" in script and "PUBLISH" in script
            marker, monitors, target, acknowledgements, channel = keys
            generation, message, lease_prefix, target_prefix, ack_prefix, ttl = argv
            previous = await self.get(marker)
            if previous:
                store._delete(target_prefix + previous, ack_prefix + previous)
            live = {token for token in store.sets[monitors] if store._alive(lease_prefix + token)}
            store.sets[monitors].intersection_update(live)
            store._delete(target, acknowledgements)
            store.sets[target].update(live)
            store._expire(target, ttl)
            store.sets[acknowledgements].add("__empty__")
            store.sets[acknowledgements].discard("__empty__")
            store._expire(acknowledgements, ttl)
            store.strings[marker] = generation
            await store.publish(channel, message)
            return len(live)
        if "return '__stale__'" in script:
            assert "SISMEMBER" in script and "EXPIRE" in script
            marker, monitors, lease = keys
            token, seconds = argv
            if generation := await self.get(marker):
                return generation
            if token not in store.sets[monitors] or not store._alive(lease):
                store.sets[monitors].discard(token)
                return "__stale__"
            store._expire(lease, seconds)
            return False
        if "local generation=redis.call('GET', KEYS[1])" in script:
            assert "SADD" in script and "SET" in script
            marker, monitors, lease = keys
            token, seconds = argv
            if generation := await self.get(marker):
                return generation
            store.sets[monitors].add(token)
            store.strings[lease] = "1"
            store._expire(lease, seconds)
            return False
        if "local generation=redis.call('GET', KEYS[3])" in script:
            assert "SREM" in script and "DEL" in script
            monitors, lease, marker = keys
            token, target_prefix, ack_prefix, ttl = argv
            if generation := await self.get(marker):
                target, acknowledgements = target_prefix + generation, ack_prefix + generation
                if token in store.sets[target]:
                    store.sets[acknowledgements].add(token)
                    store._expire(acknowledgements, ttl)
            store.sets[monitors].discard(token)
            store._delete(lease)
            return generation if (generation := await self.get(marker)) else None
        if "if redis.call('GET', KEYS[1])" in script:
            marker, target, acknowledgements = keys
            generation = argv[0]
            if await self.get(marker) == generation:
                store._delete(target, acknowledgements, marker)
                return 1
            return 0
        raise AssertionError(f"unknown relay script: {script}")

    async def aclose(self):
        self.close_calls += 1


def _fake_redis(monkeypatch):
    import huddleroom.services.agent_response_relay as relay

    backend = _FakeRedisBackend()
    client = _FakeRedisClient(backend)
    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=False, redis_url="redis://fake"))
    monkeypatch.setattr(relay, "_redis_client", lambda: client)
    return backend, client


async def _until(predicate):
    for _ in range(20):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


def _stub_sqlite_lifespan_runtime(monkeypatch):
    async def idle():
        await asyncio.Future()

    consumer_tasks = {}
    monkeypatch.setattr("huddleroom.workers.scheduler.start_scheduler", lambda: object())
    monkeypatch.setattr("huddleroom.workers.scheduler.stop_scheduler", lambda: None)
    monkeypatch.setattr("huddleroom.workers.consumers.get_consumer_tasks", lambda: consumer_tasks)
    monkeypatch.setattr("huddleroom.workers.orchestration_tasks.run_orchestration_event_supervisor", idle)
    for module in ("ws_hub", "rule_engine", "graph_engine", "meeting_engine", "optimizer"):
        monkeypatch.setattr(f"huddleroom.workers.consumers.{module}.run_{module}", idle)
    return consumer_tasks


def _event(project_id, event_type="agent_response.started", sequence=0):
    return AgentResponseEvent(
        project_id=project_id,
        call_id=uuid.uuid4(),
        sequence=sequence,
        event_type=event_type,
        emitted_at=datetime.now(timezone.utc),
        call_started_at=datetime.now(timezone.utc),
        invocation_kind="api",
        operation="task",
        parent_call_id=None,
        actor_kind="agent",
        actor_id="agent",
        payload=(
            {"request_display": {"kind": "unavailable"}, "actor_label": None, "model_or_runtime": "test"}
            if event_type == "agent_response.started"
            else {"status": "completed"}
            if event_type == "agent_response.terminal"
            else {"stream": "output", "text": "output"}
        ),
    )


@pytest.mark.asyncio
async def test_direct_transport_isolates_projects_and_does_not_write_event_log():
    from huddleroom.services.agent_response_relay import publish_agent_response, subscribe_agent_responses

    project_a = uuid.uuid4()
    received_a, received_b = [], []
    unsubscribe_a = subscribe_agent_responses(lambda event: received_a.append(event))
    unsubscribe_b = subscribe_agent_responses(lambda event: received_b.append(event))
    try:
        await publish_agent_response(_event(project_a))
    finally:
        unsubscribe_a()
        unsubscribe_b()

    assert [event.project_id for event in received_a] == [project_a]
    assert [event.project_id for event in received_b] == [project_a]


@pytest.mark.asyncio
async def test_transport_does_not_persist_event_log_rows(db_session):
    from huddleroom.services.agent_response_relay import publish_agent_response

    before = await db_session.scalar(select(func.count(EventLog.id)))
    await publish_agent_response(_event(uuid.uuid4()))
    assert await db_session.scalar(select(func.count(EventLog.id))) == before


@pytest.mark.asyncio
async def test_two_producers_increment_metrics_once_per_event_without_subscriber_double_count():
    from huddleroom.services.agent_response_relay import (
        get_agent_response_metrics,
        publish_agent_response,
        subscribe_agent_responses,
    )

    before = await get_agent_response_metrics()
    received = []
    first = subscribe_agent_responses(lambda event: received.append(event))
    second = subscribe_agent_responses(lambda event: received.append(event))
    try:
        await asyncio.gather(
            publish_agent_response(_event(uuid.uuid4())),
            publish_agent_response(_event(uuid.uuid4())),
        )
    finally:
        first()
        second()
    after = await get_agent_response_metrics()
    assert len(received) == 4
    assert after["started_total"] == before["started_total"] + 2


def test_response_event_rejects_malformed_payload():
    with pytest.raises(ValidationError):
        _event(uuid.uuid4()).model_copy(update={"payload": {"unexpected": True}}).validate_payload()


@pytest.mark.asyncio
async def test_registry_serializes_heartbeat_and_response_sends():
    registry = ConnectionRegistry()
    project_id = uuid.uuid4()

    class Socket:
        def __init__(self):
            self.messages = []
            self.sending = False

        async def send_text(self, text):
            assert not self.sending
            self.sending = True
            await asyncio.sleep(0)
            self.messages.append(text)
            self.sending = False

        async def close(self, code):
            assert code == 1013

    socket = Socket()
    registry.add(project_id, "connection", socket, None)
    await asyncio.gather(
        registry.enqueue(project_id, "connection", json.dumps({"type": "ping"}), evictable=False),
        registry.broadcast_response(_event(project_id)),
    )
    await asyncio.sleep(0)
    assert len(socket.messages) == 2
    await registry.remove(project_id, "connection")


@pytest.mark.asyncio
async def test_sqlite_reset_waiters_keep_generation_after_clear(monkeypatch):
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    first = asyncio.create_task(relay.wait_for_project_reset(project_id))
    second = asyncio.create_task(relay.wait_for_project_reset(project_id))
    while len(relay._reset_waiters.get(project_id, ())) != 2:
        await asyncio.sleep(0)
    generation = await relay.publish_project_reset(project_id)
    await relay.clear_project_reset(project_id, generation)
    assert await first == generation
    assert await second == generation


@pytest.mark.asyncio
async def test_sqlite_reset_ack_snapshot_excludes_late_and_keeps_unregistered_waiter(monkeypatch):
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    first = await relay.register_project_reset_monitor(project_id)
    generation = await relay.publish_project_reset(project_id)
    await relay.unregister_project_reset_monitor(project_id, first)
    late = await relay.register_project_reset_monitor(project_id)
    await relay.wait_for_project_reset_monitors(project_id, generation, timeout=0.01)
    assert late.token is None
    assert late.generation == generation
    await relay.unregister_project_reset_monitor(project_id, late)


@pytest.mark.asyncio
async def test_reset_acknowledgement_timeout_raises(monkeypatch):
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    await relay.register_project_reset_monitor(project_id)
    generation = await relay.publish_project_reset(project_id)
    with pytest.raises(TimeoutError, match="reset acknowledgements"):
        await relay.wait_for_project_reset_monitors(project_id, generation, timeout=0)


@pytest.mark.asyncio
async def test_call_registers_reset_monitor_before_started_is_published():
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()

    async def publish(event):
        assert relay._reset_monitor_acks[project_id]

    call = AgentResponseInvocation(
        InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"),
        publish=publish,
    ).call(messages=[])
    async with call:
        pass


@pytest.mark.asyncio
async def test_call_joining_after_reset_emits_only_project_reset_terminal():
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    generation = await relay.publish_project_reset(project_id)
    events = []

    async def publish(event):
        events.append(event)

    call = AgentResponseInvocation(
        InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"),
        publish=publish,
    ).call(messages=[])

    entered = False
    with pytest.raises(asyncio.CancelledError):
        async with call:
            entered = True

    assert [(event.event_type, event.payload.get("status")) for event in events] == [
        ("agent_response.terminal", "project_reset")
    ]
    assert events[0].payload["metadata"] == {"generation": generation}
    assert not entered


@pytest.mark.asyncio
async def test_unregistering_snapshot_monitor_acknowledges_before_removal():
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    monitor = await relay.register_project_reset_monitor(project_id)
    generation = await relay.publish_project_reset(project_id)

    await relay.unregister_project_reset_monitor(project_id, monitor)
    await relay.wait_for_project_reset_monitors(project_id, generation, timeout=0)


@pytest.mark.asyncio
async def test_reset_registration_is_fenced_after_snapshot():
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    generation = await relay.publish_project_reset(project_id)

    registration = await relay.register_project_reset_monitor(project_id)

    assert registration.token is None
    assert registration.generation == generation
    assert project_id not in relay._reset_monitor_acks


@pytest.mark.asyncio
async def test_completed_call_unregisters_snapshot_before_reset_ack_wait():
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    registration = await relay.register_project_reset_monitor(project_id)
    generation = await relay.publish_project_reset(project_id)

    await relay.unregister_project_reset_monitor(project_id, registration)

    await relay.wait_for_project_reset_monitors(project_id, generation, timeout=0)


@pytest.mark.asyncio
async def test_renewal_after_stale_registration_raises():
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    registration = await relay.register_project_reset_monitor(project_id)
    await relay.unregister_project_reset_monitor(project_id, registration)

    with pytest.raises(RuntimeError, match="stale project reset monitor"):
        await relay.renew_project_reset_monitor(project_id, registration)


@pytest.mark.asyncio
async def test_reset_waiter_skips_malformed_message_before_valid_message(monkeypatch, caplog):
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    waiter = asyncio.create_task(relay.wait_for_project_reset(project_id))
    await asyncio.sleep(0)
    await relay._deliver_reset_message("not-json")
    generation = await relay.publish_project_reset(project_id)

    assert await waiter == generation
    assert "Malformed project reset message" in caplog.text


@pytest.mark.asyncio
async def test_reset_ack_timeout_leaves_project_fenced_before_any_cancellation(monkeypatch):
    import huddleroom.services.project_reset_service as reset_service

    project_id = uuid.uuid4()
    project = type("Project", (), {"id": project_id, "name": "Fence", "status": "active"})()
    calls = []

    class Db:
        bind = None

        async def get(self, *_args, **_kwargs):
            return project

        async def commit(self):
            calls.append("commit")

        async def rollback(self):
            calls.append("rollback")

    async def lock(*_args):
        return None

    async def publish(_project_id):
        return "generation"

    async def timeout(*_args, **_kwargs):
        raise TimeoutError("reset acknowledgements")

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("cancellation must not run after acknowledgement timeout")

    monkeypatch.setattr(reset_service.ProjectService, "lock_workspace_boundary", lock)
    monkeypatch.setattr(reset_service, "publish_project_reset", publish)
    monkeypatch.setattr(reset_service, "wait_for_project_reset_monitors", timeout)
    monkeypatch.setattr(reset_service.task_runner, "cancel_project_sessions", forbidden)

    with pytest.raises(TimeoutError, match="reset acknowledgements"):
        await reset_service.ProjectResetService().reset(Db(), project_id, "Fence")

    assert project.status == "resetting"
    assert calls == ["commit"]


@pytest.mark.asyncio
async def test_fenced_terminal_does_not_make_completion_metrics_exceed_started():
    from huddleroom.services.agent_response_relay import get_agent_response_metrics, publish_agent_response

    project_id = uuid.uuid4()
    before = await get_agent_response_metrics()
    started = _event(project_id)
    fenced = _event(project_id, "agent_response.terminal", 0)
    await publish_agent_response(started)
    await publish_agent_response(fenced)
    after = await get_agent_response_metrics()

    assert after["started_total"] == before["started_total"] + 1
    assert after["terminal_total"] == before["terminal_total"]
    assert after["terminal_total"] <= after["started_total"]
    assert after["completion_rate"] <= 1.0


@pytest.mark.asyncio
async def test_reset_acknowledges_only_after_owner_cancellation_is_issued(monkeypatch):
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    order = []
    original_unregister = relay.unregister_project_reset_monitor

    async def record_unregister(*args):
        order.append("ack")
        return await original_unregister(*args)

    monkeypatch.setattr(relay, "unregister_project_reset_monitor", record_unregister)
    running = asyncio.Event()

    async def owner():
        call = AgentResponseInvocation(
            InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"),
        ).call(messages=[])
        try:
            async with call:
                running.set()
                await asyncio.Future()
        except asyncio.CancelledError:
            order.append("cancelled")
            raise

    task = asyncio.create_task(owner())
    await running.wait()
    await relay.publish_project_reset(project_id)
    with pytest.raises(asyncio.CancelledError):
        await task

    assert order.index("cancelled") < order.index("ack")


@pytest.mark.asyncio
async def test_relay_close_continues_resource_cleanup_after_call_cleanup_failure(monkeypatch):
    import huddleroom.services.agent_response_relay as relay

    closed = []

    class Resource:
        async def aclose(self):
            closed.append(self)

    loop = asyncio.get_running_loop()
    pubsub, client = Resource(), Resource()
    relay._hub_pubsubs[loop] = pubsub
    relay._redis_clients[loop] = client

    async def fail_calls():
        raise RuntimeError("call publication failed")

    monkeypatch.setattr("huddleroom.services.agent_response_stream.close_agent_response_monitors", fail_calls)
    with pytest.raises(RuntimeError, match="call publication failed"):
        await relay.close_agent_response_relay()

    assert closed == [pubsub, client]


@pytest.mark.asyncio
async def test_reset_retry_replaces_old_sqlite_target_without_losing_delivered_waiter():
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    waiter = asyncio.create_task(relay.wait_for_project_reset(project_id))
    await asyncio.sleep(0)
    first = await relay.publish_project_reset(project_id)
    second = await relay.publish_project_reset(project_id)

    assert await waiter == first
    assert (project_id, first) not in relay._reset_targets
    assert (project_id, second) in relay._reset_targets


@pytest.mark.asyncio
async def test_close_monitors_exhausts_current_loop_calls_after_a_terminal_failure(monkeypatch):
    import huddleroom.services.agent_response_stream as stream

    closed = []

    class Call:
        async def terminal(self, _status):
            closed.append(self)
            if len(closed) == 1:
                raise RuntimeError("first close failed")

    loop = asyncio.get_running_loop()
    first, second = Call(), Call()
    calls = stream._ACTIVE_RESET_CALLS.setdefault(loop, set())
    calls.update((first, second))

    with pytest.raises(RuntimeError, match="first close failed"):
        await stream.close_agent_response_monitors()

    assert set(closed) == {first, second}


@pytest.mark.asyncio
async def test_terminal_publisher_failure_untracks_and_stops_call_tasks():
    import huddleroom.services.agent_response_stream as stream

    project_id = uuid.uuid4()

    async def fail_terminal(event):
        if event.event_type == "agent_response.terminal":
            raise ConnectionError("publisher down")

    call = AgentResponseInvocation(
        InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"), publish=fail_terminal,
    ).call(messages=[])
    await call.__aenter__()
    with pytest.raises(ConnectionError, match="publisher down"):
        await call.terminal("cancelled")

    assert not call._reset_monitor and not call._reset_heartbeat
    assert all(call not in calls for calls in stream._ACTIVE_RESET_CALLS.values())


@pytest.mark.asyncio
async def test_terminal_unregister_failure_untracks_and_stops_call_tasks(monkeypatch):
    import huddleroom.services.agent_response_relay as relay
    import huddleroom.services.agent_response_stream as stream

    project_id = uuid.uuid4()
    call = AgentResponseInvocation(
        InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"),
    ).call(messages=[])
    await call.__aenter__()

    async def fail_unregister(*_args):
        raise ConnectionError("unregister down")

    monkeypatch.setattr(relay, "unregister_project_reset_monitor", fail_unregister)
    with pytest.raises(ConnectionError, match="unregister down"):
        await call.terminal("cancelled")

    assert call._reset_ack is not None
    assert not call._reset_monitor and not call._reset_heartbeat
    assert call in stream._ACTIVE_RESET_CALLS[asyncio.get_running_loop()]


@pytest.mark.asyncio
async def test_queue_eviction_keeps_queue_join_accounting_balanced():
    registry = ConnectionRegistry()
    project_id = uuid.uuid4()
    release = asyncio.Event()

    class Socket:
        async def send_text(self, _text):
            await release.wait()

        async def close(self, _code):
            raise AssertionError("unexpected close")

    registry.add(project_id, "connection", Socket(), None)
    for sequence in range(1001):
        await registry.enqueue(project_id, "connection", str(sequence), evictable=True)
    queue = registry._conns[str(project_id)]["connection"][2]
    release.set()
    await asyncio.wait_for(queue.join(), timeout=1)
    await registry.remove(project_id, "connection")


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_relay_shutdown_unregistration_uses_then_closes_current_redis_client(monkeypatch):
    import huddleroom.services.agent_response_relay as relay

    backend, client = _FakeRedisBackend(), _FakeRedisClient(_FakeRedisBackend())
    client.backend = backend
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=False, redis_url="redis://fake"))
    relay._redis_clients[loop] = client
    project_id = uuid.uuid4()
    call = AgentResponseInvocation(
        InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"),
    ).call(messages=[])
    await call.__aenter__()

    await relay.close_agent_response_relay()

    assert client.close_calls == 1
    assert not backend.sets[f"rally:project_reset_monitors:{project_id}"]


@pytest.mark.asyncio
async def test_heartbeat_network_failure_cancels_owner_when_terminal_publish_fails(monkeypatch):
    import huddleroom.services.agent_response_relay as relay
    import huddleroom.services.agent_response_stream as stream

    project_id = uuid.uuid4()
    original_sleep = asyncio.sleep

    async def immediate_sleep(_seconds):
        return None

    async def renew_failure(*_args):
        raise ConnectionError("redis lost")

    async def publish(event):
        if event.event_type == "agent_response.terminal":
            raise ConnectionError("publisher lost")

    monkeypatch.setattr(asyncio, "sleep", immediate_sleep)
    monkeypatch.setattr(relay, "renew_project_reset_monitor", renew_failure)

    async def owner():
        call = AgentResponseInvocation(
            InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"), publish=publish,
        ).call(messages=[])
        async with call:
            await asyncio.Future()

    task = asyncio.create_task(owner())
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(asyncio, "sleep", original_sleep)
    assert not stream._ACTIVE_RESET_CALLS.get(asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_terminal_cancels_blocked_heartbeat_without_deadlock(monkeypatch):
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    blocked = asyncio.Event()

    async def block_renew(*_args):
        blocked.set()
        await asyncio.Future()

    monkeypatch.setattr(relay, "_RESET_RENEW_SECONDS", 0)
    monkeypatch.setattr(relay, "renew_project_reset_monitor", block_renew)
    call = AgentResponseInvocation(
        InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"),
    ).call(messages=[])
    await call.__aenter__()
    heartbeat = call._reset_heartbeat
    await asyncio.wait_for(blocked.wait(), timeout=1)
    await asyncio.wait_for(call.terminal("cancelled"), timeout=1)

    assert heartbeat.cancelled() or heartbeat.done()


@pytest.mark.asyncio
async def test_unregister_failure_is_retried_by_relay_close(monkeypatch):
    import huddleroom.services.agent_response_relay as relay
    import huddleroom.services.agent_response_stream as stream

    project_id = uuid.uuid4()
    call = AgentResponseInvocation(
        InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"),
    ).call(messages=[])
    await call.__aenter__()
    generation = await relay.publish_project_reset(project_id)
    original = relay.unregister_project_reset_monitor
    attempts = 0

    async def fail_once(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("first unregister fails")
        return await original(*args)

    monkeypatch.setattr(relay, "unregister_project_reset_monitor", fail_once)
    with pytest.raises(ConnectionError):
        await call.terminal("cancelled")
    assert call in stream._ACTIVE_RESET_CALLS[asyncio.get_running_loop()]

    await relay.close_agent_response_relay()
    await relay.wait_for_project_reset_monitors(project_id, generation, timeout=0)
    assert attempts == 2
    assert not stream._ACTIVE_RESET_CALLS.get(asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_failed_cleanup_call_is_strongly_retained_until_relay_retry(monkeypatch):
    import gc
    import weakref
    import huddleroom.services.agent_response_relay as relay
    import huddleroom.services.agent_response_stream as stream

    project_id = uuid.uuid4()
    original = relay.unregister_project_reset_monitor
    attempts = 0

    async def fail_once(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("unregister fails")
        return await original(*args)

    monkeypatch.setattr(relay, "unregister_project_reset_monitor", fail_once)
    async def fail_cleanup():
        call = AgentResponseInvocation(
            InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"),
        ).call(messages=[])
        await call.__aenter__()
        reference = weakref.ref(call)
        with pytest.raises(ConnectionError):
            await call.terminal("cancelled")
        return reference

    reference = await asyncio.create_task(fail_cleanup())
    gc.collect()
    retained = reference()
    assert retained is not None
    assert retained in stream._ACTIVE_RESET_CALLS[asyncio.get_running_loop()]

    await relay.close_agent_response_relay()
    assert attempts == 2
    assert not stream._ACTIVE_RESET_CALLS.get(asyncio.get_running_loop())
    del retained
    gc.collect()
    assert reference() is None


@pytest.mark.asyncio
async def test_started_publish_and_unregister_failure_retains_cleanup_retry(monkeypatch):
    import huddleroom.services.agent_response_relay as relay
    import huddleroom.services.agent_response_stream as stream

    project_id = uuid.uuid4()
    original = relay.unregister_project_reset_monitor
    attempts = 0

    async def fail_once(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("unregister fails")
        return await original(*args)

    async def fail_started(event):
        if event.event_type == "agent_response.started":
            raise ConnectionError("started fails")

    monkeypatch.setattr(relay, "unregister_project_reset_monitor", fail_once)
    call = AgentResponseInvocation(
        InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"), publish=fail_started,
    ).call(messages=[])
    with pytest.raises(ConnectionError, match="started fails"):
        await call.__aenter__()
    assert not call._reset_monitor and not call._reset_heartbeat
    assert call in stream._ACTIVE_RESET_CALLS[asyncio.get_running_loop()]

    await relay.close_agent_response_relay()
    assert attempts == 2
    assert not stream._ACTIVE_RESET_CALLS.get(asyncio.get_running_loop())


@pytest.mark.asyncio
async def test_relay_close_cleans_live_call_reset_monitor():
    import huddleroom.services.agent_response_relay as relay

    project_id = uuid.uuid4()
    call = AgentResponseInvocation(
        InvocationContext(project_id, "agent", "agent", None, "api", "task", "test"),
    ).call(messages=[])
    await call.__aenter__()
    assert relay._reset_monitor_acks[project_id]

    await relay.close_agent_response_relay()

    assert project_id not in relay._reset_monitor_acks


@pytest.mark.asyncio
async def test_response_fanout_filters_projects_and_event_types():
    registry = ConnectionRegistry()
    project_a, project_b = uuid.uuid4(), uuid.uuid4()

    class Socket:
        def __init__(self):
            self.messages = []

        async def send_text(self, text):
            self.messages.append(text)

        async def close(self, code):
            raise AssertionError(code)

    accepted, filtered, other_project = Socket(), Socket(), Socket()
    registry.add(project_a, "accepted", accepted, {"agent_response.started"})
    registry.add(project_a, "filtered", filtered, {"task.created"})
    registry.add(project_b, "other", other_project, None)
    await registry.broadcast_response(_event(project_a))
    await asyncio.sleep(0)
    assert len(accepted.messages) == 1
    assert filtered.messages == []
    assert other_project.messages == []
    await registry.remove(project_a, "accepted")
    await registry.remove(project_a, "filtered")
    await registry.remove(project_b, "other")


@pytest.mark.asyncio
async def test_sender_failure_removes_connection():
    registry = ConnectionRegistry()
    project_id = uuid.uuid4()

    class Socket:
        async def send_text(self, text):
            raise RuntimeError("closed")

        async def close(self, code):
            assert code == 1013

    registry.add(project_id, "connection", Socket(), None)
    assert await registry.enqueue(project_id, "connection", "message", evictable=False)
    sender = registry._conns[str(project_id)]["connection"][3]
    await sender
    assert registry.connection_count() == 0


@pytest.mark.asyncio
async def test_queue_evicts_only_oldest_response_output_and_keeps_sequence_gap():
    registry = ConnectionRegistry()
    project_id = uuid.uuid4()
    started, release = asyncio.Event(), asyncio.Event()

    class Socket:
        def __init__(self):
            self.messages = []

        async def send_text(self, text):
            self.messages.append(text)
            started.set()
            await release.wait()

        async def close(self, code):
            raise AssertionError(code)

    socket = Socket()
    registry.add(project_id, "connection", socket, None)
    await registry.broadcast_response(_event(project_id, "agent_response.output", 0))
    await started.wait()
    for sequence in range(1, 1001):
        await registry.broadcast_response(_event(project_id, "agent_response.output", sequence))
    await registry.broadcast_response(_event(project_id, "agent_response.terminal", 1001))
    release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    sequences = [json.loads(text)["sequence"] for text in socket.messages]
    assert 1 not in sequences
    assert sequences[0] == 0
    assert sequences[-1] == 1001
    await registry.remove(project_id, "connection")


@pytest.mark.asyncio
async def test_failed_subscriber_does_not_block_other_subscribers(caplog):
    import huddleroom.services.agent_response_relay as relay

    received = []
    bad = relay.subscribe_agent_responses(lambda event: (_ for _ in ()).throw(RuntimeError("bad subscriber")))
    good = relay.subscribe_agent_responses(lambda event: received.append(event))
    try:
        await relay.publish_agent_response(_event(uuid.uuid4()))
    finally:
        bad()
        good()
    assert len(received) == 1
    assert "bad subscriber" in caplog.text


@pytest.mark.asyncio
async def test_non_evictable_overflow_removes_even_when_close_fails():
    registry = ConnectionRegistry()
    project_id = uuid.uuid4()

    class Socket:
        async def send_text(self, text):
            await asyncio.Future()

        async def close(self, code):
            assert code == 1013
            raise RuntimeError("close failed")

    registry.add(project_id, "bad", Socket(), None)
    registry.add(project_id, "good", Socket(), None)
    for sequence in range(1000):
        await registry.enqueue(project_id, "bad", str(sequence), evictable=False)
    assert not await registry.enqueue(project_id, "bad", "overflow", evictable=False)
    assert "bad" not in registry._conns.get(str(project_id), {})
    assert await registry.enqueue(project_id, "good", "still works", evictable=False)
    await registry.remove(project_id, "good")


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_fake_redis_combines_producers_ignores_malformed_pubsub_and_keeps_event_log(db_session, monkeypatch, caplog):
    """Redis publisher counters are shared, while the hub only relays validated events."""
    import huddleroom.services.agent_response_relay as relay

    backend, first = _fake_redis(monkeypatch)
    second = _FakeRedisClient(backend)
    received = []
    unsubscribe = relay.subscribe_agent_responses(received.append)
    hub = asyncio.create_task(relay.run_agent_response_hub())
    await _until(lambda: bool(backend.subscribers[relay._RESPONSE_CHANNEL]))
    before = await db_session.scalar(select(func.count(EventLog.id)))
    project_a, project_b = uuid.uuid4(), uuid.uuid4()
    try:
        await backend.publish(relay._RESPONSE_CHANNEL, "not-json")
        await relay.publish_agent_response(_event(project_a))
        monkeypatch.setattr(relay, "_redis_client", lambda: second)
        await relay.publish_agent_response(_event(project_b))
        await relay.publish_agent_response(_event(project_a, "agent_response.terminal", 1))
        await _until(lambda: len(received) == 3)
    finally:
        unsubscribe()
        hub.cancel()
        await asyncio.gather(hub, return_exceptions=True)

    metrics = await relay.get_agent_response_metrics()
    assert [event.project_id for event in received] == [project_a, project_b, project_a]
    assert metrics == {"started_total": 2, "terminal_total": 1, "omission_gap": 1, "completion_rate": 0.5}
    assert await db_session.scalar(select(func.count(EventLog.id))) == before
    assert "Malformed agent response message" in caplog.text
    assert first.pubsubs[0].close_calls == 1


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_fake_redis_reset_fence_is_atomic_and_cancels_only_matching_workers(monkeypatch, caplog):
    import huddleroom.services.agent_response_relay as relay

    backend, _ = _fake_redis(monkeypatch)
    project_a, project_b = uuid.uuid4(), uuid.uuid4()
    order, terminals = [], []
    unsubscribe = relay.subscribe_agent_responses(
        lambda event: terminals.append(event) if event.event_type == "agent_response.terminal" else None
    )
    def record_terminal(channel, data):
        if channel == relay._RESPONSE_CHANNEL:
            event = AgentResponseEvent.model_validate_json(data)
            if event.event_type == "agent_response.terminal":
                order.append(("terminal", event.project_id))

    backend.on_publish = record_terminal
    hub = asyncio.create_task(relay.run_agent_response_hub())
    await _until(lambda: bool(backend.subscribers[relay._RESPONSE_CHANNEL]))
    entered = {project_a: asyncio.Event(), project_b: asyncio.Event()}

    async def owner(project_id):
        call = AgentResponseInvocation(
            InvocationContext(project_id, "agent", "worker", None, "api", "task", "fake"),
        ).call(messages=[])
        try:
            async with call:
                entered[project_id].set()
                await asyncio.Future()
        except asyncio.CancelledError:
            order.append(("cancel", project_id))
            raise

    workers = [asyncio.create_task(owner(project_a)), asyncio.create_task(owner(project_b))]
    try:
        await asyncio.gather(*(event.wait() for event in entered.values()))
        await _until(lambda: len(backend.subscribers[relay._RESET_CHANNEL]) == 2)
        await backend.publish(relay._RESET_CHANNEL, "malformed-reset")
        generation = await relay.publish_project_reset(project_a)
        late = await relay.register_project_reset_monitor(project_a)
        await relay.wait_for_project_reset_monitors(project_a, generation, timeout=0.1)
        assert late.token is None and late.generation == generation
        await _until(lambda: any(kind == "cancel" and project == project_a for kind, project in order))
        assert not workers[1].done()
        assert [event.project_id for event in terminals if event.payload["status"] == "project_reset"] == [project_a]
        assert order.index(("terminal", project_a)) < order.index(("cancel", project_a))
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        unsubscribe()
        hub.cancel()
        await asyncio.gather(hub, return_exceptions=True)

    assert "Malformed project reset message" in caplog.text


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_fake_redis_lease_target_and_clear_generation_guards(monkeypatch):
    import huddleroom.services.agent_response_relay as relay

    backend, _ = _fake_redis(monkeypatch)
    project_id = uuid.uuid4()
    monitor = await relay.register_project_reset_monitor(project_id)
    backend.advance(59)
    assert await relay.renew_project_reset_monitor(project_id, monitor) is None
    backend.advance(59)  # past the original 60-second lease; renewal kept this monitor live.
    assert await relay.renew_project_reset_monitor(project_id, monitor) is None
    stale = await relay.register_project_reset_monitor(project_id)
    backend.advance(61)
    with pytest.raises(RuntimeError, match="stale project reset monitor"):
        await relay.renew_project_reset_monitor(project_id, stale)

    target_monitor = await relay.register_project_reset_monitor(project_id)
    generation = await relay.publish_project_reset(project_id)
    after_fence = await relay.register_project_reset_monitor(project_id)
    with pytest.raises(TimeoutError, match="reset acknowledgements"):
        await relay.wait_for_project_reset_monitors(project_id, generation, timeout=0)
    assert after_fence.token is None and after_fence.generation == generation
    assert target_monitor.token is not None
    with pytest.raises(RuntimeError, match="generation changed"):
        await relay.clear_project_reset(project_id, "wrong-generation")
    assert await relay._redis_client().get(f"rally:project_reset:{project_id}") == generation


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_fake_redis_stale_lease_terminalizes_and_cancels_the_real_owner(monkeypatch):
    import huddleroom.services.agent_response_relay as relay

    backend, _ = _fake_redis(monkeypatch)
    project_id, entered = uuid.uuid4(), asyncio.get_running_loop().create_future()
    events = []
    unsubscribe = relay.subscribe_agent_responses(events.append)
    hub = asyncio.create_task(relay.run_agent_response_hub())
    await _until(lambda: bool(backend.subscribers[relay._RESPONSE_CHANNEL]))
    monkeypatch.setattr(relay, "_RESET_RENEW_SECONDS", 0.001)

    async def owner():
        call = AgentResponseInvocation(
            InvocationContext(project_id, "agent", "worker", None, "api", "task", "fake"),
        ).call(messages=[])
        try:
            async with call:
                entered.set_result(call)
                await asyncio.Future()
        except asyncio.CancelledError:
            raise

    worker = asyncio.create_task(owner())
    try:
        await entered
        backend.advance(61)
        with pytest.raises(asyncio.CancelledError):
            await worker
        await _until(lambda: any(event.event_type == "agent_response.terminal" for event in events))
        assert [event.payload["status"] for event in events if event.event_type == "agent_response.terminal"] == ["failed"]
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        unsubscribe()
        hub.cancel()
        await asyncio.gather(hub, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.unsupported_mode
async def test_project_reset_clear_mismatch_rolls_back_and_keeps_resetting_marker(monkeypatch):
    import huddleroom.services.agent_response_relay as relay
    import huddleroom.services.project_reset_service as reset_service

    backend, _ = _fake_redis(monkeypatch)
    project_id = uuid.uuid4()
    project = SimpleNamespace(id=project_id, name="Fence", status="active")
    calls = []

    class Db:
        bind = None

        async def get(self, *_args, **_kwargs):
            return project

        async def commit(self):
            calls.append("commit")

        async def rollback(self):
            calls.append("rollback")

    async def noop(*_args, **_kwargs):
        return 0

    async def reset(*_args, **_kwargs):
        calls.append("purge")
        return {"rows": 1}

    original_eval = relay._redis_client().eval

    async def mismatched_clear(script, key_count, *values):
        if "if redis.call('GET', KEYS[1])" in script:
            return 0
        return await original_eval(script, key_count, *values)

    monkeypatch.setattr(relay._redis_client(), "eval", mismatched_clear)
    monkeypatch.setattr(reset_service.ProjectService, "lock_workspace_boundary", noop)
    monkeypatch.setattr(reset_service.ProjectService, "reset", reset)
    monkeypatch.setattr(reset_service.task_runner, "cancel_project_sessions", noop)
    monkeypatch.setattr(reset_service.meeting_tasks, "cancel_project_meeting_tasks", noop)
    monkeypatch.setattr(reset_service.meeting_tasks, "revoke_and_await_project_meeting_tasks", noop)

    with pytest.raises(RuntimeError, match="generation changed"):
        await reset_service.ProjectResetService().reset(Db(), project_id, "Fence")

    assert project.status == "resetting"
    assert calls == ["commit", "purge", "rollback"]
    assert await relay._redis_client().get(f"rally:project_reset:{project_id}")


@pytest.mark.asyncio
async def test_project_reset_service_waits_for_active_project_workers_before_purge(monkeypatch):
    """The production reset service must not reach cancellation/purge before its reset quorum."""
    import huddleroom.services.agent_response_relay as relay
    import huddleroom.services.project_reset_service as reset_service

    project_a, project_b = uuid.uuid4(), uuid.uuid4()
    project = SimpleNamespace(id=project_a, name="A", status="active")
    db_events, events, cancelled, acknowledgements = [], [], [], []
    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=True))

    class Db:
        bind = None

        async def get(self, *_args, **_kwargs):
            return project

        async def commit(self):
            db_events.append(("commit", project.status))

        async def rollback(self):
            db_events.append(("rollback", project.status))

    async def publish(event):
        events.append(event)

    async def worker(project_id, label):
        call = AgentResponseInvocation(
            InvocationContext(project_id, "agent", label, None, "api", "task", "reset"), publish=publish,
        ).call(messages=[])
        try:
            async with call:
                await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.append(project_id)
            raise

    workers = [
        asyncio.create_task(worker(project_a, "a-1")),
        asyncio.create_task(worker(project_a, "a-2")),
        asyncio.create_task(worker(project_b, "b-1")),
    ]

    async def lock(*_args):
        return None

    async def cancel_sessions(project_id):
        assert project_id == project_a
        assert cancelled.count(project_a) == 2
        assert acknowledgements == [project_a, project_a]
        terminals = [event for event in events if event.event_type == "agent_response.terminal"]
        assert [event.payload["status"] for event in terminals] == ["project_reset", "project_reset"]
        db_events.append("cancel_sessions")
        return 2

    async def cancel_meetings(project_id):
        assert project_id == project_a and db_events[-1] == "cancel_sessions"
        db_events.append("cancel_meetings")
        return 0

    async def revoke(project_id):
        assert project_id == project_a and db_events[-1] == "cancel_meetings"
        db_events.append("revoke")

    async def purge(_service, _db, project_id):
        assert project_id == project_a and db_events[-1] == "revoke"
        db_events.append("purge")
        return {"rows": 2}

    monkeypatch.setattr(reset_service.ProjectService, "lock_workspace_boundary", lock)
    monkeypatch.setattr(reset_service.task_runner, "cancel_project_sessions", cancel_sessions)
    monkeypatch.setattr(reset_service.meeting_tasks, "cancel_project_meeting_tasks", cancel_meetings)
    monkeypatch.setattr(reset_service.meeting_tasks, "revoke_and_await_project_meeting_tasks", revoke)
    monkeypatch.setattr(reset_service.ProjectService, "reset", purge)
    original_publish = reset_service.publish_project_reset

    async def publish_after_fence(project_id):
        assert db_events == [("commit", "resetting")]
        return await original_publish(project_id)

    monkeypatch.setattr(reset_service, "publish_project_reset", publish_after_fence)
    original_unregister = relay.unregister_project_reset_monitor

    async def record_unregister(project_id, monitor):
        await original_unregister(project_id, monitor)
        if project_id == project_a:
            acknowledgements.append(project_id)

    monkeypatch.setattr(relay, "unregister_project_reset_monitor", record_unregister)
    try:
        await _until(lambda: len(relay._reset_waiters.get(project_a, ())) == 2)
        await _until(lambda: len(relay._reset_waiters.get(project_b, ())) == 1)
        result = await reset_service.ProjectResetService().reset(Db(), project_a, "A")
        assert result == {"cancelled_sessions": 2, "cancelled_meeting_tasks": 0, "deletions": {"rows": 2}}
        assert project.status == "active"
        assert cancelled == [project_a, project_a]
        assert not workers[2].done()
        assert db_events == [("commit", "resetting"), "cancel_sessions", "cancel_meetings", "revoke", "purge", ("commit", "active")]
    finally:
        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)


@pytest.mark.asyncio
async def test_project_reset_service_timeout_keeps_marker_target_and_skips_shutdown_purge(monkeypatch):
    import huddleroom.services.agent_response_relay as relay
    import huddleroom.services.project_reset_service as reset_service

    project_id = uuid.uuid4()
    project = SimpleNamespace(id=project_id, name="Timeout", status="active")
    calls = []
    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=True))

    class Db:
        bind = None

        async def get(self, *_args, **_kwargs):
            return project

        async def commit(self):
            calls.append(("commit", project.status))

        async def rollback(self):
            calls.append("rollback")

    async def lock(*_args):
        return None

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("shutdown/purge/clear must not run after reset quorum timeout")

    monitor = await relay.register_project_reset_monitor(project_id)
    monkeypatch.setattr(reset_service.ProjectService, "lock_workspace_boundary", lock)
    monkeypatch.setattr(reset_service.task_runner, "cancel_project_sessions", forbidden)
    monkeypatch.setattr(reset_service.meeting_tasks, "cancel_project_meeting_tasks", forbidden)
    monkeypatch.setattr(reset_service.meeting_tasks, "revoke_and_await_project_meeting_tasks", forbidden)
    monkeypatch.setattr(reset_service.ProjectService, "reset", forbidden)
    monkeypatch.setattr(reset_service, "clear_project_reset", forbidden)
    try:
        with pytest.raises(TimeoutError, match="reset acknowledgements"):
            await reset_service.ProjectResetService().reset(Db(), project_id, "Timeout")
        generation = relay._reset_markers[project_id]
        assert project.status == "resetting"
        assert calls == [("commit", "resetting")]
        assert relay._reset_targets[project_id, generation] == (monitor.token,)
    finally:
        await relay.unregister_project_reset_monitor(project_id, monitor)
        generation = relay._reset_markers.get(project_id)
        if generation:
            await relay.clear_project_reset(project_id, generation)


@pytest.mark.asyncio
async def test_project_reset_service_clear_compare_failure_rolls_back_prepared_purge(monkeypatch):
    import huddleroom.services.agent_response_relay as relay
    import huddleroom.services.project_reset_service as reset_service

    project_id = uuid.uuid4()
    project = SimpleNamespace(id=project_id, name="Compare", status="active")
    rows, commits = ["must-survive"], []
    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=True))

    class Db:
        bind = None

        async def get(self, *_args, **_kwargs):
            return project

        async def commit(self):
            commits.append((project.status, rows.copy()))

        async def rollback(self):
            project.status, saved_rows = commits[-1]
            rows[:] = saved_rows

    async def lock(*_args):
        return None

    async def noop(*_args, **_kwargs):
        return 0

    async def purge(_service, _db, project_id):
        rows.clear()
        relay._reset_markers[project_id] = "replacement-generation"
        return {"rows": 1}

    monkeypatch.setattr(reset_service.ProjectService, "lock_workspace_boundary", lock)
    monkeypatch.setattr(reset_service.task_runner, "cancel_project_sessions", noop)
    monkeypatch.setattr(reset_service.meeting_tasks, "cancel_project_meeting_tasks", noop)
    monkeypatch.setattr(reset_service.meeting_tasks, "revoke_and_await_project_meeting_tasks", noop)
    monkeypatch.setattr(reset_service.ProjectService, "reset", purge)
    try:
        with pytest.raises(RuntimeError, match="generation changed"):
            await reset_service.ProjectResetService().reset(Db(), project_id, "Compare")
        assert rows == ["must-survive"]
        assert project.status == "resetting"
        assert relay._reset_markers[project_id] == "replacement-generation"
    finally:
        for key in tuple(relay._reset_targets):
            if key[0] == project_id:
                relay._reset_targets.pop(key)
        relay._reset_markers.pop(project_id, None)


@pytest.mark.asyncio
async def test_direct_metrics_keep_omitted_owner_and_completed_call_distinct(monkeypatch, caplog):
    import huddleroom.services.agent_response_relay as relay

    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=True))
    monkeypatch.setattr(relay, "_agent_response_metrics", {"started_total": 0, "terminal_total": 0})
    project_id = uuid.uuid4()

    async def owner():
        call = AgentResponseInvocation(
            InvocationContext(project_id, "agent", "owner", None, "api", "task", "direct"),
        ).call(messages=[])
        await call.__aenter__()
        return call

    omitted = await asyncio.create_task(owner())
    await asyncio.sleep(0)
    assert str(omitted.call_id) in caplog.text
    assert await relay.get_agent_response_metrics() == {
        "started_total": 1, "terminal_total": 0, "omission_gap": 1, "completion_rate": 0.0,
    }
    await omitted._terminal_cleanup()
    completed = AgentResponseInvocation(
        InvocationContext(project_id, "agent", "owner", None, "api", "task", "direct"),
    ).call(messages=[])
    async with completed:
        pass
    assert await relay.get_agent_response_metrics() == {
        "started_total": 2, "terminal_total": 1, "omission_gap": 1, "completion_rate": 0.5,
    }


@pytest.mark.unsupported_mode
def test_fake_redis_clients_and_hub_pubsubs_close_once_per_event_loop(monkeypatch):
    import sys
    import types
    import huddleroom.services.agent_response_relay as relay

    backend, clients = _FakeRedisBackend(), []

    def create_client(*_args, **_kwargs):
        client = _FakeRedisClient(backend)
        clients.append(client)
        return client

    fake_asyncio = types.ModuleType("redis.asyncio")
    fake_asyncio.Redis = type("Redis", (), {"from_url": staticmethod(create_client)})
    fake_redis = types.ModuleType("redis")
    fake_redis.asyncio = fake_asyncio
    monkeypatch.setitem(sys.modules, "redis", fake_redis)
    monkeypatch.setitem(sys.modules, "redis.asyncio", fake_asyncio)
    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=False, redis_url="redis://fake"))

    async def run_one_loop():
        client = relay._redis_client()
        hub = asyncio.create_task(relay.run_agent_response_hub())
        await _until(lambda: bool(backend.subscribers[relay._RESPONSE_CHANNEL]))
        hub.cancel()
        await asyncio.gather(hub, return_exceptions=True)
        await relay.close_agent_response_relay()
        assert client.pubsubs[0].close_calls == 1

    asyncio.run(run_one_loop())
    asyncio.run(run_one_loop())
    assert len(clients) == 2 and clients[0] is not clients[1]
    assert [client.close_calls for client in clients] == [1, 1]


@pytest.mark.asyncio
async def test_lifespan_closes_relay_after_earlier_shutdown_failure(monkeypatch):
    import sqlalchemy.ext.asyncio
    import huddleroom.main as main
    import huddleroom.services.agent_response_relay as relay

    order = []

    class DbCheck:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def begin(self):
            return self

        async def execute(self, *_args, **_kwargs):
            return None

        async def commit(self):
            return None

        async def rollback(self):
            return None

    async def idle_hub(ready):
        ready.set()
        try:
            await asyncio.Future()
        finally:
            order.append("hub")

    def failing_unsubscribe():
        order.append("unsubscribe")
        raise RuntimeError("unsubscribe failed")

    async def close_relay():
        order.append("relay")

    monkeypatch.setattr(main, "settings", SimpleNamespace(
        auth_enabled=False, is_sqlite=True, jwt_secret="safe", cors_origins=["https://example.test"],
    ))
    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=True))
    consumer_tasks = _stub_sqlite_lifespan_runtime(monkeypatch)
    monkeypatch.setattr(main, "validate_supported_settings", lambda _settings: None)
    monkeypatch.setattr(main, "init_db", lambda: asyncio.sleep(0))
    monkeypatch.setattr(main, "register_litellm_debug_logger", lambda: None)
    monkeypatch.setattr(sqlalchemy.ext.asyncio, "AsyncSession", DbCheck)
    monkeypatch.setattr(relay, "subscribe_agent_responses", lambda _callback: failing_unsubscribe)
    monkeypatch.setattr(relay, "run_agent_response_hub", idle_hub)
    monkeypatch.setattr(relay, "close_agent_response_relay", close_relay)

    with pytest.raises(RuntimeError, match="unsubscribe failed"):
        async with main.lifespan(None):
            await asyncio.sleep(0)
    assert order == ["unsubscribe", "hub", "relay"]


@pytest.mark.asyncio
async def test_lifespan_fails_when_response_hub_cannot_subscribe(monkeypatch):
    import sqlalchemy.ext.asyncio
    import huddleroom.main as main
    import huddleroom.services.agent_response_relay as relay

    class DbCheck:
        def __init__(self, *_args, **_kwargs):
            pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return False
        def begin(self): return self
        async def execute(self, *_args, **_kwargs): return None

    closed = []
    async def failing_hub(_ready):
        raise RuntimeError("subscribe failed")

    monkeypatch.setattr(main, "settings", SimpleNamespace(
        auth_enabled=False, is_sqlite=True, jwt_secret="safe", cors_origins=["https://example.test"],
    ))
    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=True))
    _stub_sqlite_lifespan_runtime(monkeypatch)
    monkeypatch.setattr(main, "validate_supported_settings", lambda _settings: None)
    monkeypatch.setattr(main, "init_db", lambda: asyncio.sleep(0))
    monkeypatch.setattr(main, "register_litellm_debug_logger", lambda: None)
    monkeypatch.setattr(sqlalchemy.ext.asyncio, "AsyncSession", DbCheck)
    monkeypatch.setattr(relay, "run_agent_response_hub", failing_hub)
    monkeypatch.setattr(relay, "close_agent_response_relay", lambda: closed.append(True) or asyncio.sleep(0))

    with pytest.raises(RuntimeError, match="subscribe failed"):
        async with main.lifespan(None):
            pass
    assert closed == [True]


@pytest.mark.asyncio
async def test_lifespan_restarts_response_hub_after_unexpected_exit(monkeypatch):
    import sqlalchemy.ext.asyncio
    import huddleroom.main as main
    import huddleroom.services.agent_response_relay as relay

    class DbCheck:
        def __init__(self, *_args, **_kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return False
        def begin(self): return self
        async def execute(self, *_args, **_kwargs): return None

    calls, closed = [], []
    async def hub(ready):
        calls.append(True)
        ready.set()
        if len(calls) == 1:
            return
        await asyncio.Future()

    monkeypatch.setattr(main, "settings", SimpleNamespace(
        auth_enabled=False, is_sqlite=True, jwt_secret="safe", cors_origins=["https://example.test"],
    ))
    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=True))
    _stub_sqlite_lifespan_runtime(monkeypatch)
    monkeypatch.setattr(main, "validate_supported_settings", lambda _settings: None)
    monkeypatch.setattr(main, "init_db", lambda: asyncio.sleep(0))
    monkeypatch.setattr(main, "register_litellm_debug_logger", lambda: None)
    monkeypatch.setattr(sqlalchemy.ext.asyncio, "AsyncSession", DbCheck)
    monkeypatch.setattr(main, "_RESPONSE_HUB_RESTART_DELAY", 0)
    monkeypatch.setattr(relay, "run_agent_response_hub", hub)
    monkeypatch.setattr(relay, "close_agent_response_relay", lambda: closed.append(True) or asyncio.sleep(0))

    async with main.lifespan(None):
        await _until(lambda: len(calls) == 2)
    assert closed == [True]


@pytest.mark.asyncio
async def test_sqlite_lifespan_waits_for_response_hub_and_shuts_down(monkeypatch):
    import sqlalchemy.ext.asyncio
    import huddleroom.main as main
    import huddleroom.services.agent_response_relay as relay

    class DbCheck:
        def __init__(self, *_args, **_kwargs):
            pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return False
        def begin(self): return self
        async def execute(self, *_args, **_kwargs): return None

    scheduler, subscriptions = object(), []
    monkeypatch.setattr(main, "settings", SimpleNamespace(
        auth_enabled=False, is_sqlite=True, jwt_secret="safe", cors_origins=["https://example.test"],
    ))
    monkeypatch.setattr(main, "validate_supported_settings", lambda _settings: None)
    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=True))
    consumer_tasks = _stub_sqlite_lifespan_runtime(monkeypatch)
    monkeypatch.setattr(main, "init_db", lambda: asyncio.sleep(0))
    monkeypatch.setattr(main, "register_litellm_debug_logger", lambda: None)
    monkeypatch.setattr(sqlalchemy.ext.asyncio, "AsyncSession", DbCheck)
    monkeypatch.setattr("huddleroom.workers.scheduler.start_scheduler", lambda: scheduler)
    monkeypatch.setattr("huddleroom.workers.scheduler.stop_scheduler", lambda: None)
    monkeypatch.setattr("huddleroom.workers.consumers.get_consumer_tasks", lambda: consumer_tasks)
    monkeypatch.setattr(
        relay, "subscribe_agent_responses",
        lambda callback: subscriptions.append(callback) or (lambda: subscriptions.remove(callback)),
    )

    async with asyncio.timeout(0.1):
        async with main.lifespan(None):
            assert len(subscriptions) == 1
    assert subscriptions == []


@pytest.mark.asyncio
async def test_lifespan_cancellation_before_hub_ready_cleans_startup_resources(monkeypatch):
    import sqlalchemy.ext.asyncio
    import huddleroom.main as main
    import huddleroom.services.agent_response_relay as relay

    class DbCheck:
        def __init__(self, *_args, **_kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *_args): return False
        def begin(self): return self
        async def execute(self, *_args, **_kwargs): return None

    subscribers, hub_cancelled, client_closed, scheduler_stops = [], asyncio.Event(), [], []
    async def blocked_hub(_ready):
        try:
            await asyncio.Future()
        finally:
            hub_cancelled.set()

    def subscribe(callback):
        subscribers.append(callback)
        return lambda: subscribers.remove(callback)

    async def close_relay():
        client_closed.append(True)

    monkeypatch.setattr(main, "settings", SimpleNamespace(
        auth_enabled=False, is_sqlite=True, jwt_secret="safe", cors_origins=["https://example.test"],
    ))
    monkeypatch.setattr(main, "validate_supported_settings", lambda _settings: None)
    monkeypatch.setattr(relay, "settings", SimpleNamespace(is_sqlite=True))
    _stub_sqlite_lifespan_runtime(monkeypatch)
    monkeypatch.setattr(main, "init_db", lambda: asyncio.sleep(0))
    monkeypatch.setattr(main, "register_litellm_debug_logger", lambda: None)
    monkeypatch.setattr(sqlalchemy.ext.asyncio, "AsyncSession", DbCheck)
    monkeypatch.setattr(relay, "subscribe_agent_responses", subscribe)
    monkeypatch.setattr(relay, "run_agent_response_hub", blocked_hub)
    monkeypatch.setattr(relay, "close_agent_response_relay", close_relay)
    monkeypatch.setattr("huddleroom.workers.scheduler.start_scheduler", lambda: object())
    monkeypatch.setattr("huddleroom.workers.scheduler.stop_scheduler", lambda: scheduler_stops.append(True))

    startup = asyncio.create_task(main.lifespan(None).__aenter__())
    async with asyncio.timeout(0.1):
        while not subscribers:
            await asyncio.sleep(0.001)
    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup
    await asyncio.wait_for(hub_cancelled.wait(), timeout=0.1)
    assert subscribers == []
    assert client_closed == [True]
    assert scheduler_stops == [True]


@pytest.mark.unsupported_mode
def test_celery_session_and_meeting_wrappers_close_relay_before_their_loop(monkeypatch):
    import importlib
    import sys
    import types
    import huddleroom.workers.meeting_tasks as meeting_tasks
    import huddleroom.workers.session_tasks as session_tasks
    import huddleroom.services.agent_response_relay as relay

    class App:
        def __init__(self):
            self.tasks = {}

        def task(self, **options):
            def decorate(function):
                self.tasks[options["name"]] = function
                return function
            return decorate

    class RecordingLoop:
        def __init__(self, loop, order):
            self.loop, self.order = loop, order

        def run_until_complete(self, coroutine):
            return self.loop.run_until_complete(coroutine)

        def close(self):
            self.order.append("loop")
            self.loop.close()

    order, app, original_new_loop = [], App(), asyncio.new_event_loop
    session_before = dict(session_tasks.__dict__)
    meeting_before = {name: getattr(meeting_tasks, name) for name in (
        "start_meeting", "run_meeting_turn", "resume_meeting_turn", "finalize_meeting", "meeting_timeout", "_meeting_task_app",
    )}

    async def done(*_args, **_kwargs):
        return None

    async def close_relay():
        order.append("relay")

    def new_loop():
        return RecordingLoop(original_new_loop(), order)

    try:
        monkeypatch.setattr(relay, "close_agent_response_relay", close_relay)
        monkeypatch.setattr(asyncio, "new_event_loop", new_loop)
        monkeypatch.setitem(sys.modules, "huddleroom.workers.celery_app", types.SimpleNamespace(app=app))
        celery_exceptions = types.ModuleType("celery.exceptions")
        celery_exceptions.MaxRetriesExceededError = type("MaxRetriesExceededError", (Exception,), {})
        celery_exceptions.Retry = type("Retry", (Exception,), {})
        celery = types.ModuleType("celery")
        celery.exceptions = celery_exceptions
        monkeypatch.setitem(sys.modules, "celery", celery)
        monkeypatch.setitem(sys.modules, "celery.exceptions", celery_exceptions)
        monkeypatch.setattr(session_tasks, "execute_api_session", done)
        importlib.reload(session_tasks)
        monkeypatch.setattr(session_tasks, "execute_api_session", done)
        app.tasks["rally.workers.session_tasks.run_api_session"](
            SimpleNamespace(request=SimpleNamespace(id="runner-id")), "session"
        )

        meeting_tasks.register_tasks(app)
        monkeypatch.setattr(meeting_tasks, "start_meeting_async", done)
        app.tasks["rally.meeting.start"](None, "meeting")
        assert order == ["relay", "loop", "relay", "loop"]
    finally:
        for name in ("run_api_session", "run_cli_session"):
            if name in session_before:
                setattr(session_tasks, name, session_before[name])
            else:
                session_tasks.__dict__.pop(name, None)
        for name, value in meeting_before.items():
            setattr(meeting_tasks, name, value)
