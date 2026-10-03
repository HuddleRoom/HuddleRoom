from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.artifact import Artifact
from huddleroom.models.protocol import Protocol, ProtocolInstance, ProtocolTimeout, ProtocolTransition
from huddleroom.models.session import Session
from huddleroom.models.task import Task
from huddleroom.services.action_executor import ActionExecutor
from huddleroom.services.agent_service import GLOBAL_PROJECT_ID
from huddleroom.services.actor_resolver import ActorResolver
from huddleroom.services.escalation_service import EscalationChainService
from huddleroom.services.event_bus import BusEvent, EventBusService, emit_event, emit_event_once
from huddleroom.services.guard_evaluator import GuardEvaluator
from huddleroom.services.protocol_service import ProtocolService
from huddleroom.services.project_service import ProjectService
from huddleroom.services.template_resolver import TemplateResolver

logger = logging.getLogger(__name__)

_DURATION_MAP = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _parse_duration(duration_str: str) -> int:
    value = (duration_str or "1h").strip().lower()
    for suffix, multiplier in _DURATION_MAP.items():
        if value.endswith(suffix):
            try:
                return int(value[:-1]) * multiplier
            except ValueError:
                return 3600
    return 3600


class ProtocolEngineService:
    def __init__(self, _bus: EventBusService | None = None) -> None:
        self._bus = _bus
        self._protocol_service = ProtocolService()
        self._actor_resolver = ActorResolver()
        self._guard_evaluator = GuardEvaluator()
        self._action_executor = ActionExecutor(_bus=_bus)
        self._template_resolver = TemplateResolver()

    async def process_event(self, db: AsyncSession, event: BusEvent) -> None:
        if event.project_id == GLOBAL_PROJECT_ID:
            return
        project_service = ProjectService()
        await project_service.lock_workspace_boundary(db, event.project_id)
        await project_service.require_runnable_project(db, event.project_id)
        enriched = BusEvent(
            id=event.id,
            project_id=event.project_id,
            event_type=event.event_type,
            payload=await self._enrich_event_payload(db, event.project_id, event.payload, event.event_type),
            source=event.source,
            emitted_at=event.emitted_at,
        )
        await self._check_triggers(db, enriched)
        await self._evaluate_active_instances(db, enriched)
        derived = await self._maybe_auto_emit_review_protocol_event(db, enriched)
        if derived is not None:
            await self._check_triggers(db, derived)
            await self._evaluate_active_instances(db, derived)

    async def _enrich_event_payload(
        self,
        db: AsyncSession,
        project_id: uuid.UUID,
        payload: dict,
        event_type: str | None = None,
    ) -> dict:
        enriched = dict(payload or {})
        raw_payload = dict(payload or {})
        metadata = {}
        payload_metadata = dict(enriched.get("metadata") or {})

        task_id = self._uuid_from_payload(enriched.get("task_id"))
        artifact_id = self._uuid_from_payload(enriched.get("artifact_id"))
        session_id = self._uuid_from_payload(enriched.get("session_id"))
        protocol_instance_id = self._uuid_from_payload(enriched.get("protocol_instance_id"))
        canonical_protocol_instance_id = protocol_instance_id

        session = None
        session_payload = None
        if session_id is not None:
            result = await db.execute(select(Session).where(Session.id == session_id, Session.project_id == project_id))
            session = result.scalar_one_or_none()
            if session is not None:
                task_id = task_id or session.task_id
                protocol_instance_id = protocol_instance_id or session.protocol_instance_id
                canonical_protocol_instance_id = canonical_protocol_instance_id or session.protocol_instance_id
                metadata = {**metadata, **(session.metadata_ or {})}
                enriched["session_id"] = str(session.id)
                session_payload = {
                    "id": str(session.id),
                    "task_id": str(session.task_id) if session.task_id else None,
                    "agent_id": str(session.agent_id),
                    "project_id": str(session.project_id),
                    "protocol_instance_id": (
                        str(session.protocol_instance_id) if session.protocol_instance_id else None
                    ),
                    "status": session.status,
                    "metadata": session.metadata_ or {},
                    "origin": session.origin,
                }
                enriched["session"] = session_payload

        task = None
        if task_id is not None:
            result = await db.execute(select(Task).where(Task.id == task_id, Task.project_id == project_id))
            task = result.scalar_one_or_none()
            if task is not None:
                protocol_instance_id = protocol_instance_id or task.protocol_instance_id
                canonical_protocol_instance_id = canonical_protocol_instance_id or task.protocol_instance_id
                metadata = {**(task.metadata_ or {}), **metadata}
                enriched["task_id"] = str(task.id)
                enriched["task"] = {
                    "id": str(task.id),
                    "project_id": str(task.project_id),
                    "parent_id": str(task.parent_id) if task.parent_id else None,
                    "protocol_instance_id": str(task.protocol_instance_id) if task.protocol_instance_id else None,
                    "title": task.title,
                    "status": task.status,
                    "priority": task.priority,
                    "assigned_to": str(task.assigned_to) if task.assigned_to else None,
                    "metadata": task.metadata_ or {},
                }
                if "status" in enriched and "new_status" not in enriched:
                    enriched["new_status"] = enriched["status"]

        artifact = None
        if artifact_id is not None:
            result = await db.execute(select(Artifact).where(Artifact.id == artifact_id, Artifact.project_id == project_id))
            artifact = result.scalar_one_or_none()
            if artifact is not None:
                task_id = task_id or artifact.linked_task_id
                artifact_metadata = dict(artifact.metadata_ or {})
                artifact_metadata.setdefault("type", artifact.artifact_type)
                if artifact.linked_task_id is not None:
                    artifact_metadata.setdefault("task_id", str(artifact.linked_task_id))
                metadata = {**artifact_metadata, **metadata}
                enriched["artifact_id"] = str(artifact.id)
                enriched["artifact"] = {
                    "id": str(artifact.id),
                    "project_id": str(artifact.project_id),
                    "linked_task_id": str(artifact.linked_task_id) if artifact.linked_task_id else None,
                    "name": artifact.name,
                    "artifact_type": artifact.artifact_type,
                    "status": artifact.status,
                    "path": artifact.path,
                    "url": artifact.url,
                    "metadata": artifact.metadata_ or {},
                }

        instance = None
        instance_protocol = None
        if protocol_instance_id is not None:
            result = await db.execute(
                select(ProtocolInstance).where(
                    ProtocolInstance.id == protocol_instance_id,
                    ProtocolInstance.project_id == project_id,
                )
            )
            instance = result.scalar_one_or_none()
        if instance is None and task_id is not None:
            result = await db.execute(
                select(ProtocolInstance).where(
                    ProtocolInstance.project_id == project_id,
                    ProtocolInstance.linked_task_id == task_id,
                    ProtocolInstance.status == "active",
                )
            )
            instance = result.scalars().first()
        if instance is None and artifact_id is not None:
            result = await db.execute(
                select(ProtocolInstance).where(
                    ProtocolInstance.project_id == project_id,
                    ProtocolInstance.artifact_id == artifact_id,
                    ProtocolInstance.status == "active",
                )
            )
            instance = result.scalars().first()
        if instance is not None:
            protocol_result = await db.execute(select(Protocol).where(Protocol.id == instance.protocol_id))
            instance_protocol = protocol_result.scalar_one_or_none()
            if canonical_protocol_instance_id is not None:
                enriched["protocol_instance_id"] = str(canonical_protocol_instance_id)
            enriched["protocol_instance"] = {
                "id": str(instance.id),
                "protocol_id": str(instance.protocol_id),
                "project_id": str(instance.project_id),
                "linked_task_id": str(instance.linked_task_id) if instance.linked_task_id else None,
                "artifact_id": str(instance.artifact_id) if instance.artifact_id else None,
                "current_state": instance.current_state,
                "status": instance.status,
                "context": instance.context or {},
            }
            if self._should_enrich_review_session(event_type, session, session_payload, instance, instance_protocol):
                session_payload["output"] = session.output
                enriched["review_outcome"] = self._parse_review_outcome(session.output)

        enriched["metadata"] = {**metadata, **payload_metadata}
        enriched["raw_payload"] = raw_payload

        # Write CLI session output into protocol instance context for template/guard access
        if (
            session is not None
            and instance is not None
            and session.status == "completed"
            and session.output
        ):
            instance.context = {
                **(instance.context or {}),
                "last_cli_output": session.output,
            }
            await db.flush()

        return enriched

    async def _check_triggers(self, db: AsyncSession, event: BusEvent) -> None:
        scoped = await self._protocol_service.get_active_protocols_for_event(db, event.event_type, project_id=event.project_id)
        global_protocols = await self._protocol_service.get_active_protocols_for_event(db, event.event_type, project_id=None)
        protocols = {protocol.id: protocol for protocol in [*scoped, *global_protocols]}.values()

        for protocol in protocols:
            for trigger in protocol.triggers:
                if trigger.get("event_type") != event.event_type:
                    continue
                if not self._guard_evaluator.evaluate(trigger.get("conditions") or {}, event.payload):
                    continue
                await self.start_protocol(db, protocol, event)
                break

    async def start_protocol(
        self,
        db: AsyncSession,
        protocol: Protocol,
        event: BusEvent,
    ) -> ProtocolInstance:
        definition = protocol.definition or {}
        initial_state = definition.get("initial_state", "")
        actor_assignments = await self._actor_resolver.resolve_all_slots(
            db,
            definition.get("actors") or {},
            triggering_event_payload=event.payload,
        )

        linked_task_id = self._uuid_from_payload(event.payload.get("task_id"))
        artifact_id = self._uuid_from_payload(event.payload.get("artifact_id"))
        existing = await self._find_existing_active_instance(
            db,
            protocol_id=protocol.id,
            project_id=event.project_id,
            artifact_id=artifact_id,
            linked_task_id=linked_task_id,
            triggering_event_id=event.id,
        )
        if existing is not None:
            logger.info(
                "Skipping duplicate protocol instance for protocol=%s project=%s artifact=%s task=%s",
                protocol.name,
                event.project_id,
                artifact_id,
                linked_task_id,
            )
            return existing

        await ProjectService().require_runnable_project(db, event.project_id)
        instance = await self._protocol_service.create_instance(
            db,
            protocol=protocol,
            project_id=event.project_id,
            initial_state=initial_state,
            linked_task_id=linked_task_id,
            artifact_id=artifact_id,
            triggering_event_id=event.id,
        )
        instance.actor_assignments = {
            role: assignment for role, assignment in actor_assignments.items() if assignment is not None
        }
        instance.context = {
            **(instance.context or {}),
            "trigger_event_type": event.event_type,
        }
        await db.flush()

        await emit_event(
            db,
            event.project_id,
            "protocol.instance_started",
            {
                "protocol_instance_id": str(instance.id),
                "protocol_name": protocol.name,
                "initial_state": initial_state,
                "project_id": str(event.project_id),
            },
            _bus=self._bus,
        )

        await self._record_transition(
            db,
            instance,
            from_state="",
            to_state=initial_state,
            trigger_event_id=event.id,
            trigger_reason="event",
        )

        state_def = (definition.get("states") or {}).get(initial_state, {})
        await self._execute_on_enter(db, state_def, instance)
        await self._register_timeout(db, state_def, instance)
        return instance

    async def _find_existing_active_instance(
        self,
        db: AsyncSession,
        protocol_id: uuid.UUID,
        project_id: uuid.UUID,
        artifact_id: uuid.UUID | None,
        linked_task_id: uuid.UUID | None,
        triggering_event_id: uuid.UUID | None,
    ) -> ProtocolInstance | None:
        conditions = [
            ProtocolInstance.protocol_id == protocol_id,
            ProtocolInstance.project_id == project_id,
            ProtocolInstance.status == "active",
        ]
        if artifact_id is not None:
            conditions.append(ProtocolInstance.artifact_id == artifact_id)
        elif linked_task_id is not None:
            conditions.append(ProtocolInstance.linked_task_id == linked_task_id)
        elif triggering_event_id is not None:
            conditions.append(ProtocolInstance.triggering_event_id == triggering_event_id)
        else:
            return None

        result = await db.execute(select(ProtocolInstance).where(*conditions))
        return result.scalars().first()

    async def _evaluate_active_instances(self, db: AsyncSession, event: BusEvent) -> None:
        result = await db.execute(
            select(ProtocolInstance).where(
                ProtocolInstance.project_id == event.project_id,
                ProtocolInstance.status == "active",
            )
        )
        for instance in result.scalars().all():
            await self._evaluate_instance(db, instance, event)

    async def _evaluate_instance(self, db: AsyncSession, instance: ProtocolInstance, event: BusEvent) -> None:
        event_instance_id = self._uuid_from_payload(event.payload.get("protocol_instance_id"))
        if event_instance_id is not None and event_instance_id != instance.id:
            return

        result = await db.execute(select(Protocol).where(Protocol.id == instance.protocol_id))
        protocol = result.scalar_one_or_none()
        if protocol is None:
            return

        states = (protocol.definition or {}).get("states") or {}
        current_state_def = states.get(instance.current_state) or {}
        for transition in current_state_def.get("transitions") or []:
            if transition.get("trigger_event") != event.event_type:
                continue
            resolved_guard = await self._resolve_guard(db, transition.get("guard") or {}, instance)
            if not self._guard_evaluator.evaluate(resolved_guard, event.payload):
                continue
            await self._fire_transition(
                db,
                instance,
                protocol,
                from_state=instance.current_state,
                to_state=transition["to"],
                transition_name=transition.get("name"),
                actions=transition.get("actions") or [],
                trigger_event_id=event.id,
                trigger_reason="event",
            )
            break

    async def process_timeouts(self, db: AsyncSession) -> int:
        now = datetime.now(timezone.utc)
        result = await db.execute(
            select(ProtocolTimeout).where(
                ProtocolTimeout.resolved.is_(False),
                ProtocolTimeout.expires_at <= now,
            )
        )
        timeouts = result.scalars().all()
        for timeout in timeouts:
            await self._process_timeout(db, timeout, now)
        return len(timeouts)

    async def _process_timeout(self, db: AsyncSession, timeout: ProtocolTimeout, now: datetime) -> None:
        result = await db.execute(
            select(ProtocolInstance).where(
                ProtocolInstance.id == timeout.protocol_instance_id,
                ProtocolInstance.status == "active",
            )
        )
        instance = result.scalar_one_or_none()
        if instance is None:
            timeout.resolved = True
            timeout.resolved_at = now
            await db.flush()
            return

        await ProjectService().require_runnable_project(db, instance.project_id)
        protocol_result = await db.execute(select(Protocol).where(Protocol.id == instance.protocol_id))
        protocol = protocol_result.scalar_one_or_none()
        protocol_name = protocol.name if protocol is not None else ""
        new_step = (instance.escalation_step or 0) + 1

        emitted_alert = False
        if protocol is not None and protocol.escalation_chain:
            escalation_service = EscalationChainService()
            chain = await escalation_service.get_by_name(db, protocol.escalation_chain, project_id=instance.project_id)
            if chain is None:
                chain = await escalation_service.get_by_name(db, protocol.escalation_chain, project_id=None)
            steps = chain.steps if chain is not None else []
            instance.escalation_step = min(new_step, max(len(steps), 1))
            step_def = next((step for step in steps if step.get("step") == instance.escalation_step), None)
            if step_def and step_def.get("action") == "human_notify":
                await escalation_service.notify_humans(
                    db,
                    instance.project_id,
                    instance.id,
                    protocol_name,
                    instance.current_state,
                    step_def.get("message_template", ""),
                )
                emitted_alert = True

        else:
            instance.escalation_step = new_step

        if not emitted_alert:
            await emit_event(
                db,
                instance.project_id,
                "protocol.escalated",
                {
                    "protocol_instance_id": str(instance.id),
                    "protocol_name": protocol_name,
                    "current_state": instance.current_state,
                    "timeout_id": str(timeout.id),
                    "timeout_action": timeout.timeout_action,
                    "escalation_step": instance.escalation_step,
                },
                _bus=self._bus,
            )

        timeout.resolved = True
        timeout.resolved_at = now
        await db.flush()

    async def _resolve_guard(self, db: AsyncSession, guard: dict, instance: ProtocolInstance) -> dict:
        resolved = {}
        for key, value in guard.items():
            if isinstance(value, str):
                resolved[key] = await self._template_resolver.resolve(db, value, instance)
            elif isinstance(value, dict):
                resolved[key] = await self._template_resolver.resolve_dict(db, value, instance)
            else:
                resolved[key] = value
        return resolved

    async def _fire_transition(
        self,
        db: AsyncSession,
        instance: ProtocolInstance,
        protocol: Protocol,
        from_state: str,
        to_state: str,
        transition_name: str | None,
        actions: list[dict],
        trigger_event_id: uuid.UUID | None = None,
        trigger_reason: str = "event",
    ) -> None:
        await ProjectService().require_runnable_project(db, instance.project_id)
        await self._resolve_timeout(db, instance)

        now = datetime.now(timezone.utc)
        instance.current_state = to_state
        instance.last_transitioned_at = now

        terminal_states = (protocol.definition or {}).get("terminal_states") or {}
        if to_state in (terminal_states.get("success") or []):
            instance.status = "completed"
            instance.completed_at = now
        elif to_state in (terminal_states.get("failure") or []):
            instance.status = "failed"
            instance.completed_at = now
        await db.flush()

        executed_actions = await self._action_executor.execute_all(db, actions, instance)
        transition = await self._record_transition(
            db,
            instance,
            from_state=from_state,
            to_state=to_state,
            transition_name=transition_name,
            trigger_event_id=trigger_event_id,
            trigger_reason=trigger_reason,
            actions_executed=executed_actions,
        )

        state_def = ((protocol.definition or {}).get("states") or {}).get(to_state, {})
        await self._execute_on_enter(db, state_def, instance)
        await self._register_timeout(db, state_def, instance)

        event_type = "protocol.state_transitioned"
        if instance.status == "completed":
            event_type = "protocol.completed"
        elif instance.status == "failed":
            event_type = "protocol.failed"

        await emit_event(
            db,
            instance.project_id,
            event_type,
            {
                "protocol_instance_id": str(instance.id),
                "protocol_transition_id": str(transition.id),
                "protocol_name": protocol.name,
                "from_state": from_state,
                "to_state": to_state,
                "transition_name": transition_name,
                "project_id": str(instance.project_id),
            },
            _bus=self._bus,
        )

    async def _execute_on_enter(self, db: AsyncSession, state_def: dict, instance: ProtocolInstance) -> None:
        on_enter = state_def.get("on_enter") or []
        if on_enter:
            await self._action_executor.execute_all(db, on_enter, instance)

    async def _record_transition(
        self,
        db: AsyncSession,
        instance: ProtocolInstance,
        from_state: str,
        to_state: str,
        transition_name: str | None = None,
        trigger_event_id: uuid.UUID | None = None,
        trigger_reason: str | None = None,
        actions_executed: list | None = None,
    ) -> ProtocolTransition:
        transition = ProtocolTransition(
            protocol_instance_id=instance.id,
            from_state=from_state,
            to_state=to_state,
            transition_name=transition_name,
            trigger_event_id=trigger_event_id,
            trigger_reason=trigger_reason,
            actions_executed=actions_executed or [],
        )
        db.add(transition)
        await db.flush()
        return transition

    async def _register_timeout(self, db: AsyncSession, state_def: dict, instance: ProtocolInstance) -> None:
        timeout = state_def.get("timeout")
        if not timeout:
            return
        record = ProtocolTimeout(
            protocol_instance_id=instance.id,
            state_name=instance.current_state,
            timeout_action=timeout.get("action", "escalate"),
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=_parse_duration(timeout.get("duration", "1h"))),
        )
        db.add(record)
        await db.flush()

    async def _resolve_timeout(self, db: AsyncSession, instance: ProtocolInstance) -> None:
        result = await db.execute(
            select(ProtocolTimeout).where(
                ProtocolTimeout.protocol_instance_id == instance.id,
                ProtocolTimeout.resolved.is_(False),
            )
        )
        for timeout in result.scalars().all():
            timeout.resolved = True
            timeout.resolved_at = datetime.now(timezone.utc)
        await db.flush()

    async def advance_manually(
        self,
        db: AsyncSession,
        instance_or_id: ProtocolInstance | uuid.UUID,
        to_state: str,
        reason: str | None = None,
    ) -> ProtocolInstance:
        from fastapi import HTTPException

        if isinstance(instance_or_id, ProtocolInstance):
            instance = instance_or_id
            if instance.status != "active":
                raise HTTPException(status_code=409, detail=f"Protocol instance is {instance.status}")
        else:
            result = await db.execute(select(ProtocolInstance).where(ProtocolInstance.id == instance_or_id))
            instance = result.scalar_one_or_none()
            if instance is None:
                raise HTTPException(status_code=404, detail="Protocol instance not found")
            if instance.status != "active":
                raise HTTPException(status_code=409, detail=f"Protocol instance is {instance.status}")

        proto_result = await db.execute(select(Protocol).where(Protocol.id == instance.protocol_id))
        protocol = proto_result.scalar_one_or_none()
        if protocol is None:
            raise HTTPException(status_code=404, detail="Protocol not found")
        states = (protocol.definition or {}).get("states") or {}
        if to_state not in states:
            raise HTTPException(status_code=400, detail=f"Unknown protocol state: {to_state}")
        await self._fire_transition(
            db,
            instance,
            protocol,
            from_state=instance.current_state,
            to_state=to_state,
            transition_name="manual_advance",
            actions=[],
            trigger_reason=reason or "manual",
        )
        return instance

    def _uuid_from_payload(self, value: str | None) -> uuid.UUID | None:
        if not value:
            return None
        try:
            return uuid.UUID(str(value))
        except ValueError:
            return None

    def _parse_review_outcome(self, raw_output: str | None) -> dict:
        normalized_output = (raw_output or "").upper()
        if self._is_changes_requested_review_output(normalized_output):
            verdict = "changes_requested"
        elif self._is_approved_review_output(normalized_output):
            verdict = "approved"
        else:
            verdict = "commented"
        return {
            "verdict": verdict,
            "raw_output": raw_output,
        }

    def _should_enrich_review_session(
        self,
        event_type: str | None,
        session: Session | None,
        session_payload: dict | None,
        instance: ProtocolInstance | None,
        protocol: Protocol | None,
    ) -> bool:
        return (
            event_type == "session.completed"
            and session is not None
            and session.origin == "protocol"
            and session_payload is not None
            and instance is not None
            and instance.current_state == "ready_for_review"
            and protocol is not None
            and protocol.name == "code_review"
            and self._session_matches_assigned_review_actor(instance, session)
        )

    def _is_approved_review_output(self, normalized_output: str) -> bool:
        if not normalized_output:
            return False
        if re.search(r"\b(?:DO\s+NOT|NOT)\s+APPROVE(?:D)?\b", normalized_output):
            return False
        if re.search(r"\bDISAPPROVE\b", normalized_output):
            return False
        return re.search(r"\bAPPROVE(?:D)?\b", normalized_output) is not None

    def _is_changes_requested_review_output(self, normalized_output: str) -> bool:
        if not normalized_output:
            return False
        return re.search(r"\b(?:CHANGES[_\s]+REQUESTED|REQUEST(?:ED)?\s+CHANGES)\b", normalized_output) is not None

    def _session_matches_assigned_review_actor(self, instance: ProtocolInstance, session: Session) -> bool:
        reviewer_assignment = (instance.actor_assignments or {}).get("reviewer") or {}
        if reviewer_assignment.get("kind") != "agent":
            return False
        reviewer_id = self._uuid_from_payload(reviewer_assignment.get("id"))
        return reviewer_id is not None and reviewer_id == session.agent_id

    async def _maybe_auto_emit_review_protocol_event(self, db: AsyncSession, event: BusEvent) -> BusEvent | None:
        if event.event_type != "session.completed":
            return None

        outcome = event.payload.get("review_outcome") or {}
        event_type = self._review_protocol_event_type(outcome.get("verdict"))
        if event_type is None:
            return None

        session_id = self._uuid_from_payload(event.payload.get("session_id"))
        protocol_instance_id = self._protocol_instance_uuid_from_payload(event.payload)
        artifact_id = self._artifact_uuid_from_payload(event.payload)
        if session_id is None or protocol_instance_id is None or artifact_id is None:
            return None

        result = await db.execute(select(Session).where(Session.id == session_id, Session.project_id == event.project_id))
        session = result.scalar_one_or_none()
        if session is None:
            return None
        instance_result = await db.execute(
            select(ProtocolInstance).where(
                ProtocolInstance.id == protocol_instance_id,
                ProtocolInstance.project_id == event.project_id,
            )
        )
        instance = instance_result.scalar_one_or_none()
        if instance is None or not self._session_matches_assigned_review_actor(instance, session):
            return None

        payload = {
            "artifact_id": str(artifact_id),
            "protocol_instance_id": str(protocol_instance_id),
            "session_id": str(session.id),
        }
        if event.payload.get("task_id"):
            payload["task_id"] = event.payload["task_id"]
        if outcome:
            payload["review_outcome"] = outcome

        emitted, created = await emit_event_once(
            db,
            event.project_id,
            event_type,
            payload,
            source="protocol",
            dedup_key=self._derived_review_event_dedup_key(session.id, event_type),
            _bus=self._bus,
        )
        session.metadata_ = {
            **(session.metadata_ or {}),
            "auto_review_protocol_event": {
                "event_type": event_type,
                "event_id": str(emitted.id),
                "protocol_instance_id": str(protocol_instance_id),
            },
        }
        await db.flush()
        if not created:
            return None
        return emitted

    def _review_protocol_event_type(self, verdict: str | None) -> str | None:
        if verdict == "approved":
            return "review.approved"
        if verdict == "changes_requested":
            return "review.changes_requested"
        return None

    def _derived_review_event_dedup_key(self, session_id: uuid.UUID, event_type: str) -> str:
        return f"derived_review:{event_type}:{session_id}"

    def _protocol_instance_uuid_from_payload(self, payload: dict) -> uuid.UUID | None:
        protocol_instance_id = self._uuid_from_payload(payload.get("protocol_instance_id"))
        if protocol_instance_id is not None:
            return protocol_instance_id
        instance = payload.get("protocol_instance") or {}
        return self._uuid_from_payload(instance.get("id"))

    def _artifact_uuid_from_payload(self, payload: dict) -> uuid.UUID | None:
        artifact_id = self._uuid_from_payload(payload.get("artifact_id"))
        if artifact_id is not None:
            return artifact_id
        instance = payload.get("protocol_instance") or {}
        return self._uuid_from_payload(instance.get("artifact_id"))
