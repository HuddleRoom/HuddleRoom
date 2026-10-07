from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone

import litellm
from sqlalchemy import func, insert, literal, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.models.agent import Agent
from huddleroom.models.knowledge_item import KnowledgeItem
from huddleroom.models.meeting import (
    Meeting, MeetingActionItem, MeetingAgendaItem, MeetingDecision, MeetingEvent, MeetingTurn,
)
from huddleroom.models.task import Task
from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext
from huddleroom.services.litellm_models import build_litellm_model_name
from huddleroom.services.llm_structured_repair import complete_with_repair

logger = logging.getLogger(__name__)

_OUTCOME_MODEL = settings.meeting_control_model or settings.orchestration_model
_SQLITE_LOCK_RETRIES = 3


async def _outcome_preamble(db, meeting) -> str:
    """HuddleRoom + meeting framing prepended to meeting-control orchestrator prompts."""
    from huddleroom.models.project import Project
    from huddleroom.services.orchestration_llm_decision_adapter import orchestrator_preamble
    proj = await db.get(Project, meeting.project_id)
    proj_ctx = {"name": proj.name, "description": proj.description} if proj else None
    meeting_ctx = {"title": meeting.title, "meeting_type": meeting.meeting_type}
    return orchestrator_preamble(proj_ctx, meeting=meeting_ctx) + "\n\n"


class MeetingOutcomeService:

    async def extract_action_items(
        self,
        db: AsyncSession,
        meeting: Meeting,
        decisions: list[MeetingDecision] | None = None,
    ) -> list[MeetingActionItem]:
        if meeting.meeting_type == "standup":
            return await self._fallback_action_items(db=db, meeting=meeting, decisions=decisions)

        transcript = await self._format_transcript(db, meeting.id)
        if not transcript:
            return []

        # Build participant names
        participant_names: list[str] = []
        for aid_str in meeting.participant_agent_ids:
            try:
                a = await db.get(Agent, uuid.UUID(aid_str))
                if a:
                    participant_names.append(a.name)
            except (ValueError, AttributeError):
                pass

        # Build decisions text
        decisions_block = ""
        if decisions:
            decision_lines = [
                f"- [{d.title}]: {d.chosen_option}"
                for d in decisions
            ]
            decisions_block = "DECISIONS MADE:\n" + "\n".join(decision_lines) + "\n\n"

        _pre = await _outcome_preamble(db, meeting)

        # Local parse function that raises on malformed JSON (but allows empty list)
        def parse_action_items_json(raw: str) -> list[dict]:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return []  # No JSON found is legitimate (no action items)
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise ValueError("Action items JSON is malformed") from exc

        try:
            request = {
                "model": _OUTCOME_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": _pre + (
                            "Extract concrete action items from this meeting transcript. "
                            "Each action item must: have a clear owner, be actionable, and result from "
                            "an explicit commitment or decision made in the discussion. "
                            "Be specific enough that a task can be opened from each item."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"MEETING: {meeting.title}\n"
                            f"MEETING TYPE: {meeting.meeting_type}\n"
                            f"PARTICIPANTS: {', '.join(participant_names)}\n\n"
                            f"{decisions_block}"
                            f"TRANSCRIPT:\n{transcript}\n\n"
                            'Output JSON array: [{"description": "imperative sentence, max 120 chars", '
                            '"assignee_agent_name": "<exact participant name or null>", '
                            '"priority": 70, "deadline_days": 3, '
                            '"depends_on_decision_title": "<decision title from DECISIONS MADE above, or null>"}]\n'
                            "Return [] if no action items."
                        ),
                    },
                ],
                "temperature": 0.0,
                "max_tokens": 800,
            }

            # Create invocation context for action items extraction
            action_items_request_prompt = {
                "operation": "meeting_action_items",
                "title": meeting.title,
            }
            action_items_invocation_ctx = InvocationContext(
                meeting.project_id,
                actor_kind="system",
                actor_id="orchestrator",
                actor_label="Orchestrator",
                invocation_kind="api",
                operation="meeting_action_items",
                model_or_runtime=_OUTCOME_MODEL.split("/")[-1],
                request_prompt=action_items_request_prompt,
            )
            action_items_invocation = AgentResponseInvocation(action_items_invocation_ctx)

            items_data = await complete_with_repair(
                litellm.acompletion,
                request,
                parse_action_items_json,
                invocation=action_items_invocation,
            )
        except Exception as exc:
            logger.warning("extract_action_items LLM call failed: %s", exc)
            return await self._fallback_action_items(db=db, meeting=meeting, decisions=decisions)

        if not items_data:
            return await self._fallback_action_items(db=db, meeting=meeting, decisions=decisions)

        # Build name→agent_id map
        agent_name_map: dict[str, uuid.UUID] = {}
        for aid_str in meeting.participant_agent_ids:
            try:
                a = await db.get(Agent, uuid.UUID(aid_str))
                if a:
                    agent_name_map[a.name.lower()] = a.id
            except (ValueError, AttributeError):
                pass

        # Build decision title → id map for depends_on_decision_id
        decision_title_map: dict[str, uuid.UUID] = {}
        if decisions:
            for d in decisions:
                decision_title_map[d.title.lower()] = d.id

        created: list[MeetingActionItem] = []
        for item_data in items_data:
            assignee_name = (item_data.get("assignee_agent_name") or "").lower()
            assignee_agent_id = agent_name_map.get(assignee_name)
            deadline_days = item_data.get("deadline_days")
            deadline_at: datetime | None = None
            if deadline_days is not None:
                deadline_at = datetime.now(timezone.utc) + timedelta(days=deadline_days)

            depends_title = (item_data.get("depends_on_decision_title") or "").lower()
            depends_on_decision_id = decision_title_map.get(depends_title)

            action_item = MeetingActionItem(
                meeting_id=meeting.id,
                description=item_data.get("description", ""),
                assignee_agent_id=assignee_agent_id,
                priority=item_data.get("priority", 70),
                deadline_days=deadline_days,
                deadline_at=deadline_at,
                depends_on_decision_id=depends_on_decision_id,
            )
            db.add(action_item)
            created.append(action_item)

        await db.flush()
        return created

    async def _fallback_action_items(
        self,
        db: AsyncSession,
        meeting: Meeting,
        decisions: list[MeetingDecision] | None = None,
    ) -> list[MeetingActionItem]:
        if meeting.meeting_type == "decision":
            transcript = await self._format_transcript(db, meeting.id)
            descriptions = self._extract_explicit_action_items_from_transcript(transcript)
            if not descriptions:
                descriptions = await self._derive_action_items_from_decisions(
                    db=db,
                    meeting=meeting,
                    decisions=decisions,
                )
            created: list[MeetingActionItem] = []
            seen_descriptions: set[str] = set()
            for description in descriptions:
                if description in seen_descriptions:
                    continue
                action_item = MeetingActionItem(
                    meeting_id=meeting.id,
                    description=description[:200],
                    priority=75,
                    is_partial=True,
                )
                db.add(action_item)
                created.append(action_item)
                seen_descriptions.add(description)
            await db.flush()
            return created

        if meeting.meeting_type not in {"review", "standup"}:
            return []

        result = await db.execute(
            select(MeetingAgendaItem)
            .where(MeetingAgendaItem.meeting_id == meeting.id)
            .order_by(MeetingAgendaItem.order)
        )
        agenda_items = list(result.scalars().all())
        if not agenda_items:
            return []

        created: list[MeetingActionItem] = []
        seen_descriptions: set[str] = set()
        standup_descriptions_by_key: dict[str, str] = {}
        standup_order: list[str] = []
        for item in agenda_items:
            descriptions = self._fallback_followup_descriptions(meeting=meeting, agenda_item=item)
            if meeting.meeting_type == "standup":
                for description in descriptions:
                    key = self._normalize_standup_blocker_key(description)
                    if not key:
                        continue
                    existing = standup_descriptions_by_key.get(key)
                    if existing is None:
                        standup_descriptions_by_key[key] = description
                        standup_order.append(key)
                    else:
                        standup_descriptions_by_key[key] = self._prefer_standup_blocker_description(
                            existing,
                            description,
                        )
                continue

            for description in descriptions:
                if not description or description in seen_descriptions:
                    continue

                action_item = MeetingActionItem(
                    meeting_id=meeting.id,
                    description=description[:200],
                    priority=75,
                    is_partial=True,
                )
                db.add(action_item)
                created.append(action_item)
                seen_descriptions.add(description)

        if meeting.meeting_type == "standup":
            for key in standup_order:
                description = standup_descriptions_by_key[key]
                action_item = MeetingActionItem(
                    meeting_id=meeting.id,
                    description=description[:200],
                    priority=75,
                    is_partial=True,
                )
                db.add(action_item)
                created.append(action_item)

        await db.flush()
        return created

    async def create_tasks_from_action_items(
        self,
        db: AsyncSession,
        meeting: Meeting,
        action_items: list[MeetingActionItem],
    ) -> list[Task]:
        created: list[Task] = []
        for item in action_items:
            task = Task(
                project_id=meeting.project_id,
                title=item.description[:200],
                description=f"Created from meeting: {meeting.title}",
                status="backlog",
                priority=item.priority,
                assigned_to=item.assignee_agent_id,
                created_by_meeting_id=meeting.id,
            )
            db.add(task)
            created.append(task)

        await db.flush()

        # Set status and task_id after flush to ensure task.id is available
        for i, item in enumerate(action_items):
            item.status = "task_created"
            item.task_id = created[i].id

        await db.flush()
        return created

    async def write_knowledge_items(
        self,
        db: AsyncSession,
        meeting: Meeting,
        decisions: list[MeetingDecision],
        transcript: str | None = None,
    ) -> list[KnowledgeItem]:
        created: list[KnowledgeItem] = []

        # One KI per decision
        for decision in decisions:
            content = (
                f"Decision: {decision.chosen_option}\n"
                f"Rationale: {decision.rationale}\n"
                f"Decided by: {decision.decided_by}"
            )
            ki = KnowledgeItem(
                project_id=meeting.project_id,
                title=decision.title,
                content=content,
                content_type="decision",
                tags=["meeting", "decision"],
                metadata_={
                    "source_meeting_id": str(meeting.id),
                    "decision_id": str(decision.id),
                },
            )
            db.add(ki)
            await db.flush()
            decision.knowledge_item_id = ki.id
            created.append(ki)

        if meeting.meeting_type == "standup":
            summary_text = self._build_fallback_summary(meeting=meeting, decisions=decisions)
            meeting.summary = summary_text
            summary_ki = KnowledgeItem(
                project_id=meeting.project_id,
                title=f"Meeting Summary: {meeting.title}",
                content=summary_text,
                content_type="summary",
                tags=["meeting", "summary"],
                metadata_={
                    "source_meeting_id": str(meeting.id),
                    "decision_ids": [str(d.id) for d in decisions],
                },
            )
            db.add(summary_ki)
            await db.flush()
            created.append(summary_ki)
            return created

        # Fetch transcript once if not provided
        if transcript is None:
            transcript = await self._format_transcript(db, meeting.id)

        # Meeting summary KI
        # Build decisions summary for the prompt
        decisions_summary = ""
        if decisions:
            d_lines = [
                f"- [{d.title}]: {d.chosen_option} (decided by {d.decided_by})"
                for d in decisions
            ]
            decisions_summary = "DECISIONS:\n" + "\n".join(d_lines) + "\n"

        _pre = await _outcome_preamble(db, meeting)

        # Create invocation context for summary
        summary_request_prompt = {
            "operation": "meeting_summary",
            "title": meeting.title,
        }
        summary_invocation_ctx = InvocationContext(
            meeting.project_id,
            actor_kind="system",
            actor_id="orchestrator",
            actor_label="Orchestrator",
            invocation_kind="api",
            operation="meeting_summary",
            model_or_runtime=_OUTCOME_MODEL.split("/")[-1],
            request_prompt=summary_request_prompt,
        )
        summary_invocation = AgentResponseInvocation(summary_invocation_ctx)

        try:
            summary_messages = [
                {
                    "role": "system",
                    "content": _pre + (
                        "Summarize this meeting for a project knowledge base. "
                        "Structure your response with these exact section headers. "
                        "Be concise and accurate."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"MEETING: {meeting.title}\n"
                        f"TYPE: {meeting.meeting_type}\n"
                        f"{decisions_summary}\n"
                        f"TRANSCRIPT:\n{transcript[:4000]}\n\n"
                        "Produce the summary with these sections:\n"
                        "## Outcome\n<One sentence describing the overall result.>\n\n"
                        "## Decisions\n<Bullet per decision: '- [{title}]: {chosen_option} — {rationale}'>\n\n"
                        "## Dissent and Open Questions\n<Bullet per unresolved item or dissenting position, or 'None.'>\n\n"
                        "## Action Items\n<Bullet per action item: '- [{assignee}]: {description}'>"
                    ),
                },
            ]
            async with summary_invocation.call(messages=summary_messages) as call:
                summary_resp = await call.complete(
                    litellm.acompletion,
                    {
                        "model": _OUTCOME_MODEL,
                        "messages": summary_messages,
                        "temperature": 0.0,
                        "max_tokens": 600,
                    },
                )
            summary_text = (summary_resp.choices[0].message.content or "").strip()
            if not summary_text:
                raise ValueError("meeting summary LLM returned empty content")
        except Exception as exc:
            logger.warning("meeting summary LLM call failed: %s", exc)
            summary_text = self._build_fallback_summary(meeting=meeting, decisions=decisions)
        meeting.summary = summary_text

        decision_ids = [str(d.id) for d in decisions]
        summary_ki = KnowledgeItem(
            project_id=meeting.project_id,
            title=f"Meeting Summary: {meeting.title}",
            content=summary_text,
            content_type="summary",
            tags=["meeting", "summary"],
            metadata_={
                "source_meeting_id": str(meeting.id),
                "decision_ids": decision_ids,
            },
        )
        db.add(summary_ki)
        await db.flush()
        created.append(summary_ki)
        return created

    async def run_planner_summary(
        self,
        db: AsyncSession,
        meeting: Meeting,
        decisions: list[MeetingDecision],
        transcript: str,
    ) -> str | None:
        if not meeting.planner_agent_id:
            return None

        from huddleroom.models.agent import Agent
        planner = await db.get(Agent, meeting.planner_agent_id)
        if not planner:
            logger.warning("Planner agent %s not found for meeting %s", meeting.planner_agent_id, meeting.id)
            return None

        provider = getattr(planner, "provider", None) or "openai"
        model = getattr(planner, "model", None) or "gpt-6.1-sol"
        litellm_model = build_litellm_model_name(provider, model)

        decisions_text = ""
        if decisions:
            lines = [
                f"- {d.title}: {d.chosen_option} (decided by {d.decided_by}, confidence {d.confidence or 'N/A'})"
                for d in decisions
            ]
            decisions_text = "\nDECISIONS MADE:\n" + "\n".join(lines)

        system_content = (
            (planner.system_prompt or "You produce concise, actionable summaries.")
            + "\n\n"
            + "## HuddleRoom Planner Role\n"
            + "Generate a structured summary for HuddleRoom's graph engine to create tasks and trigger processes. "
            + "Follow the format exactly. No extra text, markdown, or blank lines between sections."
        )

        user_content = (
            f"Meeting: {meeting.title}\n"
            f"Type: {meeting.meeting_type}\n\n"
            f"TRANSCRIPT:\n{transcript[:6000]}\n"
            f"{decisions_text}\n\n"
            "Produce the summary in the following schema. One entry per line. Pipe-delimited fields.\n\n"
            "DECISIONS\n"
            "DECISION|<title>|<chosen_option>|<decided_by>|<confidence or 'N/A'>\n\n"
            "ACTIONS\n"
            "ACTION|<description>|<assignee_name or 'unassigned'>|<deadline_days or 'none'>|<depends_on_decision_title or 'none'>\n\n"
            "OPEN_QUESTIONS\n"
            "OPEN_QUESTION|<question or unresolved item title>|<last_position_held>"
        )

        # Create invocation context for planner summary
        planner_request_prompt = {
            "operation": "meeting_planner_summary",
            "title": meeting.title,
        }
        planner_invocation_ctx = InvocationContext(
            meeting.project_id,
            actor_kind="agent",
            actor_id=str(planner.id),
            actor_label=planner.name,
            invocation_kind="api",
            operation="meeting_planner_summary",
            model_or_runtime=model,
            request_prompt=planner_request_prompt,
        )
        planner_invocation = AgentResponseInvocation(planner_invocation_ctx)

        try:
            planner_messages = [
                {
                    "role": "system",
                    "content": system_content,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ]
            async with planner_invocation.call(messages=planner_messages) as call:
                resp = await call.complete(
                    litellm.acompletion,
                    {
                        "model": litellm_model,
                        "messages": planner_messages,
                        "temperature": 0.0,
                        "max_tokens": 600,
                    },
                )
            summary = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Planner summary LLM call failed: %s", exc)
            return None

        meeting.planner_summary = summary

        planner_ki = KnowledgeItem(
            project_id=meeting.project_id,
            title=f"Planner Summary: {meeting.title}",
            content=summary,
            content_type="planner_summary",
            tags=["meeting", "planner_summary"],
            metadata_={
                "source_meeting_id": str(meeting.id),
                "planner_agent_id": str(meeting.planner_agent_id),
            },
        )
        db.add(planner_ki)
        await db.flush()
        return summary

    async def resolve_graph_run(
        self, db: AsyncSession, meeting: Meeting, decisions: list[MeetingDecision]
    ) -> None:
        if not meeting.source_graph_run_id or not decisions:
            return
        from huddleroom.services.event_bus import emit_event
        primary = decisions[0]
        await emit_event(
            db=db,
            project_id=meeting.project_id,
            event_type="graph.run_external_resolution",
            payload={
                "graph_run_id": str(meeting.source_graph_run_id),
                "resolution": primary.chosen_option,
                "rationale": primary.rationale,
                "meeting_id": str(meeting.id),
            },
        )

    async def finalize_meeting(
        self, db: AsyncSession, meeting: Meeting
    ) -> tuple[bool, MeetingEvent | None]:
        from huddleroom.services.meeting_service import MeetingService

        locked_meeting = (
            await db.execute(
                select(Meeting)
                .where(Meeting.id == meeting.id)
                .execution_options(populate_existing=True)
                .with_for_update()
            )
        ).scalar_one()
        meeting = locked_meeting
        if meeting.status != "concluding":
            return False, None
        pending_events, completed_events = await self._final_review_events(db, meeting.id)
        if len(pending_events) > len(completed_events):
            return False, None

        failures: list[str] = []
        result = await db.execute(
            select(MeetingDecision).where(MeetingDecision.meeting_id == meeting.id)
        )
        decisions = list(result.scalars().all())
        transcript: str | None = None
        result = await db.execute(
            select(MeetingActionItem)
            .where(MeetingActionItem.meeting_id == meeting.id)
            .order_by(MeetingActionItem.created_at, MeetingActionItem.id)
        )
        action_items = list(result.scalars().all())

        completed_waiver = bool(
            completed_events
            and completed_events[-1].payload
            and completed_events[-1].payload.get("action_items_needed") is False
        )
        if not action_items and not completed_waiver:
            try:
                await self.extract_action_items(db=db, meeting=meeting, decisions=decisions)
            except Exception as exc:
                logger.warning("extract_action_items failed during finalization: %s", exc)
                failures.append("extract_action_items")
            result = await db.execute(
                select(MeetingActionItem)
                .where(MeetingActionItem.meeting_id == meeting.id)
                .order_by(MeetingActionItem.created_at, MeetingActionItem.id)
            )
            action_items = list(result.scalars().all())

        suggestions = [
            description
            for decision in decisions
            if (description := self._decision_to_action_description(decision.chosen_option))
        ]
        review_complete = bool(pending_events and len(completed_events) >= len(pending_events))
        if suggestions and not action_items and not review_complete:
            reviewer_kind = "orchestrator"
            reviewer_id: str | None = None
            if meeting.organizer_agent_id:
                reviewer_kind = "organizer_agent"
                reviewer_id = str(meeting.organizer_agent_id)
            elif meeting.organizer_user_id:
                reviewer_kind = "organizer_user"
                reviewer_id = str(meeting.organizer_user_id)
            payload = {
                "reviewer_kind": reviewer_kind,
                "reviewer_id": reviewer_id,
                "decisions_made": bool(decisions),
                "decisions_clear": bool(decisions) and all(
                    decision.title.strip()
                    and decision.chosen_option.strip()
                    and decision.rationale.strip()
                    for decision in decisions
                ),
                "suggested_action_items": suggestions,
            }
            if settings.is_sqlite:
                pending_id = await self._claim_sqlite_final_review_event(
                    db,
                    meeting_id=meeting.id,
                    event_type="meeting_final_pass",
                    payload=payload,
                    require_no_action_items=True,
                )
                if pending_id:
                    await db.refresh(meeting)
                pending = await db.get(MeetingEvent, pending_id) if pending_id else None
            else:
                pending = MeetingEvent(
                    meeting_id=meeting.id,
                    event_type="meeting_final_pass",
                    payload=payload,
                )
                db.add(pending)
                await db.flush()
            return False, pending

        # Create tasks
        action_items_without_tasks = [item for item in action_items if item.task_id is None]
        if action_items_without_tasks:
            try:
                await self.create_tasks_from_action_items(
                    db=db, meeting=meeting, action_items=action_items_without_tasks
                )
            except Exception as exc:
                logger.warning("create_tasks_from_action_items failed during finalization: %s", exc)
                failures.append("create_tasks_from_action_items")

        # Write knowledge items
        try:
            await self.write_knowledge_items(db=db, meeting=meeting, decisions=decisions, transcript=transcript)
        except Exception as exc:
            logger.warning("write_knowledge_items failed during finalization: %s", exc)
            failures.append("write_knowledge_items")
            if not meeting.summary:
                meeting.summary = self._build_fallback_summary(meeting=meeting, decisions=decisions)

        # Resolve originating graph run if applicable
        try:
            await self.resolve_graph_run(db=db, meeting=meeting, decisions=decisions)
        except Exception as exc:
            logger.warning("resolve_graph_run failed during finalization: %s", exc)
            failures.append("resolve_graph_run")

        # Planner structured summary is skipped for standups to keep conclusion on the fast path.
        if meeting.planner_agent_id and meeting.meeting_type != "standup":
            try:
                if transcript is None:
                    transcript = await self._format_transcript(db, meeting.id)
                await self.run_planner_summary(
                    db=db, meeting=meeting, decisions=decisions, transcript=transcript
                )
            except Exception as exc:
                logger.warning("run_planner_summary failed during finalization: %s", exc)
                failures.append("run_planner_summary")

        if failures:
            meeting.is_partial = True
            await MeetingService().log_event(
                db=db,
                meeting_id=meeting.id,
                event_type="partial_finalization",
                payload={"failed_steps": failures},
            )
        await MeetingService().transition_to_concluded(db=db, meeting=meeting)
        return True, None

    async def get_pending_final_review(
        self, db: AsyncSession, meeting: Meeting
    ) -> MeetingEvent | None:
        if meeting.status != "concluding":
            return None
        pending_events, completed_events = await self._final_review_events(db, meeting.id)
        if len(pending_events) <= len(completed_events):
            return None
        return pending_events[-1]

    async def complete_final_review(
        self,
        db: AsyncSession,
        meeting: Meeting,
        *,
        decisions_made: bool,
        decisions_clear: bool,
        action_items_needed: bool,
        action_items: list[str],
        actor_agent_id: uuid.UUID | None = None,
        actor_user_id: uuid.UUID | None = None,
    ) -> bool:
        from huddleroom.services.meeting_service import MeetingTransitionError

        meeting_id = meeting.id
        meeting_query = (
            select(Meeting)
            .where(Meeting.id == meeting_id)
            .execution_options(populate_existing=True)
        )
        if not settings.is_sqlite:
            meeting_query = meeting_query.with_for_update()
        locked_meeting = (await db.execute(meeting_query)).scalar_one_or_none()
        if locked_meeting is None:
            raise MeetingTransitionError("Meeting does not exist")

        pending_events, completed_events = await self._final_review_events(db, meeting_id)
        if pending_events and len(completed_events) >= len(pending_events):
            return False
        if locked_meeting.status != "concluding" or not pending_events:
            raise MeetingTransitionError("Meeting has no active final review")

        trimmed_items = [item.strip() for item in action_items]
        if any(not item or len(item) > 200 for item in trimmed_items):
            raise ValueError("Action items must contain 1 to 200 characters")

        existing_items = list(
            (
                await db.execute(
                    select(MeetingActionItem).where(
                        MeetingActionItem.meeting_id == meeting_id
                    )
                )
            ).scalars().all()
        )
        if action_items_needed and not (trimmed_items or existing_items):
            raise ValueError("Action items are required")
        if not action_items_needed and trimmed_items:
            raise ValueError("A waiver cannot include action items")

        payload = {
            "decisions_made": decisions_made,
            "decisions_clear": decisions_clear,
            "action_items_needed": action_items_needed,
        }
        if settings.is_sqlite:
            completed_id = await self._claim_sqlite_final_review_event(
                db,
                meeting_id=meeting_id,
                event_type="final_pass_completed",
                payload=payload,
                actor_agent_id=actor_agent_id,
                actor_user_id=actor_user_id,
            )
            if completed_id is None:
                pending_events, completed_events = await self._final_review_events(db, meeting_id)
                if pending_events and len(completed_events) >= len(pending_events):
                    return False
                raise MeetingTransitionError("Meeting has no active final review")
        else:
            db.add(
                MeetingEvent(
                    meeting_id=meeting_id,
                    event_type="final_pass_completed",
                    payload=payload,
                    actor_agent_id=actor_agent_id,
                    actor_user_id=actor_user_id,
                )
            )
        for description in trimmed_items:
            db.add(MeetingActionItem(meeting_id=meeting_id, description=description))
        await db.flush()
        return True

    async def _claim_sqlite_final_review_event(
        self,
        db: AsyncSession,
        *,
        meeting_id: uuid.UUID,
        event_type: str,
        payload: dict,
        actor_agent_id: uuid.UUID | None = None,
        actor_user_id: uuid.UUID | None = None,
        require_no_action_items: bool = False,
    ) -> uuid.UUID | None:
        event = MeetingEvent.__table__
        pending_count = (
            select(func.count())  # pylint: disable=not-callable
            .select_from(event)
            .where(event.c.meeting_id == meeting_id, event.c.event_type == "meeting_final_pass")
            .scalar_subquery()
        )
        completed_count = (
            select(func.count())  # pylint: disable=not-callable
            .select_from(event)
            .where(event.c.meeting_id == meeting_id, event.c.event_type == "final_pass_completed")
            .scalar_subquery()
        )
        conditions = [
            select(Meeting.id)
            .where(Meeting.id == meeting_id, Meeting.status == "concluding")
            .exists(),
            pending_count <= completed_count
            if event_type == "meeting_final_pass"
            else pending_count > completed_count,
        ]
        if require_no_action_items:
            conditions.append(
                ~select(MeetingActionItem.id)
                .where(MeetingActionItem.meeting_id == meeting_id)
                .exists()
            )

        event_id = uuid.uuid4()
        values = select(
            literal(event_id, type_=event.c.id.type),
            literal(meeting_id, type_=event.c.meeting_id.type),
            literal(event_type, type_=event.c.event_type.type),
            literal(payload, type_=event.c.payload.type),
            literal(actor_agent_id, type_=event.c.actor_agent_id.type),
            literal(actor_user_id, type_=event.c.actor_user_id.type),
            literal(datetime.now(timezone.utc), type_=event.c.created_at.type),
        ).where(*conditions)
        claim = insert(event).from_select(
            [
                event.c.id,
                event.c.meeting_id,
                event.c.event_type,
                event.c.payload,
                event.c.actor_agent_id,
                event.c.actor_user_id,
                event.c.created_at,
            ],
            values,
        ).returning(event.c.id)

        for attempt in range(_SQLITE_LOCK_RETRIES):
            try:
                return (await db.execute(claim)).scalar_one_or_none()
            except OperationalError as exc:
                if "database is locked" not in str(exc).lower():
                    raise
                await db.rollback()
                if attempt == _SQLITE_LOCK_RETRIES - 1:
                    raise
                await asyncio.sleep(0)
        return None

    async def ask_final_reviewer(
        self, db: AsyncSession, meeting: Meeting, pending: MeetingEvent
    ) -> tuple[bool, bool, bool, list[str]]:
        payload = pending.payload or {}
        suggestions = payload.get("suggested_action_items") or []
        if payload.get("reviewer_kind") == "organizer_user":
            raise ValueError("organizer user review requires interactive submission")

        decisions = list(
            (
                await db.execute(
                    select(MeetingDecision)
                    .where(MeetingDecision.meeting_id == meeting.id)
                    .order_by(MeetingDecision.created_at, MeetingDecision.id)
                )
            ).scalars().all()
        )
        transcript = await self._format_transcript(db, meeting.id)
        decision_text = "\n".join(
            f"- {decision.title}: {decision.chosen_option}\n  Rationale: {decision.rationale}"
            for decision in decisions
        ) or "None"
        prompt = (
            f"MEETING: {meeting.title}\n\n"
            f"DECISIONS:\n{decision_text}\n\n"
            f"TRANSCRIPT TAIL:\n{transcript[-4000:]}\n\n"
            f"SUGGESTED ACTION ITEMS:\n{json.dumps(suggestions)}\n\n"
            "Answer these questions: Were decisions made? Are decisions clearly stated? "
            "Are action items needed? Return JSON with exactly these fields: "
            '"decisions_made" (boolean), "decisions_clear" (boolean), '
            '"action_items_needed" (boolean), and "action_items" (array of strings).'
        )

        reviewer_kind = payload.get("reviewer_kind")

        # Local validator for final review response JSON
        def validate_final_review_response(raw: str) -> str:
            """Validate final review response structure, raising on JSON/parse errors."""
            self._parse_final_review(raw)  # Will raise if invalid
            return raw

        if reviewer_kind == "organizer_agent":
            reviewer_id = payload.get("reviewer_id")
            agent = await db.get(Agent, uuid.UUID(reviewer_id)) if reviewer_id else None
            if agent is None:
                raise ValueError("Final reviewer agent does not exist")
            if agent.adapter_type == "cli":
                from huddleroom.adapters.cli_adapter import CliAdapter
                from huddleroom.models.project import Project

                project = await db.get(Project, meeting.project_id)
                contexts = dict(meeting.participant_contexts or {})
                cached_agent_context = contexts.get(str(agent.id))
                agent_context = (
                    dict(cached_agent_context)
                    if isinstance(cached_agent_context, dict)
                    else {"initial_ctx": cached_agent_context}
                    if isinstance(cached_agent_context, str)
                    else {}
                )
                existing_session_id = agent_context.get("cli_session_id")
                raw, new_session_id, _ = await CliAdapter().run_meeting_turn(
                    db=db,
                    meeting=meeting,
                    agent=agent,
                    project=project,
                    prompt_text=prompt,
                    existing_session_id=existing_session_id,
                    operation="meeting_final_review",
                )
                if new_session_id and new_session_id != existing_session_id:
                    agent_context["cli_session_id"] = new_session_id
                    meeting.participant_contexts = {
                        **contexts,
                        str(agent.id): agent_context,
                    }
                    await db.flush()
            else:
                # Create invocation context for agent final reviewer
                agent_review_request_prompt = {
                    "operation": "meeting_final_review",
                    "title": meeting.title,
                }
                agent_review_invocation_ctx = InvocationContext(
                    meeting.project_id,
                    actor_kind="agent",
                    actor_id=str(agent.id),
                    actor_label=agent.name,
                    invocation_kind="api",
                    operation="meeting_final_review",
                    model_or_runtime=getattr(agent, "model", "gpt-6.1-sol"),
                    request_prompt=agent_review_request_prompt,
                )
                agent_review_invocation = AgentResponseInvocation(agent_review_invocation_ctx)

                request = {
                    "model": build_litellm_model_name(agent.provider, agent.model),
                    "messages": [
                        {
                            "role": "system",
                            "content": agent.system_prompt or "You are a meeting final reviewer.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 400,
                }
                raw = await complete_with_repair(
                    litellm.acompletion,
                    request,
                    validate_final_review_response,
                    invocation=agent_review_invocation,
                )
        elif reviewer_kind == "orchestrator":
            # Create invocation context for orchestrator final reviewer
            orch_review_request_prompt = {
                "operation": "meeting_final_review",
                "title": meeting.title,
            }
            orch_review_invocation_ctx = InvocationContext(
                meeting.project_id,
                actor_kind="system",
                actor_id="orchestrator",
                actor_label="Orchestrator",
                invocation_kind="api",
                operation="meeting_final_review",
                model_or_runtime=_OUTCOME_MODEL.split("/")[-1],
                request_prompt=orch_review_request_prompt,
            )
            orch_review_invocation = AgentResponseInvocation(orch_review_invocation_ctx)

            _pre = await _outcome_preamble(db, meeting)
            request = {
                "model": _OUTCOME_MODEL,
                "messages": [
                    {"role": "system", "content": _pre + "You are HuddleRoom's meeting final reviewer."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 400,
            }
            raw = await complete_with_repair(
                litellm.acompletion,
                request,
                validate_final_review_response,
                invocation=orch_review_invocation,
            )
        else:
            raise ValueError("Unknown final reviewer kind")

        return self._parse_final_review(raw)

    async def _final_review_events(
        self, db: AsyncSession, meeting_id: uuid.UUID
    ) -> tuple[list[MeetingEvent], list[MeetingEvent]]:
        events = list(
            (
                await db.execute(
                    select(MeetingEvent)
                    .where(
                        MeetingEvent.meeting_id == meeting_id,
                        MeetingEvent.event_type.in_(("meeting_final_pass", "final_pass_completed")),
                    )
                    .order_by(MeetingEvent.created_at, MeetingEvent.id)
                )
            ).scalars().all()
        )
        return (
            [event for event in events if event.event_type == "meeting_final_pass"],
            [event for event in events if event.event_type == "final_pass_completed"],
        )

    def _parse_final_review(self, raw: str) -> tuple[bool, bool, bool, list[str]]:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("Final review response is not JSON")
        try:
            answer = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError("Final review response is not JSON") from exc

        boolean_keys = ("decisions_made", "decisions_clear", "action_items_needed")
        if any(type(answer.get(key)) is not bool for key in boolean_keys):
            raise ValueError("Final review booleans are missing or invalid")
        action_items = answer.get("action_items")
        if not isinstance(action_items, list) or any(not isinstance(item, str) for item in action_items):
            raise ValueError("Final review action_items must be a list of strings")
        return (
            answer["decisions_made"],
            answer["decisions_clear"],
            answer["action_items_needed"],
            action_items,
        )

    def _parse_json_list(self, raw: str) -> list[dict]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return []

    def _extract_explicit_action_items_from_transcript(self, transcript: str) -> list[str]:
        descriptions: list[str] = []
        for line in transcript.splitlines():
            match = re.search(r"\baction item:\s*(.+)$", line, re.IGNORECASE)
            if not match:
                continue
            description = re.sub(r"\s+", " ", match.group(1)).strip(" -\t")
            if description:
                descriptions.append(description)
        return descriptions

    async def _derive_action_items_from_decisions(
        self,
        db: AsyncSession,
        meeting: Meeting,
        decisions: list[MeetingDecision] | None = None,
    ) -> list[str]:
        if decisions is None:
            result = await db.execute(
                select(MeetingDecision).where(MeetingDecision.meeting_id == meeting.id)
            )
            decisions = list(result.scalars().all())

        descriptions: list[str] = []
        for decision in decisions:
            description = self._decision_to_action_description(decision.chosen_option)
            if description:
                descriptions.append(description)
        return descriptions

    def _decision_to_action_description(self, chosen_option: str | None) -> str | None:
        if not chosen_option:
            return None
        cleaned = re.sub(r"^\s*option:\s*", "", chosen_option, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
        if not cleaned:
            return None

        imperative_prefixes = (
            "write ", "draft ", "create ", "define ", "implement ", "spike ", "build ",
            "prepare ", "document ", "assign ", "outline ", "schedule ", "run ", "migrate ",
            "prototype ", "pilot ", "ship ", "add ", "remove ", "update ", "establish ",
            "set up ", "set ", "configure ",
        )
        if not cleaned.lower().startswith(imperative_prefixes):
            return None
        return cleaned

    def _fallback_followup_descriptions(self, meeting: Meeting, agenda_item: MeetingAgendaItem) -> list[str]:
        if meeting.meeting_type == "review":
            if agenda_item.resolution_kind not in {"rejected", "approved_with_followups", "deferred"}:
                return []
            description = (agenda_item.required_followup or "").strip()
            if description:
                return [description]
            return [self._fallback_followup_description(agenda_item)]

        if meeting.meeting_type == "standup":
            followup = (agenda_item.required_followup or "").strip()
            if not followup:
                return []
            descriptions = [
                cleaned
                for part in re.split(r"[;\n]+", followup)
                if (cleaned := self._clean_standup_blocker_fragment(part))
            ]
            return descriptions or [followup]

        return []

    def _clean_standup_blocker_fragment(self, fragment: str) -> str | None:
        cleaned = re.sub(r"\s+", " ", fragment).strip(" -.,;:")
        if not cleaned or self._is_explanatory_standup_tail(cleaned):
            return None
        return cleaned

    def _is_explanatory_standup_tail(self, fragment: str) -> bool:
        lowered = fragment.lower()
        return lowered.startswith((
            "risk persists",
            "risks persist",
            "risk remains",
            "risks remain",
            "issue persists",
            "issues persist",
            "problem persists",
            "problems persist",
            "because ",
            "since ",
            "until ",
        ))

    def _normalize_standup_blocker_key(self, description: str) -> str:
        normalized = description.lower()
        replacements = (
            (r"\bprod\b", "production"),
            (r"\bgetting\b", " "),
            (r"\bblocked\b", " "),
            (r"\bblocking\b", " "),
            (r"\bwaiting\b", " "),
            (r"\bwait\b", " "),
            (r"\bneed(?:ed|s|ing)?\b", " "),
            (r"\bmissing\b", " "),
            (r"\black(?:ing)?\b", " "),
            (r"\brequire(?:d|s|ing)?\b", " "),
            (r"\bpending\b", " "),
            (r"\bcannot\b", " "),
            (r"\bcan't\b", " "),
            (r"\bunable\b", " "),
            (r"\bapproval\b", " "),
            (r"\bgrant(?:s|ed|ing)?\b", " "),
            (r"\bpermission(?:s)?\b", " "),
            (r"\baccessing\b", "access"),
            (r"\bon\b", " "),
            (r"\bfor\b", " "),
            (r"\bto\b", " "),
            (r"\bthe\b", " "),
            (r"\ba\b", " "),
            (r"\ban\b", " "),
            (r"\bstill\b", " "),
            (r"\bcurrently\b", " "),
        )
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized)
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        tokens = list(dict.fromkeys(token for token in normalized.split() if token))
        if not tokens:
            return ""
        return " ".join(sorted(tokens))

    def _prefer_standup_blocker_description(self, current: str, candidate: str) -> str:
        if self._standup_blocker_description_score(candidate) > self._standup_blocker_description_score(current):
            return candidate
        return current

    def _standup_blocker_description_score(self, description: str) -> int:
        lowered = description.lower()
        score = 0
        if lowered.startswith("waiting on "):
            score += 40
        elif lowered.startswith("need "):
            score += 35
        elif lowered.startswith("blocked on "):
            score += 30
        elif lowered.startswith("missing "):
            score += 25

        if self._is_explanatory_standup_tail(description):
            score -= 100

        for noisy_phrase in (" getting ", " approval", " currently ", " still "):
            score -= lowered.count(noisy_phrase) * 5

        score -= max(len(description) - 24, 0) // 4
        return score

    def _fallback_followup_description(self, agenda_item: MeetingAgendaItem) -> str:
        summary = (agenda_item.resolution_summary or "").strip().rstrip(".")
        if summary:
            return f"Address review follow-ups for {agenda_item.title}: {summary}."
        return f"Address review follow-ups for {agenda_item.title}."

    async def _format_transcript(self, db: AsyncSession, meeting_id: uuid.UUID) -> str:
        from huddleroom.models.agent import Agent
        result = await db.execute(
            select(MeetingTurn)
            .where(MeetingTurn.meeting_id == meeting_id)
            .order_by(MeetingTurn.turn_number)
        )
        turns = result.scalars().all()
        lines = []
        cache: dict[uuid.UUID, str] = {}
        for t in turns:
            if t.speaker_agent_id:
                if t.speaker_agent_id not in cache:
                    a = await db.get(Agent, t.speaker_agent_id)
                    cache[t.speaker_agent_id] = a.name if a else "Agent"
                speaker = cache[t.speaker_agent_id]
            else:
                speaker = "Human"
            lines.append(f"{speaker}: {t.content}")
        return "\n".join(lines)

    def _build_fallback_summary(self, meeting: Meeting, decisions: list[MeetingDecision]) -> str:
        decision_lines = "\n".join(
            f"- [{decision.title}]: {decision.chosen_option} — {decision.rationale}"
            for decision in decisions
        ) or "None."
        outcome = f"{meeting.meeting_type.title()} meeting concluded."
        if decisions:
            outcome += " Decisions were recorded."
        else:
            outcome += " No formal decisions were recorded."
        return (
            "## Outcome\n"
            f"{outcome}\n\n"
            "## Decisions\n"
            f"{decision_lines}\n\n"
            "## Dissent and Open Questions\n"
            "None.\n\n"
            "## Action Items\n"
            "See meeting action items and linked tasks for follow-up."
        )
