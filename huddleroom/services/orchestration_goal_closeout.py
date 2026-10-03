from __future__ import annotations

import json

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import (
    OrchestrationAction,
    OrchestrationEvidence,
    OrchestrationGate,
    OrchestrationGoal,
    OrchestrationRun,
)
from huddleroom.models.orchestration_process import (
    OrchestrationAuthorityDecision,
    OrchestrationProcessRun,
)
from huddleroom.services.orchestration_agent_review_service import OrchestrationAgentReviewService
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

PROCESS_TYPE = "goal_closeout"
COMPLETION_RATIONALE_KEY = "completion_rationale"
LESSONS_LEARNED_KEY = "lessons_learned"


def _json_body(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _summary(
    status: str,
    *,
    mode: str | None = None,
    completion_authorized: bool = False,
    **extra,
) -> dict:
    return {
        "status": status,
        "mode": mode,
        "completion_authorized": completion_authorized,
        **extra,
    }


class GoalCloseoutProcess:
    """Persist the compressed automatic closeout path for trivial goals."""

    def __init__(self) -> None:
        self.process_service = OrchestrationProcessService()
        self.decision_service = OrchestrationAuthorityDecisionService()
        self.memory_service = OrchestrationMemoryService()
        self.warning_service = OrchestrationWarningService()
        self.review_service = OrchestrationAgentReviewService()

    async def advance(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        *,
        preconditions: dict | None,
    ) -> dict:
        current = await self.process_service.get_current(db, goal.id, PROCESS_TYPE)
        if (
            current is not None
            and current.status in ("completed", "skipped")
            and current.run_id != run.id
        ):
            current = None

        if current is not None and current.status == "skipped":
            return _summary("skipped", mode="completion")
        if current is not None and current.status == "completed":
            return self._completed_summary(current)
        if current is None and preconditions is None:
            return _summary("idle")
        if current is None:
            current = await self.process_service.start_process(
                db,
                goal.id,
                process_type=PROCESS_TYPE,
                trigger_reason="automatic: completion requested",
                run_id=run.id,
                input_snapshot={"mode": "completion"},
            )
        if preconditions is None:
            return _summary(current.status, mode="completion", awaiting_preconditions=True)

        full_closeout = current.input_snapshot.get("full_closeout") is True or goal.weight != "trivial"
        if full_closeout:
            return await self._advance_full_closeout(db, goal, run, current, preconditions)

        signoff = preconditions.get("signoff", {"mode": "automatic_trivial"})
        rationale = {
            "outcome": "completed",
            "objective": goal.objective,
            "run_id": str(run.id),
            "declared_success_criteria": preconditions["declared_success_criteria"],
            "criterion_evidence": preconditions["criterion_evidence"],
            "accepted_non_summary_gates": preconditions["accepted_non_summary_gates"],
            "final_summary": preconditions["final_summary"],
            "warning_disposition": preconditions["warning_disposition"],
            "signoff": signoff,
        }
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key=COMPLETION_RATIONALE_KEY,
            title="Completion rationale",
            body=_json_body(rationale),
            summary="Goal completion was authorized from accepted gates and evidence.",
            section_type="json",
            toc_order=90,
            run_id=run.id,
            created_by="orchestrator:goal_closeout",
        )
        await self.process_service.complete_process(
            db,
            current,
            outputs={
                "gates": {"closeout_completed": True},
                "completion_authorized": True,
                "full_closeout": False,
                "mode": "completion",
            },
        )
        await self._resolve_prior_skip_warnings(db, goal.id, current)
        return _summary("completed", mode="completion", completion_authorized=True, full_closeout=False)

    async def complete_no_work(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> dict | None:
        """(#1) Record a terminal completed closeout for a goal that reached closeout
        with no executable work — no gates on the run, so nothing to authorize.

        Returns None when any gate exists (defer to the normal manifest path, which
        enforces acceptance/evidence for a goal that actually executed).
        """
        gate_exists = await db.scalar(
            select(OrchestrationGate.id).where(OrchestrationGate.run_id == run.id).limit(1)
        )
        if gate_exists is not None:
            return None
        current = await self.process_service.get_current(db, goal.id, PROCESS_TYPE)
        if (
            current is not None
            and current.status == "completed"
            and current.run_id == run.id
        ):
            return self._completed_summary(current)
        if current is None or current.status in ("completed", "skipped"):
            current = await self.process_service.start_process(
                db,
                goal.id,
                process_type=PROCESS_TYPE,
                trigger_reason="automatic: no executable work to close out",
                run_id=run.id,
                input_snapshot={"mode": "completion", "no_executable_work": True},
            )
        rationale = {
            "outcome": "completed",
            "objective": goal.objective,
            "run_id": str(run.id),
            "no_executable_work": True,
            "note": "Goal reached closeout with no gates or evidence to authorize.",
        }
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key=COMPLETION_RATIONALE_KEY,
            title="Completion rationale",
            body=_json_body(rationale),
            summary="Goal closed out with no executable work to authorize.",
            section_type="json",
            toc_order=90,
            run_id=run.id,
            created_by="orchestrator:goal_closeout",
        )
        await self.process_service.complete_process(
            db,
            current,
            outputs={
                "gates": {"closeout_completed": True},
                "completion_authorized": True,
                "full_closeout": False,
                "mode": "completion",
                "no_executable_work": True,
            },
        )
        await self._resolve_prior_skip_warnings(db, goal.id, current)
        return _summary(
            "completed",
            mode="completion",
            completion_authorized=True,
            full_closeout=False,
            no_executable_work=True,
        )

    async def completion_authorization(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
    ) -> dict:
        current = await self.process_service.get_current(db, goal.id, PROCESS_TYPE)
        if current is not None:
            gates = current.outputs.get("gates")
            closeout_gate_passed = (
                isinstance(gates, dict) and gates.get("closeout_completed") is True
            )
            if (
                current.status == "completed"
                and current.run_id == run.id
                and current.outputs.get("completion_authorized") is True
                and current.outputs.get("mode") == "completion"
                and closeout_gate_passed
            ):
                return self._completed_summary(current)
        if (
            current is not None
            and current.status == "skipped"
            and current.run_id == run.id
        ):
            return _summary("skipped", mode="completion")
        raise HTTPException(status_code=409, detail="Goal closeout must authorize completion")

    async def close_cancelled(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun | None,
        *,
        cancelled_by: str,
    ) -> dict:
        if run is None:
            await self._cancel_pending_decisions(db, goal.id, reason="goal cancelled")
            return _summary("cancelled", mode="cancellation")

        actions = await self._actions(db, run)
        evidence = await db.scalar(
            select(OrchestrationEvidence.id)
            .where(
                OrchestrationEvidence.run_id == run.id,
                OrchestrationEvidence.verdict.in_(("candidate", "accepted")),
            )
            .limit(1)
        )
        plan_items = (run.plan_state or {}).get("expanded_items", [])
        significant_work = bool(actions or evidence or isinstance(plan_items, list) and plan_items)
        if not significant_work:
            await self._cancel_pending_decisions(db, goal.id, reason="goal cancelled")
            return _summary("cancelled", mode="cancellation")

        current = await self.process_service.get_current(db, goal.id, PROCESS_TYPE)
        if current is None or current.status in ("completed", "skipped"):
            current = await self.process_service.start_process(
                db,
                goal.id,
                process_type=PROCESS_TYPE,
                trigger_reason="human requested: cancellation closeout",
                run_id=run.id,
                input_snapshot={"mode": "cancellation"},
            )
        if current.status == "waiting_decision":
            signoff = await self._signoff_decision(db, goal, current)
            if signoff is not None and signoff.status == "pending":
                await self.decision_service.cancel_decision(db, signoff, reason="goal cancelled")
            await self.process_service.resume_process(db, current)

        gates = await db.execute(
            select(OrchestrationGate)
            .where(OrchestrationGate.run_id == run.id)
            .order_by(OrchestrationGate.created_at.asc(), OrchestrationGate.id.asc())
        )
        warnings = await self.warning_service.list_warnings(db, goal.id)
        rationale = {
            "outcome": "cancelled",
            "objective": goal.objective,
            "run_id": str(run.id),
            "cancellation_reason": "goal cancelled",
            "cancelled_by": cancelled_by,
            "significant_work": True,
            "gate_statuses": [
                {"gate_id": str(gate.id), "status": gate.status}
                for gate in gates.scalars().all()
            ],
            "action_statuses": [
                {"action_id": str(action.id), "action_type": action.action_type, "status": action.status}
                for action in actions
            ],
            "warning_disposition": [
                {
                    "warning_id": str(warning.id),
                    "status": "active" if warning.active else "resolved",
                    "reason": warning.resolved_reason,
                }
                for warning in warnings
            ],
        }
        cancelled_ids = await self._cancel_pending_decisions(db, goal.id, reason="goal cancelled")
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key=COMPLETION_RATIONALE_KEY,
            title="Cancellation rationale",
            body=_json_body(rationale),
            summary="Goal cancellation was recorded before terminal status.",
            section_type="json",
            toc_order=90,
            run_id=run.id,
            created_by="orchestrator:goal_closeout",
        )
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key="open_questions",
            title="Open questions",
            body=_json_body(
                {
                    "status": "closed",
                    "reason": "goal cancelled",
                    "cancelled_decision_ids": cancelled_ids,
                }
            ),
            summary="Open questions closed because the goal was cancelled.",
            section_type="json",
            toc_order=80,
            run_id=run.id,
            created_by="orchestrator:goal_closeout",
        )
        await self.process_service.complete_process(
            db,
            current,
            outputs={
                "gates": {"closeout_completed": True},
                "completion_authorized": False,
                "full_closeout": current.input_snapshot.get("full_closeout", False),
                "mode": "cancellation",
            },
        )
        return _summary("completed", mode="cancellation", significant_work=True)

    @staticmethod
    def _completed_summary(process: OrchestrationProcessRun) -> dict:
        outputs = process.outputs or {}
        return _summary(
            "completed",
            mode=outputs.get("mode", process.input_snapshot.get("mode", "completion")),
            completion_authorized=outputs.get("completion_authorized", False),
            full_closeout=outputs.get("full_closeout"),
            process_id=str(process.id),
            decision_id=outputs.get("signoff_decision_id"),
        )

    async def _advance_full_closeout(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current: OrchestrationProcessRun,
        preconditions: dict,
    ) -> dict:
        decision = await self._signoff_decision(db, goal, current)
        if current.status == "running":
            if decision is None:
                authority, authority_agent_id = await self._signoff_authority(db, goal)
                decision = await self.decision_service.create_pending(
                    db,
                    goal.id,
                    decision_key=f"goal_closeout:signoff:{current.id}",
                    title="Approve goal completion",
                    question="Does the recorded evidence and accepted risk justify completing this goal?",
                    authority=authority,
                    authority_agent_id=authority_agent_id,
                    options=[
                        {"key": "approve_completion", "description": "Authorize run completion."},
                        {"key": "keep_open", "description": "Leave the run active."},
                    ],
                    context=_json_body(
                        {
                            "declared_success_criteria": preconditions["declared_success_criteria"],
                            "criterion_evidence": preconditions["criterion_evidence"],
                            "accepted_risks": preconditions["warning_disposition"],
                            "overridden_gates": preconditions["overridden_gates"],
                        }
                    ),
                    recommendation="approve_completion",
                    consequences=(
                        "approve_completion writes closeout memory and allows complete_run; "
                        "keep_open leaves the run active."
                    ),
                    run_id=run.id,
                    source_process_run_id=current.id,
                )
                current.outputs = {
                    **(current.outputs or {}),
                    "manifest": preconditions,
                    "signoff_decision_id": str(decision.id),
                }
                await db.flush()
            await self.process_service.park_process(db, current)
            return _summary(
                "waiting_decision",
                mode="completion",
                full_closeout=True,
                process_id=str(current.id),
                decision_id=str(decision.id),
            )

        if decision is None:
            return _summary(
                "waiting_decision",
                mode="completion",
                full_closeout=True,
                inconsistency="Missing sign-off decision for closeout process.",
            )
        if decision.status == "pending":
            return _summary(
                "waiting_decision",
                mode="completion",
                full_closeout=True,
                process_id=str(current.id),
                decision_id=str(decision.id),
            )

        await self.process_service.resume_process(db, current)
        authorized = decision.status == "answered" and decision.selected_option == "approve_completion"
        if not authorized:
            await self.process_service.complete_process(
                db,
                current,
                outputs={
                    "gates": {"closeout_completed": False},
                    "completion_authorized": False,
                    "full_closeout": True,
                    "mode": "completion",
                    "signoff_decision_id": str(decision.id),
                    "selected_option": decision.selected_option,
                    "signoff_status": decision.status,
                },
            )
            return _summary(
                "completed",
                mode="completion",
                full_closeout=True,
                selected_option=decision.selected_option,
                signoff_status=decision.status,
            )

        cancelled_ids = await self._cancel_pending_decisions(db, goal.id)
        await self._write_closeout_memory(db, goal, run, preconditions, decision, cancelled_ids)
        await self.process_service.complete_process(
            db,
            current,
            outputs={
                "gates": {"closeout_completed": True},
                "completion_authorized": True,
                "full_closeout": True,
                "mode": "completion",
                "signoff_decision_id": str(decision.id),
                "selected_option": decision.selected_option,
            },
        )
        await self._resolve_prior_skip_warnings(db, goal.id, current)
        return _summary(
            "completed",
            mode="completion",
            completion_authorized=True,
            full_closeout=True,
            selected_option=decision.selected_option,
        )

    async def _resolve_prior_skip_warnings(
        self,
        db: AsyncSession,
        goal_id,
        current: OrchestrationProcessRun,
    ) -> None:
        predecessor_ids = set(
            (
                await db.execute(
                    select(OrchestrationProcessRun.id).where(
                        OrchestrationProcessRun.goal_id == goal_id,
                        OrchestrationProcessRun.process_type == PROCESS_TYPE,
                        OrchestrationProcessRun.id != current.id,
                    )
                )
            ).scalars()
        )
        for warning in await self.warning_service.list_warnings(db, goal_id, active_only=True):
            if (
                warning.warning_type == "goal_closeout_skipped"
                and warning.source_process_run_id in predecessor_ids
            ):
                await self.warning_service.resolve_warning(
                    db,
                    warning,
                    resolved_by="orchestrator:process_rerun",
                    reason="process rerun fixed the skipped-process warning",
                )

    async def _signoff_authority(
        self, db: AsyncSession, goal: OrchestrationGoal
    ) -> tuple[str, object | None]:
        if goal.authority_model == "agent_manager" and goal.manager_agent_id is not None:
            manager = await db.get(Agent, goal.manager_agent_id)
            if manager is not None and manager.is_active:
                return "manager", manager.id
        return "human", None

    async def _signoff_decision(
        self, db: AsyncSession, goal: OrchestrationGoal, current: OrchestrationProcessRun
    ) -> OrchestrationAuthorityDecision | None:
        result = await db.execute(
            select(OrchestrationAuthorityDecision)
            .where(
                OrchestrationAuthorityDecision.goal_id == goal.id,
                OrchestrationAuthorityDecision.source_process_run_id == current.id,
            )
            .order_by(
                OrchestrationAuthorityDecision.asked_at.desc(),
                OrchestrationAuthorityDecision.created_at.desc(),
                OrchestrationAuthorityDecision.id.desc(),
            )
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _cancel_pending_decisions(
        self, db: AsyncSession, goal_id, *, reason: str = "goal closeout completed"
    ) -> list[str]:
        cancelled_ids = []
        for decision in await self.decision_service.list_decisions(db, goal_id, status="pending"):
            await self.decision_service.cancel_decision(db, decision, reason=reason)
            cancelled_ids.append(str(decision.id))
        return cancelled_ids

    @staticmethod
    async def _actions(db: AsyncSession, run: OrchestrationRun) -> list[OrchestrationAction]:
        result = await db.execute(
            select(OrchestrationAction)
            .where(OrchestrationAction.run_id == run.id)
            .order_by(OrchestrationAction.created_at.asc(), OrchestrationAction.id.asc())
        )
        return list(result.scalars().all())

    async def _write_closeout_memory(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        preconditions: dict,
        signoff: OrchestrationAuthorityDecision,
        cancelled_ids: list[str],
    ) -> None:
        rationale = {
            "outcome": "completed",
            "objective": goal.objective,
            "run_id": str(run.id),
            "declared_success_criteria": preconditions["declared_success_criteria"],
            "criterion_evidence": preconditions["criterion_evidence"],
            "accepted_non_summary_gates": preconditions["accepted_non_summary_gates"],
            "final_summary": preconditions["final_summary"],
            "warning_disposition": preconditions["warning_disposition"],
            "signoff": {
                "decision_id": str(signoff.id),
                "selected_option": signoff.selected_option,
                "reason": signoff.reason,
            },
        }
        warnings = await self.warning_service.list_warnings(db, goal.id)
        decisions = await self.decision_service.list_decisions(db, goal.id)
        reviews = await self.review_service.list_reviews(db, goal.id)
        recovery_actions = await self._recovery_actions(db, run)
        human_overrides = await self._human_override_evidence(db, run)
        lessons = {
            "warnings": [
                {
                    "id": str(warning.id),
                    "type": warning.warning_type,
                    "severity": warning.severity,
                    "active": warning.active,
                    "acknowledged": warning.acknowledged_at is not None,
                    "resolved_reason": warning.resolved_reason,
                }
                for warning in warnings
            ],
            "recommendation_overrides": [
                {
                    "decision_id": str(decision.id),
                    "selected_option": decision.selected_option,
                    "recommendation": decision.recommendation,
                    "reason": decision.reason,
                }
                for decision in decisions
                if decision.overrides_recommendation
            ],
            "agent_fit": [
                {
                    "agent_id": str(review.agent_id) if review.agent_id else None,
                    "approved_work_functions": review.approved_for_work_functions,
                    "strengths": review.strengths,
                    "risks": review.risks,
                }
                for review in reviews
            ],
            "recovery_actions": recovery_actions,
            "human_gate_overrides": human_overrides,
        }
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key=COMPLETION_RATIONALE_KEY,
            title="Completion rationale",
            body=_json_body(rationale),
            summary="Goal completion was authorized from accepted gates and evidence.",
            section_type="json",
            toc_order=90,
            run_id=run.id,
            created_by="orchestrator:goal_closeout",
        )
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key=LESSONS_LEARNED_KEY,
            title="Lessons learned",
            body=_json_body(lessons),
            summary="Recorded closeout lessons from durable orchestration data.",
            section_type="json",
            toc_order=100,
            run_id=run.id,
            created_by="orchestrator:goal_closeout",
        )
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key="open_questions",
            title="Open questions",
            body=_json_body(
                {
                    "status": "closed",
                    "reason": "goal closeout completed",
                    "cancelled_decision_ids": cancelled_ids,
                }
            ),
            summary="Open questions closed by goal closeout.",
            section_type="json",
            toc_order=80,
            run_id=run.id,
            created_by="orchestrator:goal_closeout",
        )

    @staticmethod
    async def _recovery_actions(db: AsyncSession, run: OrchestrationRun) -> list[dict]:
        result = await db.execute(
            select(OrchestrationAction)
            .where(
                OrchestrationAction.run_id == run.id,
                OrchestrationAction.action_type.in_(("retry_task", "reassign_task")),
            )
            .order_by(OrchestrationAction.created_at.asc(), OrchestrationAction.id.asc())
        )
        return [
            {
                "action_id": str(action.id),
                "action_type": action.action_type,
                "status": action.status,
                "request": action.request,
            }
            for action in result.scalars().all()
        ]

    @staticmethod
    async def _human_override_evidence(db: AsyncSession, run: OrchestrationRun) -> list[dict]:
        result = await db.execute(
            select(OrchestrationEvidence)
            .where(
                OrchestrationEvidence.run_id == run.id,
                OrchestrationEvidence.source_type == "human_override",
            )
            .order_by(OrchestrationEvidence.created_at.asc(), OrchestrationEvidence.id.asc())
        )
        return [
            {
                "gate_id": str(evidence.gate_id),
                "decision": evidence.evidence_metadata.get("decision"),
                "reason": evidence.evidence_metadata.get("reason"),
                "user_id": evidence.evidence_metadata.get("user_id"),
                "details": evidence.evidence_metadata.get("details", {}),
            }
            for evidence in result.scalars().all()
        ]
