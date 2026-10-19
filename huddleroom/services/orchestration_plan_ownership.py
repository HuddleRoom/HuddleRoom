"""Accepted-team ownership checks for executable plan items (SPR237)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.schemas.orchestration import PlanReviewRecord
from huddleroom.services.orchestration_agent_definition_review import accepted_work_functions
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_team_hierarchy import accepted_role_map


class PlanOwnershipService:
    """Resolve agent work against the accepted G2 hierarchy, never plan prose."""

    async def planning_inventory(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
    ) -> dict[str, Any]:
        root, hierarchy, team = goal, None, None
        process_service = OrchestrationProcessService()
        while True:
            hierarchy = await process_service.get_current(db, root.id, "team_hierarchy")
            team = (root.orchestrator_context or {}).get("team") or None
            if hierarchy is not None and hierarchy.status == "completed":
                team = team or hierarchy.outputs
            if team is not None and hierarchy is not None and hierarchy.status == "completed":
                break
            if root.parent_goal_id is None:
                team = None
                break
            root = await db.get(OrchestrationGoal, root.parent_goal_id)
            if root is None:  # dangling parent_goal_id: no accepted team reachable
                team = None
                break
        if hierarchy is None or hierarchy.status != "completed" or not isinstance(team, dict):
            return {"agents": [], "functions": {}, "coverage": [], "conflicts": ["current accepted team hierarchy is missing"]}
        outputs = hierarchy.outputs if isinstance(hierarchy.outputs, dict) else {}
        roles = accepted_role_map(team)
        eligibility = await accepted_work_functions(db, root.id)
        review = await process_service.get_current(db, root.id, "agent_definition_review")
        if review is None or review.status != "completed" or not isinstance(eligibility, dict):
            return {"agents": [], "functions": {}, "coverage": [], "conflicts": ["current accepted agent review is missing"]}
        try:
            agent_ids = {UUID(str(value)) for value in roles.values()}
        except (TypeError, ValueError):
            agent_ids = set()
        from sqlalchemy import select

        agents = (await db.scalars(select(Agent).where(Agent.id.in_(agent_ids), Agent.is_active.is_(True)))).all() if agent_ids else []
        active = {str(agent.id): agent for agent in agents}
        functions = {
            str(function): str(agent_id)
            for function, agent_id in roles.items()
            if str(agent_id) in active and function in eligibility.get(str(agent_id), [])
        }
        coverage = [item for item in outputs.get("responsibility_coverage", []) if isinstance(item, dict)]
        gaps = {str(item) for item in outputs.get("accepted_responsibility_gap_ids", [])}
        valid_approval = outputs.get("approval") in {"approve", "approve_with_documented_gaps"}
        gates = outputs.get("gates") if isinstance(outputs.get("gates"), dict) else {}
        evidence_ok = bool(outputs.get("approval_decision_id")) and gates.get("team_structure_reviewed") is True
        conflicts = [] if valid_approval and evidence_ok else ["team hierarchy acceptance evidence is incomplete"]
        for row in coverage:
            if row.get("status") not in {"covered", "gap"} or not row.get("evidence_ref"):
                conflicts.append("responsibility coverage is not accepted evidence")
                break
            if row.get("status") == "gap" and str(row.get("responsibility_id")) not in gaps:
                conflicts.append("responsibility gap is not accepted")
                break
        return {
            "agents": [{"id": str(agent.id), "name": agent.name} for agent in active.values()],
            "functions": functions,
            "coverage": coverage,
            "accepted_gap_ids": gaps,
            "hierarchy_process_id": str(hierarchy.id),
            "conflicts": conflicts,
        }

    async def team_prompt(self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun) -> str:
        inventory = await self.planning_inventory(db, goal, run)
        if inventory["conflicts"] or not inventory["functions"]:
            return ""
        names = {agent["id"]: agent["name"] for agent in inventory["agents"]}
        functions = inventory["functions"]
        lines = ["Accepted team (use only these agents):"]
        lines += [f"- {function}: {names.get(agent_id, agent_id)} ({agent_id})" for function, agent_id in sorted(functions.items())]
        lines.append(f"Allowed work_function values: {', '.join(sorted(functions))}")
        gap_ids = [str(row.get("responsibility_id")) for row in inventory["coverage"] if row.get("status") == "gap"]
        if gap_ids:
            lines.append(
                "Documented gaps (accepted with no owner; do not create plan items for them and never invent an agent; "
                "mention them in the summary): " + ", ".join(gap_ids)
            )
        lines.append(
            "Each task item's work_function must be one of the allowed values and agent_id must be that function's "
            "accepted agent id (or null to let HuddleRoom pick it). Set owner_type explicitly to agent for every task item. "
            "Do not use labels such as documentation or research. "
            "Ignore work_function values shown in examples; use only the allowed values above."
        )
        return "\n".join(lines)

    async def validate_plan_owners(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        items: Sequence[Any],
        *,
        bind: bool,
        review: Any,
    ) -> list[dict[str, Any]]:
        """Return per-item resolved ownership or an actionable conflict.

        Human rows are retained as explicit blocked prerequisites in the reduced
        MVP.  SPR254 owns decision creation/reconciliation, so this method does
        not turn a plan row into a human workflow.
        """
        inventory = await self.planning_inventory(db, goal, run)
        results: list[dict[str, Any]] = []
        bindings: dict[str, dict[str, str]] = {}
        for item in items:
            item_id = str(getattr(item, "id", getattr(item, "item_key", "")))
            if getattr(item, "owner_type", "agent") == "human":
                results.append({"item_id": item_id, "status": "blocked", "reason": "human executable work requires SPR254 workflow"})
                continue
            if inventory["conflicts"]:
                results.append({"item_id": item_id, "status": "conflict", "reason": inventory["conflicts"][0]})
                continue
            if getattr(item, "unit_type", None) == "goal":
                results.append({"item_id": item_id, "status": "eligible", "unit_type": "goal"})
                bindings[item_id] = {"unit_type": "goal"}
                continue
            function = getattr(item, "work_function", None)
            eligible_id = inventory["functions"].get(function)
            if not eligible_id:
                results.append({"item_id": item_id, "status": "conflict", "reason": f"no accepted active agent for work_function '{function}'"})
                continue
            selected = str(getattr(item, "agent_id", None) or eligible_id)
            if selected != eligible_id:
                results.append({"item_id": item_id, "status": "conflict", "reason": "agent is outside the accepted team"})
                continue
            evidence = getattr(item, "required_evidence", {})
            responsibility_id = evidence.get("responsibility_id") if isinstance(evidence, dict) else None
            evidence_ref = evidence.get("evidence_ref") if isinstance(evidence, dict) else None
            matching_coverage = [
                row for row in inventory["coverage"]
                if row.get("work_function") == function
                and (responsibility_id is None or row.get("responsibility_id") == responsibility_id)
                and (evidence_ref is None or row.get("evidence_ref") == evidence_ref)
            ]
            if not matching_coverage:
                results.append({"item_id": item_id, "status": "conflict", "reason": "accepted responsibility coverage is missing"})
                continue
            if responsibility_id is None and evidence_ref is None and len(matching_coverage) != 1:
                results.append({"item_id": item_id, "status": "conflict", "reason": "accepted responsibility coverage is ambiguous"})
                continue
            if any(row.get("status") == "gap" for row in matching_coverage):
                results.append({"item_id": item_id, "status": "conflict", "reason": "accepted responsibility gap requires revision"})
                continue
            if any(str(row.get("agent_ref")) != eligible_id for row in matching_coverage):
                results.append({"item_id": item_id, "status": "conflict", "reason": "domain ownership is not execution-team membership"})
                continue
            result = {"item_id": item_id, "status": "eligible", "agent_id": selected, "work_function": function}
            results.append(result)
            bindings[item_id] = {"agent_id": selected, "work_function": function}
        if bind and not isinstance(review, PlanReviewRecord):
            raise ValueError("ownership binding requires the current PlanReviewRecord")
        if bind and bindings:
            state = dict(run.plan_state or {})
            # The review identity is deliberately opaque until G4 supplies its
            # current-vs-history fingerprint contract.
            state["accepted_plan_ownership"] = {
                "review_identity": review.identity.model_dump(mode="json"), "items": bindings,
            }
            run.plan_state = state
            await db.flush()
        return results
