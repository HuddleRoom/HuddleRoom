"""Server-side ingestion of planner output into a provenance-bound plan Artifact."""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import object_session

from huddleroom.models.artifact import Artifact
from huddleroom.models.session import Session
from huddleroom.models.task import Task

from huddleroom.schemas.orchestration import PLANNING_ENVELOPE_ADAPTER
from huddleroom.services.orchestration_budget_service import parent_caps
from huddleroom.services.orchestration_result_contract import PLANNING_ENVELOPE
from huddleroom.services.orchestration_roadmap_service import parse_roadmap_items
from huddleroom.services.orchestration_work_report import _DuplicateReportKey, _report_object

INGESTION_SOURCE = "planning_output_ingestion"
BUDGET_REQUIRED_CODE = "roadmap_child_budget_required"
BUDGET_REQUIRED_MESSAGE = (
    "This roadmap needs explicit budget limits before it can create child goals. "
    "Create a new goal with limits for tokens, turns, or hours, then complete its review and Start work. "
    "No child work has been created."
)


class PlanIngestionError(ValueError):
    """``correctable``: the output itself is wrong (format, schema, keys), so an output-only correction can fix it.
    Stale and provenance rejections are never correctable."""

    def __init__(self, reason: str, *, stale: bool = False, code: str | None = None,
                 diagnostics: list[str] | None = None, correctable: bool = True):
        super().__init__(reason)
        self.reason, self.stale = reason, stale
        self.code = code
        self.diagnostics = diagnostics or []
        self.correctable = correctable and not stale and code is None


def plan_artifact_id(task_id, planning_version) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"huddleroom:plan-artifact:{task_id}:{planning_version}")


def plan_output_hash(normalized: dict) -> str:
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_envelope(output: str | None, expected_kind: str, goal: Any) -> dict:
    try:
        data = json.loads(output or "", object_pairs_hook=_report_object)
    except _DuplicateReportKey as exc:
        raise PlanIngestionError(f"duplicate key: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise PlanIngestionError("output must be exactly one JSON object") from exc
    if not isinstance(data, dict):
        raise PlanIngestionError("output must be exactly one JSON object")
    budget_required = (
        expected_kind == data.get("plan_kind") == "roadmap"
        and isinstance(data.get("plan_items"), list)
        and any(isinstance(item, dict) and item.get("unit_type") == "goal" for item in data["plan_items"])
        and not parent_caps(goal)
    )
    try:
        envelope = PLANNING_ENVELOPE_ADAPTER.validate_python(data)
    except ValidationError as exc:
        errors = [
            f"field {'.'.join(str(p) for p in e.get('loc', ())) or '<root>'}: {e.get('msg', 'invalid')}"
            for e in exc.errors()
        ]
        raise PlanIngestionError(
            BUDGET_REQUIRED_MESSAGE if budget_required else "; ".join(errors[:3]),
            code=BUDGET_REQUIRED_CODE if budget_required else None, diagnostics=errors,
        ) from exc
    if envelope.plan_kind != expected_kind:
        raise PlanIngestionError(f"plan_kind '{envelope.plan_kind}' does not match expected '{expected_kind}'")
    items = [item.model_dump(mode="json") for item in envelope.plan_items]
    for item in items:
        # ponytail: model_dump drops default-valued owner_type, so stored task items read as null. Goal items are agent-only.
        if item.get("unit_type") != "goal" and item.get("owner_type") is None:
            item["owner_type"] = "agent"
    try:
        if envelope.plan_kind == "roadmap":
            parse_roadmap_items(items, set(parent_caps(goal)))
        else:
            from huddleroom.services.orchestration_service import OrchestrationService  # circular at import time

            service = OrchestrationService()
            copies = service._plan_items_from_raw(items)
            service._validate_plan_criterion_links(goal, copies)
    except HTTPException as exc:
        raise PlanIngestionError(
            BUDGET_REQUIRED_MESSAGE if budget_required else str(exc.detail),
            code=BUDGET_REQUIRED_CODE if budget_required else None, diagnostics=[str(exc.detail)],
        ) from exc
    if budget_required:
        raise PlanIngestionError(BUDGET_REQUIRED_MESSAGE, code=BUDGET_REQUIRED_CODE)
    return {
        "schema_version": envelope.schema_version,
        "plan_kind": envelope.plan_kind,
        "summary": envelope.summary,
        "plan_items": items,
    }


async def linked_correction_session(db, run, task, session, plan_version) -> bool:
    """True only for the recorded session of ``task``'s current planning result correction.

    The planning task's ledger must name that exact correction task as current and have recorded this session, the
    correction must point back to the planning task, and the ledger must belong to ``plan_version``."""
    from huddleroom.models.orchestration import OrchestrationAction  # circular at import time

    marker = await db.scalar(select(OrchestrationAction).where(
        OrchestrationAction.run_id == run.id,
        OrchestrationAction.idempotency_key == f"run:{run.id}:kind:report_consumed:task:{task.id}",
    ))
    request = marker.request if marker is not None and isinstance(marker.request, dict) else {}
    ledger = request.get("result_correction") if isinstance(request.get("result_correction"), dict) else {}
    version = ledger.get("version") if isinstance(ledger.get("version"), dict) else {}
    correction = await db.get(Task, session.task_id) if session.task_id is not None else None
    return (
        correction is not None
        and ledger.get("kind") == PLANNING_ENVELOPE
        and ledger.get("status") in {"correcting", "corrected"}
        and ledger.get("correction_task_id") == str(correction.id)
        and ledger.get("correction_session_id") == str(session.id)
        and version.get("planning_version") == plan_version
        and (correction.metadata_ or {}).get("result_correction_parent_task_id") == str(task.id)
        and session.status == "completed"
    )


def check_disposition(normalized: dict, required_task_ids) -> None:
    adopted = {str(item.get("adopt_task_id")) for item in normalized["plan_items"]}
    for task_id in required_task_ids:
        if str(task_id) not in adopted and str(task_id) not in normalized["summary"]:
            # Deciding what happens to live pre-plan work is planning, not a format fix: a revision handles it.
            raise PlanIngestionError(
                f"pre-plan task {task_id} is neither adopted (adopt_task_id) nor explained in summary",
                correctable=False,
            )


async def ingest(db, goal, run, task, winning_session_id, plan_version, output) -> Artifact:
    """Bind the winning planner output to one deterministic plan Artifact per (task, planning_version)."""
    state = run.plan_state if isinstance(run.plan_state, dict) else {}
    pending = state.get("pending_replan") if isinstance(state.get("pending_replan"), dict) else {}
    current = pending.get("task_id") if pending else state.get("planning_task_id")
    plan_meta = (task.metadata_ or {}).get("orchestration_plan") or {}
    recorded = plan_meta.get("planning_version")
    if str(task.id) != str(current):
        raise PlanIngestionError("planning task is not the current planning task", stale=True)
    if plan_version != state.get("planning_version") or (recorded is not None and recorded != plan_version):
        raise PlanIngestionError(f"planning version {plan_version} is superseded", stale=True)

    session = await db.get(Session, winning_session_id) if winning_session_id else None
    if winning_session_id and (session is None or (
        session.task_id != task.id and not await linked_correction_session(db, run, task, session, plan_version)
    )):
        raise PlanIngestionError("winning session does not belong to the planning task", correctable=False)
    author = session.agent_id if session is not None else task.assigned_to
    expected_author = state.get("plan_author_agent_id")
    if author is None or author != task.assigned_to or (expected_author and str(author) != str(expected_author)):
        raise PlanIngestionError("plan output author is not the assigned planning agent", correctable=False)

    report_schema = ((task.metadata_ or {}).get("orchestration_contract") or {}).get("report_schema") or {}
    from huddleroom.services.orchestration_service import OrchestrationService, plan_kind_for  # circular at import time

    kind = report_schema.get("plan_kind") or plan_kind_for(goal, run)
    normalized = parse_envelope(output, kind, goal)
    service = OrchestrationService()
    raw_items = normalized["plan_items"]
    try:
        parsed = (
            parse_roadmap_items(raw_items, set(parent_caps(goal))) if kind == "roadmap"
            else service._plan_items_from_raw(raw_items)
        )
        # Validated before disposition: an invalid adoption never counts as disposing of pre-plan work.
        await service._validate_plan_adoptions(db, run, parsed)
    except HTTPException as exc:
        raise PlanIngestionError(str(exc.detail)) from exc
    from huddleroom.services.orchestration_plan_ownership import PlanOwnershipService  # circular at import time
    inventory = await PlanOwnershipService().planning_inventory(db, goal, run)
    if not inventory["conflicts"] and inventory["functions"]:
        # ponytail: function/agent membership only; coverage/gap checks stay at acceptance
        functions = inventory["functions"]
        errors = []
        for item in parsed:
            item_key = getattr(item, "item_key", None) or item.id
            if getattr(item, "unit_type", "task") == "goal" or item.owner_type == "human" or item.adopt_task_id is not None:
                continue
            if item.work_function not in functions:
                errors.append(f"item '{item_key}': work_function '{item.work_function}' is not in the accepted set "
                              f"[{', '.join(sorted(functions))}]")
            elif item.agent_id is not None and str(item.agent_id) != functions[item.work_function]:
                errors.append(f"item '{item_key}': agent_id is outside the accepted team for '{item.work_function}'")
            elif hasattr(item, "required_evidence"):  # roadmap items carry no evidence field
                ids = [str(r.get("responsibility_id")) for r in inventory["coverage"]
                       if r.get("status") == "covered" and r.get("work_function") == item.work_function]
                evidence = item.required_evidence if isinstance(item.required_evidence, dict) else {}
                rid = evidence.get("responsibility_id")
                if len(ids) > 1 and (rid is None and evidence.get("evidence_ref") is None or rid is not None and str(rid) not in ids):
                    errors.append(f"item '{item_key}': work_function '{item.work_function}' has several accepted "
                                  f"responsibilities [{', '.join(ids)}]; set required_evidence.responsibility_id to one of them")
        if errors:
            raise PlanIngestionError("; ".join(errors[:3]), diagnostics=errors)
    required = [uuid.UUID(str(i)) for i in plan_meta.get("pre_plan_task_ids") or []]
    live = (
        await db.execute(select(Task.id).where(Task.id.in_(required), Task.status != "cancelled"))
    ).scalars().all() if required else []
    check_disposition(normalized, live)

    artifact_id = plan_artifact_id(task.id, plan_version)
    output_hash = plan_output_hash(normalized)
    artifact = await db.get(Artifact, artifact_id)
    if artifact is None:
        artifact_values = {
            "id": artifact_id,
            "project_id": goal.project_id,
            "name": f"plan-{task.id}-v{plan_version}",
            "artifact_type": "plan",
            "status": "draft",
            "linked_task_id": task.id,
            "created_by_agent": author,
            "content_hash": output_hash,
            "metadata_": {
                "kind": f"{kind}_plan",
                "source": INGESTION_SOURCE,
                **normalized,
                "output_hash": output_hash,
                "winning_session_id": str(winning_session_id) if winning_session_id else None,
                "planning_task_id": str(task.id),
                "planning_version": plan_version,
                "plan_author_agent_id": str(author),
            },
        }
        if db.get_bind().dialect.name == "sqlite":
            # Ordinary DML keeps the write in the caller's transaction. Releasing
            # a first-write SAVEPOINT can commit independently on legacy SQLite.
            await db.execute(
                sqlite_insert(Artifact).values(**artifact_values).on_conflict_do_nothing(index_elements=[Artifact.id])
            )
            artifact = (await db.execute(select(Artifact).where(Artifact.id == artifact_id))).scalar_one()
        else:
            artifact = Artifact(**artifact_values)
            nested = await db.begin_nested()
            try:
                db.add(artifact)
                await db.flush()
            except IntegrityError:
                await nested.rollback()
                if object_session(artifact) is not None:
                    db.expunge(artifact)
                artifact = await db.get(Artifact, artifact_id)
                if artifact is None:
                    raise
            else:
                await nested.commit()
    if (artifact.metadata_ or {}).get("output_hash") != output_hash:
        raise PlanIngestionError("conflicting plan output for task/version", correctable=False)
    persisted = artifact.metadata_ or {}
    recorded_provenance = state.get("plan_provenance") if isinstance(state.get("plan_provenance"), dict) else None
    if recorded_provenance is not None and (
        persisted.get("planning_task_id") != str(task.id)
        or persisted.get("planning_version") != plan_version
        or persisted.get("winning_session_id") != (str(winning_session_id) if winning_session_id else None)
        or persisted.get("plan_author_agent_id") != str(author)
    ):
        raise PlanIngestionError("conflicting plan provenance for task/version", correctable=False)
    provenance = {
        "artifact_id": str(artifact.id), "planning_task_id": persisted.get("planning_task_id"),
        "planning_version": persisted.get("planning_version"), "winning_session_id": persisted.get("winning_session_id"),
        "plan_author_agent_id": persisted.get("plan_author_agent_id"), "canonical_content_hash": persisted.get("output_hash"),
    }
    run.plan_state = {
        **state, "plan_artifact_id": str(artifact_id), "plan_output_hash": output_hash,
        "plan_provenance": provenance,
    }
    return artifact


def ingested_plan_hash(metadata: dict) -> str:
    return plan_output_hash(canonical_plan_content(metadata))


def canonical_plan_content(metadata: dict) -> dict:
    """The immutable envelope used by ingestion, identity, and readable projections."""
    return {key: metadata.get(key) for key in ("schema_version", "plan_kind", "summary", "plan_items")}
