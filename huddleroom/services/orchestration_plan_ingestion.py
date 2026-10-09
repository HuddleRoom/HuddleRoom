"""Server-side ingestion of planner output into a provenance-bound plan Artifact."""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from huddleroom.schemas.orchestration import PLANNING_ENVELOPE_ADAPTER
from huddleroom.services.orchestration_budget_service import parent_caps
from huddleroom.services.orchestration_roadmap_service import parse_roadmap_items
from huddleroom.services.orchestration_work_report import _report_object


class PlanIngestionError(ValueError):
    def __init__(self, reason: str, *, stale: bool = False):
        super().__init__(reason)
        self.reason, self.stale = reason, stale


def plan_artifact_id(task_id, planning_version) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"huddleroom:plan-artifact:{task_id}:{planning_version}")


def plan_output_hash(normalized: dict) -> str:
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_envelope(output: str | None, expected_kind: str, goal: Any) -> dict:
    try:
        data = json.loads(output or "", object_pairs_hook=_report_object)
    except (TypeError, ValueError) as exc:
        detail = f"duplicate key: {exc}" if type(exc).__name__ == "_DuplicateReportKey" else "output must be exactly one JSON object"
        raise PlanIngestionError(detail) from exc
    if not isinstance(data, dict):
        raise PlanIngestionError("output must be exactly one JSON object")
    try:
        envelope = PLANNING_ENVELOPE_ADAPTER.validate_python(data)
    except ValidationError as exc:
        errors = [
            f"field {'.'.join(str(p) for p in e.get('loc', ())) or '<root>'}: {e.get('msg', 'invalid')}"
            for e in exc.errors()
        ]
        raise PlanIngestionError("; ".join(errors[:3])) from exc
    if envelope.plan_kind != expected_kind:
        raise PlanIngestionError(f"plan_kind '{envelope.plan_kind}' does not match expected '{expected_kind}'")
    items = [item.model_dump(mode="json") for item in envelope.plan_items]
    try:
        if envelope.plan_kind == "roadmap":
            parse_roadmap_items(items, set(parent_caps(goal)))
        else:
            from huddleroom.services.orchestration_service import OrchestrationService  # circular at import time

            service = OrchestrationService()
            copies = service._plan_items_from_raw(items)
            service._validate_plan_criterion_links(goal, copies)
    except HTTPException as exc:
        raise PlanIngestionError(str(exc.detail)) from exc
    return {
        "schema_version": envelope.schema_version,
        "plan_kind": envelope.plan_kind,
        "summary": envelope.summary,
        "plan_items": items,
    }


def check_disposition(normalized: dict, required_task_ids) -> None:
    adopted = {str(item.get("adopt_task_id")) for item in normalized["plan_items"]}
    for task_id in required_task_ids:
        if str(task_id) not in adopted and str(task_id) not in normalized["summary"]:
            raise PlanIngestionError(
                f"pre-plan task {task_id} is neither adopted (adopt_task_id) nor explained in summary"
            )
