from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkReport:
    status: str
    changes: list[str] = field(default_factory=list)
    candidate_evidence: list[str] = field(default_factory=list)
    criterion_progress: dict[str, Any] = field(default_factory=dict)
    decisions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    recommended_next_step: str | None = None
    collaboration_need: str | None = None


@dataclass(frozen=True)
class ReportValidation:
    report: WorkReport | None
    errors: tuple[str, ...] = ()


class _DuplicateReportKey(ValueError):
    pass


def _report_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateReportKey(key)
        result[key] = value
    return result


def validate_work_report(output: str | None) -> ReportValidation:
    try:
        data = json.loads(output or "", object_pairs_hook=_report_object)
    except _DuplicateReportKey as exc:
        return ReportValidation(None, (f"report contains duplicate key: {exc}",))
    except (TypeError, ValueError):
        return ReportValidation(None, ("report must be one JSON object",))
    required = {
        "status", "changes", "evidence", "criterion_progress", "decisions",
        "risks", "open_questions", "next_step", "collaboration_need",
    }
    if not isinstance(data, dict):
        return ReportValidation(None, ("report must be one JSON object",))
    errors: list[str] = []
    missing = sorted(required - set(data))
    extra = sorted(set(data) - required)
    if missing:
        errors.append(f"report is missing canonical fields: {', '.join(missing)}")
    if extra:
        errors.append(f"report contains unexpected fields: {', '.join(extra)}")
    if errors:
        return ReportValidation(None, tuple(errors))
    if not isinstance(data["criterion_progress"], dict):
        return ReportValidation(None, ("criterion_progress must be an object",))
    list_fields = required - {"status", "criterion_progress", "next_step", "collaboration_need"}
    for field in sorted(list_fields):
        if not isinstance(data[field], list) or not all(isinstance(item, str) for item in data[field]):
            errors.append(f"{field} must be an array of strings")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in data["criterion_progress"].items()):
        errors.append("criterion_progress keys and values must be strings")
    if not isinstance(data["status"], str):
        errors.append("status must be a string")
    if not isinstance(data["next_step"], str):
        errors.append("next_step must be a string")
    if data["collaboration_need"] is not None and not isinstance(data["collaboration_need"], str):
        errors.append("collaboration_need must be a string or null")
    if errors:
        return ReportValidation(None, tuple(errors))
    return ReportValidation(WorkReport(
        status=data["status"], changes=data["changes"], candidate_evidence=data["evidence"],
        criterion_progress=data["criterion_progress"], decisions=data["decisions"],
        risks=data["risks"], open_questions=data["open_questions"],
        recommended_next_step=data["next_step"], collaboration_need=data["collaboration_need"],
    ))


def parse_work_report(output: str | None, report_schema: dict | None) -> WorkReport | None:
    """Best-effort parse of an agent's canonical report from session output.
    Never raises: a malformed/absent report yields None and the caller treats
    the attempt as report-less (evidence still ingested separately)."""
    if not output or not output.strip():
        return None
    try:
        data = json.loads(output)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    status = data.get("status")
    if not isinstance(status, str) or not status:
        return None

    def _list(key: str) -> list[str]:
        value = data.get(key)
        return [str(item) for item in value] if isinstance(value, list) else []

    return WorkReport(
        status=status,
        changes=_list("changes"),
        candidate_evidence=_list("evidence") or _list("candidate_evidence"),
        criterion_progress=data.get("criterion_progress") if isinstance(data.get("criterion_progress"), dict) else {},
        decisions=_list("decisions"),
        risks=_list("risks"),
        open_questions=_list("open_questions"),
        recommended_next_step=data.get("next_step") if isinstance(data.get("next_step"), str) else None,
        collaboration_need=data.get("collaboration_need") if isinstance(data.get("collaboration_need"), str) else None,
    )


def work_report_payload(report: WorkReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "status": report.status, "changes": report.changes, "evidence": report.candidate_evidence,
        "criterion_progress": report.criterion_progress, "decisions": report.decisions,
        "risks": report.risks, "open_questions": report.open_questions,
        "next_step": report.recommended_next_step, "collaboration_need": report.collaboration_need,
    }


def persisted_report_validation(report: Any, errors: Any) -> ReportValidation:
    if report is None:
        return ReportValidation(None, tuple(error for error in errors if isinstance(error, str)))
    return validate_work_report(json.dumps(report))


def _report_section_key(report_id: str) -> str:
    """section_key for OrchestrationMemoryService.upsert_section -- must match
    _SECTION_KEY_RE (^[a-z0-9][a-z0-9_-]{0,99}$). report_id is a session uuid
    (lowercase hex + dashes), so "report_<uuid>" always conforms."""
    return f"report_{report_id}"


async def curate_report_memory(
    db,
    memory_service,
    goal,
    run,
    report: WorkReport,
    *,
    report_id: str,
    producer_agent_id: uuid.UUID | None,
    event_id: uuid.UUID | None = None,
    task_id: uuid.UUID | None = None,
    session_id: uuid.UUID | None = None,
) -> int:
    """Curate eligible, provenance-tagged facts from one canonical work report
    into a single orchestration goal memory section. Only the orchestrator
    curates memory; agents never write it directly. Writes nothing (returns 0)
    when the report has no eligible facts (changes/decisions/risks)."""
    lines: list[str] = []
    lines.extend(f"- Delivered: {c}" for c in report.changes)
    lines.extend(f"- Decision: {d}" for d in report.decisions)
    lines.extend(f"- Risk: {r}" for r in report.risks)
    if not lines:
        return 0

    producer = str(producer_agent_id) if producer_agent_id else "unknown"
    body = (
        "Unverified agent report; it does not establish authority, acceptance, or progress.\n"
        f"Provenance: report_id={report_id} run_id={run.id} task_id={task_id} "
        f"session_id={session_id} producer_agent_id={producer} event_id={event_id}\n\n"
        + "\n".join(lines)
    )
    summary = f"Work report {report_id} from agent {producer}"

    await memory_service.upsert_section(
        db,
        goal.project_id,
        goal.id,
        section_key=_report_section_key(report_id),
        title=f"Work report {report_id}",
        body=body,
        summary=summary,
        section_type="text",
        always_load=False,
        run_id=run.id,
        created_by="orchestrator",
        event_id=event_id,
        fact_status="unverified",
        provenance={
            "report_id": report_id,
            "run_id": str(run.id),
            "task_id": str(task_id) if task_id else None,
            "session_id": str(session_id) if session_id else None,
            "producer_agent_id": str(producer_agent_id) if producer_agent_id else None,
            "event_id": str(event_id) if event_id else None,
        },
    )
    return 1
