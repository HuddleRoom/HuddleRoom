from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from typing import Any, Awaitable, Callable, Mapping
from uuid import UUID

from huddleroom.config import settings
from huddleroom.services.orchestration_completion import (
    get_orchestration_completion,
    orchestration_runtime_metadata,
)
from huddleroom.schemas.agent import AgentCreate
from huddleroom.services.orchestration_agent_definition_analyzer import (
    _required_text,
    _text_list,
    _unfence_json,
    redact_semantic_payload,
)
from huddleroom.services.llm_structured_repair import complete_with_repair
from huddleroom.services.orchestration_llm_decision_adapter import (
    _extract_content,
    _full_completion_error,
    _redact_secrets,
    _safe_completion_error,
    orchestrator_preamble,
)


@dataclass(frozen=True)
class TeamHierarchyAnalysis:
    proposed_agents: tuple[dict[str, Any], ...]
    assignments: tuple[dict[str, str], ...]
    reporting_lines: tuple[dict[str, str], ...]
    documented_gaps: tuple[str, ...]
    rationale: str
    self_review: str
    feedback_results: tuple[dict[str, Any], ...] = ()
    responsibility_coverage: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return redact_semantic_payload({
            "proposed_agents": self.proposed_agents,
            "assignments": self.assignments,
            "reporting_lines": self.reporting_lines,
            "documented_gaps": self.documented_gaps,
            "rationale": self.rationale,
            "self_review": self.self_review,
            "feedback_results": self.feedback_results,
            "responsibility_coverage": self.responsibility_coverage,
        })


class TeamHierarchyAnalysisError(RuntimeError):
    def __init__(self, category: str, error: str, request: Any, raw_response: str | None = None):
        super().__init__(error)
        self.category = category
        self.full_error = error
        self.request = request
        self.raw_response = raw_response


def canonical_agent_create(
    definition: Mapping[str, Any] | AgentCreate, *, exact_keys: bool = False
) -> AgentCreate:
    if exact_keys:
        if not isinstance(definition, Mapping):
            raise ValueError("each proposed definition must be an AgentCreate object")
        missing = [
            name for name, field in AgentCreate.model_fields.items()
            if field.is_required() and name not in definition
        ]
        extra = sorted(set(definition) - set(AgentCreate.model_fields))
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing required keys: {', '.join(missing)}")
            if extra:
                details.append(f"extra keys: {', '.join(extra)}")
            raise ValueError("each proposed definition must be a complete AgentCreate object; " + "; ".join(details))
    parsed = AgentCreate.model_validate(definition)
    if any(
        not str(value or "").strip()
        for value in (parsed.name, parsed.role, parsed.provider, parsed.model)
    ) or (
        (not exact_keys or not isinstance(definition, Mapping) or "system_prompt" in definition)
        and not str(parsed.system_prompt or "").strip()
    ):
        raise ValueError("agent definition requires complete non-empty name, role, provider, model, and system_prompt")
    return parsed


def _object_list(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must contain objects")
    return value


def parse_team_hierarchy_analysis(payload: Any, request_payload: Mapping[str, Any]) -> TeamHierarchyAnalysis:
    if not isinstance(payload, Mapping):
        raise ValueError("team hierarchy analysis must be an object")
    required_keys = {
        "proposed_agents", "assignments", "reporting_lines", "documented_gaps", "rationale", "self_review",
    }
    has_context = isinstance(request_payload.get("hierarchy_context"), Mapping)
    if has_context:
        required_keys |= {"feedback_results", "responsibility_coverage"}
    if set(payload) != required_keys:
        raise ValueError("team hierarchy analysis must contain exactly the required fields")

    active_ids = {str(agent["id"]) for agent in request_payload.get("agents", [])}
    proposed: list[dict[str, Any]] = []
    proposal_ids: set[str] = set()
    for item in _object_list(payload, "proposed_agents"):
        if set(item) != {"proposal_id", "definition"}:
            raise ValueError("each proposed agent must contain proposal_id and definition")
        proposal_id = _required_text(item, "proposal_id")
        # ponytail: models sometimes echo the "proposal:" prefix into the id itself
        # even though refs use it; normalize so both forms resolve identically.
        if proposal_id.startswith("proposal:"):
            proposal_id = proposal_id[len("proposal:"):].strip()
            if not proposal_id:
                raise ValueError("each proposed agent must contain proposal_id and definition")
        definition = item.get("definition")
        parsed = canonical_agent_create(definition, exact_keys=True)
        canonical = json.dumps(parsed.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        excluded = {
            item.get("fingerprint")
            for item in request_payload.get("resolved_proposal_exclusions", [])
            if isinstance(item, Mapping)
        }
        if fingerprint in excluded:
            raise ValueError("proposed agent repeats a resolved definition")
        if proposal_id in proposal_ids:
            raise ValueError("proposed agents require unique ids")
        proposal_ids.add(proposal_id)
        proposed.append({"proposal_id": proposal_id, "definition": parsed.model_dump(mode="json")})

    valid_refs = active_ids | {f"proposal:{proposal_id}" for proposal_id in proposal_ids}
    required_work = set(request_payload.get("required_work_functions", []))
    assignments: list[dict[str, str]] = []
    assigned_functions: set[str] = set()
    producer_refs: set[str] = set()
    reviewer_refs: set[str] = set()
    for item in _object_list(payload, "assignments"):
        if set(item) != {"work_function", "agent_ref"}:
            raise ValueError("each assignment must contain work_function and agent_ref")
        work_function = _required_text(item, "work_function")
        agent_ref = _required_text(item, "agent_ref")
        if work_function not in required_work or agent_ref not in valid_refs or work_function in assigned_functions:
            raise ValueError("assignments must uniquely reference required work and known agents")
        assigned_functions.add(work_function)
        (reviewer_refs if work_function in {"review", "validation"} else producer_refs).add(agent_ref)
        assignments.append({"work_function": work_function, "agent_ref": agent_ref})
    if producer_refs & reviewer_refs:
        raise ValueError("producer and reviewer assignments must use different agents")

    assigned_refs = {item["agent_ref"] for item in assignments}
    selected_manager = request_payload.get("selected_manager")
    selected_manager_id = (
        str(selected_manager["id"])
        if (
            isinstance(selected_manager, Mapping)
            and selected_manager.get("kind") == "agent"
            and "id" in selected_manager
        )
        else None
    )
    reporting: list[dict[str, str]] = []
    reported_refs: set[str] = set()
    for item in _object_list(payload, "reporting_lines"):
        if set(item) != {"agent_ref", "reports_to"}:
            raise ValueError("each reporting line must contain agent_ref and reports_to")
        agent_ref = _required_text(item, "agent_ref")
        reports_to = _required_text(item, "reports_to")
        if agent_ref not in valid_refs:
            raise ValueError("reporting lines must reference known agents")
        if agent_ref not in assigned_refs:
            raise ValueError("reporting line agents must be assigned")
        if reports_to == selected_manager_id and (
            agent_ref == reports_to  # manager's own self-loop
            or selected_manager_id not in assigned_refs  # unassigned manager referenced by UUID
        ):
            reports_to = "manager"
        if (
            reports_to not in assigned_refs | {"manager"}
            or agent_ref == reports_to
        ):
            raise ValueError("reporting lines must reference assigned agents or the selected manager")
        if agent_ref in reported_refs:
            raise ValueError("each assigned agent may have only one reporting line")
        reported_refs.add(agent_ref)
        reporting.append({"agent_ref": agent_ref, "reports_to": reports_to})
    if reported_refs != assigned_refs:
        raise ValueError("every assigned agent requires exactly one reporting line")

    parent_by_ref = {item["agent_ref"]: item["reports_to"] for item in reporting}
    for agent_ref in assigned_refs:
        seen: set[str] = set()
        current = agent_ref
        while current != "manager":
            if current in seen or current not in parent_by_ref:
                raise ValueError("every reporting chain must be acyclic and terminate at manager")
            seen.add(current)
            current = parent_by_ref[current]

    gaps = _text_list(payload, "documented_gaps")
    missing_work = required_work - assigned_functions
    if len(gaps) != len(set(gaps)) or set(gaps) != missing_work:
        raise ValueError("documented gaps must exactly match unassigned required work")
    feedback_results: list[dict[str, Any]] = []
    responsibility_coverage: list[dict[str, Any]] = []
    if has_context:
        context = request_payload["hierarchy_context"]
        review_agent_by_id = {
            str(item.get("review_id")): str(item.get("agent_id"))
            for item in context.get("agent_reviews", [])
            if isinstance(item, Mapping)
            and isinstance(item.get("review_id"), str)
            and isinstance(item.get("agent_id"), str)
        }
        review_agent_by_decision_id = {
            str(item.get("accepted_decision_id")): (
                str(item.get("agent_id")),
                isinstance(item.get("accepted_definition"), Mapping),
            )
            for item in context.get("agent_reviews", [])
            if isinstance(item, Mapping)
            and isinstance(item.get("accepted_decision_id"), str)
            and isinstance(item.get("agent_id"), str)
        }
        eligible_by_agent = {
            str(item.get("agent_id")): set(item.get("approved_work_functions", []))
            for item in context.get("agent_reviews", [])
            if isinstance(item, Mapping) and isinstance(item.get("agent_id"), str)
        }
        feedback_ids: set[str] = set()
        for item in _object_list(payload, "feedback_results"):
            if set(item) != {"item_id", "source_excerpt", "disposition", "reason", "change_refs"}:
                raise ValueError("each feedback result must contain item_id, source_excerpt, disposition, reason, and change_refs")
            item_id = _required_text(item, "item_id")
            if item_id in feedback_ids:
                raise ValueError("feedback results require unique item ids")
            disposition = _required_text(item, "disposition")
            if disposition not in {"applied", "declined", "impossible"}:
                raise ValueError("feedback result disposition must be applied, declined, or impossible")
            refs = item.get("change_refs")
            if not isinstance(refs, list) or any(not isinstance(ref, str) or not ref.strip() for ref in refs):
                raise ValueError("feedback result change_refs must contain text")
            feedback_ids.add(item_id)
            feedback_results.append({
                "item_id": item_id,
                "source_excerpt": _required_text(item, "source_excerpt"),
                "disposition": disposition,
                "reason": _required_text(item, "reason"),
                "change_refs": list(refs),
            })
        if context.get("revision") is None and feedback_results:
            raise ValueError("feedback results require a revision context")
        revision = context.get("revision")
        if isinstance(revision, Mapping):
            if str(revision.get("feedback_text") or "").strip() and not feedback_results:
                raise ValueError("feedback results must account for request_changes feedback")
            frozen_items = revision.get("feedback_items")
            if not isinstance(frozen_items, list):
                raise ValueError("revision feedback items are invalid")
            frozen_by_id = {
                item.get("item_id"): item.get("source_excerpt")
                for item in frozen_items
                if isinstance(item, Mapping)
                and isinstance(item.get("item_id"), str)
                and isinstance(item.get("source_excerpt"), str)
            }
            if frozen_items:
                if len(frozen_by_id) != len(frozen_items) or {item["item_id"] for item in feedback_results} != set(frozen_by_id):
                    raise ValueError("feedback results must exactly account for frozen feedback items")
                if any(item["source_excerpt"] != frozen_by_id[item["item_id"]] for item in feedback_results):
                    raise ValueError("feedback source excerpts must match frozen feedback items")
            previous = revision.get("previous_proposal")
            if not isinstance(previous, Mapping):
                raise ValueError("revision previous proposal is invalid")
            previous_assignments = {
                item.get("work_function"): item.get("agent_ref")
                for item in previous.get("assignments", []) if isinstance(item, Mapping)
            }
            changed_refs = {
                f"assignment:{work_function}"
                for work_function in set(previous_assignments) | {item["work_function"] for item in assignments}
                if previous_assignments.get(work_function) != next(
                    (item["agent_ref"] for item in assignments if item["work_function"] == work_function), None
                )
            }
            previous_reporting = {
                item.get("agent_ref"): item.get("reports_to")
                for item in previous.get("reporting_lines", []) if isinstance(item, Mapping)
            }
            current_reporting = {item["agent_ref"]: item["reports_to"] for item in reporting}
            changed_refs.update(
                f"reporting:{agent_ref}"
                for agent_ref in set(previous_reporting) | set(current_reporting)
                if previous_reporting.get(agent_ref) != current_reporting.get(agent_ref)
            )
            coverage_fact = lambda item: {
                key: value for key, value in item.items() if key not in {"reason"}
            }
            previous_coverage = {
                item.get("responsibility_id"): coverage_fact(item)
                for item in previous.get("responsibility_coverage", []) if isinstance(item, Mapping)
            }
            current_coverage = {
                item.get("responsibility_id"): coverage_fact(item)
                for item in _object_list(payload, "responsibility_coverage")
            }
            changed_refs.update(
                f"responsibility:{responsibility_id}"
                for responsibility_id in set(previous_coverage) | set(current_coverage)
                if previous_coverage.get(responsibility_id) != current_coverage.get(responsibility_id)
            )
            previous_gaps = set(previous.get("documented_gaps", []))
            changed_refs.update(
                f"generic_gap:{work_function}"
                for work_function in previous_gaps | set(gaps)
                if (work_function in previous_gaps) != (work_function in gaps)
            )
            for item in feedback_results:
                if item["disposition"] == "applied" and (
                    not item["change_refs"] or not set(item["change_refs"]) <= changed_refs
                ):
                    if not item["change_refs"]:
                        raise ValueError("applied feedback requires a changed reference")
                    raise ValueError("applied feedback references must reference actual changed hierarchy facts")

        responsibility_ids: set[str] = set()
        for item in _object_list(payload, "responsibility_coverage"):
            expected = {
                "responsibility_id", "responsibility", "source_ref", "work_function", "agent_ref",
                "status", "reason", "evidence_ref",
            }
            if set(item) != expected:
                raise ValueError("each responsibility coverage entry must contain the required fields")
            responsibility_id = _required_text(item, "responsibility_id")
            if responsibility_id in responsibility_ids:
                raise ValueError("responsibility coverage requires unique ids")
            work_function = _required_text(item, "work_function")
            if work_function not in required_work:
                raise ValueError("responsibility coverage work_function must be required")
            source_ref = _required_text(item, "source_ref")
            criteria = request_payload.get("goal", {}).get("success_criteria", [])
            criterion_refs = {f"criterion:{index}" for index, _ in enumerate(criteria, start=1)} if isinstance(criteria, list) else set()
            frozen_feedback_refs = {
                str(frozen.get("item_id")) for frozen in (revision.get("feedback_items", []) if isinstance(revision, Mapping) else [])
                if isinstance(frozen, Mapping) and isinstance(frozen.get("item_id"), str)
            }
            feedback_refs = frozen_feedback_refs or {item["item_id"] for item in feedback_results}
            if source_ref != "objective" and source_ref not in criterion_refs and source_ref not in feedback_refs:
                raise ValueError("responsibility coverage source_ref must identify the objective, criterion, or feedback")
            status = _required_text(item, "status")
            agent_ref = item.get("agent_ref")
            evidence_ref = _required_text(item, "evidence_ref")
            if status == "proposed":
                if not isinstance(agent_ref, str) or agent_ref not in valid_refs or not agent_ref.startswith("proposal:") or evidence_ref != agent_ref:
                    raise ValueError("proposed responsibility coverage requires its proposal evidence")
            elif status == "covered":
                if not isinstance(agent_ref, str) or agent_ref not in active_ids:
                    raise ValueError("covered responsibility coverage requires an active agent")
                if evidence_ref.startswith("agent_review:"):
                    review_id = evidence_ref.removeprefix("agent_review:")
                    evidence_agent = review_agent_by_id.get(review_id)
                elif evidence_ref.startswith("agent_decision:"):
                    decision_id = evidence_ref.removeprefix("agent_decision:")
                    decision_evidence = review_agent_by_decision_id.get(decision_id)
                    evidence_agent = decision_evidence[0] if decision_evidence else None
                    if not decision_evidence or not decision_evidence[1]:
                        raise ValueError("covered responsibility coverage requires the accepted definition")
                else:
                    raise ValueError("covered responsibility coverage requires accepted review evidence")
                if (
                    evidence_agent != agent_ref
                    or work_function not in eligible_by_agent.get(agent_ref, set())
                ):
                    raise ValueError("covered responsibility coverage evidence is not accepted for this work_function")
            elif status == "gap":
                if agent_ref is not None:
                    raise ValueError("gap responsibility coverage must have no owner")
            else:
                raise ValueError("responsibility coverage status must be covered, proposed, or gap")
            responsibility_ids.add(responsibility_id)
            responsibility_coverage.append({
                "responsibility_id": responsibility_id,
                "responsibility": _required_text(item, "responsibility"),
                "source_ref": source_ref,
                "work_function": work_function,
                "agent_ref": agent_ref,
                "status": status,
                "reason": _required_text(item, "reason"),
                "evidence_ref": evidence_ref,
            })

    return TeamHierarchyAnalysis(
        tuple(proposed), tuple(assignments), tuple(reporting), gaps,
        _required_text(payload, "rationale"), _required_text(payload, "self_review"),
        tuple(feedback_results), tuple(responsibility_coverage),
    )


class TeamHierarchyAnalyzer:
    def __init__(self, completion_fn: Callable[..., Awaitable[Any]] | None = None) -> None:
        self._completion_fn = get_orchestration_completion(completion_fn)

    @staticmethod
    def build_request(payload: Mapping[str, Any], project=None) -> dict[str, Any]:
        safe_payload = redact_semantic_payload(payload)
        required_agent_create_keys = ", ".join(
            name for name, field in AgentCreate.model_fields.items() if field.is_required()
        )
        allowed_agent_create_keys = ", ".join(AgentCreate.model_fields)
        messages = [
            {"role": "system", "content": (
                orchestrator_preamble(project, goal=payload.get("goal")) + "\n\n"
                "Act as the team architect. Design the smallest team and reporting structure that covers "
                "the required work using the supplied active roster before proposing new agents. "
                "The structured agents roster is authoritative for this request's agent configuration, including "
                "runtime and model. Project, goal and preparation prose is unversioned background and must not "
                "override that roster. Reviewed definition snapshots are historical review evidence; they do not "
                "override current roster settings or establish the runtime/model used by an earlier call. "
                "An unspecified or native-default model remains unspecified/default. For every "
                "new agent, design a complete AgentCreate-compatible definition. Its required keys: "
                f"{required_agent_create_keys}; allowed keys: {allowed_agent_create_keys}; defaulted AgentCreate "
                "fields may be omitted. Include a role-specific "
                "system_prompt with concrete behavior, boundaries, and handoff instructions; never include "
                "project secrets or goal-specific facts in reusable agent prompts. Assign each work function "
                "once, use only exact required_work_functions, and reference active agents by UUID or new "
                "agents as proposal:<proposal_id>. Preserve independent producer/reviewer separation. Document "
                "every uncovered gap using each exact unassigned required_work_functions identifier once. Before "
                "answering, self-review definition completeness, all references, "
                "coverage, reporting lines (each reporting line's agent_ref is an assigned agent; reports_to is "
                "either another assigned agent's ref or the literal token \"manager\" — use the literal "
                "\"manager\" for the top of the chain, never the selected manager's UUID), and reviewer "
                "independence. Return JSON only with exactly: proposed_agents [{proposal_id, definition}], assignments "
                "[{work_function, agent_ref}], reporting_lines [{agent_ref, reports_to}], documented_gaps "
                "[string], rationale "
                "(string), self_review (string). Never repeat a definition fingerprint listed in "
                "resolved_proposal_exclusions; propose only a materially different definition when still needed. "
                "When hierarchy_context is present, also return feedback_results "
                "[{item_id, source_excerpt, disposition, reason, change_refs}] and responsibility_coverage "
                "[{responsibility_id, responsibility, source_ref, work_function, agent_ref, status, reason, evidence_ref}]. "
                "Use only accepted agent_review evidence for covered existing agents, accepted agent_decision evidence for "
                "covered newly approved agents, proposal:<id> evidence for proposed agents, "
                "and a gap with no owner when evidence is missing. Treat each independent requested change as a "
                "separate feedback item and return exactly one outcome for each item."
            )},
            {"role": "user", "content": json.dumps(safe_payload, sort_keys=True, default=str)},
        ]
        return {"model": settings.orchestration_model, "messages": messages,
                "response_format": {"type": "json_object"}, "temperature": 0, "max_tokens": 32768}

    async def review(self, payload: Mapping[str, Any], project=None, *, project_id: UUID | None = None) -> TeamHierarchyAnalysis:
        return await self.review_request(self.build_request(payload, project=project), project_id=project_id)

    async def review_request(self, request: Mapping[str, Any], *, project_id: UUID | None = None) -> TeamHierarchyAnalysis:
        request = redact_semantic_payload(request)
        messages = request.get("messages", [])
        payload = json.loads(messages[1]["content"])
        completion_request = dict(request)
        if request.get("model") == "openrouter/minimax/minimax-m3":
            completion_request.pop("response_format", None)
        try:
            args = (self._completion_fn, completion_request, lambda raw: parse_team_hierarchy_analysis(json.loads(_unfence_json(raw)), payload))
            if project_id is None:
                return await complete_with_repair(*args)
            from huddleroom.services.agent_response_stream import AgentResponseInvocation, InvocationContext
            invocation_kind, runtime_model = orchestration_runtime_metadata(self._completion_fn, request["model"])
            return await complete_with_repair(*args, invocation=AgentResponseInvocation(InvocationContext(
                project_id, "system", "orchestrator", "Orchestrator", invocation_kind, "team_hierarchy",
                runtime_model,
                "Design the smallest team and reporting structure that covers the required work.",
            )))
        except Exception as exc:
            if settings.debug:
                formatter = (
                    _full_completion_error
                    if os.environ.get("HUDDLEROOM_ORCHESTRATION_BASELINE_E2E") == "true"
                    or os.environ.get("RALLY_ORCHESTRATION_BASELINE_E2E") == "true"
                    else _safe_completion_error
                )
                print(f"LLM failed model={request['model']} error={formatter(exc)}", flush=True)
            # ponytail: complete_with_repair exhausts retries and re-raises, no raw exposed
            raise TeamHierarchyAnalysisError(
                "invalid_response",
                _full_completion_error(exc), redact_semantic_payload(request),
                None,
            ) from exc
