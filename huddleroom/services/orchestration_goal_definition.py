"""Goal-weight classification and Baseline Process A (goal definition).

Spec 6.2 and 7, Phase 5 slice: classification and the clarification
process is orchestrator-owned. Adaptive analysis determines whether human
clarification is needed; never expose it through agent-facing surfaces.
"""
from __future__ import annotations

from copy import deepcopy
import logging
import re
from types import SimpleNamespace

from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.config import settings
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_goal_analyzer import GoalClarificationAnalyzer
from huddleroom.services.orchestration_llm_decision_adapter import _redact_secrets, _safe_completion_error
import json
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

logger = logging.getLogger(__name__)


def _redacted_json(value: dict) -> dict:
    return json.loads(_redact_secrets(json.dumps(value, sort_keys=True, default=str)))

GOAL_WEIGHT_ORDER = {"trivial": 0, "standard": 1, "substantial": 2}

# Weighted-signal classifier (spec 6.2, plan Deviation 1). Each signal
# contributes points; a signal strong enough to justify "substantial" on its
# own (multiple success criteria, phrase-level independent verification, a
# large budget, or multiple inferred/flagged work functions) is weighted to
# hit the threshold by itself. A tiny explicit budget is neutral per spec
# 6.2; constraints presence is the only standard-only signal here. All
# constants are module-level so the human override endpoint (Task 3) remains
# the escape hatch for anything the proxies get wrong.
SUCCESS_CRITERIA_SUBSTANTIAL_COUNT = 2
SUCCESS_CRITERIA_SUBSTANTIAL_SCORE = 3

CONSTRAINTS_PRESENT_SCORE = 1

# Budget is free-form JSON (`{"caps": {...}}`); size is read from whichever
# recognized numeric cap keys are present. Only a cap at/above its threshold
# changes weight (+3, substantial on its own). Tiny caps and unrecognized
# budget notes stay neutral so "fix typo in README" with a tiny explicit cap
# remains trivial.
BUDGET_LARGE_THRESHOLDS = {
    "max_tokens": 150_000,
    "max_cost_usd": 50,
    "max_hours": 8,
    "max_turns": 30,
}
BUDGET_LARGE_SCORE = 3

# True agent work-function inference needs Phase 8's work-function
# catalogue and is out of scope here. This is a crude proxy only: count
# distinct work-verb keywords appearing in the objective. Spec 6.2 lists
# "multiple inferred work functions" as sufficient for substantial on its
# own (same tier as multiple success criteria / independent verification),
# so two or more distinct verbs scores the full substantial threshold, not
# a partial nudge. A human can also flag this directly at creation
# (explicit_multi_work_function=True) without depending on objective
# wording -- same score either way.
WORK_FUNCTION_VERBS = (
    "design", "build", "implement", "test", "deploy", "review",
    "migrate", "integrate", "document", "research",
)
MULTI_WORK_FUNCTION_COUNT = 2
MULTI_WORK_FUNCTION_SCORE = 3  # = SUBSTANTIAL_SCORE_THRESHOLD, defined below; substantial alone

# Independent-verification signal (spec 6.2): a phrase-level scan over
# success-criteria text AND the objective. Deliberately narrow: a bare
# "verified"/"audited" with no independence qualifier is NOT a signal --
# that would overclassify any ordinary verified success criterion as
# requiring independent verification. Only "independently
# verified/reviewed/audited" (either word order), "third-party
# verified/reviewed/audited", or "externally verified/reviewed/audited"
# counts. The action-then-independent alternative allows at most 2 words
# between them (e.g. "verify the report independently") -- kept short so a
# later clause with its own verb ("review changes then merge independently")
# can't bridge into "independently" and false-positive.
INDEPENDENT_VERIFICATION_SCORE = 3
_VERIFICATION_ACTION = r"(?:review\w*|verif\w*|audit\w*)"
_VERIFICATION_PATTERN = re.compile(
    rf"\b(independent(?:ly)?\s+{_VERIFICATION_ACTION}"
    rf"|{_VERIFICATION_ACTION}\s+(?:by\s+an?\s+)?independent\w*"
    rf"|{_VERIFICATION_ACTION}(?:\s+\w+){{0,2}}\s+independent(?:ly)?"
    rf"|third[- ]party\s+{_VERIFICATION_ACTION}"
    rf"|external(?:ly)?\s+{_VERIFICATION_ACTION})\b",
    re.IGNORECASE,
)

SUBSTANTIAL_SCORE_THRESHOLD = 3


def _budget_score(budget: dict) -> int:
    if not budget:
        return 0
    caps = budget.get("caps") if isinstance(budget, dict) else None
    if isinstance(caps, dict):
        for key, threshold in BUDGET_LARGE_THRESHOLDS.items():
            value = caps.get(key)
            if isinstance(value, (int, float)) and value >= threshold:
                return BUDGET_LARGE_SCORE
    return 0


def _multi_work_function_score(objective: str, explicit_multi_work_function: bool) -> int:
    if explicit_multi_work_function:
        return MULTI_WORK_FUNCTION_SCORE
    text = (objective or "").lower()
    matched = {verb for verb in WORK_FUNCTION_VERBS if re.search(rf"\b{verb}\w*\b", text)}
    return MULTI_WORK_FUNCTION_SCORE if len(matched) >= MULTI_WORK_FUNCTION_COUNT else 0


def requires_independent_verification(success_criteria: list, objective: str) -> bool:
    if objective and _VERIFICATION_PATTERN.search(objective):
        return True
    for criterion in success_criteria or []:
        if not isinstance(criterion, dict):
            continue
        for value in criterion.values():
            if isinstance(value, str) and _VERIFICATION_PATTERN.search(value):
                return True
    return False


def classify_goal_weight(
    success_criteria: list,
    constraints: dict,
    budget: dict,
    objective: str = "",
    explicit_multi_work_function: bool = False,
) -> str:
    count = len(success_criteria or [])
    score = 0
    if count >= SUCCESS_CRITERIA_SUBSTANTIAL_COUNT:
        score += SUCCESS_CRITERIA_SUBSTANTIAL_SCORE

    if constraints:
        score += CONSTRAINTS_PRESENT_SCORE

    score += _budget_score(budget)
    score += _multi_work_function_score(objective, explicit_multi_work_function)

    if requires_independent_verification(success_criteria, objective):
        score += INDEPENDENT_VERIFICATION_SCORE

    if score == 0:
        return "trivial"
    if score >= SUBSTANTIAL_SCORE_THRESHOLD:
        return "substantial"
    return "standard"


GOAL_DEFINITION_PROCESS_VERSION = 1
DECISION_KEY_PREFIX = "goal_definition:"
MAX_CLARIFICATION_ROUNDS = 2
MEMORY_SECTION_KEY = "goal_definition"
MEMORY_TOC_ORDER = 10


def required_decision_keys_for_weight(goal: OrchestrationGoal) -> set[str]:
    """Compatibility hook: adaptive questions are not weight-derived."""
    return set()


def _repair_success_criteria_keys(goal: OrchestrationGoal) -> None:
    """Deterministic auto-repair, not a human-facing gap: final-summary
    (orchestration_service.py) requires every success criterion to have a
    non-blank, unique `key`. Assign/dedupe synthetic keys so goal-definition
    never lets a goal through that final-summary will later reject."""
    criteria = list(goal.success_criteria or [])
    if not criteria:
        return
    used: set[str] = set()
    repaired = []
    for index, criterion in enumerate(criteria):
        if not isinstance(criterion, dict):
            repaired.append(criterion)
            continue
        raw_key = criterion.get("key")
        key = raw_key.strip() if isinstance(raw_key, str) else ""
        if not key or key in used:
            base = f"criterion_{index + 1}"
            candidate = base
            suffix = 2
            while candidate in used:
                candidate = f"{base}_{suffix}"
                suffix += 1
            key = candidate
        used.add(key)
        if key != raw_key:
            criterion = {**criterion, "key": key}
        repaired.append(criterion)
    goal.success_criteria = repaired


def _append_context_value(context: dict, destination: str, value: str) -> None:
    key = destination.removeprefix("orchestrator_context.")
    if key == "resolved_clarifications":
        return
    values = list(context.get(key, []))
    values.append(value)
    context[key] = values


def _merge_answer(goal: OrchestrationGoal, context: dict, decision_key: str, question: str, answer: str) -> dict:
    destination = decision_key.rsplit(":", 1)[1]
    answer = answer.strip()
    audit = {
        "question": question,
        "answer": answer,
        "destination": destination,
        "round": int(context.get("clarification_round", 0)),
    }
    resolved = list(context.get("resolved_clarifications", []))
    resolved.append(audit)
    context["resolved_clarifications"] = resolved
    if destination == "objective":
        _append_context_value(context, "orchestrator_context.objective_notes", answer)
    elif destination == "success_criteria":
        criteria = list(goal.success_criteria or [])
        criteria.append({"key": f"clarification_{len(criteria) + 1}", "description": answer})
        goal.success_criteria = criteria
    elif destination.startswith("orchestrator_context."):
        _append_context_value(context, destination, answer)
    # an answered clarification is authoritative for its destination;
    # drop superseded provisional assumptions so they cannot contradict it
    context["assumptions"] = [
        prior for prior in context.get("assumptions", [])
        if not isinstance(prior, dict) or prior.get("destination") != destination
    ]
    return audit


def goal_objective_with_clarifications(goal: OrchestrationGoal) -> str:
    """Return goal.objective combined with any clarification notes from orchestrator_context.

    If objective_notes exist (from answered clarifications with destination=='objective'),
    appends them as a 'Clarifications:' block below the original objective. Otherwise,
    returns objective unchanged.
    """
    text = goal.objective or ""
    context = goal.orchestrator_context or {}
    notes = context.get("objective_notes")
    if notes and isinstance(notes, list) and notes:
        notes_text = "\n".join(f"- {note}" for note in notes if note)
        if notes_text:
            text = f"{text}\n\nClarifications:\n{notes_text}"
    return text


class GoalDefinitionProcess:
    """Adaptive goal-definition slice of Baseline Process A (spec 7).

    Called from the tick before anything else advances (spec 6.3). Never
    blocks the tick: a parked process returns its summary and the tick
    continues (it DOES block this goal's own forward progress until
    terminal -- see `_ensure_baseline_processes_ready`). The analyzer asks
    only material questions; a safe result opens the four 7.7 gates here.
    """

    def __init__(self, analyzer: GoalClarificationAnalyzer | None = None) -> None:
        self.analyzer = analyzer or GoalClarificationAnalyzer()
        self.process_service = OrchestrationProcessService()
        self.decision_service = OrchestrationAuthorityDecisionService()
        self.memory_service = OrchestrationMemoryService()
        self.warning_service = OrchestrationWarningService()

    async def advance(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, *, manual: bool = False
    ) -> dict:
        current = await self.process_service.get_current(db, goal.id, "goal_definition")
        if current is not None and current.status in ("completed", "skipped"):
            return {"process_type": "goal_definition", "status": current.status, "questions_created": 0}
        if current is None:
            if not (run.baseline_authorized or manual):
                return {"process_type": "goal_definition", "status": "not_started", "questions_created": 0}
            current = await self.process_service.start_process(
                db, goal.id, process_type="goal_definition",
                trigger_reason="tick: no goal_definition process run on record", run_id=run.id,
                input_snapshot={"weight": goal.weight}, process_version=GOAL_DEFINITION_PROCESS_VERSION,
            )
        retry_checkpoint = (current.outputs or {}).get("_lm_retry")
        if isinstance(retry_checkpoint, dict):
            return {
                "process_type": "goal_definition",
                "status": current.status,
                "retryable": True,
                "questions_created": 0,
                "error": retry_checkpoint.get("error"),
            }

        decisions = await self.decision_service.list_decisions(db, goal.id)
        pending = [
            decision for decision in decisions
            if decision.status == "pending" and decision.decision_key.startswith(f"{DECISION_KEY_PREFIX}adaptive:")
        ]
        if pending:
            await self.process_service.park_process(db, current)
            return {"process_type": "goal_definition", "status": current.status, "questions_created": 0}

        context, working_goal = self._continuation_state(goal, decisions)
        await self.process_service.resume_process(db, current)

        round_number = int(context.get("clarification_round", 0))
        project = await db.get(Project, goal.project_id)
        snapshot = {
            "objective": working_goal.objective,
            "original_request": goal.original_request,
            "success_criteria": working_goal.success_criteria,
            "constraints": goal.constraints,
            "budget": goal.budget,
            "orchestrator_context": context,
            "clarification_round": round_number,
        }
        request = _redacted_json(self.analyzer.build_request(snapshot, project={"name": project.name, "description": project.description} if project else None))
        try:
            analysis = await self.analyzer.analyze_request(request, project_id=goal.project_id)
        except Exception as exc:
            from huddleroom.services.orchestration_service import OrchestrationService

            safe_error = _safe_completion_error(exc)
            logger.error(
                "Goal-definition analyzer failed (model=%s, error=%s)",
                settings.orchestration_model,
                safe_error,
            )
            current.outputs = {
                **(current.outputs or {}),
                "error": safe_error,
                "retryable": True,
                "_lm_retry": {
                    "version": 1,
                    "kind": "goal_analysis",
                    "request": request,
                    "continuation": {
                        "clarification_round": round_number,
                        "context": _redacted_json(context),
                        "working_goal": _redacted_json({"objective": working_goal.objective, "success_criteria": working_goal.success_criteria}),
                    },
                    "error": safe_error,
                },
            }
            OrchestrationService._mark_run_blocked(goal, run)
            warning = await self.warning_service.create_warning(
                db,
                goal.id,
                warning_type="goal_definition_analyzer_error",
                severity="warning",
                message=f"Goal definition analysis failed: {safe_error}",
                run_id=run.id,
                source_process_run_id=current.id,
            )
            checkpoint = dict(current.outputs["_lm_retry"])
            checkpoint["warning_id"] = str(warning.id)
            current.outputs = {**current.outputs, "_lm_retry": checkpoint}
            OrchestrationService._upsert_active_blocker(run, {
                "kind": "goal_definition_analyzer_error",
                "gate_id": str(warning.id),
                "warning_id": str(warning.id),
                "reason": f"Goal analysis failed: {safe_error}",
            })
            return {
                "process_type": "goal_definition",
                "status": current.status,
                "retryable": True,
                "questions_created": 0,
                "error": safe_error,
            }

        return await self._continue_after_analysis(
            db, goal, run, current, analysis, working_goal, context, round_number,
            had_analyzer_error=False,
        )

    async def retry_failed(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, current
    ) -> dict:
        checkpoint = (current.outputs or {}).get("_lm_retry")
        if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1 or checkpoint.get("kind") != "goal_analysis":
            raise ValueError("goal definition has no retryable analyzer request")
        request = checkpoint.get("request")
        if not isinstance(request, dict):
            raise ValueError("goal definition retry request is invalid")
        try:
            analysis = await self.analyzer.analyze_request(request, project_id=goal.project_id)
        except Exception as exc:
            safe_error = _safe_completion_error(exc)
            checkpoint["error"] = safe_error
            current.outputs = {**(current.outputs or {}), "error": safe_error, "retryable": True,
                               "_lm_retry": checkpoint}
            warning_id = checkpoint.get("warning_id")
            if isinstance(warning_id, str):
                for warning in await self.warning_service.list_warnings(db, goal.id, active_only=True):
                    if str(warning.id) == warning_id:
                        warning.message = f"Goal definition analysis failed: {safe_error}"
                        break
            return {
                "process_type": "goal_definition",
                "status": current.status,
                "retryable": True,
                "questions_created": 0,
                "error": safe_error,
                "retry_failed": True,
            }

        continuation = checkpoint.get("continuation")
        if not isinstance(continuation, dict) or not isinstance(continuation.get("clarification_round"), int):
            raise ValueError("goal definition retry continuation is invalid")
        frozen_context = continuation.get("context")
        frozen_goal = continuation.get("working_goal")
        if not isinstance(frozen_context, dict) or not isinstance(frozen_goal, dict):
            raise ValueError("goal definition retry continuation is invalid")
        context = deepcopy(frozen_context)
        working_goal = SimpleNamespace(
            objective=frozen_goal.get("objective"),
            success_criteria=deepcopy(frozen_goal.get("success_criteria")),
        )
        if not isinstance(working_goal.objective, str) or not isinstance(working_goal.success_criteria, list):
            raise ValueError("goal definition retry continuation is invalid")
        round_number = continuation["clarification_round"]
        return await self._continue_after_analysis(
            db, goal, run, current, analysis, working_goal, context, round_number,
            had_analyzer_error=True,
        )

    @staticmethod
    def _continuation_state(goal: OrchestrationGoal, decisions: list) -> tuple[dict, SimpleNamespace]:
        context = deepcopy(goal.orchestrator_context or {})
        working_goal = SimpleNamespace(
            objective=goal.objective,
            success_criteria=deepcopy(goal.success_criteria or []),
        )
        merged_keys = set(context.get("merged_decision_keys", []))
        for decision in decisions:
            if (
                decision.status == "answered"
                and decision.decision_key.startswith(f"{DECISION_KEY_PREFIX}adaptive:")
                and decision.decision_key not in merged_keys
            ):
                _merge_answer(
                    working_goal, context, decision.decision_key, decision.question,
                    decision.selected_option or decision.reason or "",
                )
                merged_keys.add(decision.decision_key)
        context["merged_decision_keys"] = sorted(merged_keys)
        return context, working_goal

    async def _continue_after_analysis(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current,
        analysis,
        working_goal: SimpleNamespace,
        context: dict,
        round_number: int,
        *,
        had_analyzer_error: bool,
    ) -> dict:
        retry_checkpoint = (current.outputs or {}).get("_lm_retry") if had_analyzer_error else None
        goal.success_criteria = working_goal.success_criteria
        assumptions = list(context.get("assumptions", []))
        assumptions.extend(
            {"text": assumption.text, "destination": assumption.destination, "round": round_number}
            for assumption in analysis.assumptions
        )
        context["assumptions"] = assumptions
        goal.orchestrator_context = context

        if analysis.questions and round_number <= MAX_CLARIFICATION_ROUNDS + int(context.get("clarification_round_bonus", 0)):
            next_round = round_number + 1
            context["clarification_round"] = next_round
            goal.orchestrator_context = context
            for index, question in enumerate(analysis.questions):
                await self.decision_service.create_pending(
                    db, goal.id,
                    decision_key=f"{DECISION_KEY_PREFIX}adaptive:{next_round}:{index}:{question.destination}",
                    title="Clarify goal definition", question=question.question, authority="human",
                    context=f"{question.rationale} Destination: {question.destination}", run_id=run.id,
                    source_process_run_id=current.id, options=None,
                )
            await self.process_service.park_process(db, current)
            await self._clear_retry_checkpoint(db, goal, run, current, retry_checkpoint)
            return {"process_type": "goal_definition", "status": current.status,
                    "questions_created": len(analysis.questions)}

        if analysis.unsafe_unresolved or analysis.questions:
            from huddleroom.services.orchestration_service import OrchestrationService

            OrchestrationService._upsert_active_blocker(run, {
                "kind": "goal_definition_clarification_limit",
                "reason": "Material goal ambiguity remains after clarification rounds; all prior answers are preserved. Proceed with current understanding or request another round.",
            })
            OrchestrationService._mark_run_blocked(goal, run)
            await self.process_service.complete_process(db, current, outputs={
                "clarification_limit_reached": True,
                "unsafe_unresolved": True,
                "unresolved_questions": [question.question for question in analysis.questions],
            })
            await self._clear_retry_checkpoint(db, goal, run, current, retry_checkpoint)
            return {"process_type": "goal_definition", "status": "completed", "questions_created": 0,
                    "clarification_limit_reached": True}

        result = await self._finalize_goal_definition(db, goal, run, context, current)
        await self._clear_retry_checkpoint(db, goal, run, current, retry_checkpoint)
        return result

    async def _clear_retry_checkpoint(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, current, checkpoint: dict | None
    ) -> None:
        if checkpoint is None:
            return
        outputs = dict(current.outputs or {})
        outputs.pop("error", None)
        outputs.pop("retryable", None)
        outputs.pop("_lm_retry", None)
        current.outputs = outputs
        from huddleroom.services.orchestration_service import OrchestrationService

        warning_id = checkpoint.get("warning_id")
        if isinstance(warning_id, str):
            run.active_blockers = [
                item
                for item in run.active_blockers
                if not (
                    isinstance(item, dict)
                    and item.get("kind") == "goal_definition_analyzer_error"
                    and item.get("warning_id") == warning_id
                )
            ]
            for warning in await self.warning_service.list_warnings(db, goal.id, active_only=True):
                if str(warning.id) == warning_id:
                    await self.warning_service.resolve_warning(
                        db,
                        warning,
                        resolved_by="orchestrator:goal_definition",
                        reason="goal definition analysis succeeded after analyzer failure",
                    )
        if not run.active_blockers:
            if goal.status == "blocked":
                goal.status = "active"
            if run.status == "blocked":
                run.status = "running"

    async def _finalize_goal_definition(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun,
        context: dict, current: "OrchestrationProcessRun",  # noqa: F821
    ) -> dict:
        """Finalize goal definition by applying weight, gates, and memory."""
        _repair_success_criteria_keys(goal)
        if goal.weight_overridden_by is None:
            goal.weight = classify_goal_weight(
                goal.success_criteria, goal.constraints, goal.budget, goal_objective_with_clarifications(goal),
                explicit_multi_work_function=goal.explicit_multi_work_function,
            )
        gates = {
            "goal_definition_reviewed": True,
            "success_criteria_confirmed": True,
            "constraints_confirmed": True,
            "evidence_expectations_confirmed": True,
        }
        all_clarifications = list(context.get("resolved_clarifications", []))
        await self.memory_service.upsert_section(
            db, goal.project_id, goal.id, section_key=MEMORY_SECTION_KEY, title="Goal definition",
            body=self._memory_body(goal, all_clarifications),
            summary=f"Weight {goal.weight}; {len(all_clarifications)} clarification(s) recorded.",
            toc_order=MEMORY_TOC_ORDER, run_id=run.id, created_by="orchestrator:goal_definition",
        )
        await self.process_service.complete_process(db, current, outputs={
            "weight": goal.weight, "gaps": [], "questions_created": 0,
            "clarifications": all_clarifications, "compressed": False, "gates": gates,
        })
        return {"process_type": "goal_definition", "status": "completed", "questions_created": 0}

    @staticmethod
    def _memory_body(goal: OrchestrationGoal, clarifications: list[dict]) -> str:
        lines = [
            f"Objective: {goal.objective}",
            f"Original request: {goal.original_request}",
            f"Weight: {goal.weight}",
            f"Success criteria: {len(goal.success_criteria or [])}",
            f"Constraints: {'recorded' if goal.constraints else 'none provided'}",
            f"Budget: {'recorded' if goal.budget else 'none provided'}",
        ]
        if clarifications:
            lines.append("Accepted clarifications:")
            lines.extend(
                f"- {c['question']} -> {c.get('answer', c.get('selected_option')) or '(no option recorded)'}"
                f" ({c.get('reason') or 'no reason given'})"
                for c in clarifications
            )
        return "\n".join(lines)
