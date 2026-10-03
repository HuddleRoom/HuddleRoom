"""Baseline Process B: manager selection (Spec 8, Phase 6 slice).

Candidate detection, fit ranking, and the human approval path are
deterministic code over the roster mapper's scoring -- no LLM calls. The
manager-as-agent decision-request path (spec 8.3.1) belongs to the phases
that route decisions *to* the selected manager; Process B only selects one.
Rerun triggers (spec 8.1) are the human force-start endpoint for now.
Orchestrator-only: never expose through agent-facing surfaces.
"""
from __future__ import annotations

import re
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from huddleroom.models.agent import Agent
from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.project import Project
from huddleroom.models.user import User
from huddleroom.services.orchestration_authority_service import OrchestrationAuthorityDecisionService
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService
from huddleroom.services.orchestration_manager_analyzer import ManagerSelectionAnalyzer
from huddleroom.services.orchestration_process_service import OrchestrationProcessService
from huddleroom.services.orchestration_roster_mapper import OrchestrationRosterMapper, RosterFit
from huddleroom.services.orchestration_warning_service import OrchestrationWarningService

MANAGER_SELECTION_PROCESS_VERSION = 1
DECISION_KEY_PREFIX = "manager_selection:"
SELECT_MANAGER_DECISION_KEY = f"{DECISION_KEY_PREFIX}select_manager"
REVIEW_OVERRIDE_DECISION_KEY = f"{DECISION_KEY_PREFIX}review_override"
MEMORY_SECTION_KEY = "manager_authority"
MEMORY_TOC_ORDER = 20
MANAGER_WORK_FUNCTION = "management"
MAX_AGENT_OPTIONS = 3


HUMAN_AS_MANAGER_OPTION = "human_as_manager"
NO_MANAGER_OPTION = "no_manager"
NO_MANAGER_WARNING_TYPE = "manager_selection_no_manager"
REDACTED_CONFIG_KEYS = ("provider_extras", "cli_env_extras")
# Spec 8.7, verbatim.
NO_MANAGER_WARNING_MESSAGE = (
    "No manager / main point of contact was selected. The orchestrator may "
    "need to ask the human more often, and coordination decisions may be "
    "slower or less consistent."
)

# Spec 8.3 step 3: "instructions, model configuration, personality, tools."
# These are local, manager-selection-only signal terms -- they do not
# change the shared roster_mapper PROFILES used by other work functions.
# Split into the three distinct categories spec 8.3 step 3/step 5 names
# ("ownership, coordination, and decision-making") rather than one flat
# term list (review finding, HIGH): an agent definition that establishes
# ALL THREE on its own must be able to qualify as a strong candidate even
# when its role/capability terms carry no management signal at all -- a
# single flat bonus tied to any one term firing could never do that if the
# roster-mapper base score is low, silently pushing the recommendation to
# human-as-manager despite the instructions clearly saying otherwise.
DEFINITION_OWNERSHIP_TERMS = ("own", "owns", "ownership", "point of contact")
DEFINITION_COORDINATION_TERMS = ("coordinate", "coordination", "lead the")
DEFINITION_DECISION_TERMS = ("decide", "decision", "prioritize", "priority", "tradeoffs")
DEFINITION_CATEGORY_TERMS = (
    DEFINITION_OWNERSHIP_TERMS,
    DEFINITION_COORDINATION_TERMS,
    DEFINITION_DECISION_TERMS,
)
# Negation is scoped PER CATEGORY, evaluated at each individual term match
# (review finding, MEDIUM): a global "any negation phrase anywhere in the
# text cancels every category" flag let one negated clause -- e.g. "do not
# manage decisions" -- wipe out unrelated positive ownership/coordination
# evidence sitting elsewhere in the same mixed-responsibility definition.
# `_term_negated` below only looks at the text immediately preceding the
# matched term, so a negation only suppresses the category it actually
# qualifies.
# Match complete cue words: a substring check mistakes the end of "Bruno "
# for the "no" cue. `n't` intentionally has no leading word boundary so
# contractions such as "doesn't" remain covered.
NEGATION_CUE_RE = re.compile(
    r"(?:\bnot\b(?!\s+only\b)|\bnever\b|\bno\b|\bwithout\b|\bcannot\b|n't\b)",
    re.IGNORECASE,
)
# "not only" reads as affirmative emphasis ("not only owns X but also...."),
# not negation -- the regex excludes it from cue matching.
NEGATION_WINDOW_CHARS = 24
# Support context cues indicate subordinate/advisory roles, not exercised authority.
# Phrases like "supports the owner" or "assists the team" should not trigger
# management-authority bonuses even if they mention authority-related keywords
# (owner, coordination, decisions). Example: "supports the owner, coordinates
# logistics, prepares decision support" describes a support role, not a manager,
# and must not score as establishing management authority (finding #8).
# Added "decision support to", "support to" phrasing as advisory (finding #8).
SUPPORT_CONTEXT_CUES = (
    "supports the", "support for", "assists the", "helps the", "advises the",
    "reports to", "on behalf of", "decision support to", "support to"
)
# Contrastive conjunctions that end a support-context framing mid-clause
# (review finding, HIGH): "supports the team, but owns delivery, coordinates
# execution" must not have the earlier "supports the" suppress the later
# affirmative "owns"/"coordinates" -- the cue only applies to the semantic
# segment it actually sits in, not the whole clause.
CONTRAST_SPLIT_RE = re.compile(
    r"\b(?:but|however|although|though|whereas|except)\b", re.IGNORECASE
)
# Per matched, non-negated category (ownership / coordination /
# decision-making), up to all three -- see _inspect_candidate_definition and
# the strong-override below.
DEFINITION_CATEGORY_BONUS = 10
# Real Agent.config keys (see huddleroom/adapters/api_adapter.py, meeting_runner.py,
# tool_executor.py). These are inspected and reported to the human, but NOT
# scored here: OrchestrationRosterMapper already scores memory_enabled, and
# permissions/runtime/model knobs are boundaries, not management fitness.
DEFINITION_CONTEXT_CONFIG_KEYS = (
    "allow_global_scope", "memory_enabled", "temperature", "max_tokens",
    "provider_extras", "context_message_window", "cli_runtime",
    "session_timeout_seconds", "meeting_turn_timeout_seconds", "cli_env_extras",
)
# Roster-mapper score + definition bonus combined must clear this to count
# as a "strong" candidate for recommendation/asking purposes (spec 8.3 step
# 4) -- mirrors OrchestrationRosterMapper's own weak threshold (score < 45)
# but is evaluated on the COMBINED score, never on RosterFit.weak alone, so
# a candidate the bare roster score would call weak can still be promoted
# once its definition is inspected.
MANAGEMENT_STRONG_THRESHOLD = 45


def _is_strong_candidate(score: int, definition_bonus: int, definition_strong: bool) -> bool:
    # `definition_strong` (all three of ownership/coordination/decision-
    # making established by the agent's own instructions, spec 8.3 step 3)
    # overrides a low combined score outright (review finding, HIGH): an
    # agent definition that clearly establishes management must never be
    # graded "weak" just because its role/capability terms carried no
    # management signal for the roster mapper to score.
    return definition_strong or (score + definition_bonus) >= MANAGEMENT_STRONG_THRESHOLD


def _candidate_option(
    fit: RosterFit,
    definition_bonus: int,
    definition_signals: list[str],
    definition_strong: bool = False,
) -> dict:
    # Only "key" is contractual (validated by the decision service on
    # answer); the rest is context for whoever renders the decision
    # (dashboard, Phase 9 checkpoint).
    combined_score = fit.score + definition_bonus
    return {
        "key": f"agent:{fit.agent_id}",
        "label": f"{fit.name} ({fit.role})",
        "score": fit.score,
        "definition_bonus": definition_bonus,
        "combined_score": combined_score,
        # Recomputed from the COMBINED score, with the instruction-only
        # strong-override applied (review finding) -- NOT fit.weak, which
        # only reflects the pre-inspection roster score and would
        # misclassify a definition-qualified candidate as weak.
        "weak": not _is_strong_candidate(fit.score, definition_bonus, definition_strong),
        "signals": list(fit.matched_signals) + definition_signals,
    }


class ManagerSelectionProcess:
    """Deterministic Phase 6 slice of Baseline Process B (spec 8).

    Called from the tick once goal_definition is terminal (spec 6.3 order).
    Never blocks the tick: a parked process returns its summary and the tick
    continues. It DOES block this goal's own forward progress until terminal
    (the shared baseline-process gate in orchestration_service). Trivial
    goals auto-complete with the human as implicit manager (spec 6.2) --
    zero questions.
    """

    def __init__(self, analyzer: ManagerSelectionAnalyzer | None = None) -> None:
        self.process_service = OrchestrationProcessService()
        self.decision_service = OrchestrationAuthorityDecisionService()
        self.memory_service = OrchestrationMemoryService()
        self.warning_service = OrchestrationWarningService()
        self.roster_mapper = OrchestrationRosterMapper()
        self.analyzer = analyzer or ManagerSelectionAnalyzer()

    async def advance(
        self, db: AsyncSession, goal: OrchestrationGoal, run: OrchestrationRun, *, manual: bool = False
    ) -> dict:
        current = await self.process_service.get_current(db, goal.id, "manager_selection")
        if current is None and not (run.baseline_authorized or manual):
            return {"process_type": "manager_selection", "status": "not_started", "questions_created": 0}
        if current is not None and current.status == "skipped":
            # Pre-Phase-6 skipped rows may not have backfilled authority_model=no_manager
            # and the spec warning. Check if goal is un-healed and backfill idempotently.
            if (
                goal.authority_model != "no_manager"
                or goal.manager_agent_id is not None
                or goal.manager_user_id is not None
            ):
                await self.handle_skip(db, goal, current)
            return {
                "process_type": "manager_selection",
                "status": current.status,
                "questions_created": 0,
            }
        if current is not None and current.status == "completed":
            stale_reason = await self._stale_manager_reason(db, goal, current)
            roster_gained_reason = await self._roster_gained_candidates_reason(db, goal, current)
            # ponytail: check weight tier change at process completion time
            # (finding #1). persist weight in outputs to compare on future advances.
            tier_changed = False
            if current.outputs.get("weight") is not None:
                recorded_weight = current.outputs.get("weight")
                tier_changed = recorded_weight != goal.weight
            if stale_reason is None and roster_gained_reason is None and not tier_changed:
                return {
                    "process_type": "manager_selection",
                    "status": current.status,
                    "questions_created": 0,
                }
            rerun_reason = stale_reason or roster_gained_reason or "goal weight tier changed"
            if run.phase == "baseline":
                # Baseline phase: spec 8.1 rerun triggers -- "current manager
                # is removed or inactive", weight tier changed, roster gained
                # candidates (bug #89) -- all convert to a one-time
                # suggestion (item 3, orchestrator override): no auto-rerun,
                # no clearing manager_agent_id/authority_model, no new
                # process row. The completed selection stands until a human
                # approves a rerun via the suggestion.
                await self.warning_service.suggest_stale_inputs(
                    db,
                    goal.id,
                    process_type="manager_selection",
                    process_run_id=current.id,
                    step_label=f"manager selection ({rerun_reason})",
                    run_id=run.id,
                )
                return {
                    "process_type": "manager_selection",
                    "status": current.status,
                    "questions_created": 0,
                }
            # Post-baseline: execution-time task assignments are part of
            # _fingerprint and must keep coverage in sync silently -- a
            # human is no longer actively reviewing the baseline, so restore
            # the original auto-rerun (spec 8.1) instead of leaving a stale
            # suggestion.
            current = await self.process_service.start_process(
                db,
                goal.id,
                process_type="manager_selection",
                trigger_reason=f"auto-rerun: {rerun_reason} (spec 8.1)",
                run_id=run.id,
                input_snapshot={"weight": goal.weight, "reason": rerun_reason},
                process_version=MANAGER_SELECTION_PROCESS_VERSION,
            )
            # Bug #42: concurrent force-start can return a terminal row; short-circuit if not active
            if current.status not in ("running", "waiting_decision"):
                return {
                    "process_type": "manager_selection",
                    "status": current.status,
                    "questions_created": 0,
                }
            # Only the auto-rerun path clears the goal columns: the selected
            # manager is genuinely gone (removed/inactive), so leaving stale
            # manager_agent_id/authority_model would misrepresent the goal. A
            # human force-start (a fresh `running` run reached via the
            # `current is None` branch below, not here) deliberately leaves the
            # existing manager in place until the human picks a replacement --
            # its manager was never invalid. Do not "unify" these by clearing
            # columns on every rerun.
            goal.manager_agent_id = None
            goal.manager_user_id = None
            goal.authority_model = None
            await db.flush()
            # Rerun outputs supersede prior outputs (review finding): without
            # this, the manager_authority memory section keeps naming the
            # removed manager for the entire time the process is re-parked,
            # which is stale the moment we clear the goal columns above.
            await self.memory_service.upsert_section(
                db,
                goal.project_id,
                goal.id,
                section_key=MEMORY_SECTION_KEY,
                title="Manager and authority model",
                body=self._memory_body(
                    goal,
                    "reselection in progress",
                    f"previous manager selection superseded: {stale_reason}",
                    [],
                ),
                summary="Manager: reselection in progress.",
                toc_order=MEMORY_TOC_ORDER,
                run_id=run.id,
                created_by="orchestrator:manager_selection",
            )

        # Spec 8.3 steps 1-2: rank ALL active agents by role/capability fit.
        fits = await self.roster_mapper.rank_agents(db, goal.project_id, MANAGER_WORK_FUNCTION)
        # Spec 8.3 step 3: inspect EVERY candidate's definition, not a
        # pre-inspection top slice (review finding) -- a candidate the bare
        # roster score ranks low can still be definition-qualified, and a
        # slice taken before inspection could drop it before it's ever
        # looked at. Rosters are agent-roster-sized (small), so this is
        # cheap; it's the pool of candidates OFFERED as options (below)
        # that stays bounded, not the pool inspected.
        definition_signals: dict[uuid.UUID, tuple[int, list[str], bool]] = {}
        for fit in fits:
            candidate_agent = await db.get(Agent, fit.agent_id)
            definition_signals[fit.agent_id] = self._inspect_candidate_definition(
                candidate_agent
            )
        # Spec 8.3 step 4 "rank manager fit" combines the roster-mapper role/
        # capability score with the step-3 definition-inspection bonus over
        # the FULL candidate set, so a strong definition can promote a
        # candidate the bare roster score alone would never surface.
        #
        # Strong-first, THEN combined score (review finding, HIGH): sorting
        # purely by combined score before slicing to MAX_AGENT_OPTIONS let a
        # candidate that only qualifies as strong through the instruction-
        # only `_is_strong_candidate` override -- but has a low combined
        # score -- fall outside the top-N cut. That silently dropped a
        # strong candidate from both the offered options and the
        # recommendation, degrading to `human_as_manager` in violation of
        # spec 8.3 even though a strong candidate existed. Sorting on
        # "is strong" first guarantees every strong candidate sorts ahead of
        # every weak one regardless of raw score, so MAX_AGENT_OPTIONS can
        # only ever truncate excess weak candidates.
        ranked = sorted(
            fits,
            key=lambda fit: (
                0
                if _is_strong_candidate(
                    fit.score,
                    definition_signals[fit.agent_id][0],
                    definition_signals[fit.agent_id][2],
                )
                else 1,
                -(fit.score + definition_signals[fit.agent_id][0]),
                fit.load.penalty,
                fit.name.lower(),
                str(fit.agent_id),
            ),
        )
        top_fits = ranked[:MAX_AGENT_OPTIONS]

        if current is None:
            current = await self.process_service.start_process(
                db,
                goal.id,
                process_type="manager_selection",
                trigger_reason="tick: no manager_selection process run on record",
                run_id=run.id,
                input_snapshot={
                    "weight": goal.weight,
                    "candidate_count": len(fits),
                    "strong_candidate_count": sum(
                        1
                        for fit in fits
                        if _is_strong_candidate(
                            fit.score,
                            definition_signals[fit.agent_id][0],
                            definition_signals[fit.agent_id][2],
                        )
                    ),
                },
                process_version=MANAGER_SELECTION_PROCESS_VERSION,
            )
            # Bug #42: concurrent force-start can return a terminal row; short-circuit if not active
            if current.status not in ("running", "waiting_decision"):
                return {
                    "process_type": "manager_selection",
                    "status": current.status,
                    "questions_created": 0,
                }

        if goal.weight == "trivial":
            # Spec 6.2 / 8.3 step 6: the human is the implicit manager for
            # trivial goals, applied automatically without a question. A
            # question raised while the goal was still standard/substantial
            # (weight forced lighter mid-process) is now moot: cancel it so
            # nothing parks on a stale decision.
            for decision in await self.decision_service.list_decisions(
                db, goal.id, status="pending"
            ):
                if decision.decision_key == SELECT_MANAGER_DECISION_KEY:
                    await self.decision_service.cancel_decision(
                        db,
                        decision,
                        reason=(
                            "goal weight is trivial: the human is the implicit "
                            "manager (spec 6.2)"
                        ),
                    )
            if current.status == "waiting_decision":
                await self.process_service.resume_process(db, current)
            # Bug fix A: before assigning the creator as implicit manager, verify
            # the creator exists and is active. If creator was deleted or
            # deactivated, pass None instead to avoid assigning a nonexistent/stale
            # manager or triggering an infinite rerun loop (spec 8.1 review finding).
            effective_manager_user_id = None
            if goal.created_by_user_id is not None:
                creator = await db.get(User, goal.created_by_user_id)
                if creator is not None and creator.is_active:
                    effective_manager_user_id = goal.created_by_user_id
            # Build candidates list from live roster for this non-answered path
            candidates = [
                _candidate_option(fit, *definition_signals.get(fit.agent_id, (0, [], False)))
                for fit in top_fits
            ]
            return await self._complete(
                db, goal, run, current,
                selected_option=HUMAN_AS_MANAGER_OPTION,
                manager_user_id=effective_manager_user_id,
                candidates=candidates,
                rationale="trivial goal: the human is the implicit manager (spec 6.2)",
                compressed=True,
            )

        orchestrated = await self._orchestrate_selection(
            db, goal, run, current, top_fits, definition_signals
        )
        if orchestrated is not None:
            return orchestrated

        decisions = [
            d
            for d in await self.decision_service.list_decisions(db, goal.id)
            if d.decision_key == SELECT_MANAGER_DECISION_KEY
        ]
        pending_decisions = [d for d in decisions if d.status == "pending"]
        pending = next(
            (d for d in pending_decisions if d.source_process_run_id == current.id),
            None,
        )
        # Latest answer wins, scoped to THIS process run: a superseded run's
        # answer must not leak into a forced rerun (spec 6.3: a rerun re-asks
        # its own questions), and after an invalid-agent re-ask both the
        # stale and the fresh answer exist -- only the newest reflects the
        # human's current choice (plan Deviation 11).
        answered = next(
            (
                d
                for d in reversed(decisions)
                if d.status == "answered" and d.source_process_run_id == current.id
            ),
            None,
        )

        manager_agent: Agent | None = None
        stale_agent_answer = False
        stale_human_answer = False
        if answered is not None:
            option = answered.selected_option or ""
            if option.startswith("agent:"):
                manager_agent = await db.get(
                    Agent, uuid.UUID(option.removeprefix("agent:"))
                )
                if manager_agent is None or not manager_agent.is_active:
                    # The chosen agent vanished or was deactivated between
                    # ask and answer. The answered row is terminal, so the
                    # decision key is free again: re-ask with refreshed
                    # options instead of recording an unusable manager.
                    manager_agent = None
                    stale_agent_answer = True
            elif option == HUMAN_AS_MANAGER_OPTION:
                # Parallel check for human manager: verify the selected user
                # still exists and is active. If the user vanished or was
                # deactivated between ask and answer, re-ask with refreshed
                # options instead of recording an unusable manager.
                # Finding #2: treat NULL decided_by_user_id (FK ON DELETE SET NULL)
                # as stale -- no actual user to record.
                if answered.decided_by_user_id is None:
                    stale_human_answer = True
                else:
                    manager_user = await db.get(User, answered.decided_by_user_id)
                    if manager_user is None or not manager_user.is_active:
                        stale_human_answer = True

        if answered is None or stale_agent_answer or stale_human_answer:
            questions_created = 0
            if pending is None:
                for stale_pending in pending_decisions:
                    await self.decision_service.cancel_decision(
                        db,
                        stale_pending,
                        reason="superseded by a newer manager-selection process run",
                    )
                await self._ask(db, goal, run, current, top_fits, definition_signals)
                questions_created = 1
            await self.process_service.park_process(db, current)
            return {
                "process_type": "manager_selection",
                "status": current.status,
                "questions_created": questions_created,
            }

        if current.status == "waiting_decision":
            await self.process_service.resume_process(db, current)

        # Review finding #6: use the stored option snapshot (what was shown to
        # the human) instead of recomputing from live roster, so persisted
        # candidates and memory body reflect the decision-time roster state, not
        # a mutated roster after the human answered.
        # Finding #13: filter to agent-only entries; exclude human_as_manager/no_manager
        # control entries that lack candidate score fields (maintain uniform shape).
        candidates = [
            opt for opt in answered.options
            if opt.get("key", "").startswith("agent:")
        ]

        option = answered.selected_option or ""
        if manager_agent is not None:
            # Use the stored option snapshot to build rationale instead of
            # recomputing from live fits
            rationale = "selected by the human"
            stored_option = next(
                (o for o in answered.options if o["key"] == option),
                None,
            )
            if stored_option:
                signal_bits = stored_option.get("signals", [])
                rationale = (
                    f"score {stored_option['score']} + definition bonus {stored_option['definition_bonus']}; "
                    f"signals: {', '.join(signal_bits) or 'none'}"
                )
            return await self._complete(
                db, goal, run, current,
                selected_option=option,
                manager_agent=manager_agent,
                candidates=candidates,
                rationale=rationale,
                compressed=False,
            )
        if option == HUMAN_AS_MANAGER_OPTION:
            return await self._complete(
                db, goal, run, current,
                selected_option=option,
                manager_user_id=answered.decided_by_user_id,
                candidates=candidates,
                rationale="human chose to act as manager / main POC (spec 8.3 step 6)",
                compressed=False,
            )
        # NO_MANAGER_OPTION -- the only other offered key.
        return await self._complete(
            db, goal, run, current,
            selected_option=NO_MANAGER_OPTION,
            candidates=candidates,
            rationale="human approved proceeding without a manager (spec 8.7)",
            compressed=False,
        )

    def _inspect_candidate_definition(self, agent: Agent | None) -> tuple[int, list[str], bool]:
        """Spec 8.3 step 3: examine candidate agent definitions --
        instructions, model configuration, personality, tools -- not just
        role/capability terms. Deliberately small and local to manager
        selection; it never touches the shared roster_mapper PROFILES used
        by other work functions. A deeper generic agent-definition review
        (instruction quality, whether THIS model reasons well for the
        goal's specific domain, whether tool/personality fit the goal) is
        Phase 7's job -- this is the Phase 6 selection's own evidence, not a
        stand-in for it. What this method does NOT skip: model/provider,
        adapter_type/cli_runtime (tools/permission surface), and a
        description/system_prompt personality excerpt are all always
        inspected and reported (never silently ignored, review finding)
        even though no bonus is tied to any of them -- every agent has a
        non-null provider/model/adapter_type, so "has one" carries no
        signal, and inventing quality tiers for model, tools, or
        personality here would just be a guess Phase 7's real review
        should make."""
        if agent is None:
            return 0, [], False
        bonus = 0
        signals: list[str] = []
        definition_text = "\n".join(
            part for part in (agent.description, agent.system_prompt) if part
        )
        prompt = definition_text.lower()
        # Scored per distinct category (ownership / coordination / decision-
        # making), not as one flat yes/no term hit (review finding, HIGH):
        # an agent whose instructions establish all three on their own must
        # be able to out-rank a low roster-mapper base score, not just add
        # a fixed +10 that a weak role/capability match can still bury
        # below MANAGEMENT_STRONG_THRESHOLD.
        #
        # Negation is checked per matched term, not once globally over the
        # whole text (review finding, MEDIUM): a mixed-responsibility
        # definition like "Do not manage decisions. Own the roadmap and
        # coordinate the team." must lose only the decision-making signal
        # ("decisions" sits right after "do not manage") while still
        # earning ownership and coordination credit for the affirmative
        # sentence that follows. A blanket "any negation phrase anywhere
        # cancels every category" flag would silently drop all three.
        matched_categories = 0
        category_labels = ("ownership", "coordination", "decision-making")
        for label, terms in zip(category_labels, DEFINITION_CATEGORY_TERMS):
            matched_term, negated = self._first_category_match(prompt, terms)
            if matched_term is None:
                continue
            if negated:
                signals.append(f"instructions explicitly limit {label}")
                continue
            matched_categories += 1
            bonus += DEFINITION_CATEGORY_BONUS
            signals.append(f"instructions describe {label}")
        # Spec 8.3 step 3/step 5: instructions that clearly establish ALL
        # THREE of ownership, coordination, and decision-making are strong
        # evidence of management fit on their own -- this overrides the
        # combined-score threshold outright (see _is_strong_candidate)
        # rather than relying on the bonus alone to cross it, since the
        # bonus is capped and a very low roster base score could otherwise
        # still leave a definitively management-oriented agent "weak".
        definition_strong = matched_categories == len(DEFINITION_CATEGORY_TERMS)
        # Real Agent.config keys (see huddleroom/adapters/api_adapter.py,
        # meeting_runner.py, tool_executor.py). These are surfaced, not
        # scored: memory_enabled is already scored by OrchestrationRosterMapper,
        # and permissions/runtime/model knobs are boundaries, not management
        # fitness.
        config = agent.config or {}
        for key in DEFINITION_CONTEXT_CONFIG_KEYS:
            if key not in config:
                continue
            if key == "allow_global_scope":
                value = config[key]
                if value:
                    signals.append(f"config: allow_global_scope={value!r} (broad permission boundary)")
                else:
                    signals.append(f"config: allow_global_scope={value!r}")
            elif key == "memory_enabled":
                value = config[key]
                if value:
                    signals.append(f"config: memory_enabled={value!r} (context persists across the goal)")
                else:
                    signals.append(f"config: memory_enabled={value!r}")
            elif key in REDACTED_CONFIG_KEYS:
                value = config[key]
                if isinstance(value, dict):
                    signals.append(f"config: {key} is set (keys: {', '.join(sorted(value.keys()))})")
                else:
                    signals.append(f"config: {key} is set")
            else:
                signals.append(f"config: {key}={config[key]!r}")
        # Spec 8.3 step 3 "tools": adapter_type is what actually determines
        # the candidate's runtime tool/permission surface. Always surfaced,
        # never scored (review finding: tool/permission fields must not be
        # silently ignored) -- which tool surface best fits the goal is
        # Phase 7's agent-definition-review job.
        #
        # Review finding (MEDIUM): the permission story is per-runtime, not
        # a blanket "every CLI agent is unrestricted" claim, and the
        # runtime that actually executes is `Agent.config["cli_runtime"]`
        # when set, NOT the bare `agent.cli_runtime` column -- CliAdapter
        # resolves it the same way (huddleroom/adapters/cli_adapter.py
        # `_build_command` callers): `agent.config.get("cli_runtime",
        # agent.cli_runtime or "claude_code")`. `claude_code`, `copilot`,
        # `opencode`, and `pi` bypass interactive permission approval;
        # `codex` and `aider` do not. Displaying the resolved runtime (not the
        # raw column) also matters for human-approval context (review
        # finding): approving "the agent" should reflect what will actually
        # run, not a stale/overridden column value.
        if agent.adapter_type == "cli":
            effective_cli_runtime = (agent.config or {}).get(
                "cli_runtime", agent.cli_runtime or "claude_code"
            )
            if effective_cli_runtime in {"claude_code", "copilot", "opencode", "pi"}:
                signals.append(
                    f"tools: cli adapter ({effective_cli_runtime}), runs with "
                    "unrestricted local tool permissions (non-interactive approval bypass)"
                )
            else:
                signals.append(
                    f"tools: cli adapter ({effective_cli_runtime}), permission surface is "
                    "runtime-specific (no blanket unrestricted-access flag)"
                )
        else:
            signals.append(f"tools: {agent.adapter_type} adapter (scoped to configured tools)")
        # Spec 8.3 step 3 "personality": Phase 3's agent-definition-review
        # precedent already treats description/system_prompt AS the
        # personality signal (no separate column exists). Always surfaced
        # here too -- even when no ownership-term bonus applies -- so the
        # human genuinely sees personality/tone (review finding), not just
        # a binary management-keyword bonus standing in for it.
        personality_excerpt = (agent.description or agent.system_prompt or "").strip()
        if personality_excerpt:
            signals.append(f"personality: {personality_excerpt[:120]}")
        # Always surfaced, never scored: transparency for the human
        # answering the decision, not a ranking signal.
        signals.append(f"model: {agent.provider}/{agent.model}")
        return bonus, signals, definition_strong

    @staticmethod
    def _first_category_match(prompt: str, terms: tuple[str, ...]) -> tuple[str | None, bool]:
        """Match of any `term` from `terms` in `prompt`, preferring an
        unnegated occurrence. Scans every occurrence of every term and
        returns the first with no negation cue (`NEGATION_CUE_RE`) within
        `NEGATION_WINDOW_CHARS` before it in the same clause; only if every
        occurrence is negated is the match reported negated. Scoped per-term
        and per-clause rather than a document-wide flag, and scanning past
        an early negated hit (review finding, MEDIUM) so 'do not own
        deployments' does not suppress a later affirmative ownership
        statement for the same category. Negation window never crosses clause
        boundaries (sentence, semicolon, newline) to avoid negation in one
        sentence incorrectly canceling an affirmative in another."""
        first_negated: str | None = None
        # Split prompt into clauses on sentence/clause boundaries
        clauses = re.split(r'[.!?;\n]+', prompt)
        for clause in clauses:
            # Support-context cues frame their segment's subject as
            # subordinate/advisory (e.g. "supports the owner, prepares
            # decision support") -- unlike negation, which only cancels a
            # term sitting right next to it, so this check scans the whole
            # segment, not just the narrow NEGATION_WINDOW_CHARS window
            # immediately before the term. Scoped to the segment BETWEEN
            # contrastive conjunctions (review finding, HIGH), not the whole
            # clause: "supports the team, but owns delivery, coordinates
            # execution" starts a new segment at "but", so the earlier
            # "supports the" no longer suppresses "owns"/"coordinates" in
            # the contrasted segment that follows.
            bounds = {0, len(clause)}
            for contrast in CONTRAST_SPLIT_RE.finditer(clause):
                bounds.add(contrast.start())
            bounds = sorted(bounds)
            segments = list(zip(bounds, bounds[1:])) or [(0, len(clause))]
            for term in terms:
                # Use word-boundary anchored regex to match term and its suffixes
                pattern = rf'\b{re.escape(term)}\w*\b'
                for match in re.finditer(pattern, clause, re.IGNORECASE):
                    idx = match.start()
                    segment = clause
                    segment_start = 0
                    for seg_start, seg_end in segments:
                        if seg_start <= idx < seg_end:
                            segment = clause[seg_start:seg_end]
                            segment_start = seg_start
                            break
                    # Keep the bounded negation window within this contrastive
                    # segment: "do not manage budgets, but own the roadmap"
                    # affirms ownership after "but".
                    window_start = max(segment_start, idx - NEGATION_WINDOW_CHARS)
                    window = clause[window_start:idx]
                    negated_by_cue = NEGATION_CUE_RE.search(window) is not None
                    # ponytail: make support-context cue matching position-aware (finding #8).
                    # a support cue suppresses a term if: (1) cue appears BEFORE the term in segment,
                    # or (2) term appears WITHIN a support cue phrase (e.g., "decision" in "decision support to").
                    negated_by_support = False
                    if not negated_by_cue:
                        for seg_start, seg_end in segments:
                            if seg_start <= idx < seg_end:
                                segment_text = clause[seg_start:seg_end]
                                term_in_segment = segment_text[idx - seg_start:].split()[0].lower()
                                # Check all support cues for this match
                                for cue in SUPPORT_CONTEXT_CUES:
                                    cue_idx = segment_text.find(cue.lower())
                                    if cue_idx != -1:
                                        cue_pos = seg_start + cue_idx
                                        # Suppress if cue appears before term OR term is contained in cue phrase
                                        if cue_pos < idx or (cue_idx <= (idx - seg_start) < cue_idx + len(cue)):
                                            negated_by_support = True
                                            break
                                if negated_by_support:
                                    break
                    negated = negated_by_cue or negated_by_support
                    if not negated:
                        return term, False
                    if first_negated is None:
                        first_negated = term
        return first_negated, first_negated is not None

    async def _stale_manager_reason(
        self, db: AsyncSession, goal: OrchestrationGoal, current
    ) -> str | None:
        """Spec 8.1 rerun trigger: 'current manager is removed or
        inactive.' Agents track both deletion and is_active. Users also
        track is_active.

        A `compressed` completion (trivial-goal implicit human manager, spec
        6.2) only has a NULL manager_user_id when the goal itself has no
        creator (goal.created_by_user_id is nullable) -- there was never a
        concrete person to validate. That generic-"human" case is
        indistinguishable from 'the human manager was deleted' and would
        otherwise auto-rerun/re-park forever. But a trivial goal WITH a
        creator persists that creator's concrete user id as manager_user_id
        (see the `trivial` branch above), and deleting/deactivating that
        creator must still trigger a rerun (review finding, HIGH) -- so the
        bypass only applies when no concrete id was ever recorded, never as
        a blanket skip for every compressed completion.

        Bug fix B: the compressed bypass must key off the persisted
        `selected_manager` output string, not the live `goal.manager_user_id`
        column (which FK SET NULL can null even for concrete creators). If
        selected_manager == "human", it's the generic case with no concrete
        id ever recorded, so no rerun needed. If it matches "human:<uuid>",
        extract and verify that uuid's user -- if missing or inactive,
        return a stale reason so a rerun is triggered."""
        if current.outputs.get("compressed"):
            selected_manager = current.outputs.get("selected_manager", "")
            if selected_manager == "human":
                # Generic implicit manager, no concrete id ever recorded.
                return None
            # Check if it's "human:<uuid>" -- concrete creator was recorded.
            if selected_manager.startswith("human:"):
                try:
                    user_id_str = selected_manager.removeprefix("human:")
                    user_id = uuid.UUID(user_id_str)
                    manager = await db.get(User, user_id)
                    if manager is None or not manager.is_active:
                        return "the selected human manager is removed or inactive"
                except (ValueError, AttributeError):
                    # Invalid UUID format in selected_manager -- malformed,
                    # treat as stale to trigger rerun and fix.
                    return "selected manager has invalid format"
                return None
        if goal.authority_model == "agent_manager":
            manager = (
                await db.get(Agent, goal.manager_agent_id)
                if goal.manager_agent_id is not None
                else None
            )
            if manager is None or not manager.is_active:
                return "the selected manager agent is removed or inactive"
        if goal.authority_model == "human_manager":
            manager = (
                await db.get(User, goal.manager_user_id)
                if goal.manager_user_id is not None
                else None
            )
            if manager is None or not manager.is_active:
                return "the selected human manager is removed or inactive"
        return None

    async def _roster_gained_candidates_reason(
        self, db: AsyncSession, goal: OrchestrationGoal, current
    ) -> str | None:
        """Rerun trigger (bug #89): a completed run that fell back to
        human/no_manager because the roster had ZERO agent candidates at
        selection time must re-evaluate once the roster gains a candidate --
        agent_definition_review/team_hierarchy run AFTER manager_selection and
        can create the very agents this process never got to compare. Trivial
        goals are excluded (the human is always assigned regardless of roster).
        A non-empty outputs["candidates"] means agents WERE compared and the
        human still chose not to use one -- that explicit outcome is not revisited.
        """
        if goal.weight == "trivial":
            return None
        if goal.authority_model not in ("human_manager", "no_manager", None):
            return None
        if current.outputs.get("candidates"):
            return None
        fits = await self.roster_mapper.rank_agents(db, goal.project_id, MANAGER_WORK_FUNCTION)
        if not fits:
            return None
        return "the agent roster now has candidates for management fit (none existed at selection)"

    async def handle_skip(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        skipped_run,
    ) -> None:
        """Spec 8.7: 'The human may proceed without a manager. If skipped,
        create an active warning: <verbatim text>.' Skipping and choosing
        the explicit no_manager option are the same real-world outcome, so
        this reuses the same warning type/message/consequence as
        `_complete`'s NO_MANAGER_OPTION branch. Called from the skip REST
        route (Task 4) right after the generic Phase 2 skip succeeds; the
        generic `manager_selection_skipped` warning is unaffected and both
        warnings remain active side by side.

        Run linkage is always derived from `skipped_run.run_id` (review
        finding, LOW), never from a caller-passed run: a later idempotent
        skip retry looks up the warning by `skipped_run.run_id` for its
        idempotency key, so any other run id here would let a second active
        no-manager warning slip past that check."""
        run_id = skipped_run.run_id
        goal.manager_agent_id = None
        goal.manager_user_id = None
        goal.authority_model = "no_manager"
        await db.flush()
        await self.warning_service.create_warning(
            db,
            goal.id,
            warning_type=NO_MANAGER_WARNING_TYPE,
            severity="warning",
            message=NO_MANAGER_WARNING_MESSAGE,
            run_id=run_id,
            source_process_run_id=skipped_run.id,
        )
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key=MEMORY_SECTION_KEY,
            title="Manager and authority model",
            body=self._memory_body(
                goal, "none selected", "human skipped manager selection (spec 8.7)", []
            ),
            summary="Manager: none selected; authority model no_manager.",
            toc_order=MEMORY_TOC_ORDER,
            run_id=run_id,
            created_by="orchestrator:manager_selection",
        )

    async def _ask(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current,
        top_fits: list[RosterFit],
        definition_signals: dict[uuid.UUID, tuple[int, list[str], bool]],
    ) -> None:
        # Strength is judged on the COMBINED score (review finding), never
        # on RosterFit.weak alone -- that field only reflects the
        # pre-inspection roster score and would misclassify a
        # definition-qualified candidate as weak.
        strong = []
        for fit in top_fits:
            bonus, _signals, definition_strong = definition_signals.get(
                fit.agent_id, (0, [], False)
            )
            if _is_strong_candidate(fit.score, bonus, definition_strong):
                strong.append(fit)
        options = [
            _candidate_option(fit, *definition_signals.get(fit.agent_id, (0, [], False)))
            for fit in top_fits
        ]
        options.append(
            {"key": HUMAN_AS_MANAGER_OPTION, "label": "The human acts as manager / main POC"}
        )
        options.append(
            {"key": NO_MANAGER_OPTION, "label": "Proceed without a manager (creates a warning)"}
        )
        # top_fits is already sorted by combined score (roster fit + spec
        # 8.3 step 3 definition-inspection bonus), so the first strong
        # candidate here is the overall best-fit recommendation.
        recommendation = (
            f"agent:{strong[0].agent_id}" if strong else HUMAN_AS_MANAGER_OPTION
        )
        context_lines = [
            "Deterministic manager-fit ranking over active agents (spec 8.3 "
            "steps 1-4, including instructions/config inspection):"
        ]
        if top_fits:
            for fit in top_fits:
                bonus, def_signals, definition_strong = definition_signals.get(
                    fit.agent_id, (0, [], False)
                )
                is_strong = _is_strong_candidate(fit.score, bonus, definition_strong)
                extra = f"; {', '.join(def_signals)}" if def_signals else ""
                context_lines.append(
                    f"- agent:{fit.agent_id} {fit.name} ({fit.role}): score {fit.score}"
                    f" + definition bonus {bonus}"
                    f"{'' if is_strong else ' [weak]'}{extra}"
                )
        else:
            context_lines.append("- no active agents with management fit")
        await self.decision_service.create_pending(
            db,
            goal.id,
            decision_key=SELECT_MANAGER_DECISION_KEY,
            title="Select manager / main point of contact",
            question=(
                "Who should act as manager / main point of contact for this "
                "goal? The orchestrator routes authority decisions to the "
                "manager instead of deciding silently (spec 8.2)."
            ),
            authority="human",
            options=options,
            context="\n".join(context_lines),
            recommendation=recommendation,
            consequences=(
                f"Choosing '{NO_MANAGER_OPTION}' creates a warning: "
                f"{NO_MANAGER_WARNING_MESSAGE}"
            ),
            run_id=run.id,
            source_process_run_id=current.id,
        )

    async def _orchestrate_selection(
        self, db, goal, run, current, top_fits, definition_signals
    ) -> dict | None:
        outputs = current.outputs or {}
        review = outputs.get("manager_review")
        if review is None and isinstance(outputs.get("_lm_retry"), dict):
            return {"process_type": "manager_selection", "status": current.status,
                    "questions_created": 0, "retryable": True, "error": outputs.get("error")}

        frozen = outputs.get("manager_candidates")
        recommendation = outputs.get("deterministic_recommendation")
        if frozen is None:
            frozen = []
            for fit in top_fits:
                agent = await db.get(Agent, fit.agent_id)
                bonus, signals, strong = definition_signals[fit.agent_id]
                frozen.append({
                    "key": f"agent:{fit.agent_id}", "label": f"{fit.name} ({fit.role})",
                    "role": fit.role, "description": agent.description, "system_prompt": agent.system_prompt,
                    "capabilities": list(fit.capabilities), "provider": fit.provider, "model": fit.model,
                    "workload": {"active_tasks": fit.load.active_tasks, "active_goals": fit.load.active_sessions},
                    "ranking_evidence": {"base_score": fit.score, "definition_bonus": bonus,
                                         "combined_score": fit.score + bonus, "signals": list(fit.matched_signals) + signals,
                                         "weak": not _is_strong_candidate(fit.score, bonus, strong)},
                })
            frozen.extend([
                {"key": HUMAN_AS_MANAGER_OPTION, "label": "The human acts as manager / main POC", "role": None,
                 "description": None, "system_prompt": None, "capabilities": [], "provider": None, "model": None,
                 "workload": {"active_tasks": 0, "active_goals": 0},
                 "ranking_evidence": {"base_score": 0, "definition_bonus": 0, "combined_score": 0,
                                      "signals": [], "weak": False}},
                {"key": NO_MANAGER_OPTION, "label": "Proceed without a manager", "role": None,
                 "description": None, "system_prompt": None, "capabilities": [], "provider": None, "model": None,
                 "workload": {"active_tasks": 0, "active_goals": 0},
                 "ranking_evidence": {"base_score": 0, "definition_bonus": 0, "combined_score": 0,
                                      "signals": [], "weak": False}},
            ])
            strong = next((fit for fit in top_fits if not next(
                candidate for candidate in frozen if candidate["key"] == f"agent:{fit.agent_id}"
            )["ranking_evidence"]["weak"]), None)
            recommendation = f"agent:{strong.agent_id}" if strong else HUMAN_AS_MANAGER_OPTION
            project = await db.get(Project, goal.project_id)
            project_dict = {"name": project.name, "description": project.description} if project else None
            payload = {
                "schema_version": 1,
                "goal": {"id": str(goal.id), "objective": goal.objective,
                         "success_criteria": [{"key": item["key"], "description": item["description"]}
                                              for item in goal.success_criteria],
                         "constraints": [f"{key}: {value}" for key, value in sorted((goal.constraints or {}).items())],
                         "weight": goal.weight},
                "candidates": frozen,
                "deterministic_recommendation": recommendation,
            }
            candidate_outputs = [
                {"key": item["key"], "label": item["label"],
                 "score": item["ranking_evidence"]["base_score"],
                 "definition_bonus": item["ranking_evidence"]["definition_bonus"],
                 "combined_score": item["ranking_evidence"]["combined_score"],
                 "weak": item["ranking_evidence"]["weak"],
                 "signals": [signal for signal in item["ranking_evidence"]["signals"]
                             if not signal.startswith("personality:")]}
                for item in frozen if item["key"].startswith("agent:")
            ]
            try:
                assessment = await self.analyzer.review(
                    payload, project=project_dict, project_id=goal.project_id
                )
            except Exception as exc:
                from huddleroom.services.orchestration_llm_decision_adapter import _safe_completion_error
                safe_error = _safe_completion_error(exc)
                request = getattr(exc, "request", self.analyzer.build_request(payload, project=project_dict))
                current.outputs = {**outputs, "manager_candidates": candidate_outputs,
                                   "deterministic_recommendation": recommendation,
                                   "_lm_retry": {"kind": "manager_selection", "version": 1,
                                                 "request": request},
                                   "retryable": True, "error": safe_error}
                await db.flush()
                return {"process_type": "manager_selection", "status": current.status,
                        "questions_created": 0, "retryable": True, "error": safe_error}
            review = {"verdict": assessment.verdict, "selected_key": assessment.selected_key,
                      "rationale": assessment.rationale}
            current.outputs = {**outputs, "manager_candidates": candidate_outputs,
                               "deterministic_recommendation": recommendation, "manager_review": review}
            await db.flush()

        if "candidate_outputs" not in locals():
            candidate_outputs = frozen
        if review["verdict"] == "confirm":
            return await self._apply_frozen_choice(
                db, goal, run, current, review["selected_key"], candidate_outputs, review["rationale"]
            )

        decisions = [d for d in await self.decision_service.list_decisions(db, goal.id)
                     if d.decision_key == REVIEW_OVERRIDE_DECISION_KEY and d.source_process_run_id == current.id]
        answered = next((d for d in reversed(decisions) if d.status == "answered"), None)
        if answered is not None:
            selected = review["selected_key"] if answered.selected_option == "approve" else recommendation
            rationale = review["rationale"] if answered.selected_option == "approve" else "human rejected LLM override"
            return await self._apply_frozen_choice(
                db, goal, run, current, selected, candidate_outputs, rationale
            )
        pending = next((d for d in decisions if d.status == "pending"), None)
        if pending is None:
            await self.decision_service.create_pending(
                db, goal.id, decision_key=REVIEW_OVERRIDE_DECISION_KEY,
                title="Review manager recommendation override",
                question="Approve the LLM manager choice?", authority="human",
                options=[{"key": "approve", "label": "Approve"}, {"key": "reject", "label": "Reject"}],
                context=(f"Deterministic: {recommendation}\nLLM: {review['selected_key']}\n"
                         f"Rationale: {review['rationale']}"), recommendation="approve",
                run_id=run.id, source_process_run_id=current.id,
            )
        await self.process_service.park_process(db, current)
        return {"process_type": "manager_selection", "status": current.status,
                "questions_created": 1 if pending is None else 0}

    async def _apply_frozen_choice(self, db, goal, run, current, key, candidates, rationale):
        if current.status == "waiting_decision":
            await self.process_service.resume_process(db, current)
        if key.startswith("agent:"):
            agent = await db.get(Agent, uuid.UUID(key.removeprefix("agent:")))
            if agent is None or not agent.is_active:
                return {"process_type": "manager_selection", "status": current.status,
                        "questions_created": 0, "retryable": False, "error": "frozen candidate is no longer active"}
            return await self._complete(db, goal, run, current, selected_option=key, manager_agent=agent,
                                        candidates=candidates, rationale=rationale, compressed=False)
        return await self._complete(
            db, goal, run, current, selected_option=key,
            manager_user_id=goal.created_by_user_id if key == HUMAN_AS_MANAGER_OPTION else None,
            candidates=candidates, rationale=rationale, compressed=False,
        )

    async def retry_failed(self, db, goal, run, current):
        checkpoint = (current.outputs or {}).get("_lm_retry")
        if not isinstance(checkpoint, dict) or checkpoint.get("kind") != "manager_selection":
            raise ValueError("manager selection has no valid retry checkpoint")
        try:
            assessment = await self.analyzer.review_request(checkpoint["request"], project_id=goal.project_id)
        except Exception as exc:
            from huddleroom.services.orchestration_llm_decision_adapter import _safe_completion_error
            safe_error = _safe_completion_error(exc)
            current.outputs = {key: value for key, value in (current.outputs or {}).items()
                               if key not in {"_lm_retry", "retryable", "error"}}
            current.outputs["_lm_retry"] = {"kind": "manager_selection", "version": 1, "request": checkpoint["request"]}
            current.outputs["retryable"] = True
            current.outputs["error"] = safe_error
            await db.flush()
            return {"process_type": "manager_selection", "status": current.status,
                    "questions_created": 0, "retryable": True, "error": safe_error}
        current.outputs = {key: value for key, value in (current.outputs or {}).items()
                           if key not in {"_lm_retry", "retryable", "error"}}
        current.outputs["manager_review"] = {"verdict": assessment.verdict,
                                             "selected_key": assessment.selected_key,
                                             "rationale": assessment.rationale}
        await db.flush()
        return await self.advance(db, goal, run)

    async def _complete(
        self,
        db: AsyncSession,
        goal: OrchestrationGoal,
        run: OrchestrationRun,
        current,
        *,
        selected_option: str,
        candidates: list[dict],
        rationale: str,
        compressed: bool,
        manager_agent: Agent | None = None,
        manager_user_id: uuid.UUID | None = None,
    ) -> dict:
        if selected_option == NO_MANAGER_OPTION:
            goal.manager_agent_id = None
            goal.manager_user_id = None
            goal.authority_model = "no_manager"
            selected_manager = None
            await self.warning_service.create_warning(
                db,
                goal.id,
                warning_type=NO_MANAGER_WARNING_TYPE,
                severity="warning",
                message=NO_MANAGER_WARNING_MESSAGE,
                run_id=run.id,
                source_process_run_id=current.id,
            )
        elif manager_agent is not None:
            goal.manager_agent_id = manager_agent.id
            goal.manager_user_id = None
            goal.authority_model = "agent_manager"
            selected_manager = f"agent:{manager_agent.id}"
        else:
            goal.manager_agent_id = None
            goal.manager_user_id = manager_user_id
            goal.authority_model = "human_manager"
            selected_manager = (
                f"human:{manager_user_id}" if manager_user_id is not None else "human"
            )
        if goal.authority_model in ("agent_manager", "human_manager"):
            # Review finding: a manager selected via rerun must supersede a
            # prior no-manager warning -- otherwise the spec-8.7 warning
            # stays active forever even after the human proceeds to pick a
            # real manager.
            for warning in await self.warning_service.list_warnings(
                db, goal.id, active_only=True
            ):
                if warning.warning_type == NO_MANAGER_WARNING_TYPE:
                    await self.warning_service.resolve_warning(
                        db,
                        warning,
                        resolved_by="orchestrator:manager_selection",
                        reason=f"manager selected: {selected_manager}",
                    )
        await db.flush()

        gates = {
            "manager_selected": goal.authority_model in ("agent_manager", "human_manager"),
            "authority_model_confirmed": True,
        }
        manager_label = selected_manager or "none selected"
        if manager_agent is not None:
            manager_label = f"{manager_agent.name} ({selected_manager})"
        await self.memory_service.upsert_section(
            db,
            goal.project_id,
            goal.id,
            section_key=MEMORY_SECTION_KEY,
            title="Manager and authority model",
            body=self._memory_body(goal, manager_label, rationale, candidates),
            summary=f"Manager: {manager_label}; authority model {goal.authority_model}.",
            toc_order=MEMORY_TOC_ORDER,
            run_id=run.id,
            created_by="orchestrator:manager_selection",
        )
        await self.process_service.complete_process(
            db,
            current,
            outputs={
                **{key: value for key, value in (current.outputs or {}).items()
                   if key in {"manager_review", "deterministic_recommendation"}},
                "selected_manager": selected_manager,
                "authority_model": goal.authority_model,
                "manager_fit_rationale": rationale,
                "candidates": candidates,
                "compressed": compressed,
                "gates": gates,
                "weight": goal.weight,  # Finding #1: persist weight for tier-change detection
            },
        )
        return {
            "process_type": "manager_selection",
            "status": "completed",
            "questions_created": 0,
        }

    @staticmethod
    def _memory_body(
        goal: OrchestrationGoal,
        manager_label: str,
        rationale: str,
        candidates: list[dict],
    ) -> str:
        lines = [
            f"Manager / main POC: {manager_label}",
            f"Authority model: {goal.authority_model}",
            f"Selection rationale: {rationale}",
            # Spec 8.5: fallback decision path when no manager exists.
            "Fallback decision path: authority decisions the manager cannot "
            "or does not answer go to the human; with no manager selected, "
            "every authority decision goes to the human.",
        ]
        # Filter to agent candidates only (exclude human_as_manager/no_manager options)
        agent_candidates = [c for c in candidates if c.get("key", "").startswith("agent:")]
        if agent_candidates:
            lines.append("Candidates considered (management fit):")
            for candidate in agent_candidates:
                lines.append(
                    f"- {candidate['label']}: score {candidate['score']}"
                    f" + definition bonus {candidate['definition_bonus']}"
                    f"{'' if not candidate['weak'] else ' [weak]'}"
                )
        else:
            lines.append("No agent candidates with management fit were found.")
        return "\n".join(lines)
