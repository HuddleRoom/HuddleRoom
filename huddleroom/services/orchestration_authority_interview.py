from __future__ import annotations

from datetime import datetime, timezone

from huddleroom.models.orchestration import OrchestrationGoal, OrchestrationRun
from huddleroom.models.orchestration_process import OrchestrationAuthorityDecision
from huddleroom.services.orchestration_memory_service import OrchestrationMemoryService

OPEN_QUESTIONS_SECTION_KEY = "open_questions"
OPEN_QUESTIONS_TOC_ORDER = 90


def _newest_first_key(decision: OrchestrationAuthorityDecision) -> tuple[float, str]:
    occurred_at: datetime = decision.asked_at or decision.created_at
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=timezone.utc)
    else:
        occurred_at = occurred_at.astimezone(timezone.utc)
    return (-occurred_at.timestamp(), str(decision.id))


def build_checkpoint(
    pending: list[OrchestrationAuthorityDecision],
    *,
    max_questions: int,
) -> tuple[list[OrchestrationAuthorityDecision], list[OrchestrationAuthorityDecision]]:
    """Split pending decisions into (checkpoint, deferred) for the human
    kickoff checkpoint (spec 6.4).

    Only human-authority decisions are checkpoint material — manager/
    team_lead/agent decisions are delivered to their agent by
    `OrchestrationService._sync_agent_authority_decisions`, never shown to
    the human as a question to answer directly.

    Non-proposal questions are ordered newest-first before the cap is applied.
    Naive SQLite timestamps are interpreted as UTC, and equal timestamps use
    the immutable decision id as a deterministic tie break.

    Exception: `agent_definition_review:proposal:` decisions are pinned to the
    front of the checkpoint ahead of the independent questions and are never deferred,
    because they are answered as one atomic all-or-nothing batch (see below).
    This assumes a single review batch is pending at a time — the invariant the
    review's cancel-on-rerun logic maintains.
    """
    if max_questions < 1:
        raise ValueError(f"max_questions must be >= 1, got {max_questions}")
    human_pending = sorted(
        (d for d in pending if d.authority == "human" and d.status == "pending"),
        key=_newest_first_key,
    )
    # agent_definition_review proposals are answered as one atomic batch by a
    # dedicated all-or-nothing endpoint; splitting them across the max_questions
    # cap deadlocks that endpoint (submitted subset can never equal the full
    # pending set), so they are never deferred. The cap applies only to
    # independent questions.
    # ponytail: whole-batch bypasses the UX cap — fine for roster-sized batches;
    # if a review batch ever gets very large, make the endpoint accept subsets.
    batch = [d for d in human_pending if d.decision_key.startswith("agent_definition_review:proposal:")]
    rest = [d for d in human_pending if not d.decision_key.startswith("agent_definition_review:proposal:")]
    return batch + rest[:max_questions], rest[max_questions:]


async def sync_deferred_questions_memory(
    db,
    goal: OrchestrationGoal,
    run: OrchestrationRun,
    deferred: list[OrchestrationAuthorityDecision],
) -> None:
    """Spec 6.4 point 3: unanswered lower-value checkpoint questions become
    open questions in memory, not blockers. Full-replace on every call that
    actually writes (memory writes are always full-replace, spec 5.5) so a
    question that gets answered or re-prioritized out of the deferred set
    disappears from here instead of rotting.

    Called once per tick per goal, so a goal with zero deferred
    questions and no prior `open_questions` section must be a true no-op --
    otherwise every goal gets a DB write every tick forever for a section
    that would only ever say "no open questions". Once a section exists
    (there was ever at least one deferred question), we keep writing so a
    goal that clears its last deferred question gets the stale content
    replaced instead of left behind.
    """
    memory_service = OrchestrationMemoryService()
    if not deferred:
        existing = await memory_service.get_section(db, goal.project_id, goal.id, OPEN_QUESTIONS_SECTION_KEY)
        if existing is None:
            return
        body = "No open questions are currently deferred from the kickoff checkpoint."
    else:
        body = "\n".join(f"- {d.title}: {d.question}" for d in deferred)
    await memory_service.upsert_section(
        db,
        goal.project_id,
        goal.id,
        section_key=OPEN_QUESTIONS_SECTION_KEY,
        title="Open questions",
        body=body,
        summary=f"{len(deferred)} question(s) deferred from the kickoff checkpoint.",
        toc_order=OPEN_QUESTIONS_TOC_ORDER,
        run_id=run.id,
    )
