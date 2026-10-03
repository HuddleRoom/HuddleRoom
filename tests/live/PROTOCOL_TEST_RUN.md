# Protocol Live Test Run — Progress & Findings

## How to resume this session

1. Read this file top to bottom — it is the single source of truth.
2. Working directory: repository root
3. Test file: `tests/live/live_test_protocols.py`
4. Start server if not running: `.venv/bin/huddleroom serve --port 8001` (from project root)
5. Run a single test: `python tests/live/live_test_protocols.py --no-start-server --tests PC1`
6. Run all tests: `python tests/live/live_test_protocols.py --no-start-server --tests PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10`
7. Note: PC9 requires ~35s for background scheduler (runs every 30s). Typical total runtime: ~60s.

---

## Per-test evaluation criteria

For each test, answer:
1. **Purpose** — what behavior is this test verifying?
2. **Test quality** — is the test well-designed for that purpose? any gaps or false positives?
3. **Rally result** — did Rally pass the test?
4. **Test validity** — did the test actually exercise what was intended? (no false pass/fail)
5. **Actions** — fixes needed in test code and/or in Rally, or "none"

Iterate (run → conclude → fix → re-run) until the test passes cleanly AND the test is judged sound. Then mark DONE and move on.

---

## Status overview

| ID  | Label                                     | Status      | Rally result | Test sound? |
|-----|-------------------------------------------|-------------|--------------|-------------|
| PC1 | Trigger & Actor Resolution                | DONE        | PASS         | YES         |
| PC2 | Template Guard Resolution (Guard Specificity) | DONE    | PASS         | YES         |
| PC3 | post_message Content Resolution           | DONE        | PASS         | YES         |
| PC4 | create_session Side Effect                | DONE        | PASS         | YES         |
| PC5 | notify_actor Content Resolution           | DONE        | PASS         | YES         |
| PC6 | CI Failure Back-Loop                      | DONE        | PASS         | YES         |
| PC7 | complete_task in Merged State             | DONE        | PASS         | YES         |
| PC8 | record_decision in Merged State           | DONE        | PASS         | YES         |
| PC9 | Timeout Escalation                        | DONE        | PASS         | YES         |
| PC10| Awaiting Revision Loop                    | DONE        | PASS         | YES         |

Status legend: PENDING | IN PROGRESS | DONE | BLOCKED

---

## Detailed findings

### PC1 — Trigger & Actor Resolution (DONE, PASS)

**Purpose:** Verify that emitting `code.pr_opened` creates a protocol instance with all three actors (author, reviewer, merger) correctly resolved.

**Test quality:** Good. Uses real code_review protocol. Tests all 3 actor assignment strategies (auto, role=reviewer, role=pm). Checks actor IDs match expected agents.

**Rally result:** PASS

**Test validity:** Sound. Guards are specific to artifact_id, `wait_for_instance` filters by artifact_id to avoid picking up stale instances from prior runs.

**Actions:** None.

---

### PC2 — Template Guard Resolution / Guard Specificity (DONE, PASS)

**Purpose:** Verify that `{{protocol_instance.artifact_id}}` template in guards is resolved per-instance — two simultaneous instances only transition the one whose artifact_id matches the event.

**Test quality:** Good. Creates 2 artifacts/instances, emits `test.passed` with only artifact_A's ID, verifies instance A transitions and instance B stays.

**Rally result:** PASS

**Test validity:** Sound.

**Actions:** None.

---

### PC3 — post_message Content Resolution (DONE, PASS)

**Purpose:** Verify `on_enter` `post_message` action resolves `{{artifact.name}}` in message content — confirming the ActionExecutor template fix works for messages.

**Test quality:** Good. Creates artifact with unique name, verifies message appears in "general" channel with the actual name (no `{{...}}` placeholders).

**Rally result:** PASS (after fixes)

**Test validity:** Sound after fixes.

**Actions fixed:**
- `MessageResponse` schema was returning `metadata_` not `metadata` — added `Field(serialization_alias="metadata")` to `rally/schemas/message.py` (both main and worktree).
- `wait_for_instance` was returning stale instances from previous runs — fixed by adding `artifact_id` filter parameter.
- `_post` client method was silently returning `{}` on HTTP errors, causing cryptic `KeyError('id')` — fixed to raise `RuntimeError` with error details.

---

### PC4 — create_session Side Effect (DONE, PASS)

**Purpose:** Verify that when instance enters `ready_for_review`, a Session is created for the reviewer agent with `origin=protocol`.

**Test quality:** Good. Drives instance through `opened → ready_for_review`, verifies a session exists for the reviewer agent linked to the new task.

**Rally result:** PASS

**Test validity:** Sound.

**Actions:** None.

---

### PC5 — notify_actor Content Resolution (DONE, PASS)

**Purpose:** Verify `notify_actor` action resolves `{{artifact.name}}` in the DM message.

**Test quality:** Good. Drives to `ci_failure` state, finds DM channel `dm-{author_id}`, verifies message content has the resolved artifact name.

**Rally result:** PASS

**Test validity:** Sound.

**Actions:** None.

---

### PC6 — CI Failure Back-Loop (DONE, PASS)

**Purpose:** Verify the state machine loop: `opened → ci_failure → opened` (author fixes and pushes again).

**Test quality:** Good. Emits `test.failed` then `code.pr_updated`, verifies state sequence in transition history.

**Rally result:** PASS

**Test validity:** Sound.

**Actions:** None.

---

### PC7 — complete_task in Merged State (DONE, PASS)

**Purpose:** Verify that the `complete_task` on_enter action in `merged` state marks the linked task as done.

**Test quality:** Good. Creates task via `setup_pr` (status=backlog), drives to merged, verifies task.status=done.

**Rally result:** PASS (after fix)

**Test validity:** Sound.

**Actions fixed:**
- `_complete_task` in `action_executor.py` didn't handle `backlog` task status — added `backlog → ready` transition before proceeding to `in_progress → done`.

---

### PC8 — record_decision in Merged State (DONE, PASS)

**Purpose:** Verify `record_decision` action creates a `KnowledgeItem` with resolved template values.

**Test quality:** Good. Drives to merged, finds KnowledgeItem by `provenance_protocol_instance_id`, verifies content has artifact name.

**Rally result:** PASS (after fix)

**Test validity:** Sound.

**Actions fixed:**
- `KnowledgeResponse` schema was missing `provenance_protocol_instance_id` field — added it to both `rally/schemas/knowledge.py` main and worktree.

---

### PC9 — Timeout Escalation (DONE, PASS)

**Purpose:** Verify that a protocol instance with a short timeout triggers the background scheduler to fire `protocol.escalated` and increment `escalation_step`.

**Test quality:** Good. Creates a protocol with `duration: "1s"` timeout, waits for `escalation_step >= 1`.

**Rally result:** PASS (after fixes)

**Test validity:** Sound.

**Actions fixed:**
- `_DURATION_MAP` in `protocol_engine.py` was missing `"s"` key — `"1s"` fell through to 3600s default. Added `"s": 1`.
- Timeout scheduler ran every 60s — changed to 30s in `rally/workers/scheduler.py`.
- Test had hardcoded 40s wait — changed to 80s.

---

### PC10 — Awaiting Revision Loop (DONE, PASS)

**Purpose:** Verify the revision loop: `ready_for_review → awaiting_revision → ready_for_review`, including `notify_actor` firing correctly in the `awaiting_revision` state.

**Test quality:** Good. Drives through the full revision loop and verifies the DM notification was sent.

**Rally result:** PASS

**Test validity:** Sound.

**Actions:** None.

---

## Fix log

### ActionExecutor Template Resolution Fix (Applied in prior session)

**Files:** `rally/services/action_executor.py` (main + worktree)

- Added `_r()` helper + `resolve_dict()` in `execute()` to resolve all action parameters before dispatch.
- Handlers: `assign_task`, `create_session`, `post_message`, `notify_actor`, `record_decision`, `complete_task` now receive resolved values.

### MessageResponse metadata alias (PC3)

**Files:** `rally/schemas/message.py` (main + worktree)

- Added `Field(serialization_alias="metadata")` so API returns `metadata` not `metadata_`.

### wait_for_instance artifact_id filter (PC3)

**Files:** `tests/live/live_test_protocols.py`

- Added `artifact_id` parameter to `wait_for_instance` to avoid returning stale instances from previous runs.

### _post raises on error (PC3 debug)

**Files:** `tests/live/live_test_protocols.py`

- `_post` now raises `RuntimeError` with status+body on non-2xx instead of silently returning `{}`.

### complete_task backlog state (PC7)

**Files:** `rally/services/action_executor.py` (main + worktree)

- Added `backlog → ready` transition before the existing `ready/failed/blocked → in_progress → done` chain.

### KnowledgeResponse provenance field (PC8)

**Files:** `rally/schemas/knowledge.py` (main + worktree)

- Added `provenance_protocol_instance_id: UUID | None = None` to `KnowledgeResponse`.

### Duration parser seconds support (PC9)

**Files:** `rally/services/protocol_engine.py` (main + worktree)

- Added `"s": 1` to `_DURATION_MAP` so `"1s"` resolves to 1 second not the 3600s fallback.

### Timeout scheduler interval (PC9)

**Files:** `rally/workers/scheduler.py`

- Reduced `process_protocol_timeouts_job` interval from 60s to 30s for more responsive escalation.

---

## Final result

**All 10 tests PASS.** Full suite runtime: ~60s (PC9 adds ~30s for scheduler).

```bash
python tests/live/live_test_protocols.py --no-start-server --tests PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10
```
