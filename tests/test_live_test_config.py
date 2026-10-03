from __future__ import annotations

from pathlib import Path

import pytest

from tests.live import live_test


def _live_agents_by_name() -> dict[str, dict[str, str]]:
    return {
        "live-architect": {"id": "agent-1"},
        "live-pragmatist": {"id": "agent-2"},
        "live-security": {"id": "agent-3"},
        "live-pm": {"id": "agent-4"},
    }


def _standup_turn(turn_number: int, speaker_agent_id: str, content: str) -> dict:
    return {
        "turn_number": turn_number,
        "speaker_agent_id": speaker_agent_id,
        "content": content,
        "agenda_item_id": "agenda-1",
    }



def test_project_venv_executable_prefers_repo_dotvenv(tmp_path: Path):
    executable = tmp_path / ".venv" / "bin" / "huddleroom"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")

    assert live_test.project_venv_executable(tmp_path, "huddleroom") == executable


def test_project_venv_executable_requires_repo_venv(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Project venv executable not found"):
        live_test.project_venv_executable(tmp_path, "huddleroom")


def test_collect_verification_errors_accepts_valid_t10_standup():
    verify = next(test for test in live_test.MEETING_TESTS if test["id"] == "T10")["verify"]
    agents_by_name = _live_agents_by_name()
    turns = [
        _standup_turn(1, "agent-1", "DONE: reviewed API docs\nNOW: drafting interface changes\nBLOCKERS: none"),
        _standup_turn(2, "agent-2", "DONE: fixed the flaky test\nNOW: validating the patch\nBLOCKERS: none"),
        _standup_turn(3, "agent-3", "DONE: checked auth logs\nNOW: reviewing token scopes\nBLOCKERS: none"),
        _standup_turn(4, "agent-4", "DONE: updated the sprint notes\nNOW: confirming release scope\nBLOCKERS: waiting on signoff"),
    ]
    errors = live_test.collect_verification_errors(
        verify=verify,
        turns=turns,
        decisions=[],
        events=[{"event_type": "meeting.concluded"}],
        final_meeting={
            "agenda_items": [
                {
                    "id": "agenda-1",
                    "title": "Daily standup",
                    "status": "resolved",
                    "resolution_kind": "updates_shared",
                }
            ],
            "is_partial": False,
        },
        final_status="concluded",
        action_items=[{"id": "followup-1"}],
        agent_id_to_name={details["id"]: name for name, details in agents_by_name.items()},
        agents_by_name=agents_by_name,
        concluded=True,
    )

    assert errors == []


def test_collect_verification_errors_rejects_incomplete_t10_standup():
    verify = next(test for test in live_test.MEETING_TESTS if test["id"] == "T10")["verify"]
    agents_by_name = _live_agents_by_name()
    turns = [
        _standup_turn(1, "agent-1", "DONE: reviewed API docs\nNOW: drafting interface changes"),
        _standup_turn(2, "agent-2", "DONE: fixed the flaky test\nNOW: validating the patch\nBLOCKERS: none"),
        _standup_turn(3, "agent-3", "DONE: checked auth logs\nNOW: reviewing token scopes\nBLOCKERS: none"),
        _standup_turn(4, "agent-4", "DONE: updated the sprint notes\nNOW: confirming release scope\nBLOCKERS: none"),
    ]
    errors = live_test.collect_verification_errors(
        verify=verify,
        turns=turns,
        decisions=[{"id": "decision-1"}],
        events=[],
        final_meeting={
            "agenda_items": [
                {
                    "id": "agenda-1",
                    "title": "Daily standup",
                    "status": "active",
                    "resolution_kind": None,
                },
                {
                    "id": "agenda-2",
                    "title": "Yesterday carryover",
                    "status": "resolved",
                    "resolution_kind": "consensus",
                }
            ],
            "is_partial": True,
        },
        final_status="active",
        action_items=[],
        agent_id_to_name={details["id"]: name for name, details in agents_by_name.items()},
        agents_by_name=agents_by_name,
        concluded=False,
    )

    assert any("Expected ≤0 decisions, got 1" in error for error in errors)
    assert any("Missing DONE/NOW/BLOCKERS lines" in error for error in errors)
    assert any("Agenda items not completed" in error for error in errors)
    assert any("Agenda items missing required resolution kinds" in error for error in errors)
    assert any("Expected terminal WS event 'meeting.concluded'" in error for error in errors)
    assert any("Final meeting status is 'active', expected 'concluded'" in error for error in errors)
    assert "Meeting was marked partial, but a full conclusion is required" in errors


def test_generate_html_report_renders_action_items_and_moderator_trace():
    run_log = live_test.RunLog()
    run_log.results = [
        live_test.TestResult(
            test_id="T11",
            label="Architecture Design Follow-up",
            passed=True,
            message="PASS",
            turns=1,
            decisions=1,
            meeting_type="decision",
            turn_strategy="moderated",
        )
    ]
    run_log.test_conversations["T11"] = {
        "turns": [
            {
                "turn_number": 1,
                "speaker_agent_id": "agent-1",
                "agenda_item_id": "agenda-1",
                "content": "POSITION: Event-driven architecture",
                "model_used": "openai/gpt-5-nano",
                "latency_ms": 120,
                "moderator_note": None,
                "prompt_messages": [],
                "raw_response": "POSITION: Event-driven architecture",
                "reasoning_content": None,
                "organizer_selection": {
                    "selector_type": "moderator",
                    "selected_by": "orchestration_model",
                    "next_speaker_id": "agent-1",
                    "reason": "Architect should frame the trade-offs first.",
                    "model_used": "openai/gpt-4o-mini",
                    "messages": [{"role": "system", "content": "You are a meeting moderator."}],
                    "raw_response": '{"next_speaker_id":"agent-1","reason":"Architect should frame the trade-offs first.","close_item":false,"close_reason":null}',
                },
            }
        ],
        "decisions": [
            {
                "agenda_item_id": "agenda-1",
                "title": "Architecture direction",
                "chosen_option": "Event-driven architecture",
                "rationale": "Scales better for async workflows.",
                "decided_by": "consensus",
            }
        ],
        "action_items": [
            {
                "description": "Draft an ADR for the event-driven split",
                "status": "task_created",
            }
        ],
        "agenda_items": [
            {
                "id": "agenda-1",
                "title": "Architecture direction",
                "resolution_kind": "consensus",
                "resolution_summary": "Team aligned on an event-driven split.",
            }
        ],
        "meeting": {"id": "meeting-1"},
        "agent_id_to_name": {"agent-1": "live-architect"},
    }

    html = live_test.generate_html_report(run_log, "2026-05-18 10:00:00")

    assert "Action Items" in html
    assert "Draft an ADR for the event-driven split" in html
    assert "Moderator selected" in html
    assert "orchestration_model" in html
    assert "cheap_model" not in html
    assert "openai/gpt-4o-mini" in html


def test_generate_html_report_renders_signal_probing_section():
    run_log = live_test.RunLog()
    run_log.results = [
        live_test.TestResult(
            test_id="T11",
            label="Architecture Design Follow-up",
            passed=True,
            message="PASS",
            turns=1,
            decisions=0,
            meeting_type="decision",
            turn_strategy="moderated",
        )
    ]
    run_log.test_conversations["T11"] = {
        "turns": [],
        "decisions": [],
        "action_items": [],
        "agenda_items": [],
        "meeting": {"id": "meeting-1"},
        "agent_id_to_name": {"agent-2": "live-pragmatist"},
    }
    run_log.add(
        "ws_event",
        "T11",
        {
            "meeting_id": "meeting-1",
            "event_type": "meeting.signal_probe",
            "payload": {
                "event_type": "meeting.signal_probe",
                "payload": {
                    "meeting_id": "meeting-1",
                    "probed_agent_id": "agent-2",
                    "speaking_agent_id": "agent-1",
                    "probe_response": "YES: I want to challenge the migration cost.",
                    "signaled": True,
                    "signal_message": "I want to challenge the migration cost.",
                },
            },
        },
    )

    html = live_test.generate_html_report(run_log, "2026-05-18 10:00:00")

    assert "Signal Probing" in html
    assert "live-pragmatist" in html
    assert "YES: I want to challenge the migration cost." in html
    assert "Signal emitted" in html
