"""Pure unit tests for tests/live/orchestration_baseline_e2e.py.

No live server, no network, no DB. Imports the module directly via importlib
since tests/live is not a package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent / "orchestration_baseline_e2e.py"
_spec = importlib.util.spec_from_file_location("orchestration_baseline_e2e", _MODULE_PATH)
obe = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = obe
_spec.loader.exec_module(obe)


def make_process(process_type, status="completed", outputs=None, process_id="p1", **extra):
    return {
        "id": process_id,
        "process_type": process_type,
        "status": status,
        "outputs": outputs if outputs is not None else {},
        **extra,
    }


ORIGINAL_TYPES = ("goal_definition", "manager_selection", "agent_definition_review", "team_hierarchy")
JUDGEABLE_TYPES = ("effectiveness_review", "goal_closeout")


class TestPhaseAction:
    def test_none_process_returns_step(self):
        assert obe._phase_action(None, []) == "step"

    @pytest.mark.parametrize("process_type", JUDGEABLE_TYPES)
    def test_judgeable_waiting_decision_no_verdict_is_judge_existing(self, process_type):
        process = make_process(process_type, status="waiting_decision")
        assert obe._phase_action(process, []) == "judge_existing"

    @pytest.mark.parametrize("process_type", ORIGINAL_TYPES)
    def test_original_types_waiting_decision_still_steps(self, process_type):
        process = make_process(process_type, status="waiting_decision")
        assert obe._phase_action(process, []) == "step"

    @pytest.mark.parametrize("process_type", obe.PROCESS_ORDER)
    def test_completed_with_reusable_pass_verdict_skips(self, process_type):
        outputs = {"k": "v"}
        process = make_process(process_type, status="completed", outputs=outputs, process_id="p1")
        fingerprint = obe.canonical_fingerprint(outputs)
        verdicts = [
            {"process_id": "p1", "fingerprint": fingerprint, "verdict": "pass"},
        ]
        assert obe._phase_action(process, verdicts) == "skip"

    def test_completed_no_verdict_returns_judge_existing(self):
        process = make_process("goal_definition", status="completed", outputs={"a": 1})
        assert obe._phase_action(process, []) == "judge_existing"

    def test_clean_no_work_closeout_skips_without_a_verdict(self):
        process = make_process(
            "goal_closeout",
            outputs={
                "no_executable_work": True,
                "full_closeout": False,
                "completion_authorized": True,
                "mode": "completion",
                "gates": {"closeout_completed": True},
            },
        )
        assert obe._phase_action(process, []) == "no_work_skip"

    @pytest.mark.parametrize(
        "outputs",
        [
            {"no_executable_work": True, "full_closeout": True},
            {"no_executable_work": False, "full_closeout": False},
            {"full_closeout": False},
            {
                "no_executable_work": True,
                "full_closeout": False,
                "completion_authorized": False,
                "mode": "completion",
                "gates": {"closeout_completed": True},
            },
            {
                "no_executable_work": True,
                "full_closeout": False,
                "completion_authorized": True,
                "mode": "completion",
                "gates": {"closeout_completed": False},
            },
            {
                "no_executable_work": True,
                "full_closeout": False,
                "completion_authorized": True,
                "mode": "completion",
                "gates": {"closeout_completed": True},
                "warning_ids": ["warning-1"],
            },
        ],
    )
    def test_other_closeout_shapes_remain_judgeable(self, outputs):
        process = make_process("goal_closeout", outputs=outputs)
        assert obe._phase_action(process, []) == "judge_existing"

    def test_needs_human_judgment_same_fingerprint_reruns(self):
        outputs = {"a": 1}
        process = make_process("goal_definition", status="completed", outputs=outputs, process_id="p1")
        fingerprint = obe.canonical_fingerprint(outputs)
        verdicts = [{"process_id": "p1", "fingerprint": fingerprint, "verdict": "needs_human_judgment"}]
        assert obe._phase_action(process, verdicts) == "rerun"

    def test_needs_human_judgment_different_fingerprint_judge_existing(self):
        outputs = {"a": 2}
        process = make_process("goal_definition", status="completed", outputs=outputs, process_id="p1")
        verdicts = [{"process_id": "p1", "fingerprint": "stale-fingerprint", "verdict": "needs_human_judgment"}]
        assert obe._phase_action(process, verdicts) == "judge_existing"

    def test_clean_no_work_closeout_with_malformed_id_blocks(self):
        process = make_process(
            "goal_closeout",
            process_id="",
            outputs={
                "no_executable_work": True,
                "full_closeout": False,
                "completion_authorized": True,
                "mode": "completion",
                "gates": {"closeout_completed": True},
            },
        )
        with pytest.raises(obe.Blocked):
            obe._phase_action(process, [])


class TestCheckBlockers:
    @pytest.mark.parametrize("process_type", JUDGEABLE_TYPES)
    def test_judgeable_waiting_decision_not_a_blocker(self, process_type):
        process = make_process(process_type, status="waiting_decision")
        assert obe._check_blockers([process]) is None

    @pytest.mark.parametrize("process_type", ORIGINAL_TYPES)
    def test_original_types_waiting_decision_is_blocker(self, process_type):
        process = make_process(process_type, status="waiting_decision")
        assert obe._check_blockers([process]) == process

    def test_blocker_reason_waiting_decision(self):
        process = make_process("manager_selection", status="waiting_decision")
        assert obe.blocker_reason(process) == "waiting for approval"

    def test_current_processes_rejects_duplicate_types(self):
        processes = [
            make_process("goal_definition", process_id="a"),
            make_process("goal_definition", process_id="b"),
        ]
        with pytest.raises(obe.Blocked):
            obe.current_processes(processes)

    def test_current_processes_skips_superseded(self):
        processes = [
            make_process("goal_definition", process_id="a", superseded_by_id="b"),
            make_process("goal_definition", process_id="b"),
        ]
        current = obe.current_processes(processes)
        assert current["goal_definition"]["id"] == "b"


class FakeResponse:
    def __init__(self, status_code, body=None, json_raises=False):
        self.status_code = status_code
        self._body = body
        self._json_raises = json_raises

    def json(self):
        if self._json_raises:
            raise ValueError("bad json")
        return self._body


class TestCloseoutNotReady:
    @pytest.mark.parametrize(
        "detail",
        [
            "goal_definition, manager_selection, agent_definition_review, and team_hierarchy must all be terminal before goal_closeout can advance",
            "All orchestration gates must be accepted",
            "Accepted gate evidence is missing",
            "Final summary evidence is missing",
        ],
    )
    def test_409_with_exact_not_ready_detail_returns_true(self, detail):
        response = FakeResponse(409, {"detail": detail})
        assert obe._closeout_not_ready(response) is True

    @pytest.mark.parametrize(
        "detail",
        [
            "Active blocker warning 'x' must be resolved before completion",
            "Active warning 'x' must be acknowledged or resolved before completion",
            "Orchestration budget is exceeded",
            "Final summary evidence is invalid",
            "Final summary output must be valid JSON",
            "Final summary session is missing",
            "goal is 'blocked'; cannot advance a process",
            "active run is 'running'; not tickable",
            "goal has no active run to advance",
            "Retry the failed LM request via baseline/retry",
        ],
    )
    def test_409_with_blocking_or_unrelated_detail_returns_false(self, detail):
        response = FakeResponse(409, {"detail": detail})
        assert obe._closeout_not_ready(response) is False

    @pytest.mark.parametrize(
        "partial",
        [
            "Active ",
            "All orchestration gates",
            "Accepted gate evidence",
            "Final summary",
        ],
    )
    def test_409_with_partial_prefix_does_not_match_returns_false(self, partial):
        """Partial/truncated strings must not match; only exact strings count."""
        response = FakeResponse(409, {"detail": partial})
        assert obe._closeout_not_ready(response) is False

    @pytest.mark.parametrize("status_code", [200, 500])
    def test_non_409_returns_false(self, status_code):
        response = FakeResponse(status_code, {"detail": "goal_definition, manager_selection"})
        assert obe._closeout_not_ready(response) is False

    def test_409_json_raises_returns_false(self):
        response = FakeResponse(409, json_raises=True)
        assert obe._closeout_not_ready(response) is False

    def test_409_non_string_detail_returns_false(self):
        response = FakeResponse(409, {"detail": None})
        assert obe._closeout_not_ready(response) is False


class TestNoWorkCloseout:
    @staticmethod
    def process():
        return make_process(
            "goal_closeout",
            outputs={
                "no_executable_work": True,
                "full_closeout": False,
                "completion_authorized": True,
                "mode": "completion",
                "gates": {"closeout_completed": True},
            },
        )

    def test_fresh_no_work_closeout_clears_stale_packet(self, monkeypatch, capsys, tmp_path):
        process = self.process()
        packet_path = tmp_path / "judgment-packet.json"
        packet_path.write_text("stale", encoding="utf-8")
        states = iter([[], [process]])
        monkeypatch.setattr(obe, "_processes", lambda *args: next(states))
        monkeypatch.setattr(obe, "_load_verdicts", lambda: [])
        monkeypatch.setattr(obe, "_request", lambda *args, **kwargs: {"process": process})
        monkeypatch.setattr(obe, "_write_checkpoint", lambda *args: None)
        monkeypatch.setattr(obe, "_append_event", lambda *args, **kwargs: None)
        monkeypatch.setattr(obe, "_emit_packet", lambda packet: pytest.fail("unexpected judgment packet"))
        monkeypatch.setattr(obe, "PACKET_PATH", packet_path)

        assert obe._advance(object(), "project", "goal", ("goal_closeout",), 0, 0) == 0
        assert "SKIP: goal_closeout completed (no executable work)" in capsys.readouterr().out
        assert not packet_path.exists()

    def test_resumed_no_work_closeout_makes_no_post_and_clears_stale_packet(
        self, monkeypatch, capsys, tmp_path
    ):
        process = self.process()
        packet_path = tmp_path / "judgment-packet.json"
        packet_path.write_text("stale", encoding="utf-8")
        monkeypatch.setattr(obe, "_processes", lambda *args: [process])
        monkeypatch.setattr(obe, "_load_verdicts", lambda: [])
        monkeypatch.setattr(obe, "_request", lambda *args, **kwargs: pytest.fail("unexpected POST"))
        monkeypatch.setattr(obe, "_write_checkpoint", lambda *args: None)
        monkeypatch.setattr(obe, "_append_event", lambda *args, **kwargs: None)
        monkeypatch.setattr(obe, "PACKET_PATH", packet_path)

        assert obe._advance(object(), "project", "goal", ("goal_closeout",), 0, 0) == 0
        assert "SKIP: goal_closeout completed (no executable work)" in capsys.readouterr().out
        assert not packet_path.exists()


class TestPhasesConsistency:
    @pytest.mark.parametrize("process_type", obe.PROCESS_ORDER)
    def test_process_order_has_phase_entry(self, process_type):
        phase = obe.PHASES.get(process_type)
        assert phase is not None
        assert isinstance(phase["purpose"], str) and phase["purpose"]
        assert len(phase["acceptance_criteria"]) == 3
