import pytest

from tests.conftest import adopt_session, owned_session
from huddleroom.services.orchestration_baseline_runtime import owns_control_transaction


# Existing un-adopted autobegin 409 guard (covers the plain-session transaction-shape refusal):
# tests/test_orchestration_llm_decision_adapter.py::test_request_llm_decision_rejects_preflushed_committed_run_caller


async def test_adopt_session_marks_owned_in_test_body(owned_session_factory):
    async with owned_session_factory() as db:
        adopt_session(db)
        assert owns_control_transaction(db) is True


async def test_owned_session_context_marks_owned(owned_session_factory):
    async with owned_session(owned_session_factory) as s:
        assert owns_control_transaction(s) is True


async def test_adopt_session_rejects_explicit_begin(owned_session_factory):
    async with owned_session_factory() as db, db.begin():
        with pytest.raises(AssertionError, match="autobegin"):
            adopt_session(db)
