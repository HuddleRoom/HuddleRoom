from types import SimpleNamespace

from huddleroom.services import orchestration_review_brief as rb


def _build(**kw):
    args = dict(
        scope="Implement login", deliverable="Login page", success_evidence=["tests pass"],
        criterion_lines=["Users can log in"], report_excerpt="Did the thing", marker_ref="m-1",
        artifacts=[{"id": "a1", "version": 3, "content_hash": "abc"}],
    )
    args.update(kw)
    return rb.build_review_inputs(**args)


def test_sections_present_and_no_stale_wording_by_default():
    out = _build()
    text = "\n".join([out["scope"], out["deliverable"], *out["inputs"]])
    assert "Implement login" in out["scope"] and out["deliverable"] == "Login page"
    assert rb.JUDGE_INSTRUCTION in out["scope"]
    for needle in ("tests pass", "Users can log in", "Did the thing", "marker m-1", "Artifact a1, version 3, hash abc"):
        assert needle in text
    assert "stale orchestration gate" not in text


def test_stale_wording_only_when_stale():
    out = _build(stale_evidence=True)
    assert "stale orchestration gate" in out["scope"]
    assert rb.JUDGE_INSTRUCTION in out["scope"]


def test_artifact_objects_and_unknowns():
    out = _build(artifacts=[SimpleNamespace(id="a2", version=None, content_hash=None)])
    assert "Artifact a2, version unknown, hash unknown" in "\n".join(out["inputs"])


def test_truncation_is_stated():
    out = _build(
        scope="s" * (rb.MAX_SCOPE_CHARS + 1),
        report_excerpt="r" * (rb.MAX_REPORT_EXCERPT_CHARS + 1),
        success_evidence=["e"] * (rb.MAX_EVIDENCE_ITEMS + 1),
        criterion_lines=["c"] * (rb.MAX_CRITERIA + 1),
        artifacts=[{"id": str(i)} for i in range(rb.MAX_ARTIFACTS + 1)],
    )
    text = "\n".join([out["scope"], *out["inputs"]])
    for label in ("Scope truncated", "Producer report truncated", "Success evidence truncated",
                  "Criteria truncated", "Artifacts truncated"):
        assert label in text


def test_untruncated_has_no_truncation_note():
    out = _build()
    assert "truncated" not in "\n".join([out["scope"], *out["inputs"]])
