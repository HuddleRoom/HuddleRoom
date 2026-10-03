import pytest
from huddleroom.services.orchestration_effectiveness_analyzer import (
    parse_effectiveness_analysis,
    EffectivenessAnalysis,
    VALID_DISPOSITIONS,
)


def test_extra_key_ignored():
    """Extra keys in payload are ignored, parsing succeeds."""
    result = parse_effectiveness_analysis({
        "disposition": "continue",
        "findings": [],
        "rationale": "ok",
        "unexpected": 123,
        "another_extra": "should be ignored",
    })
    assert result.disposition == "continue"
    assert result.findings == ()
    assert result.rationale == "ok"


def test_disposition_coercion_whitespace():
    """Disposition is trimmed."""
    result = parse_effectiveness_analysis({"disposition": "  CONTINUE  "})
    assert result.disposition == "continue"


def test_disposition_coercion_case():
    """Disposition is lowercased."""
    result = parse_effectiveness_analysis({"disposition": "Revise"})
    assert result.disposition == "revise"


def test_disposition_coercion_mixed_case():
    """Disposition case and whitespace both normalized."""
    result = parse_effectiveness_analysis({"disposition": "  PAUSE  "})
    assert result.disposition == "pause"


def test_missing_findings_defaults_to_empty():
    """Payload without findings key results in empty findings tuple."""
    result = parse_effectiveness_analysis({"disposition": "continue"})
    assert result.findings == ()


def test_missing_rationale_defaults_to_empty():
    """Payload without rationale key results in empty string rationale."""
    result = parse_effectiveness_analysis({"disposition": "revise"})
    assert result.rationale == ""


def test_malformed_findings_dropped_wellformed_kept():
    """Malformed finding entries are dropped; well-formed ones kept."""
    result = parse_effectiveness_analysis({
        "disposition": "pause",
        "findings": [
            {"name": "a", "detail": "b"},
            {"bad": "x"},  # missing name/detail
            "notadict",    # not a mapping
            {"name": "c"},  # missing detail
            {"detail": "d"},  # missing name
            {"name": "e", "detail": "f"},
        ],
    })
    assert result.findings == (
        {"name": "a", "detail": "b"},
        {"name": "e", "detail": "f"},
    )


def test_finding_requires_both_name_and_detail():
    """Finding must have both name and detail as strings."""
    result = parse_effectiveness_analysis({
        "disposition": "split",
        "findings": [
            {"name": "x", "detail": "y"},
            {"name": 123, "detail": "y"},  # name not string
            {"name": "x", "detail": 456},  # detail not string
        ],
    })
    assert result.findings == ({"name": "x", "detail": "y"},)


def test_invalid_disposition_raises():
    """Invalid disposition raises ValueError with descriptive message."""
    with pytest.raises(ValueError, match=r"must be one of.*proceed"):
        parse_effectiveness_analysis({"disposition": "proceed"})


def test_invalid_disposition_in_error_message():
    """Error message includes the offending disposition value."""
    with pytest.raises(ValueError) as exc_info:
        parse_effectiveness_analysis({"disposition": "invalid"})
    assert "invalid" in str(exc_info.value)


def test_non_mapping_list_raises():
    """List payload raises ValueError."""
    with pytest.raises(ValueError, match=r"must be an object.*list"):
        parse_effectiveness_analysis(["not", "a", "dict"])


def test_non_mapping_string_raises():
    """String payload raises ValueError."""
    with pytest.raises(ValueError, match=r"must be an object.*str"):
        parse_effectiveness_analysis("not a dict")


def test_non_mapping_none_raises():
    """None payload raises ValueError."""
    with pytest.raises(ValueError, match=r"must be an object.*NoneType"):
        parse_effectiveness_analysis(None)


def test_non_mapping_int_raises():
    """Integer payload raises ValueError."""
    with pytest.raises(ValueError, match=r"must be an object.*int"):
        parse_effectiveness_analysis(42)


def test_valid_disposition_continue():
    """Disposition 'continue' is accepted."""
    result = parse_effectiveness_analysis({"disposition": "continue"})
    assert result.disposition == "continue"


def test_valid_disposition_revise():
    """Disposition 'revise' is accepted."""
    result = parse_effectiveness_analysis({"disposition": "revise"})
    assert result.disposition == "revise"


def test_valid_disposition_split():
    """Disposition 'split' is accepted."""
    result = parse_effectiveness_analysis({"disposition": "split"})
    assert result.disposition == "split"


def test_valid_disposition_pause():
    """Disposition 'pause' is accepted."""
    result = parse_effectiveness_analysis({"disposition": "pause"})
    assert result.disposition == "pause"


def test_all_valid_dispositions_accepted():
    """All four valid dispositions are accepted."""
    for disposition in VALID_DISPOSITIONS:
        result = parse_effectiveness_analysis({"disposition": disposition})
        assert result.disposition == disposition


def test_returns_effectiveness_analysis_dataclass():
    """Result is an EffectivenessAnalysis dataclass."""
    result = parse_effectiveness_analysis({
        "disposition": "continue",
        "findings": [{"name": "test", "detail": "data"}],
        "rationale": "test rationale",
    })
    assert isinstance(result, EffectivenessAnalysis)
    assert result.disposition == "continue"
    assert len(result.findings) == 1
    assert result.rationale == "test rationale"


def test_findings_is_tuple():
    """Findings are returned as a tuple, not a list."""
    result = parse_effectiveness_analysis({
        "disposition": "continue",
        "findings": [{"name": "a", "detail": "b"}],
    })
    assert isinstance(result.findings, tuple)


def test_empty_findings_list():
    """Empty findings list results in empty tuple."""
    result = parse_effectiveness_analysis({
        "disposition": "continue",
        "findings": [],
    })
    assert result.findings == ()


def test_non_string_rationale_defaults_to_empty():
    """Non-string rationale defaults to empty string."""
    result = parse_effectiveness_analysis({
        "disposition": "continue",
        "rationale": 123,
    })
    assert result.rationale == ""


def test_non_string_rationale_list_defaults_to_empty():
    """List rationale defaults to empty string."""
    result = parse_effectiveness_analysis({
        "disposition": "continue",
        "rationale": ["not", "string"],
    })
    assert result.rationale == ""


def test_non_list_findings_ignored():
    """Non-list findings are ignored (defaults to empty)."""
    result = parse_effectiveness_analysis({
        "disposition": "continue",
        "findings": "not a list",
    })
    assert result.findings == ()


def test_non_list_findings_dict_ignored():
    """Dict findings are ignored (defaults to empty)."""
    result = parse_effectiveness_analysis({
        "disposition": "continue",
        "findings": {"name": "a", "detail": "b"},
    })
    assert result.findings == ()


def test_disposition_not_string_raises():
    """Non-string disposition raises ValueError."""
    with pytest.raises(ValueError, match=r"disposition must be a string"):
        parse_effectiveness_analysis({"disposition": 123})


def test_disposition_list_raises():
    """List disposition raises ValueError."""
    with pytest.raises(ValueError, match=r"disposition must be a string"):
        parse_effectiveness_analysis({"disposition": ["continue"]})


def test_full_valid_payload():
    """Complete valid payload with all fields."""
    result = parse_effectiveness_analysis({
        "disposition": "revise",
        "findings": [
            {"name": "finding1", "detail": "detail1"},
            {"name": "finding2", "detail": "detail2"},
        ],
        "rationale": "Goal needs revision based on findings.",
    })
    assert result.disposition == "revise"
    assert len(result.findings) == 2
    assert result.findings[0] == {"name": "finding1", "detail": "detail1"}
    assert result.findings[1] == {"name": "finding2", "detail": "detail2"}
    assert result.rationale == "Goal needs revision based on findings."


def test_minimal_valid_payload():
    """Minimal valid payload with only required disposition."""
    result = parse_effectiveness_analysis({"disposition": "continue"})
    assert result.disposition == "continue"
    assert result.findings == ()
    assert result.rationale == ""
