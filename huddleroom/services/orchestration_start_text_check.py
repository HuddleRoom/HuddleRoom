"""Detect goal text that tells the orchestrator not to start yet.

Pure string check: no DB, no service imports. Used by Start to warn the owner
when description/notes/constraints contradict the start action.
"""

import re
from collections.abc import Mapping

# ponytail: phrase list covers the known wording; extend the alternation when new phrasing shows up.
_START_CONTRADICTION_RE = re.compile(
    r"\bprepare\s+only\b"
    r"|\b(?:do\s+not|don['’]?t|never)\s+(?:(?:yet|actually|the|this|that)\s+)?(?:start|begin|launch|execute)\b"
    r"|\bhold\s+off\b",
    re.IGNORECASE,
)


def find_start_contradictions(sources: Mapping[str, str | None]) -> list[tuple[str, str]]:
    """Return (source_name, matched_phrase) pairs in source order, deduped per source."""
    found: list[tuple[str, str]] = []
    for name, text in sources.items():
        if not text:
            continue
        seen: set[str] = set()
        for match in _START_CONTRADICTION_RE.finditer(text):
            phrase = match.group(0).strip()
            if phrase in seen:
                continue
            seen.add(phrase)
            found.append((name, phrase))
    return found
