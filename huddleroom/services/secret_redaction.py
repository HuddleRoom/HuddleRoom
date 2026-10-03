"""Utilities for redacting secrets from text (API keys, tokens, passwords, etc.)"""

import re

_JSON_SECRET_PATTERN = re.compile(
    r'(?i)(["\'](?:api[_-]?key|authorization|bearer|token|secret|password)["\']\s*:\s*["\'])([^"\']+)(["\'])'
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization)\s*[:=]\s*(bearer\s+)?([^\s,;]+)"),
    re.compile(r"(?i)(bearer)\s+([^\s,;]+)"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*([^\s,;]+)"),
    re.compile(r"(?i)(api[_-]?key|authorization|bearer|token|secret|password)=([^&\s]+)"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\brk-(?:live|test)-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bxai-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
)


def redact_secrets(text: str) -> str:
    """Redact API keys, tokens, passwords, and other secrets from text.

    Matches common patterns like:
    - JSON key-value pairs (api_key, authorization, bearer, token, secret, password)
    - Environment variable-style assignments (key=value)
    - Bearer tokens (authorization: bearer ...)
    - Provider-specific token formats (sk-*, rk-*, xai-*, AIza...)

    Args:
        text: Text that may contain secrets

    Returns:
        Text with secrets replaced with [REDACTED]
    """
    text = _JSON_SECRET_PATTERN.sub(r"\1[REDACTED]\3", text)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text
