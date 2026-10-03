from __future__ import annotations

from typing import Any


class GuardEvaluator:
    def evaluate(self, guard: dict | None, payload: dict) -> bool:
        if not guard:
            return True
        for field, condition in guard.items():
            actual = self._get_field(payload, field)
            if isinstance(condition, dict) and "operator" in condition:
                if not self._apply_operator(actual, condition["operator"], condition.get("value")):
                    return False
            elif not self._str_equal(actual, condition):
                return False
        return True

    def _str_equal(self, actual: object, condition: object) -> bool:
        if isinstance(condition, bool) or isinstance(actual, bool):
            return str(actual).lower() == str(condition).lower()
        return str(actual) == str(condition)

    def _get_field(self, payload: dict, dotpath: str) -> Any:
        current: Any = payload
        for part in dotpath.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    def _apply_operator(self, actual: Any, operator: str, value: Any) -> bool:
        if operator == "eq":
            return str(actual) == str(value)
        if operator == "neq":
            return str(actual) != str(value)
        if operator == "gte":
            try:
                return float(actual or 0) >= float(value)
            except (TypeError, ValueError):
                return False
        if operator == "lte":
            try:
                return float(actual or 0) <= float(value)
            except (TypeError, ValueError):
                return False
        if operator == "contains":
            if isinstance(actual, list):
                return value in actual
            return str(value) in str(actual or "")
        if operator == "not_contains":
            if isinstance(actual, list):
                return value not in actual
            return str(value) not in str(actual or "")
        return False
