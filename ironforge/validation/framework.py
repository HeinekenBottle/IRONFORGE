"""Lightweight validation framework primitives.

Defines a consistent result shape and helpers to compose multiple validators.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Protocol


@dataclass
class ValidationCheck:
    name: str
    status: str  # "pass" | "fail" | "warning"
    message: str | None = None
    data: dict[str, Any] | None = None


@dataclass
class ValidationResult:
    status: str
    checks: dict[str, ValidationCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": {k: asdict(v) for k, v in self.checks.items()},
        }


class Validator(Protocol):
    def __call__(self, *args, **kwargs) -> ValidationResult | dict[str, Any]:
        ...


def combine_results(results: dict[str, ValidationResult]) -> dict[str, Any]:
    overall = "pass" if all(r.status == "pass" for r in results.values()) else "fail"
    return {
        "status": overall,
        "validations": {k: v.to_dict() for k, v in results.items()},
    }

