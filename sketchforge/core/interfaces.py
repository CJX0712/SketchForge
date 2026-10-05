"""Behavioral protocol and budget-honesty checks for sketches.

Author: 晨星
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Protocol, Self, runtime_checkable

import numpy as np

from sketchforge.core.errors import IndistinguishableBudgetError
from sketchforge.core.types import Estimate, Query


@runtime_checkable
class Sketch(Protocol):
    """Runtime-checkable contract shared by all sketch backends."""

    SUPPORTS_MERGE: bool
    DETERMINISTIC: bool
    HASH_SEED_INJECTABLE: bool
    TIER: int
    PARAM_AXIS: str

    def update(self, item: object) -> None: ...

    def update_batch(self, items: np.ndarray) -> None: ...

    def estimate(self, query: Query) -> Estimate: ...

    def merge(self, other: Self) -> Self: ...

    def memory_slots(self) -> int: ...

    def memory_bytes(self) -> int: ...

    def serialize(self) -> bytes: ...

    @classmethod
    def params_for_budget(cls, budget_bytes: int) -> dict[str, object]: ...


def assert_budget_honest(sketch: Sketch) -> None:
    """Require declared memory to agree with serialized size within two percent."""
    declared = sketch.memory_bytes()
    if isinstance(declared, bool) or not isinstance(declared, Real):
        raise IndistinguishableBudgetError(
            "memory_bytes() must return a real number", declared=declared
        )
    declared_value = float(declared)
    if not math.isfinite(declared_value) or declared_value <= 0:
        raise IndistinguishableBudgetError(
            "memory_bytes() must be finite and positive", declared=declared
        )
    payload = sketch.serialize()
    if not isinstance(payload, bytes) or not payload:
        raise IndistinguishableBudgetError(
            "serialize() must return non-empty bytes", payload_type=type(payload).__name__
        )
    actual = len(payload)
    relative_error = abs(declared_value - actual) / actual
    if relative_error > 0.02:
        raise IndistinguishableBudgetError(
            "declared memory differs from serialized size by more than 2%",
            declared=declared_value,
            serialized=actual,
            relative_error=relative_error,
        )
