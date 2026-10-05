"""Core guard tests beyond the focused module suites.

Author: 晨星
"""

from dataclasses import dataclass

import numpy as np
import pytest

from sketchforge.core.errors import IndistinguishableBudgetError, InvalidConfigError
from sketchforge.core.interfaces import assert_budget_honest
from sketchforge.core.types import BudgetPlan, Estimate, Query, QueryType


@dataclass
class DummySketch:
    size: int

    SUPPORTS_MERGE = True
    DETERMINISTIC = True
    HASH_SEED_INJECTABLE = True
    TIER = 1
    PARAM_AXIS = "size"

    def update(self, item: object) -> None:
        del item

    def update_batch(self, items: np.ndarray) -> None:
        del items

    def estimate(self, query: Query) -> Estimate:
        del query
        return Estimate(0.0)

    def merge(self, other: "DummySketch") -> "DummySketch":
        return DummySketch(self.size + other.size)

    def memory_slots(self) -> int:
        return self.size

    def memory_bytes(self) -> int:
        return self.size

    def serialize(self) -> bytes:
        return bytes(self.size)

    @classmethod
    def params_for_budget(cls, budget_bytes: int) -> dict[str, object]:
        return {"size": budget_bytes}


def test_budget_plan_enforces_conservation_with_e101() -> None:
    plan = BudgetPlan({QueryType.CARDINALITY.value: 100}, total_bytes=128, strategy="fixed")
    assert plan.get(QueryType.CARDINALITY) == 100
    assert plan.get(QueryType.QUANTILE) == 0
    with pytest.raises(InvalidConfigError) as captured:
        BudgetPlan({"a": 65, "b": 64}, total_bytes=128, strategy="invalid")
    assert captured.value.code == "E101"


def test_budget_honesty_accepts_exact_and_rejects_mismatch() -> None:
    assert_budget_honest(DummySketch(100))
    dishonest = DummySketch(100)
    dishonest.serialize = lambda: bytes(50)  # type: ignore[method-assign]
    with pytest.raises(IndistinguishableBudgetError) as captured:
        assert_budget_honest(dishonest)
    assert captured.value.code == "E302"
