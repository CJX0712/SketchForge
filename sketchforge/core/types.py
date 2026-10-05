"""Shared immutable value objects for SketchForge.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sketchforge.core.errors import InvalidConfigError


class QueryType(str, Enum):  # noqa: UP042 - public contract requires ``str, Enum``
    """Supported stream-query families."""

    CARDINALITY = "cardinality"
    QUANTILE = "quantile"
    FREQUENCY = "frequency"
    HEAVY_HITTERS = "heavy_hitters"


@dataclass(frozen=True)
class Query:
    """A normalized query request."""

    kind: QueryType
    q: float | None = None
    key: object | None = None
    top_k: int | None = None


@dataclass(frozen=True)
class Estimate:
    """An estimate and optional confidence bounds."""

    value: float | list
    lower: float | None = None
    upper: float | None = None
    n_updates: int = 0


@dataclass(frozen=True)
class MetricSpec:
    """Metadata and optimization direction for an evaluation metric."""

    name: str
    unit: str
    higher_is_better: bool

    def score(self, value: float) -> float:
        """Map a metric value to a common higher-is-better score."""
        return value if self.higher_is_better else -value


@dataclass(frozen=True)
class MethodSpec:
    """Static capabilities advertised by a sketch implementation."""

    name: str
    tier: int
    backend: str
    query_types: tuple[QueryType, ...]
    supports_merge: bool
    deterministic: bool
    hash_seed_injectable: bool


@dataclass(frozen=True)
class BudgetPlan:
    """Byte allocation across query families."""

    alloc: dict[str, int]
    total_bytes: int
    strategy: str

    def __post_init__(self) -> None:
        if self.total_bytes <= 0:
            raise InvalidConfigError("total_bytes must be positive", total_bytes=self.total_bytes)
        if any(value < 0 for value in self.alloc.values()):
            raise InvalidConfigError("budget allocations must be non-negative", alloc=self.alloc)
        allocated = sum(self.alloc.values())
        if allocated > self.total_bytes:
            raise InvalidConfigError(
                "budget allocations exceed total_bytes",
                allocated=allocated,
                total_bytes=self.total_bytes,
            )

    def get(self, query_type: QueryType | str) -> int:
        """Return allocated bytes for a query family, or zero when absent."""
        key = query_type.value if isinstance(query_type, QueryType) else query_type
        return self.alloc.get(key, 0)
