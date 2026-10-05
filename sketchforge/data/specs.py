"""Declarative synthetic stream specifications.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from sketchforge.core.errors import InvalidSpecError

StreamKind = Literal[
    "uniform",
    "zipf",
    "pareto",
    "gmm",
    "drift_abrupt",
    "drift_gradual",
    "low_card",
    "high_card",
    "elephant_mice",
]
_VALID_KINDS = frozenset(
    {
        "uniform",
        "zipf",
        "pareto",
        "gmm",
        "drift_abrupt",
        "drift_gradual",
        "low_card",
        "high_card",
        "elephant_mice",
    }
)


@dataclass(frozen=True)
class StreamSpec:
    """Complete, serializable description of a synthetic stream."""

    kind: StreamKind
    n: int
    domain: int
    seed: int
    zipf_alpha: float = 1.1
    drift_rate: float = 0.0
    value_dist: str = "normal"

    def __post_init__(self) -> None:
        if self.kind not in _VALID_KINDS:
            raise InvalidSpecError("unsupported stream kind", kind=self.kind)
        if self.n <= 0:
            raise InvalidSpecError("stream length must be positive", n=self.n)
        if self.domain <= 0:
            raise InvalidSpecError("stream domain must be positive", domain=self.domain)
        if self.zipf_alpha <= 0:
            raise InvalidSpecError("zipf_alpha must be positive", zipf_alpha=self.zipf_alpha)
        if not 0.0 <= self.drift_rate <= 1.0:
            raise InvalidSpecError(
                "drift_rate must be between zero and one", drift_rate=self.drift_rate
            )
        if not self.value_dist:
            raise InvalidSpecError("value_dist must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)
