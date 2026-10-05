"""Exact full-stream oracle used as the evaluation gold standard.

Author: 晨星
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from sketchforge.core.errors import EmptyStreamError, InvalidSpecError
from sketchforge.data.streams import Stream


class ExactOracle:
    """Own read-only copies of a stream and answer queries exactly."""

    def __init__(self, stream: Stream) -> None:
        if len(stream.keys) == 0:
            raise EmptyStreamError("exact oracle requires a non-empty stream")
        self._keys = np.array(stream.keys, dtype=np.int64, copy=True)
        self._values = np.array(stream.values, dtype=np.float64, copy=True)
        self._keys.setflags(write=False)
        self._values.setflags(write=False)
        self._sorted_values = np.sort(self._values)
        self._sorted_values.setflags(write=False)
        unique, counts = np.unique(self._keys, return_counts=True)
        self._frequencies = Counter(
            {int(key): int(count) for key, count in zip(unique, counts, strict=True)}
        )

    def cardinality(self) -> int:
        """Return the exact number of distinct keys."""
        return int(np.unique(self._keys).size)

    def quantile(self, q: float) -> float:
        """Return the exact linearly interpolated full-stream quantile."""
        if not 0.0 <= q <= 1.0:
            raise InvalidSpecError("quantile must be between zero and one", q=q)
        return float(np.quantile(self._sorted_values, q))

    def rank(self, value: float) -> float:
        """Return the exact inclusive normalized rank of a numeric value."""
        position = np.searchsorted(self._sorted_values, value, side="right")
        return float(position / len(self._sorted_values))

    def frequency(self, key: object) -> int:
        """Return the exact occurrence count for a key."""
        return self._frequencies.get(int(key), 0)

    def top_k(self, k: int) -> list[tuple[int, int]]:
        """Return exact heavy hitters ordered by count then key."""
        if k <= 0:
            raise InvalidSpecError("top-k size must be positive", k=k)
        ordered = sorted(self._frequencies.items(), key=lambda pair: (-pair[1], pair[0]))
        return ordered[:k]

    def total(self) -> int:
        """Return the exact number of updates."""
        return len(self._keys)
