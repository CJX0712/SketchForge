"""Point-query frequency sketches for Q3.

``CountMinNumpy`` implements the classic over-estimating Count-Min invariant
(I-1: estimate >= true count, zero tolerance). ``CuSketchNumpy`` applies
conservative update to shrink the over-estimate, ``CountSketchNumpy`` uses signed
projections, and ``ExactHashTruncate`` is the deterministic exact reference.

All integer-key hashing routes through :func:`hash_array_u64` so the single-key
``estimate`` path and the batched ``update`` path are byte-for-byte consistent.

Author: 晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sketchforge.core.errors import EstimateUnavailableError
from sketchforge.core.hashing import hash_array_u64
from sketchforge.core.types import Estimate, Query, QueryType
from sketchforge.sketches.backend import CountMinSketch  # noqa: F401 - register T0 ref
from sketchforge.sketches.base import AbstractSketch
from sketchforge.sketches.registry import register

DEPTH_CM = 4
DEPTH_CS = 5


def _row_seeds(seed_state, depth: int):
    return [seed_state.derive(f"row:{r}") for r in range(depth)]


@register
class CountMinNumpy(AbstractSketch):
    """Classic Count-Min sketch (guaranteed over-estimate of point frequency)."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = True
    HASH_SEED_INJECTABLE = True
    QUERY_TYPES = (QueryType.FREQUENCY,)
    PARAM_AXIS = "width_power"
    PARAM_RANGE = (5, 20)

    def __init__(self, *, width_power: int = 12, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._width_power = int(width_power)
        self._width = 1 << self._width_power
        self._counts = np.zeros((DEPTH_CM, self._width), dtype=np.int64)
        self._row_seeds = _row_seeds(self._seed_state, DEPTH_CM)

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"width_power": int(value)}

    def _fold(self, items: np.ndarray) -> None:
        arr = np.asarray(items, dtype=np.int64)
        for r in range(DEPTH_CM):
            idx = (hash_array_u64(arr, self._row_seeds[r]) % np.uint64(self._width)).astype(np.int64)
            np.add.at(self._counts, (r, idx), 1)

    def update(self, item: object) -> None:
        self._fold(np.array([int(item)], dtype=np.int64))

    def update_batch(self, items: np.ndarray) -> None:
        self._fold(items)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.FREQUENCY:
            raise EstimateUnavailableError("CountMinNumpy answers frequency only")
        key = np.array([int(query.key)], dtype=np.int64)
        best = np.inf
        for r in range(DEPTH_CM):
            idx = int((hash_array_u64(key, self._row_seeds[r]) % np.uint64(self._width))[0])
            best = min(best, int(self._counts[r, idx]))
        return Estimate(value=float(best), n_updates=0)

    def memory_slots(self) -> int:
        return DEPTH_CM * self._width

    def _merge_same_type(self, other: AbstractSketch) -> None:
        np.add(self._counts, other._counts, out=self._counts)


@register
class CuSketchNumpy(AbstractSketch):
    """Count-Min with conservative update (smaller over-estimate than plain CM)."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = True
    HASH_SEED_INJECTABLE = True
    QUERY_TYPES = (QueryType.FREQUENCY,)
    PARAM_AXIS = "width_power"
    PARAM_RANGE = (5, 20)

    def __init__(self, *, width_power: int = 12, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._width_power = int(width_power)
        self._width = 1 << self._width_power
        self._counts = np.zeros((DEPTH_CM, self._width), dtype=np.int64)
        self._row_seeds = _row_seeds(self._seed_state, DEPTH_CM)

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"width_power": int(value)}

    def _fold(self, items: np.ndarray) -> None:
        arr = np.asarray(items, dtype=np.int64)
        for r in range(DEPTH_CM):
            idx = (hash_array_u64(arr, self._row_seeds[r]) % np.uint64(self._width)).astype(np.int64)
            target = 0
            for _j, i in enumerate(idx):
                current = int(self._counts[r, i])
                target = max(target, current + 1)
                if current < target:
                    self._counts[r, i] = target

    def update(self, item: object) -> None:
        self._fold(np.array([int(item)], dtype=np.int64))

    def update_batch(self, items: np.ndarray) -> None:
        arr = np.asarray(items, dtype=np.int64)
        for item in arr:
            self._fold(np.array([int(item)], dtype=np.int64))

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.FREQUENCY:
            raise EstimateUnavailableError("CuSketchNumpy answers frequency only")
        key = np.array([int(query.key)], dtype=np.int64)
        best = np.inf
        for r in range(DEPTH_CM):
            idx = int((hash_array_u64(key, self._row_seeds[r]) % np.uint64(self._width))[0])
            best = min(best, int(self._counts[r, idx]))
        return Estimate(value=float(best), n_updates=0)

    def memory_slots(self) -> int:
        return DEPTH_CM * self._width

    def _merge_same_type(self, other: AbstractSketch) -> None:
        np.add(self._counts, other._counts, out=self._counts)


@register
class CountSketchNumpy(AbstractSketch):
    """Count-Sketch: signed projections, median aggregation (unbiased)."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = True
    HASH_SEED_INJECTABLE = True
    QUERY_TYPES = (QueryType.FREQUENCY,)
    PARAM_AXIS = "width_power"
    PARAM_RANGE = (5, 20)

    def __init__(self, *, width_power: int = 12, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._width_power = int(width_power)
        self._width = 1 << self._width_power
        self._counts = np.zeros((DEPTH_CS, self._width), dtype=np.int64)
        self._row_seeds = _row_seeds(self._seed_state, DEPTH_CS)

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"width_power": int(value)}

    def _fold(self, items: np.ndarray) -> None:
        arr = np.asarray(items, dtype=np.int64)
        for r in range(DEPTH_CS):
            h = hash_array_u64(arr, self._row_seeds[r])
            idx = (h % np.uint64(self._width)).astype(np.int64)
            signs = np.where((h & np.uint64(1)) == np.uint64(0), 1, -1).astype(np.int64)
            np.add.at(self._counts, (r, idx), signs)

    def update(self, item: object) -> None:
        self._fold(np.array([int(item)], dtype=np.int64))

    def update_batch(self, items: np.ndarray) -> None:
        self._fold(items)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.FREQUENCY:
            raise EstimateUnavailableError("CountSketchNumpy answers frequency only")
        key = np.array([int(query.key)], dtype=np.int64)
        vals = np.empty(DEPTH_CS, dtype=np.int64)
        for r in range(DEPTH_CS):
            h = hash_array_u64(key, self._row_seeds[r])
            idx = int((h % np.uint64(self._width))[0])
            sign = 1 if int(h[0] & np.uint64(1)) == 0 else -1
            vals[r] = int(self._counts[r, idx]) * sign
        return Estimate(value=float(np.median(vals)), n_updates=0)

    def memory_slots(self) -> int:
        return DEPTH_CS * self._width

    def _merge_same_type(self, other: AbstractSketch) -> None:
        np.add(self._counts, other._counts, out=self._counts)


@register
class ExactHashTruncate(AbstractSketch):
    """Exact frequency counts up to a capacity (deterministic reference oracle)."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.FREQUENCY,)
    PARAM_AXIS = "capacity"
    PARAM_RANGE = (1, 500_000)

    def __init__(self, *, capacity: int = 4096, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._capacity = int(capacity)
        self._lookup: dict[int, int] = {}
        self._total = 0

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"capacity": int(value)}

    def update(self, item: object) -> None:
        key = int(item)
        self._total += 1
        if key in self._lookup or len(self._lookup) < self._capacity:
            self._lookup[key] = self._lookup.get(key, 0) + 1

    def update_batch(self, items: np.ndarray) -> None:
        for item in np.asarray(items):
            self.update(item)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.FREQUENCY:
            raise EstimateUnavailableError("ExactHashTruncate answers frequency only")
        return Estimate(value=float(self._lookup.get(int(query.key), 0)), n_updates=self._total)

    def memory_slots(self) -> int:
        return self._capacity

    def _state(self) -> tuple[dict, dict]:
        keys = np.array(list(self._lookup.keys()), dtype=np.int64)
        vals = np.array(list(self._lookup.values()), dtype=np.int64)
        return {"_k": keys, "_v": vals}, {"_capacity": self._capacity, "_total": self._total}

    def _restore(self, arrays: dict, scalars: dict) -> None:
        self._capacity = int(scalars["_capacity"])
        self._total = int(scalars["_total"])
        self._lookup = {int(k): int(v) for k, v in zip(arrays["_k"].tolist(), arrays["_v"].tolist(), strict=True)}
