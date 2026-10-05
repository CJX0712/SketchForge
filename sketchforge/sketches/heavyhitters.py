"""Heavy-hitter (top-k frequency) sketches for Q4.

Tier-1 deterministic references: ``SpaceSaving``, ``MisraGries``, ``LossyCounting``,
``CmPlusHeap``, ``ExactCountTruncate``. Tier-0 reference: ``FrequentStringsSketch``.

Author: 晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sketchforge.core.errors import EstimateUnavailableError
from sketchforge.core.hashing import hash_index, stable_key
from sketchforge.core.types import Estimate, Query, QueryType
from sketchforge.sketches.backend import FrequentStringsSketch  # noqa: F401 - register T0 ref
from sketchforge.sketches.base import AbstractSketch
from sketchforge.sketches.registry import register


def _pack_counts(lookup: dict[int, int]) -> tuple[dict, dict]:
    keys = np.array(sorted(lookup.keys()), dtype=np.int64)
    vals = np.array([lookup[k] for k in keys.tolist()], dtype=np.int64)
    return {"_k": keys, "_v": vals}, {}


def _unpack_counts(arrays: dict, scalars: dict) -> dict[int, int]:
    return {int(k): int(v) for k, v in zip(arrays["_k"].tolist(), arrays["_v"].tolist(), strict=True)}


@register
class SpaceSaving(AbstractSketch):
    """Space-Saving: keep ``width`` counters, evict the lightest on overflow."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.HEAVY_HITTERS,)
    PARAM_AXIS = "width_power"
    PARAM_RANGE = (4, 20)

    def __init__(self, *, width_power: int = 12, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._width_power = int(width_power)
        self._width = 1 << self._width_power
        self._counts: dict[int, int] = {}

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"width_power": int(value)}

    def update(self, item: object) -> None:
        key = int(item)
        if key in self._counts:
            self._counts[key] += 1
            return
        if len(self._counts) < self._width:
            self._counts[key] = 1
            return
        min_key = min(self._counts, key=lambda k: self._counts[k])
        del self._counts[min_key]
        self._counts[key] = 1

    def update_batch(self, items: np.ndarray) -> None:
        for item in np.asarray(items):
            self.update(item)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.HEAVY_HITTERS:
            raise EstimateUnavailableError("SpaceSaving answers heavy hitters only")
        top_k = query.top_k or 10
        ordered = sorted(self._counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return Estimate(value=[(int(k), int(v)) for k, v in ordered], n_updates=0)

    def memory_slots(self) -> int:
        return self._width

    def _state(self) -> tuple[dict, dict]:
        return _pack_counts(self._counts)

    def _restore(self, arrays: dict, scalars: dict) -> None:
        self._counts = _unpack_counts(arrays, scalars)


@register
class MisraGries(AbstractSketch):
    """Misra-Gries summary: decrement-all on overflow."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.HEAVY_HITTERS,)
    PARAM_AXIS = "width_power"
    PARAM_RANGE = (4, 20)

    def __init__(self, *, width_power: int = 12, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._width_power = int(width_power)
        self._width = (1 << self._width_power) - 1
        self._counts: dict[int, int] = {}

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"width_power": int(value)}

    def update(self, item: object) -> None:
        key = int(item)
        if key in self._counts:
            self._counts[key] += 1
        elif len(self._counts) < self._width:
            self._counts[key] = 1
        else:
            for k in list(self._counts.keys()):
                self._counts[k] -= 1
                if self._counts[k] <= 0:
                    del self._counts[k]

    def update_batch(self, items: np.ndarray) -> None:
        for item in np.asarray(items):
            self.update(item)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.HEAVY_HITTERS:
            raise EstimateUnavailableError("MisraGries answers heavy hitters only")
        top_k = query.top_k or 10
        ordered = sorted(self._counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return Estimate(value=[(int(k), int(v)) for k, v in ordered], n_updates=0)

    def memory_slots(self) -> int:
        return self._width

    def _state(self) -> tuple[dict, dict]:
        return _pack_counts(self._counts)

    def _restore(self, arrays: dict, scalars: dict) -> None:
        self._counts = _unpack_counts(arrays, scalars)


@register
class LossyCounting(AbstractSketch):
    """Lossy counting: bucketed summaries with error-tolerant eviction."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.HEAVY_HITTERS,)
    PARAM_AXIS = "width_power"
    PARAM_RANGE = (6, 20)

    def __init__(self, *, width_power: int = 12, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._width_power = int(width_power)
        self._bucket = max(1, 1 << self._width_power)
        self._n = 0
        self._counts: dict[int, tuple[int, int]] = {}

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"width_power": int(value)}

    def update(self, item: object) -> None:
        key = int(item)
        self._n += 1
        bucket = self._n // self._bucket
        entry = self._counts.get(key)
        if entry is not None:
            f, delta = entry
            self._counts[key] = (f + 1, delta)
        elif len(self._counts) < self._bucket:
            self._counts[key] = (1, bucket)
        else:
            for k in list(self._counts.keys()):
                f, delta = self._counts[k]
                if f <= delta:
                    del self._counts[k]

    def update_batch(self, items: np.ndarray) -> None:
        for item in np.asarray(items):
            self.update(item)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.HEAVY_HITTERS:
            raise EstimateUnavailableError("LossyCounting answers heavy hitters only")
        top_k = query.top_k or 10
        # (key, frequency estimate)
        scored = [(k, f) for k, (f, _d) in self._counts.items()]
        ordered = sorted(scored, key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return Estimate(value=[(int(k), int(v)) for k, v in ordered], n_updates=0)

    def memory_slots(self) -> int:
        return self._bucket

    def _state(self) -> tuple[dict, dict]:
        keys = np.array(sorted(self._counts.keys()), dtype=np.int64)
        f = np.array([self._counts[k][0] for k in keys.tolist()], dtype=np.int64)
        d = np.array([self._counts[k][1] for k in keys.tolist()], dtype=np.int64)
        return {"_k": keys, "_f": f, "_d": d}, {"_bucket": self._bucket, "_n": self._n}

    def _restore(self, arrays: dict, scalars: dict) -> None:
        self._bucket = int(scalars["_bucket"])
        self._n = int(scalars["_n"])
        self._counts = {
            int(k): (int(f), int(d))
            for k, f, d in zip(arrays["_k"].tolist(), arrays["_f"].tolist(), arrays["_d"].tolist(), strict=True)
        }


@register
class CmPlusHeap(AbstractSketch):
    """Count-Min backing store + bounded exact tracking of heavy candidates."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = True
    QUERY_TYPES = (QueryType.HEAVY_HITTERS,)
    PARAM_AXIS = "width_power"
    PARAM_RANGE = (6, 20)

    def __init__(self, *, width_power: int = 12, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._width_power = int(width_power)
        self._width = 1 << self._width_power
        self._row_seeds = [
            self._seed_state.derive(f"cmh:{r}") for r in range(4)
        ]
        self._counts = np.zeros((4, self._width), dtype=np.int64)
        self._exact: dict[int, int] = {}
        self._exact_cap = self._width

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"width_power": int(value)}

    def _cm_idx(self, key: bytes) -> list[int]:
        return [hash_index(key, self._row_seeds[r], self._width) for r in range(4)]

    def update(self, item: object) -> None:
        key = int(item)
        kb = stable_key(key)
        idxs = self._cm_idx(kb)
        for r in range(4):
            self._counts[r, idxs[r]] += 1
        if key in self._exact:
            self._exact[key] += 1
        elif len(self._exact) < self._exact_cap:
            self._exact[key] = 1
        else:
            est = min(int(self._counts[r, idxs[r]]) for r in range(4))
            min_key = min(self._exact, key=lambda k: self._exact[k])
            if self._exact[min_key] < est:
                del self._exact[min_key]
                self._exact[key] = 1

    def update_batch(self, items: np.ndarray) -> None:
        for item in np.asarray(items):
            self.update(item)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.HEAVY_HITTERS:
            raise EstimateUnavailableError("CmPlusHeap answers heavy hitters only")
        top_k = query.top_k or 10
        ordered = sorted(self._exact.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return Estimate(value=[(int(k), int(v)) for k, v in ordered], n_updates=0)

    def memory_slots(self) -> int:
        return self._width

    def _state(self) -> tuple[dict, dict]:
        arrays, _ = _pack_counts(self._exact)
        arrays["_counts"] = self._counts
        return arrays, {"_width_power": self._width_power}

    def _restore(self, arrays: dict, scalars: dict) -> None:
        self._width_power = int(scalars["_width_power"])
        self._width = 1 << self._width_power
        self._counts = arrays["_counts"]
        self._exact = _unpack_counts(arrays, scalars)
        self._exact_cap = self._width


@register
class ExactCountTruncate(AbstractSketch):
    """Exact top-k counts up to a capacity (deterministic reference oracle)."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.HEAVY_HITTERS,)
    PARAM_AXIS = "capacity"
    PARAM_RANGE = (1, 500_000)

    def __init__(self, *, capacity: int = 4096, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._capacity = int(capacity)
        self._counts: dict[int, int] = {}

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"capacity": int(value)}

    def update(self, item: object) -> None:
        key = int(item)
        if key in self._counts or len(self._counts) < self._capacity:
            self._counts[key] = self._counts.get(key, 0) + 1

    def update_batch(self, items: np.ndarray) -> None:
        for item in np.asarray(items):
            self.update(item)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.HEAVY_HITTERS:
            raise EstimateUnavailableError("ExactCountTruncate answers heavy hitters only")
        top_k = query.top_k or 10
        ordered = sorted(self._counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return Estimate(value=[(int(k), int(v)) for k, v in ordered], n_updates=0)

    def memory_slots(self) -> int:
        return self._capacity

    def _state(self) -> tuple[dict, dict]:
        return _pack_counts(self._counts)

    def _restore(self, arrays: dict, scalars: dict) -> None:
        self._counts = _unpack_counts(arrays, scalars)
