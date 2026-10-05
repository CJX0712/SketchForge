"""Cardinality (distinct-count) sketches for Q1.

Tier-1 implementations (HyperLogLogNumpy, KmvSketch, LinearCountingBitmap,
ExactTruncate) are deterministic and form the reproducible backbone of the main gate.
Tier-0 references wrap ``datasketches`` and are registered for non-inferiority
comparison only.

Author: 晨星
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np

from sketchforge.core.errors import EstimateUnavailableError
from sketchforge.core.hashing import hash_array_u64
from sketchforge.core.types import Estimate, Query, QueryType
from sketchforge.sketches.backend import CpcSketch, HllSketch  # noqa: F401 - register T0 refs
from sketchforge.sketches.base import AbstractSketch, bit_length_u64
from sketchforge.sketches.registry import register


@register
class HyperLogLogNumpy(AbstractSketch):
    """Classic HyperLogLog with deterministic seed-injectable hashing."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = True
    HASH_SEED_INJECTABLE = True
    QUERY_TYPES = (QueryType.CARDINALITY,)
    PARAM_AXIS = "p"
    PARAM_RANGE = (4, 18)

    _ALPHA_SMALL: ClassVar[dict[int, float]] = {16: 0.673, 32: 0.697, 64: 0.709}

    def __init__(self, *, p: int = 12, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._p = int(p)
        self._m = 1 << self._p
        self._registers = np.zeros(self._m, dtype=np.uint8)

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"p": int(value)}

    def _rho_batch(self, items: np.ndarray) -> None:
        hashes = hash_array_u64(np.asarray(items, dtype=np.int64), self._seed_state.derive("hll"))
        mask = np.uint64(self._m - 1)
        idx = (hashes & mask).astype(np.int64)
        w = hashes >> np.uint64(self._p)
        rho = (64 - self._p) - bit_length_u64(w).astype(np.int32) + 1
        np.maximum.at(self._registers, idx, rho)

    def update(self, item: object) -> None:
        self._rho_batch(np.array([int(item)], dtype=np.int64))

    def update_batch(self, items: np.ndarray) -> None:
        self._rho_batch(items)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.CARDINALITY:
            raise EstimateUnavailableError("HyperLogLogNumpy answers cardinality only")
        m = self._m
        alpha = 0.7213 / (1.0 + 1.079 / m) if m >= 128 else self._ALPHA_SMALL.get(m, 0.715)
        with np.errstate(divide="ignore"):
            sum_inv = float(np.sum(np.power(2.0, -self._registers.astype(np.float64))))
        if sum_inv <= 0:
            return Estimate(value=0.0, n_updates=0)
        estimate = alpha * m * m / sum_inv
        zeros = int(np.count_nonzero(self._registers == 0))
        if estimate <= 0.0:
            return Estimate(value=0.0, n_updates=0)
        if estimate < 2.5 * m and zeros > 0:
            estimate = m * math.log(m / zeros)
        return Estimate(value=float(estimate), n_updates=0)

    def memory_slots(self) -> int:
        return self._m

    def _merge_same_type(self, other: AbstractSketch) -> None:
        np.maximum(self._registers, other._registers, out=self._registers)


@register
class KmvSketch(AbstractSketch):
    """K-minimum-values distinct counter (shared bottom-k core primitive)."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = True
    HASH_SEED_INJECTABLE = True
    QUERY_TYPES = (QueryType.CARDINALITY,)
    PARAM_AXIS = "k"
    PARAM_RANGE = (16, 8192)

    _MAX = np.uint64((1 << 64) - 1)

    def __init__(self, *, k: int = 1024, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._k = int(k)
        self._heap = np.empty(0, dtype=np.uint64)
        self._seen = 0

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"k": int(value)}

    def _fold(self, items: np.ndarray) -> None:
        hashes = hash_array_u64(np.asarray(items, dtype=np.int64), self._seed_state.derive("kmv"))
        combined = np.unique(np.concatenate((self._heap, hashes)))
        combined.sort()
        self._heap = combined[: self._k]
        self._seen += len(hashes)

    def update(self, item: object) -> None:
        self._fold(np.array([int(item)], dtype=np.int64))

    def update_batch(self, items: np.ndarray) -> None:
        self._fold(items)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.CARDINALITY:
            raise EstimateUnavailableError("KmvSketch answers cardinality only")
        full = int(self._heap.size)
        if full == 0:
            return Estimate(value=0.0, n_updates=self._seen)
        if full < self._k:
            return Estimate(value=float(full), n_updates=self._seen)
        kth = int(self._heap[self._k - 1]) + 1
        estimate = (self._k - 1) * (1 << 64) / kth
        return Estimate(value=float(estimate), n_updates=self._seen)

    def memory_slots(self) -> int:
        return self._k

    def _merge_same_type(self, other: AbstractSketch) -> None:
        combined = np.concatenate((self._heap, other._heap))
        combined.sort()
        self._heap = combined[: self._k]


@register
class LinearCountingBitmap(AbstractSketch):
    """Linear-counting bitmap distinct estimator."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = True
    HASH_SEED_INJECTABLE = True
    QUERY_TYPES = (QueryType.CARDINALITY,)
    PARAM_AXIS = "bits_exp"
    PARAM_RANGE = (8, 24)

    def __init__(self, *, bits_exp: int = 14, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._bits_exp = int(bits_exp)
        self._m = 1 << self._bits_exp
        self._bytes = (self._m + 7) >> 3
        self._bitmap = np.zeros(self._bytes, dtype=np.uint8)

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"bits_exp": int(value)}

    def _set_bits(self, bits: np.ndarray) -> None:
        bytes_idx = (bits >> 3).astype(np.int64)
        bit_mask = (np.uint8(1) << (bits & 7)).astype(np.uint8)
        np.bitwise_or.at(self._bitmap, bytes_idx, bit_mask)

    def update(self, item: object) -> None:
        self.update_batch(np.array([int(item)], dtype=np.int64))

    def update_batch(self, items: np.ndarray) -> None:
        hashes = hash_array_u64(np.asarray(items, dtype=np.int64), self._seed_state.derive("lc"))
        bits = (hashes % np.uint64(self._m)).astype(np.int64)
        self._set_bits(bits)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.CARDINALITY:
            raise EstimateUnavailableError("LinearCountingBitmap answers cardinality only")
        set_bits = int(np.unpackbits(self._bitmap).astype(np.int64)[: self._m].sum())
        zero_bits = self._m - set_bits
        if zero_bits <= 0:
            return Estimate(value=float(self._m), n_updates=0)
        estimate = -self._m * math.log(zero_bits / self._m)
        return Estimate(value=float(estimate), n_updates=0)

    def memory_slots(self) -> int:
        return self._m

    def _merge_same_type(self, other: AbstractSketch) -> None:
        np.bitwise_or(self._bitmap, other._bitmap, out=self._bitmap)


@register
class ExactTruncate(AbstractSketch):
    """Exact distinct count up to a capacity; a deterministic reference oracle."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.CARDINALITY,)
    PARAM_AXIS = "capacity"
    PARAM_RANGE = (1, 500_000)

    def __init__(self, *, capacity: int = 4096, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._capacity = int(capacity)
        self._seen = 0
        self._keys: list[int] = []
        self._lookup: set[int] = set()

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"capacity": int(value)}

    def update(self, item: object) -> None:
        key = int(item)
        self._seen += 1
        if key in self._lookup or len(self._lookup) >= self._capacity:
            return
        self._lookup.add(key)
        self._keys.append(key)

    def update_batch(self, items: np.ndarray) -> None:
        for item in np.asarray(items):
            self.update(item)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.CARDINALITY:
            raise EstimateUnavailableError("ExactTruncate answers cardinality only")
        exact = len(self._lookup)
        return Estimate(value=float(exact), n_updates=self._seen)

    def memory_slots(self) -> int:
        return self._capacity

    def _state(self) -> tuple[dict, dict]:
        arrays = {"_stored": np.asarray(sorted(self._lookup), dtype=np.int64)}
        scalars = {"_capacity": self._capacity, "_seen": self._seen}
        return arrays, scalars

    def _restore(self, arrays: dict, scalars: dict) -> None:
        self._capacity = int(scalars["_capacity"])
        self._seen = int(scalars["_seen"])
        self._lookup = set(int(x) for x in arrays["_stored"].tolist())
        self._keys = list(self._lookup)
