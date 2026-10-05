"""Quantile / rank sketches for Q2.

All Tier-1 methods are deterministic. ``TDigestNumpy`` is an in-house centroid
t-digest (no external dependency) used as a reproducible reference. Tier-0
references (``KllSketch`` / ``ReqSketch``) are non-deterministic and excluded
from the main gate.

Author: 晨星
"""

from __future__ import annotations

import bisect
from typing import Any

import numpy as np

from sketchforge.core.errors import EstimateUnavailableError
from sketchforge.core.types import Estimate, Query, QueryType
from sketchforge.sketches.backend import KllSketch, ReqSketch  # noqa: F401 - register T0 refs
from sketchforge.sketches.base import AbstractSketch
from sketchforge.sketches.registry import register


def _quantile_from_sorted(sorted_vals: np.ndarray, q: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


@register
class ReservoirQuantile(AbstractSketch):
    """Reservoir sampler returning approximate quantiles from a fixed memory window."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = True
    QUERY_TYPES = (QueryType.QUANTILE,)
    PARAM_AXIS = "k"
    PARAM_RANGE = (16, 8192)

    def __init__(self, *, k: int = 1024, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._k = int(k)
        self._reservoir = np.empty(self._k, dtype=np.float64)
        self._count = 0
        self._index = 0

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"k": int(value)}

    def update(self, item: object) -> None:
        value = float(item)
        n = self._count
        if n < self._k:
            self._reservoir[n] = value
        else:
            j = int(self._generator("reservoir").integers(0, n + 1))
            if j < self._k:
                self._reservoir[j] = value
        self._count += 1
        self._index = self._count

    def update_batch(self, items: np.ndarray) -> None:
        rng = self._generator("reservoir")
        for value in np.asarray(items, dtype=np.float64):
            n = self._count
            if n < self._k:
                self._reservoir[n] = value
            else:
                j = int(rng.integers(0, n + 1))
                if j < self._k:
                    self._reservoir[j] = value
            self._count += 1

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.QUANTILE:
            raise EstimateUnavailableError("ReservoirQuantile answers quantile only")
        q = 0.5 if query.q is None else float(query.q)
        filled = min(self._count, self._k)
        if filled == 0:
            return Estimate(value=0.0, n_updates=self._count)
        sorted_vals = np.sort(self._reservoir[:filled])
        return Estimate(value=_quantile_from_sorted(sorted_vals, q), n_updates=self._count)

    def memory_slots(self) -> int:
        return self._k


@register
class EqualWidthHistogram(AbstractSketch):
    """Equal-width histogram with interpolation for quantile queries."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = True
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.QUANTILE,)
    PARAM_AXIS = "bins"
    PARAM_RANGE = (8, 4096)

    def __init__(self, *, bins: int = 256, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._bins = int(bins)
        self._lo = np.inf
        self._hi = -np.inf
        self._counts = np.zeros(self._bins, dtype=np.int64)
        self._total = 0

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"bins": int(value)}

    def _width(self) -> float:
        if not np.isfinite(self._lo) or not np.isfinite(self._hi) or self._hi <= self._lo:
            return 1.0
        return (self._hi - self._lo) / self._bins

    def update(self, item: object) -> None:
        value = float(item)
        if value < self._lo:
            self._lo = value
        if value > self._hi:
            self._hi = value
        width = self._width()
        if self._hi > self._lo:
            idx = int((value - self._lo) / width)
            idx = min(max(idx, 0), self._bins - 1)
            self._counts[idx] += 1
        self._total += 1

    def update_batch(self, items: np.ndarray) -> None:
        for value in np.asarray(items, dtype=np.float64):
            self.update(value)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.QUANTILE:
            raise EstimateUnavailableError("EqualWidthHistogram answers quantile only")
        q = 0.5 if query.q is None else float(query.q)
        if self._total == 0:
            return Estimate(value=0.0, n_updates=0)
        if self._hi <= self._lo:
            return Estimate(value=float(self._lo), n_updates=self._total)
        cum = np.cumsum(self._counts)
        target = q * (self._total - 1)
        bin_idx = int(np.searchsorted(cum, target, side="right"))
        bin_idx = min(bin_idx, self._bins - 1)
        width = self._width()
        base = self._lo + bin_idx * width
        return Estimate(value=float(base + width * 0.5), n_updates=self._total)

    def memory_slots(self) -> int:
        return self._bins

    def _merge_same_type(self, other: AbstractSketch) -> None:
        self._lo = min(self._lo, other._lo)
        self._hi = max(self._hi, other._hi)
        self._counts = self._counts + other._counts
        self._total += other._total


@register
class ExactSortTruncate(AbstractSketch):
    """Exact quantile from a truncated sorted buffer (deterministic reference)."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.QUANTILE,)
    PARAM_AXIS = "capacity"
    PARAM_RANGE = (64, 500_000)

    def __init__(self, *, capacity: int = 4096, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._capacity = int(capacity)
        self._buf = np.empty(0, dtype=np.float64)
        self._total = 0

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"capacity": int(value)}

    def update(self, item: object) -> None:
        self._total += 1
        if self._buf.size < self._capacity:
            self._buf = np.append(self._buf, float(item))

    def update_batch(self, items: np.ndarray) -> None:
        for value in np.asarray(items, dtype=np.float64):
            self.update(value)

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.QUANTILE:
            raise EstimateUnavailableError("ExactSortTruncate answers quantile only")
        q = 0.5 if query.q is None else float(query.q)
        if self._buf.size == 0:
            return Estimate(value=0.0, n_updates=self._total)
        return Estimate(value=_quantile_from_sorted(np.sort(self._buf), q), n_updates=self._total)

    def memory_slots(self) -> int:
        return self._capacity


@register
class TDigestNumpy(AbstractSketch):
    """Deterministic centroid t-digest with bounded merge (in-house, Tier-1)."""

    TIER = 1
    BACKEND = "numpy"
    DETERMINISTIC = True
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.QUANTILE,)
    PARAM_AXIS = "delta_exp"
    PARAM_RANGE = (3, 12)

    def __init__(self, *, delta_exp: int = 8, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._delta = float(1 << int(delta_exp))
        self._max_centroids = int(self._delta)
        self._c = np.empty((0, 2), dtype=np.float64)

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"delta_exp": int(value)}

    def update(self, item: object) -> None:
        x = float(item)
        if self._c.size == 0:
            self._c = np.array([[x, 1.0]], dtype=np.float64)
        else:
            self._c = np.vstack((self._c, [x, 1.0]))
        self._compress()

    def update_batch(self, items: np.ndarray) -> None:
        for value in np.asarray(items, dtype=np.float64):
            self.update(value)

    def _compress(self) -> None:
        if self._c.shape[0] <= self._max_centroids:
            self._c = self._c[np.argsort(self._c[:, 0], kind="stable")]
            return
        self._c = self._c[np.argsort(self._c[:, 0], kind="stable")]
        while self._c.shape[0] > self._max_centroids:
            n = self._c.shape[0]
            best_i = 0
            best_dist = np.inf
            for i in range(n - 1):
                dist = abs(self._c[i + 1, 0] - self._c[i, 0])
                if dist < best_dist:
                    best_dist = dist
                    best_i = i
            left = self._c[best_i]
            right = self._c[best_i + 1]
            merged = np.array(
                [[(left[0] * left[1] + right[0] * right[1]) / (left[1] + right[1]), left[1] + right[1]]],
                dtype=np.float64,
            )
            self._c = np.vstack((self._c[:best_i], merged, self._c[best_i + 2 :]))

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.QUANTILE:
            raise EstimateUnavailableError("TDigestNumpy answers quantile only")
        q = 0.5 if query.q is None else float(query.q)
        if self._c.shape[0] == 0:
            return Estimate(value=0.0, n_updates=0)
        order = np.argsort(self._c[:, 0], kind="stable")
        c = self._c[order]
        total = float(c[:, 1].sum())
        if total <= 0:
            return Estimate(value=float(c[0, 0]), n_updates=0)
        target = q * total
        cum = np.cumsum(c[:, 1])
        idx = int(bisect.bisect_left(cum, target))
        idx = min(max(idx, 0), c.shape[0] - 1)
        return Estimate(value=float(c[idx, 0]), n_updates=int(total))

    def memory_slots(self) -> int:
        return self._max_centroids
