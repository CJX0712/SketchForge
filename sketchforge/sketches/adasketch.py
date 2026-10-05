"""AdaSketch: the SketchForge flagship fixed-budget streaming system.

AdaSketch splits a total byte budget ``B`` across three modules via an
:class:`AllocationStrategy`:

* ``M_shared`` -- a single shared *bottom-k* core (by min-hash) that serves **both**
  Q1 (cardinality, via a KMV estimate) and Q3 (point frequency, via exact counts of
  the retained items). One parameter therefore moves two error curves.
* ``M_q2`` -- an independent quantile module.
* ``M_q4`` -- an independent heavy-hitter module.

The shared core is the budget-efficiency claim: because its marginal utility is the
sum of the elasticities of Q1 and Q3, it structurally deserves more budget than any
single-query module (architecture §7.4).

Author: 晨星
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from sketchforge.core.errors import EstimateUnavailableError
from sketchforge.core.hashing import hash_array_u64
from sketchforge.core.seed import SeedState
from sketchforge.core.types import Estimate, Query, QueryType
from sketchforge.sketches.heavyhitters import SpaceSaving
from sketchforge.sketches.quantile import TDigestNumpy

_BYTES_PER_ENTRY = 24  # int64 key + int64 count + uint64 hash


class _BottomKCore:
    """Min-hash bottom-k sample retaining exact counts of the retained distinct keys.

    Keeps the ``k`` smallest per-key hashes. The retained keys are mirrored in a
    max-heap (via ``heapq`` on ``(-hash, key)``) so a full-core insert is O(log k)
    instead of an O(k) max scan -- this is what keeps large-budget builds tractable.
    The heap top is always the largest retained hash, which is exactly the eviction
    candidate and the KMV threshold.
    """

    def __init__(self, k: int, seed: int) -> None:
        self._k = max(1, int(k))
        self._hseed = int(SeedState(int(seed)).derive("adasketch:shared"))
        self._entries: dict[int, list] = {}  # key -> [count, hash]
        self._heap: list[tuple[int, int]] = []  # max-heap via (-hash, key)
        self._seen = 0

    def update(self, key: int) -> None:
        self.update_batch(np.array([int(key)], dtype=np.int64))

    def update_batch(self, keys: np.ndarray) -> None:
        arr = np.asarray(keys, dtype=np.int64).ravel()
        if arr.size == 0:
            return
        hashes = hash_array_u64(arr, self._hseed)
        self._seen += int(arr.size)
        for key, h in zip(arr.tolist(), hashes.tolist(), strict=True):
            h = int(h)
            entry = self._entries.get(key)
            if entry is not None:
                entry[0] += 1
                continue
            if len(self._entries) < self._k:
                self._entries[key] = [1, h]
                heapq.heappush(self._heap, (-h, key))
                continue
            neg_max, evict_key = self._heap[0]
            if h < -neg_max:
                heapq.heapreplace(self._heap, (-h, key))
                del self._entries[evict_key]
                self._entries[key] = [1, h]

    def cardinality(self) -> float:
        if not self._entries:
            return 0.0
        if len(self._entries) < self._k:
            return float(len(self._entries))
        kth = -self._heap[0][0] + 1
        return (len(self._entries) - 1) * (1 << 64) / kth

    def frequency(self, key: int) -> float:
        entry = self._entries.get(int(key))
        return float(entry[0]) if entry else 0.0

    def memory_bytes(self) -> int:
        return self._k * _BYTES_PER_ENTRY


@dataclass
class AdaSketchConfig:
    """Construction options for :class:`AdaSketch`."""

    q2_method: type = TDigestNumpy
    q4_method: type = SpaceSaving
    b_min_shared: int = 256


class AdaSketch:
    """Fixed-budget streaming system that routes the four query families to modules."""

    def __init__(
        self,
        budget: int,
        strategy,
        laws: dict[QueryType, object],
        ctx,
        seed: int = 0,
        config: AdaSketchConfig | None = None,
    ) -> None:
        self._budget = int(budget)
        self._strategy = strategy
        self._laws = laws
        self._ctx = ctx
        self._seed = int(seed)
        self._config = config or AdaSketchConfig()
        self._plan: dict[str, int] = {}
        self._shared: _BottomKCore | None = None
        self._q2 = None
        self._q4 = None
        self._build()

    def _allocate(self) -> dict[str, int]:
        return dict(self._strategy.allocate(self._laws, self._budget, self._ctx))

    def _build(self) -> None:
        plan = self._allocate()
        self._plan = plan
        k_shared = max(16, min(8192, plan.get("M_shared", self._config.b_min_shared) // _BYTES_PER_ENTRY))
        self._shared = _BottomKCore(k=k_shared, seed=self._seed)
        q2_params = self._config.q2_method.params_for_budget(plan.get("M_q2", 0), seed=self._seed)
        self._q2 = self._config.q2_method(seed=self._seed, **q2_params)
        q4_params = self._config.q4_method.params_for_budget(plan.get("M_q4", 0), seed=self._seed)
        self._q4 = self._config.q4_method(seed=self._seed, **q4_params)

    def update(self, stream) -> None:
        if self._shared is None or self._q2 is None or self._q4 is None:
            raise EstimateUnavailableError("AdaSketch was not built")
        self._shared.update_batch(stream.keys)
        self._q2.update_batch(stream.values)
        self._q4.update_batch(stream.keys)

    def estimate(self, query: Query) -> Estimate:
        if self._shared is None or self._q2 is None or self._q4 is None:
            raise EstimateUnavailableError("AdaSketch was not built")
        if query.kind is QueryType.CARDINALITY:
            return Estimate(value=self._shared.cardinality(), n_updates=self._shared._seen)
        if query.kind is QueryType.FREQUENCY:
            return Estimate(value=self._shared.frequency(int(query.key)), n_updates=self._shared._seen)
        if query.kind is QueryType.QUANTILE:
            return self._q2.estimate(query)
        if query.kind is QueryType.HEAVY_HITTERS:
            return self._q4.estimate(query)
        raise EstimateUnavailableError("unsupported query", kind=str(query.kind))

    def memory_bytes(self) -> int:
        total = 0
        if self._shared is not None:
            total += self._shared.memory_bytes()
        if self._q2 is not None:
            total += self._q2.memory_bytes()
        if self._q4 is not None:
            total += self._q4.memory_bytes()
        return total

    def plan(self) -> dict[str, int]:
        return dict(self._plan)
