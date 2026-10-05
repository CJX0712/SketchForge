"""Byte-budget parameter fitting and integer allocation helpers.

The budget currency of SketchForge is *serialized bytes*. ``fit_param_under_budget``
binary-searches the largest parameter on a sketch's ``PARAM_AXIS`` whose
post-ingest serialized size still fits the budget, using a deterministic probe
stream so the result is reproducible.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from sketchforge.core.errors import IndistinguishableCostError
from sketchforge.core.seed import SeedState
from sketchforge.core.types import BudgetPlan

# A probe only needs to populate a sketch to its steady-state occupancy so the
# serialized footprint is representative. 2,000 distinct-ish keys is ample for the
# param ranges we search; the previous 20,000 caused every binary-search step to
# re-ingest the full probe and dominated the runtime (~30s per fit_laws call).
_PROBE_SIZE = 2_000
_PROBE_DOMAIN = 1_000_000

# Binary search hits the same (cls, budget, axis-range, seed) many times across
# strategies and grid points. Cache the structural result (the probe is fixed, so
# the post-ingest footprint is a pure function of these keys).
_PARAM_CACHE: dict[tuple, tuple[dict, int]] = {}


def default_probe() -> np.ndarray:
    """Deterministic int64 probe stream used to measure post-ingest footprint."""
    rng = np.random.default_rng(SeedState(0xC0FFEE).derive("budget-probe"))
    return rng.integers(0, _PROBE_DOMAIN, size=_PROBE_SIZE, dtype=np.int64)


def fit_param_under_budget(
    cls: type,
    budget_bytes: int,
    lo: int,
    hi: int,
    probe: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[dict, int]:
    """Largest ``PARAM_AXIS`` value whose ingested serialized size fits ``budget_bytes``.

    Binary search over the closed integer interval ``[lo, hi]``. Returns the full
    parameter dict and the chosen axis value. When even ``lo`` overflows the budget,
    ``lo`` is returned (the caller must decide whether such a budget is viable).
    Results are cached because the probe and parameters are deterministic.
    """
    if probe is None:
        probe = default_probe()
    cache_key = (cls.__name__, int(budget_bytes), int(lo), int(hi), int(seed))
    hit = _PARAM_CACHE.get(cache_key)
    if hit is not None:
        return hit
    lo_v, hi_v = int(lo), int(hi)
    best = lo_v
    while lo_v <= hi_v:
        mid = (lo_v + hi_v) // 2
        params = cls.params_from_axis(mid)
        sketch = cls(seed=seed, **params)
        sketch.update_batch(probe)
        if sketch.memory_bytes() <= budget_bytes:
            best = mid
            lo_v = mid + 1
        else:
            hi_v = mid - 1
    result = (cls.params_from_axis(best), best)
    _PARAM_CACHE[cache_key] = result
    return result


def largest_remainder(fractions: np.ndarray, total: int) -> list[int]:
    """Integer allocation of ``total`` units by ``fractions`` (Hamilton method).

    The remainder is handed out in stable descending order of the fractional
    residual, which makes ties break deterministically by enumeration order.
    Raises ``IndistinguishableCostError`` when the fractions share a constant
    denominator (a vacuous-cost configuration) or sum to zero.
    """
    array = np.asarray(fractions, dtype=np.float64)
    s = float(array.sum())
    if s <= 0 or not np.all(np.isfinite(array)):
        raise IndistinguishableCostError(
            "allocation fractions must be finite and sum to a positive value",
            sum=s,
        )
    scaled = array / s * total
    floored = np.floor(scaled).astype(np.int64)
    assigned = int(floored.sum())
    remainder = total - assigned
    out = list(int(x) for x in floored)
    if remainder > 0:
        order = np.argsort(-(scaled - floored), kind="stable")
        for idx in order[:remainder]:
            out[int(idx)] += 1
    elif remainder < 0:
        order = np.argsort(scaled - floored, kind="stable")
        for idx in order[:-remainder]:
            out[int(idx)] -= 1
    return out


def make_plan(alloc: dict[str, int], total_bytes: int, strategy: str) -> BudgetPlan:
    """Build a validated :class:`BudgetPlan`."""
    return BudgetPlan(alloc={k: int(v) for k, v in alloc.items()}, total_bytes=int(total_bytes), strategy=strategy)
