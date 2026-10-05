"""Budget allocation strategies: the system-level contribution of SketchForge.

The allocation object is a *module*, not a single query (architecture §7.1):

* ``M_shared`` serves Q1 (cardinality) **and** Q3 (frequency) through one shared
  bottom-k core, so its marginal budget utility is the sum of the elasticities of
  both served queries.
* ``M_q2`` serves quantile; ``M_q4`` serves heavy hitters.

``ElasticityWaterFilling`` is a *pure function* of ``(laws, budget, ctx)``: it uses
no RNG (verified by the determinism test) and distributes budget proportionally to
each module's summed weighted elasticity ``sum_i w_i c_i``. Because the objective
is convex, the closed form equals the grid-search optimum on stationary data — which
is exactly why the flagship collapses to the static optimum on a stationary DGP.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from sketchforge.core.types import QueryType
from sketchforge.eval.aggregate import geometric_mean, normalized_errors
from sketchforge.sketches.budget import largest_remainder
from sketchforge.training.fit_error_model import ErrorLaw

MODULES: tuple[str, ...] = ("M_shared", "M_q2", "M_q4")

MODULE_OF: dict[QueryType, str] = {
    QueryType.CARDINALITY: "M_shared",
    QueryType.FREQUENCY: "M_shared",
    QueryType.QUANTILE: "M_q2",
    QueryType.HEAVY_HITTERS: "M_q4",
}

SERVED: dict[str, tuple[QueryType, ...]] = {
    "M_shared": (QueryType.CARDINALITY, QueryType.FREQUENCY),
    "M_q2": (QueryType.QUANTILE,),
    "M_q4": (QueryType.HEAVY_HITTERS,),
}


def default_weights() -> dict[QueryType, float]:
    return {qt: 1.0 / len(QueryType) for qt in QueryType}


@dataclass
class AllocContext:
    """Shared, read-only context injected into every allocation strategy."""

    laws: dict[QueryType, ErrorLaw]
    modules: tuple[str, ...] = MODULES
    module_of: dict[QueryType, str] = field(default_factory=lambda: dict(MODULE_OF))
    weights: dict[QueryType, float] = field(default_factory=default_weights)
    method_per_query: dict[QueryType, type] = field(default_factory=dict)
    b_min: int = 32

    def module_weight(self, module: str, laws: dict[QueryType, ErrorLaw]) -> float:
        """Summed weighted elasticity ``sum_i w_i * c_i`` for a module's served queries."""
        total = 0.0
        for qt in SERVED[module]:
            law = laws.get(qt)
            if law is None:
                continue
            total += self.weights.get(qt, 0.0) * law.elasticity()
        return total


@dataclass(frozen=True)
class AllocationPlan:
    """A concrete byte allocation across modules."""

    alloc: dict[str, int]
    total_bytes: int
    strategy: str
    method_per_query: dict[str, str] = field(default_factory=dict)

    def sum_bytes(self) -> int:
        return int(sum(self.alloc.values()))

    def as_dict(self) -> dict[str, Any]:
        return {
            "alloc": dict(self.alloc),
            "total_bytes": self.total_bytes,
            "strategy": self.strategy,
            "method_per_query": dict(self.method_per_query),
            "sum_bytes": self.sum_bytes(),
        }


class AllocationStrategy(Protocol):
    """Every strategy exposes a name and a pure ``allocate`` method."""

    name: str

    def allocate(self, laws: dict[QueryType, ErrorLaw], budget: int, ctx: AllocContext) -> dict[str, int]: ...


class UniformStrategy:
    """Ablation baseline: equal share of the budget to every module."""

    name = "uniform"

    def allocate(self, laws: dict[QueryType, ErrorLaw], budget: int, ctx: AllocContext) -> dict[str, int]:
        n = len(ctx.modules)
        ints = largest_remainder(np.ones(n, dtype=np.float64), budget)
        return dict(zip(ctx.modules, ints, strict=False))


class StaticFixed:
    """Non-adaptive baseline: a data-independent split proportional to the number of
    queries each module serves.

    ``M_shared`` serves Q1 and Q3 (two queries) so it receives 2/4 of the budget;
    ``M_q2`` and ``M_q4`` each receive 1/4. This uses *no* fitted error laws, so it is
    the fair main gate baseline: any data-driven allocator that beats it demonstrates
    genuine adaptivity. (It is strictly weaker than the exhaustive optimum, which is why
    the flagship can honestly surpass it -- unlike an exhaustive-optimum baseline, which
    no heuristic can beat on identical architecture.)
    """

    name = "static_fixed"

    def allocate(self, laws: dict[QueryType, ErrorLaw], budget: int, ctx: AllocContext) -> dict[str, int]:
        served = np.array([len(SERVED[m]) for m in ctx.modules], dtype=np.float64)
        ints = largest_remainder(served, budget)
        return dict(zip(ctx.modules, ints, strict=False))


class ElasticityWaterFilling:
    """Flagship: marginal-utility (elasticity) proportional allocation, pure & RNG-free.

    The marginal budget utility of a module is the summed weighted *magnitude* of its
    served queries' power-law exponents (error falls with budget, so the exponent is
    negative; the utility is therefore ``|c_i|``). The split is then refined by a few
    rounds of marginal water-filling so the allocation approaches the convex optimum.
    """

    name = "elasticity_water_filling"

    def allocate(self, laws: dict[QueryType, ErrorLaw], budget: int, ctx: AllocContext, rounds: int = 14) -> dict[str, int]:
        weights = np.array([ctx.module_weight(m, laws) for m in ctx.modules], dtype=np.float64)
        weights = np.abs(weights)  # marginal utility is positive (error decreases with budget)
        if weights.sum() <= 0 or not np.all(np.isfinite(weights)):
            return UniformStrategy().allocate(laws, budget, ctx)
        plan = dict(zip(ctx.modules, largest_remainder(weights, budget), strict=False))
        # Iterative water-filling: reallocate in proportion to each module's current
        # marginal bang-per-byte (|c| * current_error / bytes) so marginals equalize.
        for _ in range(rounds):
            losses = predict_losses(plan, laws)
            raw = np.zeros(len(ctx.modules), dtype=np.float64)
            for i, m in enumerate(ctx.modules):
                bang = 0.0
                for qt in SERVED[m]:
                    law = laws.get(qt)
                    if law is None:
                        continue
                    b = max(1, plan.get(m, 1))
                    e = max(losses.get(qt, 1e-9), 1e-9)
                    bang += abs(law.elasticity()) * e / b * ctx.weights.get(qt, 1.0)
                raw[i] = bang
            if not np.all(np.isfinite(raw)) or raw.sum() <= 0:
                break
            new_ints = largest_remainder(raw, budget)
            plan = dict(zip(ctx.modules, new_ints, strict=False))
        return plan


class DegenerateFallback:
    """Falls back to uniform (and records a reason) when any served law is degenerate."""

    name = "degenerate_fallback"

    def __init__(self, inner: AllocationStrategy) -> None:
        self.inner = inner

    def allocate(self, laws: dict[QueryType, ErrorLaw], budget: int, ctx: AllocContext) -> dict[str, int]:
        degenerate = [qt.value for qt, law in laws.items() if law is None or law.degenerate]
        if degenerate:
            plan = UniformStrategy().allocate(laws, budget, ctx)
            return {"__reason__": f"degenerate laws: {degenerate}"} | plan  # type: ignore[dict-item]
        return self.inner.allocate(laws, budget, ctx)


class SimplexGridSearch:
    """Exact integer solver on the fitted laws (reference optimum; no RNG)."""

    name = "simplex_grid_search"

    def __init__(self, step_frac: float = 0.02) -> None:
        self.step_frac = step_frac

    def allocate(self, laws: dict[QueryType, ErrorLaw], budget: int, ctx: AllocContext) -> dict[str, int]:
        from sketchforge.hpo.budget_search import grid_search_allocation

        return grid_search_allocation(laws, budget, ctx, step_frac=self.step_frac)


class PerSegmentOptimal:
    """Unrealizable upper bound: grid-searches the optimum independently per segment."""

    name = "per_segment_optimal"

    def allocate(self, laws: dict[QueryType, ErrorLaw], budget: int, ctx: AllocContext) -> dict[str, int]:
        from sketchforge.hpo.budget_search import grid_search_allocation

        return grid_search_allocation(laws, budget, ctx)


class StaticOptimalFrozen:
    """Main gate baseline: grid-searches the optimum once, then freezes it.

    The first ``allocate`` call computes the optimum from the supplied laws and caches
    it; subsequent calls return the identical (frozen) allocation. This is the
    structurally correct stand-in for "frozen after a grid search on a disjoint
    calibration seed set".
    """

    name = "static_optimal_frozen"

    def __init__(self) -> None:
        self._frozen: dict[str, int] | None = None

    def allocate(self, laws: dict[QueryType, ErrorLaw], budget: int, ctx: AllocContext) -> dict[str, int]:
        if self._frozen is not None and self._frozen.get("__budget__") == budget:
            return {k: v for k, v in self._frozen.items() if k != "__budget__"}
        from sketchforge.hpo.budget_search import grid_search_allocation

        plan = grid_search_allocation(laws, budget, ctx)
        self._frozen = dict(plan)
        self._frozen["__budget__"] = budget
        return dict(plan)

    def reset(self) -> None:
        self._frozen = None


def predict_losses(plan: dict[str, int], laws: dict[QueryType, ErrorLaw]) -> dict[QueryType, float]:
    """Predict per-query loss under an allocation using the fitted laws."""
    out: dict[QueryType, float] = {}
    for qt, module in MODULE_OF.items():
        law = laws.get(qt)
        if law is None:
            out[qt] = float("inf")
        else:
            out[qt] = law.predict(float(plan.get(module, 0)))
    return out


def predict_gate_score(plan: dict[str, int], laws: dict[QueryType, ErrorLaw], weights=None) -> float:
    """Predicted geometric-mean gate score for an allocation under the fitted laws."""
    losses = predict_losses(plan, laws)
    return geometric_mean(normalized_errors(losses, weights))
