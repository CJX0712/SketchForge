"""Grid-search solver for the integer allocation that minimizes predicted error.

The objective is the geometric mean of normalized errors, which (under power-law
laws ``err_i = max(a_i b_m^{-c_i}, floor_i)``) is convex in the module budgets. This
coarse grid search recovers the closed-form optimum to within the integerization
resolution and serves as the reference solver for :class:`StaticOptimalFrozen` and
:class:`PerSegmentOptimal`.

Author: 晨星
"""

from __future__ import annotations

import itertools

import numpy as np

from sketchforge.core.errors import InvalidConfigError
from sketchforge.core.types import QueryType
from sketchforge.eval.aggregate import geometric_mean, normalized_errors
from sketchforge.hpo.allocation import (
    AllocContext,
    predict_gate_score,
    predict_losses,
)
from sketchforge.training.fit_error_model import ErrorLaw


def grid_search_allocation(
    laws: dict[QueryType, ErrorLaw],
    budget: int,
    ctx: AllocContext,
    step_frac: float = 0.02,
) -> dict[str, int]:
    """Enumerate integer allocations summing to ``budget`` and keep the best by predicted G.

    Every module is guaranteed at least ``ctx.b_min`` bytes. Without this floor a
    mis-fitted law can starve a module to 0 bytes, which collapses that query's
    accuracy catastrophically and makes the data-driven allocator far *worse* than a
    balanced fixed split. The floor is the architecture's existing ``b_min`` contract.
    """
    modules = list(ctx.modules)
    n = len(modules)
    if budget <= 0:
        raise InvalidConfigError("budget must be positive for allocation", budget=budget)
    if n == 0:
        raise InvalidConfigError("no modules to allocate across")
    b_min = int(getattr(ctx, "b_min", 0) or 0)
    if n * b_min > budget:
        b_min = budget // n  # cannot honour the floor; split as evenly as possible
    step = max(1, round(budget * step_frac))
    steps = [s for s in range(0, budget + 1, step) if s >= b_min]
    if not steps or steps[0] != b_min:
        steps = [b_min, *steps]
    n_steps = len(steps)

    best_plan: dict[str, int] | None = None
    best_score = float("inf")
    # n-1 free coordinates; last derived so the sum equals budget.
    for combo in itertools.product(range(n_steps), repeat=n - 1):
        prefix = [steps[i] for i in combo]
        remainder = budget - sum(prefix)
        if remainder < b_min:
            continue
        plan = dict(zip(modules, [*prefix, remainder], strict=False))
        losses = predict_losses(plan, laws)
        if any(not np.isfinite(v) for v in losses.values()):
            continue
        try:
            score = geometric_mean(normalized_errors(losses))
        except Exception:
            continue
        if score < best_score:
            best_score = score
            best_plan = plan
    if best_plan is None:
        # Degenerate fallback: split evenly (integerized to exactly budget).
        from sketchforge.sketches.budget import largest_remainder

        ints = largest_remainder(np.ones(n, dtype=np.float64), budget)
        best_plan = dict(zip(modules, ints, strict=False))
    return dict(best_plan)


def predicted_ratio(
    laws_flagship: dict[QueryType, ErrorLaw],
    laws_baseline: dict[QueryType, ErrorLaw],
    budget: int,
    ctx_flagship: AllocContext,
    ctx_baseline: AllocContext,
    grid_kwargs: dict | None = None,
) -> float:
    """Predicted G(Flagship) / G(StaticFixed baseline) from the fitted laws (gate G-4).

    Mirrors the observed comparison in run_benchmark: the data-driven optimal flagship
    versus the non-adaptive fixed baseline. Using the same two allocators keeps the
    model-consistency gate (G-4) well-defined.
    """
    grid_kwargs = grid_kwargs or {}
    from sketchforge.hpo.allocation import (
        SimplexGridSearch,
        StaticFixed,
    )

    f_plan = SimplexGridSearch().allocate(laws_flagship, budget, ctx_flagship)
    b_plan = StaticFixed().allocate(laws_baseline, budget, ctx_baseline)
    g_f = predict_gate_score(f_plan, laws_flagship)
    g_b = predict_gate_score(b_plan, laws_baseline)
    if g_b <= 0:
        return float("nan")
    return g_f / g_b
