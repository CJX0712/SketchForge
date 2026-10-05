"""Aggregation of per-query losses into a single scalar gate score.

The primary score is the *geometric mean* of normalized errors ``e_i = err_i / eps_i``:

    G = prod_i e_i^{w_i}

The geometric mean is immune to the choice of normalization constants eps_i (they
factor out as an additive constant in log space and therefore cannot be tuned to
make a strategy win) — this is the mathematical, not aesthetic, reason it is the
primary scoring rule (architecture §7.4 / §8 of the model card).

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from sketchforge.core.errors import MetricOutOfBoundsError
from sketchforge.core.types import QueryType
from sketchforge.eval.metrics import SPEC_TOLERANCE

_EPS_FLOOR = 1e-12


def normalized_errors(
    losses: dict[QueryType, float], weights: dict[QueryType, float] | None = None
) -> dict[QueryType, float]:
    """Map raw losses to ``e_i = loss_i / eps_i`` (eps from ``SPEC_TOLERANCE``)."""
    eps = {qt: SPEC_TOLERANCE[qt] for qt in losses}
    if weights is None:
        weights = {qt: 1.0 / len(losses) for qt in losses}
    out: dict[QueryType, float] = {}
    for qt, loss in losses.items():
        if loss < 0:
            raise MetricOutOfBoundsError("loss must be non-negative", query=qt.value, loss=loss)
        out[qt] = max(loss, 0.0) / max(eps[qt], _EPS_FLOOR)
    return out


def geometric_mean(errors: dict[QueryType, float], weights: dict[QueryType, float] | None = None) -> float:
    """Geometric mean of normalized errors with equal-or-weighted exponents."""
    keys = list(errors.keys())
    if not keys:
        return float("nan")
    if weights is None:
        weights = {qt: 1.0 / len(keys) for qt in keys}
        w = np.array([1.0 / len(keys) for _ in keys])
    else:
        w = np.array([float(weights[qt]) for qt in keys], dtype=np.float64)
        s = w.sum()
        w = np.ones(len(keys)) / len(keys) if s <= 0 else w / s
    vals = np.array([float(errors[qt]) for qt in keys], dtype=np.float64)
    vals = np.maximum(vals, _EPS_FLOOR)
    log_g = float(np.sum(w * np.log(vals)))
    return float(np.exp(log_g))


def gate_score(
    losses: dict[QueryType, float], weights: dict[QueryType, float] | None = None
) -> float:
    """Convenience: normalized errors -> geometric mean gate score ``G``."""
    return geometric_mean(normalized_errors(losses, weights), weights)
