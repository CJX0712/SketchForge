"""Error-law fitting: power-law ``err = max(a * b^{-c}, floor)`` via closed-form OLS.

Per the architecture contract the main fitting path is a *closed-form* log-log
least squares (``c = -Cov(x,y)/Var(x)``, ``a = exp(ybar + c*xbar)``), never
``scipy.optimize.curve_fit`` (which is iteration-order dependent and therefore not
a well-defined function). ``crosscheck_with_curve_fit`` exists only as an offline
audit cross-check.

Author: 晨星
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_FLOOR_CLAMP = 1e-6
_DEGENERATE_R2 = 0.9
_MIN_POINTS = 4


@dataclass(frozen=True)
class ErrorLaw:
    """Fitted power law ``err = max(a * b^{-c}, floor)``."""

    a: float
    c: float
    floor: float
    r2: float
    n_points: int
    n_points_clamped: int
    degenerate: bool
    reason: str

    def predict(self, budget: float) -> float:
        """Predict the error at ``budget`` bytes (never below ``floor``)."""
        if budget <= 0:
            return float(self.a + self.floor)
        raw = self.a * math.pow(budget, -self.c)
        return max(raw, self.floor)

    def elasticity(self) -> float:
        """Return ``c`` (the budget elasticity); 0.0 when degenerate."""
        return 0.0 if self.degenerate else float(self.c)


def fit_error_law_closed_form(
    budgets: np.ndarray, errors: np.ndarray, floor_floor: float = _FLOOR_CLAMP
) -> ErrorLaw:
    """Fit ``err = max(a*b^{-c}, floor)`` by closed-form log-log OLS.

    ``budgets`` and ``errors`` are 1-D arrays of equal length. Values at or below
    ``floor_floor`` are clamped (and reported via ``n_points_clamped``) so that the
    logarithm is well defined; the clamp is reported honestly in the returned law.
    """
    budgets = np.asarray(budgets, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if budgets.size < 2:
        return ErrorLaw(
            a=float("nan"),
            c=0.0,
            floor=float("nan"),
            r2=0.0,
            n_points=int(budgets.size),
            n_points_clamped=0,
            degenerate=True,
            reason="fewer than two points",
        )
    positive = budgets > 0
    budgets = budgets[positive]
    errors = errors[positive]
    n = budgets.size
    if n < _MIN_POINTS:
        return ErrorLaw(
            a=float("nan"),
            c=0.0,
            floor=float(np.min(errors)) if errors.size else float("nan"),
            r2=0.0,
            n_points=int(n),
            n_points_clamped=0,
            degenerate=True,
            reason=f"fewer than {_MIN_POINTS} positive points",
        )
    clamped = errors <= _FLOOR_CLAMP
    n_clamped = int(np.count_nonzero(clamped))
    y = np.log(np.maximum(errors, _FLOOR_CLAMP))
    x = np.log(budgets)
    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    sxx = float(np.sum((x - xbar) ** 2))
    if sxx <= 0:
        return ErrorLaw(
            a=float(np.exp(ybar)),
            c=0.0,
            floor=float(np.min(errors)) if errors.size else 0.0,
            r2=0.0,
            n_points=int(n),
            n_points_clamped=n_clamped,
            degenerate=True,
            reason="zero variance in log-budget",
        )
    sxy = float(np.sum((x - xbar) * (y - ybar)))
    c = -sxy / sxx  # err = a*b^{-c} => slope = -c
    a = math.exp(ybar + c * xbar)
    ss_res = float(np.sum((y - (ybar + (-c) * (x - xbar))) ** 2))
    ss_tot = float(np.sum((y - ybar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    floor = float(np.min(errors))
    degenerate = (c <= 0.0) or (r2 < _DEGENERATE_R2)
    reason = "" if not degenerate else (f"non-positive elasticity c={c:.4f}" if c <= 0 else f"low r2={r2:.4f}")
    return ErrorLaw(
        a=a,
        c=c,
        floor=floor,
        r2=r2,
        n_points=int(n),
        n_points_clamped=n_clamped,
        degenerate=degenerate,
        reason=reason,
    )


def crosscheck_with_curve_fit(law: ErrorLaw, budgets: np.ndarray, errors: np.ndarray) -> dict[str, float]:
    """Offline cross-check against an iterative fitter; results enter audits only."""
    try:
        from scipy.optimize import curve_fit

        def model(b: np.ndarray, a: float, c: float) -> np.ndarray:
            return a * np.power(np.asarray(b, dtype=np.float64), -c)

        budgets = np.asarray(budgets, dtype=np.float64)
        errors = np.asarray(errors, dtype=np.float64)
        positive = (budgets > 0) & (errors > _FLOOR_CLAMP)
        popt, _ = curve_fit(model, budgets[positive], errors[positive], p0=[1.0, 0.5], maxfev=20000)
        return {"a_curvefit": float(popt[0]), "c_curvefit": float(popt[1])}
    except Exception as exc:
        return {"error": str(exc)}
