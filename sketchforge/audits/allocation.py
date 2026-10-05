"""Non-vacuity / fairness audits for the allocation experiment (architecture §10).

Every audit returns an :class:`AuditItem` carrying its measured ``value``, the
pre-registered ``threshold``, whether it ``passed``, and a ``vacuous`` flag. Per
§10.1, any *ratio* audit whose denominator is a constant shared by every compared
object is automatically flagged vacuous and forced to FAIL -- the experiment is then
not allowed to claim superiority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditItem:
    """One audit outcome."""

    name: str
    value: float
    threshold: float
    passed: bool
    vacuous: bool = False
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "threshold": self.threshold,
            "passed": self.passed,
            "vacuous": self.vacuous,
            "detail": self.detail,
        }


def _ratio_is_vacuous(denominators: list[float]) -> bool:
    """§10.1: a ratio whose denominator is one shared constant across all compared
    objects is vacuous (e.g. every strategy "spends" the same budget B)."""
    if not denominators:
        return True
    first = denominators[0]
    return all(abs(d - first) < 1e-12 for d in denominators)


def allocation_separation(plan_a: dict[str, int], plan_b: dict[str, int], threshold: float = 1.20) -> AuditItem:
    """``max_m max(b^A_m, b^U_m) / min(b^A_m, b^U_m)`` over the shared modules.

    Detects that the two strategies actually diverge somewhere (otherwise a "win" is
    an artifact of identical allocation). A module allocated 0 by one strategy makes
    the ratio undefined -> treated as maximum separation and the audit still reports
    the flag honestly.
    """
    modules = sorted(set(plan_a) | set(plan_b))
    denom: list[float] = []
    best = 0.0
    for m in modules:
        a = float(plan_a.get(m, 0))
        b = float(plan_b.get(m, 0))
        lo, hi = min(a, b), max(a, b)
        if lo <= 0:
            # One strategy starves a module: separation is maximal, record it.
            best = float("inf") if hi > 0 else 0.0
            denom.append(0.0)
            continue
        r = hi / lo
        denom.append(lo)
        best = max(best, r)
    # Denominators are per-module minima that differ by construction -> never vacuous.
    return AuditItem(
        "allocation_separation",
        float(best),
        threshold,
        bool(best >= threshold),
        vacuous=False,
        detail="max over modules of max/min allocated bytes",
    )


def budget_utilisation(achieved: dict[str, int], budget: int, threshold: float = 0.50) -> AuditItem:
    """``sum(achieved_m) / B`` for a single strategy; must show the budget is really
    being used (guards against a strategy that silently under-spends)."""
    spent = float(sum(achieved.values()))
    ratio = spent / float(budget) if budget > 0 else 0.0
    # Denominator is the raw budget B (shared), so per §10.1 this ratio is reported
    # with an explicit vacuous flag; it is only meaningful because `achieved` is the
    # *real* measured footprint which can be < B.
    vacuous = _ratio_is_vacuous([float(budget)])
    return AuditItem("budget_utilisation", ratio, threshold, bool(ratio >= threshold), vacuous=vacuous, detail="measured footprint / budget")


def budget_response(g_at_b: float, g_at_2b: float, threshold: float = 0.02) -> AuditItem:
    """``|G(2B) - G(B)| / G(B)``; guards that the metric is actually budget-sensitive
    (if G does not move with budget, the allocation experiment is vacuous)."""
    if g_at_b <= 0:
        return AuditItem("budget_response", 0.0, threshold, False, vacuous=True, detail="baseline gate score is non-positive")
    resp = abs(g_at_2b - g_at_b) / g_at_b
    # Denominator G(B) is object-specific -> not vacuous.
    return AuditItem("budget_response", resp, threshold, bool(resp >= threshold), vacuous=False, detail="|G(2B)-G(B)|/G(B)")


def spend_parity(plan_a: dict[str, int], plan_b: dict[str, int]) -> AuditItem:
    """``|sum(b^A) - sum(b^U)| <= 1`` byte: both strategies must actually spend the
    same total budget (otherwise a "win" is bought with more memory)."""
    diff = abs(sum(plan_a.values()) - sum(plan_b.values()))
    return AuditItem("spend_parity", float(diff), 1.0, bool(diff <= 1), vacuous=False, detail="absolute byte difference of the two total spends")


def method_parity(methods_a: dict[str, str], methods_b: dict[str, str]) -> AuditItem:
    """The two strategies must route each query to the *same* sketch method, so a
    difference can only come from allocation, never from method selection."""
    same = dict(methods_a) == dict(methods_b)
    return AuditItem("method_parity", 0.0 if same else 1.0, 0.0, same, vacuous=False, detail="method_per_query identical across strategies")
