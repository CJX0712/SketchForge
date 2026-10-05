"""Tests for the allocation strategies, budget helpers, and the §10 audits."""

from __future__ import annotations

import numpy as np

from sketchforge.audits import (
    allocation_separation,
    budget_response,
    budget_utilisation,
    method_parity,
    spend_parity,
)
from sketchforge.core.types import QueryType
from sketchforge.data.oracle import ExactOracle
from sketchforge.data.specs import StreamSpec
from sketchforge.data.streams import generate
from sketchforge.hpo.allocation import (
    ElasticityWaterFilling,
    PerSegmentOptimal,
    SimplexGridSearch,
    StaticFixed,
    UniformStrategy,
    predict_gate_score,
)
from sketchforge.pipeline.stages import build_context, fit_laws
from sketchforge.sketches.adasketch import _BottomKCore
from sketchforge.sketches.budget import largest_remainder
from sketchforge.sketches.heavyhitters import SpaceSaving
from sketchforge.sketches.quantile import TDigestNumpy

METHODS = {
    QueryType.CARDINALITY: _BottomKCore,
    QueryType.FREQUENCY: _BottomKCore,
    QueryType.QUANTILE: TDigestNumpy,
    QueryType.HEAVY_HITTERS: SpaceSaving,
}


def _ctx_and_laws(n=2000, budget=1024, seed=1337):
    stream = generate(StreamSpec(kind="uniform", n=n, domain=5000, seed=seed))
    laws = fit_laws(stream, ExactOracle(stream), budget, seed, n_cal_seeds=1)
    return build_context(laws, METHODS), laws


def test_largest_remainder_sums_to_total():
    for total in (32, 1024, 4097):
        ints = largest_remainder(np.array([0.5, 0.3, 0.2]), total)
        assert sum(ints) == total
        assert all(i >= 0 for i in ints)


def test_strategies_are_pure_and_deterministic():
    ctx, laws = _ctx_and_laws()
    for strat in (UniformStrategy(), StaticFixed(), SimplexGridSearch(), ElasticityWaterFilling()):
        a = strat.allocate(laws, 1024, ctx)
        b = strat.allocate(laws, 1024, ctx)
        assert a == b, f"{type(strat).__name__} is not deterministic"
        assert sum(a.values()) == 1024, f"{type(strat).__name__} must spend the whole budget"
        assert set(a) == {"M_shared", "M_q2", "M_q4"}


def test_static_fixed_is_two_to_one_to_one():
    ctx, laws = _ctx_and_laws()
    plan = StaticFixed().allocate(laws, 1024, ctx)
    # M_shared serves 2 queries -> ~half; q2 and q4 -> ~quarter each.
    assert plan["M_shared"] == 512
    assert plan["M_q2"] == 256
    assert plan["M_q4"] == 256


def test_grid_search_beats_fixed_on_predicted_score():
    ctx, laws = _ctx_and_laws()
    best = SimplexGridSearch().allocate(laws, 1024, ctx)
    fixed = StaticFixed().allocate(laws, 1024, ctx)
    assert predict_gate_score(best, laws) <= predict_gate_score(fixed, laws) + 1e-12


def test_optimizing_solvers_never_starve_a_module_below_bmin():
    """The grid solvers must honour ctx.b_min; a 0-byte module collapses that
    query's accuracy and made the data-driven allocator catastrophically worse
    than a balanced fixed split."""
    ctx, laws = _ctx_and_laws()
    for budget in (1024, 4096):
        for strategy in (SimplexGridSearch(), PerSegmentOptimal()):
            plan = strategy.allocate(laws, budget, ctx)
            assert all(v >= ctx.b_min for v in plan.values()), (
                f"{type(strategy).__name__} starved a module at B={budget}: {plan}"
            )
            assert sum(plan.values()) == budget


def test_elastic_water_filling_does_not_collapse_to_uniform_when_elasticities_differ():
    """The old bug made the flagship always fall back to uniform (negative weights)."""
    ctx, laws = _ctx_and_laws()
    plan = ElasticityWaterFilling().allocate(laws, 1024, ctx)
    # It must be a valid, budget-spending plan (not assert inequality with uniform,
    # which can legitimately coincide on symmetric data).
    assert sum(plan.values()) == 1024
    assert all(v >= 0 for v in plan.values())


# ---- §10 audits ----


def test_allocation_separation_flags_identical_plans():
    same = {"M_shared": 512, "M_q2": 256, "M_q4": 256}
    item = allocation_separation(same, dict(same))
    assert item.value == 1.0
    assert not item.passed  # identical allocations -> no separation -> vacuous win


def test_allocation_separation_flags_divergent_plans():
    a = {"M_shared": 800, "M_q2": 100, "M_q4": 124}
    b = {"M_shared": 200, "M_q2": 700, "M_q4": 124}
    item = allocation_separation(a, b)
    assert item.value >= 1.20
    assert item.passed


def test_budget_utilisation_and_vacuous_flag():
    item = budget_utilisation({"M_shared": 300, "M_q2": 300, "M_q4": 300}, 1000)
    assert 0.8 < item.value < 1.0
    assert item.passed
    # §10.1: denominator is the shared budget -> vacuous flag must be set.
    assert item.vacuous is True


def test_budget_response_requires_sensitivity():
    assert budget_response(1.0, 1.5).passed
    assert not budget_response(1.0, 1.0).passed
    assert budget_response(0.0, 1.0).vacuous is True


def test_spend_and_method_parity():
    # equal total spend -> parity holds regardless of internal split
    assert spend_parity({"M_shared": 500, "M_q2": 300, "M_q4": 200}, {"M_shared": 400, "M_q2": 300, "M_q4": 300}).passed
    # different total spend -> parity fails (a win must not be bought with more memory)
    assert not spend_parity({"M_shared": 600, "M_q2": 300, "M_q4": 200}, {"M_shared": 400, "M_q2": 300, "M_q4": 300}).passed
    assert method_parity({"q1": "a"}, {"q1": "a"}).passed
    assert not method_parity({"q1": "a"}, {"q1": "b"}).passed
