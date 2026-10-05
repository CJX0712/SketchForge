"""Tests for the AdaSketch fixed-budget system."""

from __future__ import annotations

from sketchforge.core.types import Query, QueryType
from sketchforge.data.oracle import ExactOracle
from sketchforge.data.specs import StreamSpec
from sketchforge.data.streams import generate
from sketchforge.hpo.allocation import SimplexGridSearch, StaticFixed
from sketchforge.pipeline.stages import build_context, fit_laws
from sketchforge.sketches.adasketch import AdaSketch, _BottomKCore
from sketchforge.sketches.heavyhitters import SpaceSaving
from sketchforge.sketches.quantile import TDigestNumpy

METHODS = {
    QueryType.CARDINALITY: _BottomKCore,
    QueryType.FREQUENCY: _BottomKCore,
    QueryType.QUANTILE: TDigestNumpy,
    QueryType.HEAVY_HITTERS: SpaceSaving,
}


def _system(kind="uniform", n=2000, budget=1024, seed=1337):
    calib = generate(StreamSpec(kind=kind, n=n, domain=5000, seed=seed))
    laws = fit_laws(calib, ExactOracle(calib), budget, seed, n_cal_seeds=1)
    ctx = build_context(laws, METHODS)
    ev = generate(StreamSpec(kind=kind, n=n, domain=5000, seed=seed + 7_000_003))
    return AdaSketch(budget, SimplexGridSearch(), laws, ctx, seed=seed), ev, ExactOracle(ev)


def test_plan_spends_whole_budget_and_answers_all_four_queries():
    system, ev, oracle = _system()
    system.update(ev)
    plan = system.plan()
    assert set(plan) == {"M_shared", "M_q2", "M_q4"}
    assert sum(plan.values()) == 1024
    assert system.memory_bytes() > 0
    # All four families answer.
    assert system.estimate(Query(QueryType.CARDINALITY)).value >= 0
    hk = oracle.top_k(1)[0][0]
    assert system.estimate(Query(QueryType.FREQUENCY, key=hk)).value >= 0
    assert system.estimate(Query(QueryType.QUANTILE, q=0.5)).value is not None
    assert system.estimate(Query(QueryType.HEAVY_HITTERS, top_k=10)).value is not None


def test_adasketch_is_deterministic_for_same_seed():
    a, ev, _ = _system()
    b, _, _ = _system()
    a.update(ev)
    b.update(ev)
    assert a.plan() == b.plan()
    for q in (Query(QueryType.CARDINALITY), Query(QueryType.QUANTILE, q=0.5)):
        assert a.estimate(q).value == b.estimate(q).value


def test_shared_core_serves_cardinality_and_frequency():
    """The flagship's shared bottom-k module is one structure answering Q1 and Q3."""
    system, ev, oracle = _system()
    system.update(ev)
    plan = system.plan()
    assert plan["M_shared"] > 0
    # Q3 point frequency is exact when the key is retained by the core.
    hk = oracle.top_k(1)[0][0]
    est = system.estimate(Query(QueryType.FREQUENCY, key=hk)).value
    truth = oracle.frequency(hk)
    assert est == truth or est == 0.0  # retained -> exact; otherwise bottom-k returns 0

def test_baseline_static_fixed_also_answers_all_four():
    calib = generate(StreamSpec(kind="uniform", n=2000, domain=5000, seed=1337))
    laws = fit_laws(calib, ExactOracle(calib), 1024, 1337, n_cal_seeds=1)
    ctx = build_context(laws, METHODS)
    ev = generate(StreamSpec(kind="uniform", n=2000, domain=5000, seed=1337 + 7_000_003))
    system = AdaSketch(1024, StaticFixed(), laws, ctx, seed=1337)
    system.update(ev)
    assert system.estimate(Query(QueryType.CARDINALITY)).value >= 0
    assert sum(system.plan().values()) == 1024
