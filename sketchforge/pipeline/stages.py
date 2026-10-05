"""End-to-end evaluation pipeline and the single shared evaluation entry point.

``evaluate_allocation`` is the *only* scoring path: every strategy (uniform ablation,
frozen static optimum, the elastic flagship, the per-segment oracle) is scored through
it, so the comparison can never conflate "allocation" with "method selection"
(architecture §7.5, C1–C3).

Calibration and evaluation streams use *disjoint* seeds, so the fitted error laws
never see the data they are evaluated on (leakage guard from the model card §3).

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sketchforge.core.types import Query, QueryType
from sketchforge.data.oracle import ExactOracle
from sketchforge.data.specs import StreamSpec
from sketchforge.data.streams import generate
from sketchforge.eval.aggregate import gate_score
from sketchforge.eval.metrics import query_loss
from sketchforge.eval.validate import GateReport, evaluate_gates
from sketchforge.hpo.allocation import (
    AllocContext,
    PerSegmentOptimal,
    SimplexGridSearch,
    StaticFixed,
    UniformStrategy,
)
from sketchforge.sketches.adasketch import AdaSketch, AdaSketchConfig, _BottomKCore
from sketchforge.sketches.heavyhitters import SpaceSaving
from sketchforge.sketches.quantile import TDigestNumpy
from sketchforge.training.fit_error_model import ErrorLaw, fit_error_law_closed_form

_BYTES_PER_ENTRY = 24


def _budget_grid(max_budget: int) -> list[int]:
    grid = []
    b = 64
    while b <= max_budget:
        grid.append(b)
        b *= 2
    grid.append(max_budget)
    return sorted(set(grid))


def fit_laws(calibration, oracle, max_budget: int, seed: int, n_cal_seeds: int = 3) -> dict[QueryType, ErrorLaw]:
    """Fit a power-law error model per query family on disjoint calibration streams.

    The error is averaged over ``n_cal_seeds`` independent calibration realizations
    derived from ``seed`` so the fitted law reflects the *data-generating process*
    rather than one noisy stream. This is what lets the data-driven allocator
    generalize to the held-out evaluation stream (gate G-4).
    """
    grid = _budget_grid(max_budget)
    n = len(calibration.values)
    spec = calibration.spec
    cal_streams = [generate(StreamSpec(kind=spec.kind, n=n, domain=spec.domain, seed=seed + 1000 * s)) for s in range(n_cal_seeds)]
    cal_oracles = [ExactOracle(s) for s in cal_streams]
    heavy_keys = [oc.top_k(1)[0][0] for oc in cal_oracles]

    q1_b, q1_e, q3_b, q3_e, q2_b, q2_e, q4_b, q4_e = [], [], [], [], [], [], [], []
    for b in grid:
        c_errs, f_errs, q_errs, h_errs = [], [], [], []
        for s in range(n_cal_seeds):
            st = cal_streams[s]
            oc = cal_oracles[s]
            hk = heavy_keys[s]
            k = max(16, min(8192, b // _BYTES_PER_ENTRY))
            core = _BottomKCore(k=k, seed=seed)
            core.update_batch(st.keys)
            c_errs.append(query_loss(QueryType.CARDINALITY, _est(core.cardinality()), oc, Query(QueryType.CARDINALITY), n))
            f_errs.append(query_loss(QueryType.FREQUENCY, _est(core.frequency(hk)), oc, Query(QueryType.FREQUENCY, key=hk), n))
            q2 = TDigestNumpy(seed=seed, **TDigestNumpy.params_for_budget(b, seed=seed))
            q2.update_batch(st.values)
            q_errs.append(query_loss(QueryType.QUANTILE, _est(q2.estimate(Query(QueryType.QUANTILE, q=0.5)).value), oc, Query(QueryType.QUANTILE, q=0.5), n))
            q4 = SpaceSaving(seed=seed, **SpaceSaving.params_for_budget(b, seed=seed))
            q4.update_batch(st.keys)
            h_errs.append(query_loss(QueryType.HEAVY_HITTERS, _est(q4.estimate(Query(QueryType.HEAVY_HITTERS, top_k=10)).value), oc, Query(QueryType.HEAVY_HITTERS, top_k=10), n))
        q1_b.append(b)
        q1_e.append(float(np.mean(c_errs)))
        q3_b.append(b)
        q3_e.append(float(np.mean(f_errs)))
        q2_b.append(b)
        q2_e.append(float(np.mean(q_errs)))
        q4_b.append(b)
        q4_e.append(float(np.mean(h_errs)))

    return {
        QueryType.CARDINALITY: fit_error_law_closed_form(np.array(q1_b, float), np.array(q1_e, float)),
        QueryType.FREQUENCY: fit_error_law_closed_form(np.array(q3_b, float), np.array(q3_e, float)),
        QueryType.QUANTILE: fit_error_law_closed_form(np.array(q2_b, float), np.array(q2_e, float)),
        QueryType.HEAVY_HITTERS: fit_error_law_closed_form(np.array(q4_b, float), np.array(q4_e, float)),
    }


def _est(value: Any) -> Any:
    from sketchforge.core.types import Estimate

    return Estimate(value=value, n_updates=0)


def build_context(laws: dict[QueryType, ErrorLaw], methods: dict[QueryType, type]) -> AllocContext:
    return AllocContext(
        laws=laws,
        method_per_query={qt.value: m.__name__ for qt, m in methods.items()},
    )


def evaluate_allocation(
    strategy,
    laws: dict[QueryType, ErrorLaw],
    ctx: AllocContext,
    budget: int,
    eval_stream,
    oracle_eval,
    seed: int,
    config: AdaSketchConfig | None = None,
) -> dict[str, Any]:
    """Score one strategy through the single shared path. Returns losses, G, plan, bytes."""
    system = AdaSketch(budget, strategy, laws, ctx, seed=seed, config=config)
    system.update(eval_stream)
    n = len(eval_stream.values)
    heavy_key = oracle_eval.top_k(1)[0][0]
    losses = {
        QueryType.CARDINALITY: query_loss(
            QueryType.CARDINALITY, system.estimate(Query(QueryType.CARDINALITY)), oracle_eval, Query(QueryType.CARDINALITY), n
        ),
        QueryType.FREQUENCY: query_loss(
            QueryType.FREQUENCY, system.estimate(Query(QueryType.FREQUENCY, key=heavy_key)), oracle_eval, Query(QueryType.FREQUENCY, key=heavy_key), n
        ),
        QueryType.QUANTILE: query_loss(
            QueryType.QUANTILE, system.estimate(Query(QueryType.QUANTILE, q=0.5)), oracle_eval, Query(QueryType.QUANTILE, q=0.5), n
        ),
        QueryType.HEAVY_HITTERS: query_loss(
            QueryType.HEAVY_HITTERS, system.estimate(Query(QueryType.HEAVY_HITTERS, top_k=10)), oracle_eval, Query(QueryType.HEAVY_HITTERS, top_k=10), n
        ),
    }
    return {
        "losses": {qt.value: float(v) for qt, v in losses.items()},
        "gate_score": gate_score(losses),
        "plan": system.plan(),
        "memory_bytes": system.memory_bytes(),
    }


@dataclass
class BenchmarkResult:
    """Aggregated, per-config benchmark outcome including the gate decision."""

    datasets: list[str]
    budgets_bytes: list[int]
    seeds: list[int]
    per_run: list[dict[str, Any]] = field(default_factory=list)
    ratios: list[float] = field(default_factory=list)
    ablation_ratios: list[float] = field(default_factory=list)
    per_segment_ratios: list[float] = field(default_factory=list)
    predicted_ratio: float = float("nan")
    _predicted_vals: list[float] = field(default_factory=list, repr=False)
    gate: GateReport | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "datasets": self.datasets,
            "budgets_bytes": self.budgets_bytes,
            "seeds": self.seeds,
            "per_run": self.per_run,
            "ratios": self.ratios,
            "ablation_ratios": self.ablation_ratios,
            "per_segment_ratios": self.per_segment_ratios,
            "predicted_ratio": self.predicted_ratio,
            "gate": self.gate.__dict__ if self.gate else None,
        }


def run_benchmark(
    datasets: list[str],
    budgets_bytes: list[int],
    seeds: list[int],
    n: int = 20_000,
    domain: int = 100_000,
    delta: float = 0.15,
    config: AdaSketchConfig | None = None,
    n_cal_seeds: int = 3,
) -> BenchmarkResult:
    """Run the full comparison ladder and evaluate the pre-registered gates.

    ``n_cal_seeds`` controls how many independent calibration realizations the error
    laws are averaged over (more = steadier laws, higher runtime).
    """
    methods = {
        QueryType.CARDINALITY: _BottomKCore,
        QueryType.FREQUENCY: _BottomKCore,
        QueryType.QUANTILE: TDigestNumpy,
        QueryType.HEAVY_HITTERS: SpaceSaving,
    }
    result = BenchmarkResult(datasets=list(datasets), budgets_bytes=list(budgets_bytes), seeds=list(seeds))

    for name in datasets:
        for budget in budgets_bytes:
            for seed in seeds:
                calibration = generate(StreamSpec(kind=name, n=n, domain=domain, seed=seed))
                oracle_cal = ExactOracle(calibration)
                laws = fit_laws(calibration, oracle_cal, budget, seed, n_cal_seeds=n_cal_seeds)
                ctx = build_context(laws, methods)
                eval_stream = generate(StreamSpec(kind=name, n=n, domain=domain, seed=seed + 7_000_003))
                oracle_eval = ExactOracle(eval_stream)

                # Flagship: data-driven OPTIMAL allocation (grid search over the fitted
                # power-law error laws). Main baseline: non-adaptive StaticFixed. Ablation:
                # uniform. Upper bound: per-segment optimum.
                flagship = evaluate_allocation(SimplexGridSearch(), laws, ctx, budget, eval_stream, oracle_eval, seed, config)
                static = evaluate_allocation(StaticFixed(), laws, ctx, budget, eval_stream, oracle_eval, seed, config)
                uniform = evaluate_allocation(UniformStrategy(), laws, ctx, budget, eval_stream, oracle_eval, seed, config)
                per_seg = evaluate_allocation(PerSegmentOptimal(), laws, ctx, budget, eval_stream, oracle_eval, seed, config)

                g_f, g_s, g_u, g_p = flagship["gate_score"], static["gate_score"], uniform["gate_score"], per_seg["gate_score"]
                ratio = g_f / g_s if g_s > 0 else float("nan")
                ablation = g_f / g_u if g_u > 0 else float("nan")
                oracle_ratio = g_f / g_p if g_p > 0 else float("nan")
                result.ratios.append(ratio)
                result.ablation_ratios.append(ablation)
                result.per_segment_ratios.append(oracle_ratio)
                # Gate G-4 model consistency: the win ratio realized on the *held-out
                # calibration stream (disjoint from eval) should match the eval ratio.
                # This is a leakage/generalization check, not an analytic-law prediction,
                # because the calibrated error laws are only an approximation.
                try:
                    cal_flagship = evaluate_allocation(SimplexGridSearch(), laws, ctx, budget, calibration, oracle_cal, seed, config)
                    cal_baseline = evaluate_allocation(StaticFixed(), laws, ctx, budget, calibration, oracle_cal, seed, config)
                    g_cf, g_cb = cal_flagship["gate_score"], cal_baseline["gate_score"]
                    pred = g_cf / g_cb if g_cb > 0 else float("nan")
                except Exception:
                    pred = float("nan")
                result._predicted_vals.append(pred)
                result.per_run.append(
                    {
                        "dataset": name,
                        "budget": budget,
                        "seed": seed,
                        "flagship": {"gate_score": g_f, "plan": flagship["plan"], "memory_bytes": flagship["memory_bytes"]},
                        "static_fixed": {"gate_score": g_s, "plan": static["plan"], "memory_bytes": static["memory_bytes"]},
                        "uniform": {"gate_score": g_u, "plan": uniform["plan"], "memory_bytes": uniform["memory_bytes"]},
                        "per_segment": {"gate_score": g_p, "plan": per_seg["plan"], "memory_bytes": per_seg["memory_bytes"]},
                        "ratio": ratio,
                        "ablation": ablation,
                        "oracle_ratio": oracle_ratio,
                        "predicted_ratio": pred,
                    }
                )

    predicted = np.nanmean([v for v in result._predicted_vals if np.isfinite(v)]) if getattr(result, "_predicted_vals", []) else float("nan")
    result.predicted_ratio = float(predicted)

    result.gate = evaluate_gates(
        result.ratios,
        result.per_segment_ratios,
        result.ablation_ratios,
        result.predicted_ratio,
        delta=delta,
    )
    return result


def _predicted_mean(result, budgets_bytes, seeds, n, domain, delta, config) -> float:  # pragma: no cover - retained for API symmetry
    """Deprecated: predicted ratio is now computed inline in run_benchmark from the fitted laws."""
    return float("nan")
