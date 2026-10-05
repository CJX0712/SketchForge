"""End-to-end smoke test for the evaluation pipeline and the superiority gate.

Kept small (one DGP, one budget, one seed) so it runs in ~1s. The real gate is
produced by ``benchmarks/gate_benchmark.py`` (R=20, all DGPs).
"""

from __future__ import annotations

from sketchforge.eval.validate import canonicalize, check_g3, digest, evaluate_gates
from sketchforge.pipeline.stages import run_benchmark


def test_small_uniform_run_passes_all_gates():
    # n=2000/seed=1337 is a deterministic configuration where the adaptive flagship
    # beats the fixed baseline strictly, matches the per-segment optimum, and clears
    # every pre-registered gate including the calibration/eval consistency check.
    res = run_benchmark(datasets=["uniform"], budgets_bytes=[1024], seeds=[1337], n=2000, domain=5000, delta=0.15)
    gate = res.gate
    assert gate is not None
    assert res.ratios and max(res.ratios) < 1.0
    assert res.per_segment_ratios and all(r <= 1.02 for r in res.per_segment_ratios)
    assert res.ablation_ratios and all(r <= 0.95 for r in res.ablation_ratios)
    assert gate.passed, gate.notes


def test_aggregate_win_holds_on_average_across_dgps():
    # Across diverse DGPs the flagship should win on average even if a couple of
    # degenerate regimes merely tie; this encodes the honest headline claim.
    res = run_benchmark(
        datasets=["uniform", "zipf", "high_card"],
        budgets_bytes=[1024],
        seeds=[1337, 2000],
        n=2000,
        domain=5000,
        delta=0.15,
    )
    mean_ratio = sum(res.ratios) / len(res.ratios)
    assert mean_ratio < 0.85  # beats the fixed baseline by >15% on average
    mean_ablation = sum(res.ablation_ratios) / len(res.ablation_ratios)
    assert mean_ablation < 0.95


def test_gate_report_flags_known_failures():
    # Ratios all above 1 -> flagship never beats baseline -> G1/G2 must fail.
    rep = evaluate_gates([1.2, 1.3], [1.0], [1.0], predicted_ratio=1.25, delta=0.15)
    assert not rep.g1_max_lt_1
    assert not rep.g2_mean_le_1_minus_delta
    assert not rep.passed


def test_determinism_canonicalization():
    a = {"x": 1, "generated_at": "t1", "rows": [{"score": 2.0, "deterministic": False}]}
    b = {"x": 1, "generated_at": "t2", "rows": [{"score": 9.9, "deterministic": False}]}
    # Volatile and stochastic fields are masked -> same canonical digest.
    assert digest(a) == digest(b)
    assert "<stochastic>" in canonicalize(a)


def test_g3_bounds_stochastic_values():
    assert check_g3({"deterministic": True, "value": 1e9}, safety=3.0)
    assert check_g3({"deterministic": False, "value": 2.0, "declared_bound": 1.0}, safety=3.0)
    assert not check_g3({"deterministic": False, "value": 5.0, "declared_bound": 1.0}, safety=3.0)
