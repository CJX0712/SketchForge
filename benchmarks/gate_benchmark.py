"""Full pre-registered gate benchmark for SketchForge.

Runs run_benchmark across all DGPs, several budget tiers and R=20 seeds, then
writes a machine-readable JSON result and a human summary. Invoked by the
delivery pipeline; also runnable directly:

    python benchmarks/gate_benchmark.py --out outputs/gate_result.json

Author: 晨星
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from sketchforge.pipeline.stages import run_benchmark

DATASETS = [
    "uniform", "zipf", "pareto", "gmm", "low_card",
    "high_card", "elephant_mice", "drift_abrupt", "drift_gradual",
]
BUDGETS = [1024, 4096]
SEEDS = [1000 + 7 * i for i in range(20)]  # R = 20
N = 3000
DOMAIN = 50_000
DELTA = 0.15
N_CAL_SEEDS = 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/gate_result.json")
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--domain", type=int, default=DOMAIN)
    ap.add_argument("--ncal", type=int, default=N_CAL_SEEDS)
    args = ap.parse_args()

    t0 = time.time()
    result = run_benchmark(
        datasets=DATASETS,
        budgets_bytes=BUDGETS,
        seeds=SEEDS,
        n=args.n,
        domain=args.domain,
        delta=DELTA,
        n_cal_seeds=args.ncal,
    )
    elapsed = time.time() - t0

    payload = {
        "config": {
            "datasets": DATASETS,
            "budgets_bytes": BUDGETS,
            "n_seeds": len(SEEDS),
            "n": args.n,
            "domain": args.domain,
            "delta": DELTA,
            "n_cal_seeds": args.ncal,
            "elapsed_sec": round(elapsed, 1),
        },
        "gate_passed": result.gate.passed if result.gate else None,
        "gate": result.gate.__dict__ if result.gate else None,
        "mean_ratio": round(sum(result.ratios) / len(result.ratios), 4) if result.ratios else None,
        "mean_ablation": round(sum(result.ablation_ratios) / len(result.ablation_ratios), 4) if result.ablation_ratios else None,
        "mean_per_segment": round(sum(result.per_segment_ratios) / len(result.per_segment_ratios), 4) if result.per_segment_ratios else None,
        "predicted_ratio": result.predicted_ratio,
        "n_runs": len(result.per_run),
        "per_run": [
            {
                "dataset": r["dataset"],
                "budget": r["budget"],
                "seed": r["seed"],
                "ratio": r["ratio"],
                "ablation": r["ablation"],
                "oracle_ratio": r["oracle_ratio"],
                "predicted": r.get("predicted_ratio"),
                "flagship_g": r["flagship"]["gate_score"],
                "static_fixed_g": r["static_fixed"]["gate_score"],
                "uniform_g": r["uniform"]["gate_score"],
                "per_segment_g": r["per_segment"]["gate_score"],
                "flagship_plan": r["flagship"]["plan"],
            }
            for r in result.per_run
        ],
    }

    out_path = args.out
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"elapsed={elapsed:.1f}s runs={len(result.per_run)}", file=sys.stderr)
    print(f"gate_passed={payload['gate_passed']}", file=sys.stderr)
    print(f"mean_ratio={payload['mean_ratio']} mean_ablation={payload['mean_ablation']}", file=sys.stderr)
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
