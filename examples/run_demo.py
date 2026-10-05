"""Fast end-to-end demo of SketchForge.

This is the *demo track*, NOT gate evidence: it uses a small N and few replications
so it runs in seconds. The pre-registered gate is produced by
``benchmarks/gate_benchmark.py`` (R=20, larger N).

    python examples/run_demo.py

Author: 晨星
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sketchforge.core.types import Query, QueryType  # noqa: E402
from sketchforge.data.oracle import ExactOracle  # noqa: E402
from sketchforge.data.specs import StreamSpec  # noqa: E402
from sketchforge.data.streams import generate  # noqa: E402
from sketchforge.hpo.allocation import SimplexGridSearch, StaticFixed, UniformStrategy  # noqa: E402
from sketchforge.pipeline.stages import (  # noqa: E402
    build_context,
    evaluate_allocation,
    fit_laws,
)
from sketchforge.sketches.adasketch import _BottomKCore  # noqa: E402
from sketchforge.sketches.heavyhitters import SpaceSaving  # noqa: E402
from sketchforge.sketches.quantile import TDigestNumpy  # noqa: E402

BADGE = "DEMO TRACK | R=2 | N small | NOT A GATE EVIDENCE"


def run_demo(n: int = 5000, budget: int = 2048, seed: int = 1337) -> None:
    print("=" * 64)
    print(f"SketchForge demo -- {BADGE}")
    print("=" * 64)

    methods = {
        QueryType.CARDINALITY: _BottomKCore,
        QueryType.FREQUENCY: _BottomKCore,
        QueryType.QUANTILE: TDigestNumpy,
        QueryType.HEAVY_HITTERS: SpaceSaving,
    }

    for kind in ("uniform", "zipf", "high_card"):
        calib = generate(StreamSpec(kind=kind, n=n, domain=50_000, seed=seed))
        laws = fit_laws(calib, ExactOracle(calib), budget, seed, n_cal_seeds=2)
        ctx = build_context(laws, methods)
        ev = generate(StreamSpec(kind=kind, n=n, domain=50_000, seed=seed + 7_000_003))
        ore = ExactOracle(ev)

        fs = evaluate_allocation(SimplexGridSearch(), laws, ctx, budget, ev, ore, seed)
        sf = evaluate_allocation(StaticFixed(), laws, ctx, budget, ev, ore, seed)
        uf = evaluate_allocation(UniformStrategy(), laws, ctx, budget, ev, ore, seed)

        ratio = fs["gate_score"] / sf["gate_score"] if sf["gate_score"] > 0 else float("nan")
        print(f"\n[{kind}] budget={budget}B  distinct={ore.cardinality():.0f}")
        print(f"  flagship (data-driven opt)  G={fs['gate_score']:.3f}  plan={fs['plan']}  bytes={fs['memory_bytes']}")
        print(f"  baseline (fixed, non-adapt) G={sf['gate_score']:.3f}  plan={sf['plan']}  bytes={sf['memory_bytes']}")
        print(f"  ablation (uniform)          G={uf['gate_score']:.3f}  plan={uf['plan']}  bytes={uf['memory_bytes']}")
        print(f"  flagship/baseline ratio = {ratio:.3f}  (lower is better)")

        # Show a single raw estimate vs truth for transparency.
        from sketchforge.sketches.adasketch import AdaSketch

        sysd = AdaSketch(budget, SimplexGridSearch(), laws, ctx, seed=seed)
        sysd.update(ev)
        card = sysd.estimate(Query(QueryType.CARDINALITY)).value
        hk = ore.top_k(1)[0][0]
        freq = sysd.estimate(Query(QueryType.FREQUENCY, key=hk)).value
        print(f"  sanity: cardinality est={card:.1f} truth={ore.cardinality():.1f} | top-key freq est={freq:.0f} truth={ore.frequency(hk):.0f}")

    print("\nDone. For gate evidence run benchmarks/gate_benchmark.py (R=20).")


if __name__ == "__main__":
    run_demo()
