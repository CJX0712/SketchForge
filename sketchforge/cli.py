"""SketchForge command-line interface.

Usage:
    python -m sketchforge info
    python -m sketchforge registry
    python -m sketchforge demo
    python -m sketchforge bench --n 2000 --budgets 1024 4096

Author: 晨星
"""

from __future__ import annotations

import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import argparse  # noqa: E402

from sketchforge.core.runtime import env_probe  # noqa: E402
from sketchforge.core.types import QueryType  # noqa: E402
from sketchforge.sketches.backend import backend_available  # noqa: E402
from sketchforge.sketches.registry import all_names, specs  # noqa: E402


def _cmd_info() -> int:
    probe = env_probe()
    print("SketchForge environment")
    for k, v in probe.items():
        print(f"  {k}: {v}")
    print(f"  tier0_backend(datasketches): {backend_available()}")
    return 0


def _cmd_registry() -> int:
    rows = specs()
    by_qt: dict[QueryType, list] = {}
    for spec in rows:
        for qt in spec.query_types:
            by_qt.setdefault(qt, []).append(spec)
    for qt in QueryType:
        items = by_qt.get(qt, [])
        print(f"[{qt.value}] ({len(items)} methods)")
        for spec in items:
            tier = "T0" if spec.tier == 0 else "T1"
            det = "det" if spec.deterministic else "sto"
            print(f"  - {spec.name:<26} tier={tier} {det} backend={spec.backend}")
    print(f"total registered classes: {len(all_names())}")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from examples.run_demo import run_demo  # type: ignore[import-not-found]

    run_demo(n=args.n, budget=args.budget, seed=args.seed)
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from sketchforge.pipeline.stages import run_benchmark

    datasets = args.datasets.split(",") if args.datasets else ["uniform", "zipf", "high_card"]
    budgets = [int(b) for b in args.budgets.split(",")] if args.budgets else [1024, 4096]
    seeds = [1000 + 7 * i for i in range(args.seeds)]
    res = run_benchmark(datasets=datasets, budgets_bytes=budgets, seeds=seeds, n=args.n, domain=args.domain, delta=args.delta)
    print(f"datasets={datasets} budgets={budgets} seeds={len(seeds)}")
    print(f"mean_ratio(flagship/static)   = {sum(res.ratios)/len(res.ratios):.4f}")
    print(f"mean_ablation(flagship/uniform) = {sum(res.ablation_ratios)/len(res.ablation_ratios):.4f}")
    print(f"mean_oracle(flagship/segopt)   = {sum(res.per_segment_ratios)/len(res.per_segment_ratios):.4f}")
    g = res.gate
    if g is not None:
        print(f"GATE passed = {g.passed}")
        for note in g.notes:
            print(f"  - {note}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sketchforge", description="Fixed-budget data-stream sketching toolkit")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="print environment and backend status")
    sub.add_parser("registry", help="list all registered sketch methods")

    d = sub.add_parser("demo", help="run a fast end-to-end demo (not gate evidence)")
    d.add_argument("--n", type=int, default=5000)
    d.add_argument("--budget", type=int, default=2048)
    d.add_argument("--seed", type=int, default=1337)

    b = sub.add_parser("bench", help="run the pre-registered gate benchmark")
    b.add_argument("--datasets", default="", help="comma-separated DGP kinds (default subset)")
    b.add_argument("--budgets", default="", help="comma-separated byte budgets")
    b.add_argument("--seeds", type=int, default=5, help="number of seeds (R)")
    b.add_argument("--n", type=int, default=2000)
    b.add_argument("--domain", type=int, default=50_000)
    b.add_argument("--delta", type=float, default=0.15)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "info":
        return _cmd_info()
    if args.cmd == "registry":
        return _cmd_registry()
    if args.cmd == "demo":
        return _cmd_demo(args)
    if args.cmd == "bench":
        return _cmd_bench(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
