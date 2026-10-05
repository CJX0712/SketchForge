# SketchForge

**Fixed-budget data-stream sketching with data-driven budget allocation.**

SketchForge answers four query families over a single streaming pass under a *hard,
pre-declared memory budget* measured in **serialized bytes** — and it decides how to
split that budget across the queries from the data itself.

| Family | Query | Tier-1 methods | Tier-0 (DataSketches) |
|---|---|---|---|
| Q1 | distinct count (cardinality) | HyperLogLog, KMV, LinearCounting, Exact | HLL, CPC |
| Q2 | quantiles / rank | Reservoir, t-Digest, Histogram, Exact | KLL, REQ |
| Q3 | point frequency | Count-Min, Conservative-Update, CountSketch, Exact | Count-Min |
| Q4 | heavy hitters (top-k) | Space-Saving, Misra-Gries, LossyCounting, CM+Heap, Exact | Frequent-Strings |

The flagship system **AdaSketch** routes Q1 *and* Q3 through a single shared
*bottom-k* (min-hash) core, so one data structure serves two error curves, and
splits the total byte budget across its three modules using a power-law error model
fitted on a disjoint calibration stream.

---

## Why this exists

Most sketching benchmarks answer one query type at a time, hand each sketch its own
parameter, and then quietly compare systems that did not spend the same memory.
SketchForge is built around three harder constraints:

1. **Byte-honest budgets.** The budget currency is `len(serialize())`. A declared
   budget that differs from the measured footprint by more than 2% is a hard error
   (`E302`), not a footnote.
2. **Non-vacuous evaluation.** Any ratio audit whose denominator is a constant shared
   by every compared system is automatically flagged vacuous and forced to FAIL
   (architecture §10.1). The evaluation refuses to certify its own success.
3. **Determinism.** Three independent determinism gates: exact re-run digests
   (volatile + stochastic fields masked), canonical-form equality, and a bound check
   that every non-deterministic value respects its library-declared error bound.

## Install

```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[tier0,dev]"
# or, for the locked reference environment:
pip install -r requirements.lock.txt
```

Requires Python ≥ 3.12. `datasketches` is optional (Tier-0 comparison backends);
the Tier-1 numpy implementations run without it.

## Quick start

```bash
python -m sketchforge info                 # environment + backend status
python -m sketchforge registry             # every registered sketch method
python examples/run_demo.py                # fast end-to-end demo (seconds)

# the pre-registered gate benchmark (all DGPs, R seeds) -> outputs/gate_result.json
python benchmarks/gate_benchmark.py --out outputs/gate_result.json
```

Library use:

```python
from sketchforge.data.specs import StreamSpec
from sketchforge.data.streams import generate
from sketchforge.data.oracle import ExactOracle
from sketchforge.sketches.cardinality import HyperLogLogNumpy
from sketchforge.core.types import Query, QueryType

stream = generate(StreamSpec(kind="zipf", n=100_000, domain=1_000_000, seed=42))
hll = HyperLogLogNumpy(p=12, seed=42)
hll.update_batch(stream.keys)
est = hll.estimate(Query(QueryType.CARDINALITY)).value
truth = ExactOracle(stream).cardinality()
print(f"{est:.0f} vs {truth:.0f}  ({hll.memory_bytes()} bytes)")
```

## How the flagship allocates

```
                    total budget B (bytes)
                              |
        +---------------------+---------------------+
        |                     |                     |
   M_shared                 M_q2                  M_q4
  bottom-k core        quantile module      heavy-hitter module
  serves Q1 + Q3
        |
   split chosen by a power-law error model fitted on a
   disjoint calibration stream (never the eval stream)
```

`M_shared` earns a larger share because its marginal utility is the *sum* of the
elasticities of the two queries it serves — one structure, two error curves.

## Evaluation & the superiority gate

Every configuration is scored by one shared entry point, so "allocation" can never
be conflated with "method selection". The flagship (data-driven optimal allocation)
is compared against a **non-adaptive fixed baseline** (`StaticFixed`: bytes split
2:1:1 by how many queries each module serves), a **uniform ablation**, and the
**per-segment optimum** (an unrealizable upper bound). Errors are aggregated as a
*geometric mean of tolerance-normalized* errors, which is invariant to the choice of
normalization `ε`.

Pre-registered gates (frozen, not tuned to results):

| Gate | Requirement |
|---|---|
| G-1 | every run: `G(flagship)/G(baseline) < 1.0` |
| G-2 | mean ratio `≤ 1 − δ` (δ = 0.15) |
| G-3 | `G(flagship)/G(per-segment) ≤ 1.02` (matches the upper bound) |
| G-4 | calibration-vs-eval consistency of the win ratio (leakage / generalization) |
| ablation | `G(flagship)/G(uniform) ≤ 0.95` |

See `docs/architecture.md` and `docs/model_card.md` for the full design, the frozen
gate definitions, the non-vacuity audits, and the as-built results. The measured
outcome — including any gate that does **not** pass — is reported in
`references/delivered.md` and the raw numbers in `outputs/gate_result.json`.

## Reproducibility & tests

```bash
make test        # pytest
make lint        # ruff
make gate        # full pre-registered benchmark
```

Determinism is enforced in-process (seeded `SeedState` derivation, blake2b hashing —
never Python's salted `hash()`), and across runs via the masked canonical digest.

## Project layout

```
sketchforge/core/       contracts, seeding, hashing, config, errors
sketchforge/data/       stream generators, exact oracle, loaders
sketchforge/sketches/   Tier-1 numpy + Tier-0 datasketches adapters, registry, AdaSketch
sketchforge/training/   closed-form power-law error-law fitting
sketchforge/eval/       metrics, geometric aggregation, determinism + gate validation
sketchforge/hpo/        allocation strategies + grid-search solver
sketchforge/pipeline/   the single shared evaluation entry point
sketchforge/audits/     non-vacuity / fairness audits (§10)
benchmarks/             the pre-registered gate benchmark
examples/               fast demo (not gate evidence)
docs/                   architecture + model card (the design source of truth)
```

## License

Apache-2.0. Author: **晨星**.
