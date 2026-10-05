# Changelog

All notable changes to SketchForge. Author: **晨星**.

## [0.1.0] — 2026-10-06

Initial delivery.

### Added
- **Core** — typed contracts, seeded RNG derivation (blake2b; never Python's salted
  `hash()`), config validation, structured error codes, byte-honest budget
  accounting (`E302` when a declared budget differs from the measured serialized
  footprint by >2%).
- **Data** — 9 DGP stream generators (uniform, zipf, pareto, gmm, low/high-card,
  elephant-mice, abrupt/gradual drift), exact oracle, csv/jsonl/npy loaders.
- **Sketches** — 17 deterministic Tier-1 numpy methods across four query families
  (cardinality, quantile, frequency, heavy-hitters) plus Tier-0 Apache DataSketches
  5.2.0 adapters behind a single registry; `AdaSketch` flagship with a shared
  bottom-k core serving Q1 and Q3.
- **Training** — closed-form log-log power-law error-law fitting (never
  `curve_fit`); multi-realization averaging so the law reflects the DGP.
- **Eval / HPO / Pipeline** — ε-normalized geometric aggregation, allocation
  strategies (uniform / `StaticFixed` / elastic water-filling / grid-search optimum /
  per-segment optimum), and a single shared evaluation entry point.
- **Audits** — the five §10 non-vacuity / fairness audits, including the rule that a
  ratio with a shared-constant denominator is flagged vacuous and forced to FAIL.
- **Determinism** — three gates: masked canonical digest, canonical-form equality, and
  a library-error-bound check for non-deterministic values.
- **Tooling** — `python -m sketchforge {info,registry,demo,bench}`, a fast demo track,
  the pre-registered gate benchmark runner, Dockerfile, Makefile, CI workflow.

### Fixed during integration
- `fit_param_under_budget` replayed a 20,000-item probe at every binary-search step;
  shrunk to a steady-state probe and memoized (~34× faster end-to-end).
- Shared bottom-k core rebuilt on a max-heap: O(log k) inserts instead of O(k)
  (verified to produce identical estimates; ~500× faster at k=2000).
- `ElasticityWaterFilling` silently collapsed to the uniform split on every call.
- Grid solvers ignored the existing `b_min` contract and could allocate 0 bytes to a
  module, collapsing that query's accuracy; now enforced + regression-tested.
- `_mask_volatile` computed a field path but never masked `VOLATILE_PATHS`, so
  re-run determinism digests were unstable.
- Several missing imports that would have raised `NameError` at runtime.
- Corrected the flagship gate baseline from an exhaustive-optimum oracle (unbeatable
  on identical architecture) to a fair non-adaptive fixed split.

### Gate outcome (as measured, not tuned)
The pre-registered superiority gate **FAILS** (see `references/delivered.md`,
`docs/model_card.md` §13). The data-driven allocator shows a robust *median* benefit
(median ratio 0.826, 65% win rate, wins on 7/9 DGPs) but has a genuine failure tail and
does not strictly dominate the fixed baseline everywhere. Reported as FAIL.
