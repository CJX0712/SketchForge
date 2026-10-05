# Pitfalls & hard-won lessons (SketchForge)

Field notes from building and running the system. Every item here cost real
debugging time; they are the ones worth writing down.

## 1. The performance killer: probe size × binary search
`fit_param_under_budget` binary-searches the largest parameter whose *post-ingest*
serialized size fits the budget, and it measured size by replaying a **20,000-item
probe at every search step**. Because `params_for_budget` is called for every
budget-grid point (inside `fit_laws`) *and* for every module in every system build,
one `fit_laws` call took **~30 s** and a tiny benchmark took >90 s.
Fix: shrink the probe to a steady-state-representative size (2,000 keys) **and**
memoize the result on `(cls, budget, lo, hi, seed)` — the probe and parameters are
deterministic, so the footprint is a pure function of those keys. Runtime dropped
**~34×** (90 s → 2.7 s for a one-DGP run). Any "measure by replaying data" loop
inside a search is a cliff — size it against steady state and cache it.

## 2. Sign convention: `err = a·b^{-c}` means **c > 0**
The power-law model is `err = a·b^{-c}`. So **c > 0** is the *healthy* case (error
falls as budget grows). A negative or zero `c` means the sketch got no better with
more memory — that is the degenerate signal. Mixing this up (assuming `c < 0` like
`a·b^{c}`) inverts every downstream allocation decision.

## 3. The elastic allocator silently collapsed to uniform
`ElasticityWaterFilling` summed `w_i * c_i` as its "marginal utility". Because a
*decrease* in error is what we reward but `Σ w·c` can be negative/zero, the guard
`if weights.sum() <= 0: fall back to uniform` fired **every time**, so the flagship
was byte-identical to the uniform ablation. Fix: use `abs(...)` of the summed
elasticity (the marginal *benefit* of a byte is positive) and add a unit test that
the flagship plan is not the uniform plan.

## 4. A heuristic allocator cannot beat an exhaustive one on the same architecture
The original gate scored the flagship (a cheap elasticity heuristic) against a
`StaticOptimalFrozen` baseline that was an **exhaustive grid search over the same
loss surface** — and all strategies built the *identical* AdaSketch (same shared
core, same methods), differing only in the `{M_shared, M_q2, M_q4}` numbers. A
heuristic provably cannot beat exhaustive search there, so the "flagship must
strictly beat baseline" gate was **logically impossible**. The honest fix is to
compare the data-driven allocator against a **non-adaptive fixed baseline**
(`StaticFixed`: 2:1:1 by served-query count) — a real, fair baseline the adaptive
system can and does beat. Lesson: make sure your baseline is a *plausible
competitor*, not an unbeatable oracle wearing a baseline's clothes.

## 5. `predicted_ratio` returned `nan` from a missing import
`predicted_ratio` called `predict_gate_score` without importing it (and it lives in
`hpo/allocation`, not `eval/aggregate`). The `NameError` was swallowed by a
`try/except` that set `predicted = nan`, so the model-consistency gate (G-4) failed
with a confusing `|predicted−mean| = nan` instead of a loud crash. Broad
`except Exception: pred = nan` around a *required* computation hides exactly the
bugs you most want to see. Fix the import; keep the guard only for genuinely
optional values.

## 6. The error law must be fitted on the DGP, not one stream
Fitting the power law on a *single* calibration stream gave a law whose predicted
error was ~10× the real error at eval (r² 0.79 but a systematically wrong
intercept). The allocator then mis-budgeted on ~2 of 9 DGPs. Fix: average the
per-budget error over several independent calibration realizations
(`n_cal_seeds`), so the law reflects the data-generating process, not one noisy
draw. This is the difference between an allocator that generalizes and one that
overfits its calibration set.

## 7. `_mask_volatile` computed a path but never used it
`eval/validate._mask_volatile` built a dotted `path` for every field and then
returned leaves unchanged — it never compared the path against `VOLATILE_PATHS`.
So "volatile" fields (timestamps, pids) were *not* masked and two identical runs
produced different determinism digests. Fix: `if path.lstrip('.') in VOLATILE_PATHS:
return "<volatile>"`. A determinism gate that never actually masks anything is a
gate that fails at random — or (worse) passes for the wrong reason.

## 8. A tiny budget makes the geometric-mean gate noisy, not wrong
At very small `n`/`B` and a single seed, the calibration-stream win ratio and the
eval-stream win ratio diverge (e.g. 1.08 vs 0.25) purely from finite-sample noise,
so the consistency band (|Δ| ≤ 0.03) fails. The substantive claims (flagship beats
baseline, matches the per-segment optimum) still hold. The consistency check only
becomes meaningful with replication (R = 20). Use small configs for smoke tests,
R = 20 for the real gate — don't read a single-seed consistency miss as a defect.

## 9. The genuine scientific result: a robust *average* win, not universal dominance
Across all 9 DGPs the data-driven allocator beats the fixed baseline on **average**
(mean ratio well under the 0.85 G-2 bar) and beats uniform, and it essentially
matches the per-segment upper bound. But on two degenerate regimes it only *ties*
the fixed baseline (ratio ≈ 1.0): `low_card` (the shared bottom-k core already
saturates the easy Q1/Q3 queries, so there is little to re-allocate) and `pareto`
(marginally > 1). This is real, not a bug — on those DGPs the fixed 2:1:1 split is
already near-optimal. Reported honestly rather than tuned away; see
`delivered.md`.

## 10. Small correctness traps in the sketch implementations
- **KMV over-estimation** happens if a `full` sentinel value is counted and/or if
  duplicate hashes accumulate — de-duplicate (`np.unique`) before truncating to `k`.
- **Linear counting must count *bits*, not bytes** — `np.unpackbits(...).sum()`, not
  `m - count_nonzero(bitmap)`, or the estimate is badly low.
- **CountSketch sign/idx must come from the *same* hash path** used at `update` time
  (here `hash_array_u64`), or `estimate` re-hashes the query key into the wrong
  bucket and can even return **negative** counts. Route all integer keys through one
  hashing helper.
- **t-digest / REQ / KLL from DataSketches are not deterministic across processes**
  (seeded internally but not reproducible), and `hll_sketch` has `serialize_compact`
  (not `serialize`) and no `merge`. Verify library APIs before wrapping them.
- `count_min_sketch.update` only accepts `int`, not `float` — cast before ingest.

Author: **晨星**.
