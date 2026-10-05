"""Probe DataSketches 5.2.0 real API behaviour (part 5)."""

from collections import Counter

import datasketches as ds
import numpy as np

print("--- frequent_strings inflation ordering ---")
rng = np.random.default_rng(3)
# 3 heavy items + long tail so the sketch must evict/compress
stream = ["a"] * 900 + ["b"] * 500 + ["c"] * 200 + [f"u{i}" for i in range(4000)]
rng.shuffle(stream)
cnt = Counter(stream)
fs = ds.frequent_strings_sketch(lg_max_k=7)
for s in stream:
    fs.update(s)
nfp = fs.get_frequent_items(ds.NO_FALSE_POSITIVES)[:5]
print("size", fs.get_serialized_size_bytes(), "epsilon", fs.epsilon, "N", fs.total_weight)
for row in nfp:
    print("row", row[0], "true", cnt[row[0]], "fields", row[1:])
print("est a", fs.get_estimate("a"), "lb", fs.get_lower_bound("a"), "ub", fs.get_upper_bound("a"))

print("--- hll bounds signature ---")
h = ds.hll_sketch(10)
for v in rng.integers(0, 5000, 5000):
    h.update(int(v))
print("est", h.get_estimate(), "lb", h.get_lower_bound(1), "ub", h.get_upper_bound(1))
print("rel_err", h.get_rel_err(), "is_compact", h.is_compact(), "empty", h.is_empty())
h2 = ds.hll_sketch.deserialize(h.serialize_compact())
print("roundtrip", h2.get_estimate(), "updatable", h2.get_updatable_serialization_bytes())

print("--- cpc bounds ---")
c = ds.cpc_sketch(lg_k=10, seed=7)
for v in rng.integers(0, 5000, 5000):
    c.update(int(v))
print("est", c.get_estimate(), "lb", c.get_lower_bound(1), "ub", c.get_upper_bound(1), "lg_k", c.lg_k, "empty", c.is_empty())

print("--- kll full ---")
k = ds.kll_floats_sketch(k=200)
vals = rng.integers(0, 10000, 5000).astype(np.float64)
k.update(vals)
print("q", k.get_quantile(0.5), "rank", k.get_rank(5000.0), "n", k.n, "retained", k.num_retained)
print("min", k.get_min_value(), "max", k.get_max_value(), "estimation_mode", k.is_estimation_mode())
print("nre", k.get_normalized_rank_error(200, False), ds.kll_floats_sketch.get_normalized_rank_error(200, True))
print("quantiles", k.get_quantiles([0.25, 0.75]))
k2 = ds.kll_floats_sketch.deserialize(k.serialize())
print("roundtrip q", k2.get_quantile(0.5), "n", k2.n)

print("--- tdigest ---")
t = ds.tdigest_double(k=100)
t.update(vals)
print("q", t.get_quantile(0.5), "rank", t.get_rank(5000.0), "size", t.get_serialized_size_bytes(), "weight", t.get_total_weight())
t2 = ds.tdigest_double.deserialize(t.serialize())
print("roundtrip", t2.get_quantile(0.5))

print("--- req ---")
r = ds.req_floats_sketch(k=6, is_hra=True)
r.update(vals)
print("q", r.get_quantile(0.5), "size", len(r.serialize()), "n", r.n, "retained", r.num_retained, "hra", r.is_hra)
print("RSE", ds.req_floats_sketch.get_RSE(6, 0.5, True, r.n))
r2 = ds.req_floats_sketch.deserialize(r.serialize())
print("roundtrip", r2.get_quantile(0.5))

print("--- cms ---")
cm = ds.count_min_sketch(4, 1024, 9001)
for v in rng.integers(0, 500, 5000).astype(np.int64):
    cm.update(int(v))
print("est", cm.get_estimate(7), "size", cm.get_serialized_size_bytes(), "nh", cm.num_hashes, "nb", cm.num_buckets, "seed", cm.seed)
print("lb", cm.get_lower_bound(7), "ub", cm.get_upper_bound(7), "rel", cm.get_relative_error(), "total", cm.total_weight)
cm2 = ds.count_min_sketch.deserialize(cm.serialize())
print("roundtrip", cm2.get_estimate(7))
cm3 = ds.count_min_sketch(4, 1024, 9001)
cm4 = ds.count_min_sketch(4, 1024, 9001)
for v in rng.integers(0, 500, 100).astype(np.int64):
    cm3.update(int(v))
for v in rng.integers(500, 1000, 100).astype(np.int64):
    cm4.update(int(v))
cm3.merge(cm4)
print("cms merge", cm3.get_estimate(7), cm3.total_weight)
