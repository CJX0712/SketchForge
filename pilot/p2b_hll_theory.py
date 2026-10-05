"""pilot §B-2b: 交叉验证 HLL 的 1.04/sqrt(m) 常数。

p2 实测 DataSketches HLL 的 relSD 只有理论值的 0.36~0.61 倍，I-4 全线 FAIL。
两种可能：(a) 1.04 常数记错；(b) DataSketches 用了比 Flajolet 原始估计量更强的估计量。
判别方法：自己实现一个 **vanilla HLL**（Flajolet 2007 原始估计量，无偏差修正），
测其 SD，与理论对照；再与 DataSketches 同参数对照。

同时修正 p2 的一处不公平：DataSketches 默认可能是 HLL_8（1 字节/寄存器），
而工业默认应为 HLL_4。这里显式测 HLL_4 / HLL_6 / HLL_8 / CPC。
"""
import sys
import time

import datasketches as ds
import dgp
import numpy as np

MASK = (1 << 64) - 1


def splitmix64(x, seed):
    z = (np.asarray(x, dtype=np.uint64) + np.uint64(seed)) & np.uint64(MASK)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(MASK)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(MASK)
    return (z ^ (z >> np.uint64(31))) & np.uint64(MASK)


def nlz64(x):
    """前导零个数（对 uint64 numpy 数组），值域 [0,64]。"""
    x = np.asarray(x, dtype=np.uint64)
    n = np.zeros(x.shape, dtype=np.int64)
    nz = x != 0
    for shift in (32, 16, 8, 4, 2, 1):
        hi = x >> np.uint64(shift)
        take = nz & (hi != 0)
        n[take] += shift
        x = np.where(take, hi, x)
    n[~nz] = 64
    return n


def alpha_m(m):
    if m == 16:
        return 0.673
    if m == 32:
        return 0.697
    if m == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / m)


class VanillaHLL:
    """Flajolet 2007 原始估计量：小范围用 Linear Counting，大范围修正，无经验偏差修正表。"""

    def __init__(self, lg_k, seed=0):
        self.m = 1 << lg_k
        self.b = lg_k
        self.seed = seed
        self.M = np.zeros(self.m, dtype=np.int64)

    def update(self, keys):
        h = splitmix64(np.asarray(keys, dtype=np.uint64), self.seed)
        idx = (h & np.uint64(self.m - 1)).astype(np.int64)
        rho = nlz64(h >> np.uint64(self.b)) + 1
        np.maximum.at(self.M, idx, rho)

    def estimate(self):
        m = self.m
        s = np.sum(np.exp2(-self.M.astype(np.float64)))
        E = alpha_m(m) * m * m / s
        # 小范围修正（Flajolet 2007 eq.）
        if 2.5 * m >= E:
            V = int(np.sum(self.M == 0))
            if V != 0:
                E = m * np.log(m / V)
        # 大范围修正
        if E > (1 << 32) / 30.0:
            E = -(1 << 32) * np.log(1.0 - E / (1 << 32))
        return float(E)


def ds_build(kind, lg_k, lst):
    if kind == "cpc":
        sk = ds.cpc_sketch(lg_k)
        for v in lst:
            sk.update(v)
        return sk.get_estimate(), len(sk.serialize()), None
    tt = {"hll_4": ds.HLL_4, "hll_6": ds.HLL_6, "hll_8": ds.HLL_8}[kind]
    sk = ds.hll_sketch(lg_k, tt)
    for v in lst:
        sk.update(v)
    return (sk.get_estimate(), sk.get_compact_serialization_bytes(),
            sk.get_rel_err(False, False, lg_k, 1))


def main():
    t0 = time.perf_counter()
    R = 20
    N = 100_000
    LGKS = [6, 8, 10, 12, 14]
    KINDS = ["hll_4", "hll_6", "hll_8", "cpc"]

    print(f"[p2b] HLL 理论常数交叉验证   R={R} n={N} lg_k={LGKS}")
    print("  vanilla = 自研 Flajolet 2007 原始估计量（numpy）")
    print("  ds_*    = DataSketches 实现（含偏差修正）")
    print()
    print(f"{'lg_k':>5}{'m':>7}{'theory':>10}{'vanillaSD':>11}{'van/theory':>12}"
          f"{'ds_hll4':>11}{'4/theory':>10}{'ds_hll8':>11}{'8/theory':>10}{'ds_cpc':>11}{'cpc/theory':>12}")
    print("-" * 121)

    streams = [dgp.perm_unique(N, dgp.spawn(s)[0]) for s in range(R)]
    lsts = [s.tolist() for s in streams]
    van = {}
    dsres = {}
    for lg_k in LGKS:
        m = 1 << lg_k
        th = 1.04 / np.sqrt(m)
        # vanilla
        ests = []
        for i, s in enumerate(streams):
            h = VanillaHLL(lg_k, seed=1000 + i)
            h.update(s)
            ests.append(h.estimate())
        ests = np.array(ests)
        vsd = float(ests.std(ddof=1) / N)
        van[lg_k] = vsd
        row = {}
        for kind in KINDS:
            e = np.array([ds_build(kind, lg_k, lst)[0] for lst in lsts])
            row[kind] = float(e.std(ddof=1) / N)
            dsres[(lg_k, kind)] = row[kind]
        print(f"{lg_k:>5}{m:>7}{th:>10.6f}{vsd:>11.6f}{vsd/th:>12.4f}"
              f"{row['hll_4']:>11.6f}{row['hll_4']/th:>10.4f}"
              f"{row['hll_8']:>11.6f}{row['hll_8']/th:>10.4f}"
              f"{row['cpc']:>11.6f}{row['cpc']/th:>12.4f}")

    print()
    print("[p2b] --- 结论判别 ---")
    r_van = np.array([van[k] / (1.04 / np.sqrt(1 << k)) for k in LGKS])
    r_4 = np.array([dsres[(k, "hll_4")] / (1.04 / np.sqrt(1 << k)) for k in LGKS])
    r_cpc = np.array([dsres[(k, "cpc")] / (1.04 / np.sqrt(1 << k)) for k in LGKS])
    print(f"  vanilla / theory : mean={r_van.mean():.4f}  range=[{r_van.min():.4f}, {r_van.max():.4f}]")
    print(f"  ds_hll4 / theory : mean={r_4.mean():.4f}  range=[{r_4.min():.4f}, {r_4.max():.4f}]")
    print(f"  ds_cpc  / theory : mean={r_cpc.mean():.4f}  range=[{r_cpc.min():.4f}, {r_cpc.max():.4f}]")
    print()
    print("  若 vanilla/theory ≈ 1 而 ds/theory ≪ 1  => 1.04 常数正确，")
    print("     DataSketches 的估计量确实强于原始 HLL（偏差修正 + 更强估计量）。")

    print()
    print("[p2b] --- DataSketches 自报 rel_err vs 实测（lg_k, num_std_devs=1） ---")
    print(f"{'lg_k':>5}{'ds_rel_err':>12}{'measured_SD_hll4':>20}{'ratio':>9}")
    for lg_k in LGKS:
        re = ds_build("hll_4", lg_k, lsts[0])[2]
        print(f"{lg_k:>5}{re:>12.6f}{dsres[(lg_k,'hll_4')]:>20.6f}"
              f"{dsres[(lg_k,'hll_4')]/re:>9.4f}")

    print()
    print("[p2b] --- 等误差口径字节数（HLL_4 vs CPC），log-log 插值 ---")
    print(f"{'target_SD':>11}{'bytes_hll4':>12}{'bytes_cpc':>11}{'cpc/hll4':>10}")
    small_lg = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    b4, bc = {}, {}
    for lg_k in small_lg:
        ests = np.array([ds_build("hll_4", lg_k, lst)[0] for lst in lsts])
        b4[lg_k] = (ests.std(ddof=1) / N, ds_build("hll_4", lg_k, lsts[0])[1])
        ests = np.array([ds_build("cpc", lg_k, lst)[0] for lst in lsts])
        bc[lg_k] = (ests.std(ddof=1) / N, ds_build("cpc", lg_k, lsts[0])[1])
    for tgt in [0.05, 0.02, 0.01, 0.005]:
        def bf(d):
            xs = np.array([d[k][0] for k in small_lg])
            bs = np.array([d[k][1] for k in small_lg], dtype=float)
            o = np.argsort(xs)
            xs, bs = xs[o], bs[o]
            if tgt < xs.min() or tgt > xs.max():
                return np.nan
            return float(np.exp(np.interp(np.log(tgt), np.log(xs), np.log(bs))))
        a, b = bf(b4), bf(bc)
        if np.isnan(a) or np.isnan(b):
            print(f"{tgt:>11.4f}{'o-o-r':>12}{'o-o-r':>11}{'-':>10}")
        else:
            print(f"{tgt:>11.4f}{a:>12.0f}{b:>11.0f}{b/a:>10.4f}")

    print()
    print(f"[p2b] done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
