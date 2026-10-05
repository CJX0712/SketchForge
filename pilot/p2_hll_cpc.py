"""pilot §B-2: HLL / CPC 相对误差与 SD vs m；对照理论 1.04/sqrt(m)；
并给出 CPC 相对 HLL 的实测空间节省比（等误差口径）。

要点：序列化字节必须在 ingest 之后测（HLL 稀疏/CPC 压缩，字节随基数增长）。
"""
import sys
import time

import datasketches as ds
import dgp
import numpy as np

LGKS = [8, 9, 10, 11, 12, 13, 14]
# (n, R)：n=1e6 时降 R 以控时间；R 逐档明示，不做隐式统一
BLOCKS = [(10_000, 20), (100_000, 20), (1_000_000, 8)]


def build(kind, lg_k, lst):
    if kind == "hll":
        sk = ds.hll_sketch(lg_k)
        for v in lst:
            sk.update(v)
        return sk, sk.get_estimate(), len(sk.serialize_updatable()), \
            sk.get_compact_serialization_bytes(), str(sk.tgt_type)
    sk = ds.cpc_sketch(lg_k)
    for v in lst:
        sk.update(v)
    b = len(sk.serialize())
    return sk, sk.get_estimate(), b, b, "cpc"


def main():
    t0 = time.perf_counter()
    print(f"[p2] HLL/CPC calibration  blocks(n,R)={BLOCKS}  lg_k={LGKS}")
    print("     真基数精确已知（0..n-1 的置换）；relSD = SD(est)/n；theory = 1.04/sqrt(m)")
    print()
    print(f"{'n':>9}{'R':>4}{'lg_k':>6}{'m':>7}{'kind':>5}{'relbias':>10}{'relSD':>11}"
          f"{'theory':>10}{'SD/theory':>11}{'b_upd':>8}{'b_cmp':>8}")
    print("-" * 89)
    table = {}
    for n, R in BLOCKS:
        lsts = [dgp.perm_unique(n, dgp.spawn(s)[0]).tolist() for s in range(R)]
        for lg_k in LGKS:
            m = 2 ** lg_k
            for kind in ("hll", "cpc"):
                ests = []
                bu = bc = None
                for lst in lsts:
                    _, e, u, c, _ = build(kind, lg_k, lst)
                    ests.append(e)
                    bu, bc = u, c
                ests = np.array(ests)
                rel = ests / n - 1.0
                bias = float(rel.mean())
                sd = float(ests.std(ddof=1) / n)
                th = 1.04 / np.sqrt(m)
                table[(n, lg_k, kind)] = dict(bias=bias, sd=sd, bu=bu, bc=bc,
                                              theory=th, R=R)
                print(f"{n:>9}{R:>4}{lg_k:>6}{m:>7}{kind:>5}{bias:>+10.5f}{sd:>11.6f}"
                      f"{th:>10.6f}{sd/th:>11.4f}{bu:>8}{bc:>8}")

    print()
    print("[p2] --- I-4 判定：SD/theory ∈ [0.85,1.15]，仅对 rho=n/m ≥ 2.5 的渐近区 ---")
    print(f"{'n':>9}{'lg_k':>6}{'kind':>5}{'rho':>9}{'SD/theory':>11}  verdict")
    n_pass = n_fail = 0
    for (n, lg_k, kind), d in sorted(table.items()):
        rho = n / (2 ** lg_k)
        ratio = d["sd"] / d["theory"]
        if rho < 2.5:
            print(f"{n:>9}{lg_k:>6}{kind:>5}{rho:>9.2f}{ratio:>11.4f}  n/a (rho<2.5)")
            continue
        ok = 0.85 <= ratio <= 1.15
        n_pass += ok
        n_fail += (not ok)
        print(f"{n:>9}{lg_k:>6}{kind:>5}{rho:>9.2f}{ratio:>11.4f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  => PASS={n_pass} FAIL={n_fail}")

    print()
    print("[p2] --- I-18 标度律（基数族要求 expo ∈ [-0.55,-0.45]，SD ∝ m^-0.5） ---")
    for n, R in BLOCKS:
        for kind in ("hll", "cpc"):
            sds = np.array([table[(n, lgk, kind)]["sd"] for lgk in LGKS])
            ms = np.array([2.0 ** lgk for lgk in LGKS], dtype=float)
            b, a = np.polyfit(np.log(ms), np.log(sds), 1)
            pred = b * np.log(ms) + a
            r2 = 1 - np.sum((np.log(sds) - pred) ** 2) / np.sum(
                (np.log(sds) - np.log(sds).mean()) ** 2)
            ok = (-0.55 <= b <= -0.45) and r2 >= 0.95
            print(f"  n={n:<9} {kind:<4} expo={b:+.4f} R2={r2:.4f} -> {'PASS' if ok else 'FAIL'}")

    print()
    print("[p2] --- CPC vs HLL 等误差口径空间节省比（compact 字节，log-log 插值） ---")
    print(f"{'n':>9}{'target_SD':>12}{'bytes_hll':>11}{'bytes_cpc':>11}{'cpc/hll':>10}")
    for n, R in BLOCKS:
        for tgt in [0.05, 0.02, 0.01]:
            def bf(kind):
                xs = np.array([table[(n, lgk, kind)]["sd"] for lgk in LGKS])
                bs = np.array([table[(n, lgk, kind)]["bc"] for lgk in LGKS], dtype=float)
                o = np.argsort(xs)
                xs, bs = xs[o], bs[o]
                if tgt < xs.min() or tgt > xs.max():
                    return np.nan
                return float(np.exp(np.interp(np.log(tgt), np.log(xs), np.log(bs))))
            bh, bc_ = bf("hll"), bf("cpc")
            if np.isnan(bh) or np.isnan(bc_):
                print(f"{n:>9}{tgt:>12.4f}{'o-o-r':>11}{'o-o-r':>11}{'-':>10}")
            else:
                print(f"{n:>9}{tgt:>12.4f}{bh:>11.0f}{bc_:>11.0f}{bc_/bh:>10.4f}")

    print()
    print(f"[p2] done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
