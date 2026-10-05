"""pilot §B-1: KLL 归一化秩误差 vs k 的实测常数 c_KLL。

标定目标：err(k) ≈ c_KLL / k 中的 c_KLL；并验证标度指数 ≈ -1（不变量 I-18）。
真值：全量排序 + np.searchsorted（暴力 oracle）。
"""

import sys
import time

import datasketches as ds
import dgp
import numpy as np

R = 20  # seeds
N = 50_000  # 流长
KS = [50, 100, 200, 400, 800]
DGPS = ["uniform", "pareto_a1", "pareto_a3", "gauss_mix5"]


def make(seed, name):
    g, _, _ = dgp.spawn(seed)
    if name == "uniform":
        return g.random(N)  # 连续均匀，便于取分位
    if name == "pareto_a1":
        return dgp.pareto(N, 1.0, g)
    if name == "pareto_a3":
        return dgp.pareto(N, 3.0, g)
    if name == "gauss_mix5":
        return dgp.gauss_mix(N, 5, 1.0, g)
    raise ValueError(name)


def main():
    t0 = time.perf_counter()
    print(f"[p1] KLL calibration  R={R} N={N} k={KS} dgp={DGPS}")
    print("[p1] note: err = mean over P-grid of |rank/N - p|;  err_max = max over seeds&P")
    print()
    hdr = (
        f"{'dgp':<12}{'k':>6}{'err_mean':>12}{'err_p95':>12}{'err_max':>12}"
        f"{'err*k':>10}{'ds_nre':>10}{'retained':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for name in DGPS:
        for k in KS:
            errs = []
            retained = None
            ds_nre = None
            for seed in range(R):
                stream = make(seed, name)
                sk = ds.kll_floats_sketch(k)
                for v in stream:
                    sk.update(float(v))
                srt = np.sort(stream)
                e_p = []
                for p in dgp.P_GRID:
                    qh = sk.get_quantile(p)
                    e_p.append(abs(dgp.normalized_rank_error(srt, qh) - p))
                errs.append(e_p)
                if retained is None:
                    retained = sk.num_retained
                    ds_nre = sk.get_normalized_rank_error(k, False)
            E = np.array(errs)  # (R, |P|)
            err_mean = E.mean()
            err_p95 = np.quantile(E, 0.95)
            err_max = E.max()
            results[(name, k)] = dict(
                err_mean=err_mean,
                err_p95=err_p95,
                err_max=err_max,
                retained=retained,
                ds_nre=ds_nre,
            )
            print(
                f"{name:<12}{k:>6}{err_mean:>12.6f}{err_p95:>12.6f}"
                f"{err_max:>12.6f}{err_mean * k:>10.4f}{ds_nre:>10.6f}"
                f"{retained:>10}"
            )

    print()
    print("[p1] --- c_KLL 拟合（err_mean = c / k，log-log 回归） ---")
    print(
        f"{'dgp':<12}{'c_mean':>10}{'c_p95':>10}{'c_max':>10}"
        f"{'expo_mean':>11}{'expo_max':>10}{'R2_mean':>9}"
    )
    ks = np.array(KS, dtype=float)
    for name in DGPS:
        em = np.array([results[(name, k)]["err_mean"] for k in KS])
        ep = np.array([results[(name, k)]["err_p95"] for k in KS])
        ex = np.array([results[(name, k)]["err_max"] for k in KS])
        lk = np.log(ks)
        out = []
        for arr in (em, ep, ex):
            b, a = np.polyfit(lk, np.log(arr), 1)  # log err = b*log k + a
            pred = b * lk + a
            ss_res = np.sum((np.log(arr) - pred) ** 2)
            ss_tot = np.sum((np.log(arr) - np.log(arr).mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            out.append((np.exp(a), b, r2))  # c = exp(a)
        print(
            f"{name:<12}{out[0][0]:>10.4f}{out[1][0]:>10.4f}{out[2][0]:>10.4f}"
            f"{out[0][1]:>11.4f}{out[2][1]:>10.4f}{out[0][2]:>9.4f}"
        )

    print()
    print("[p1] --- I-18 标度律判定（分位族要求 expo ∈ [-1.15,-0.85], R2 ≥ 0.95） ---")
    for name in DGPS:
        em = np.array([results[(name, k)]["err_mean"] for k in KS])
        b, a = np.polyfit(np.log(ks), np.log(em), 1)
        pred = b * np.log(ks) + a
        ss_res = np.sum((np.log(em) - pred) ** 2)
        ss_tot = np.sum((np.log(em) - np.log(em).mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        ok = (-1.15 <= b <= -0.85) and (r2 >= 0.95)
        print(f"  {name:<12} expo={b:+.4f} R2={r2:.4f}  -> {'PASS' if ok else 'FAIL'}")

    print()
    print(f"[p1] done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
