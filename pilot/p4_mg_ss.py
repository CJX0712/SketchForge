"""pilot §B-4: MG / SS 的 recall@K、误差界覆盖率，并验证 I-10 夹逼的 100% 覆盖。

自研实现：
  MG = Misra-Gries（欠估计 → 给下界），offset 技巧，O(1) 摊还
  SS = Space-Saving（过估计 → 给上界），bucket Stream-Summary，O(1) 摊还
I-10 区间：[mg_lb(x), ss_ub(x)] 必须包含真值 f(x)，覆盖率 = 100%
  mg_lb(x) = stored-offset  若被跟踪；否则 0
  ss_ub(x) = count          若被跟踪；否则 min_count（表未满时为 0）
"""
import sys
import time
from collections import Counter

import dgp
import numpy as np


class MisraGries:
    """k 个计数器。保证：误差 ≤ N/(k+1)；f > N/(k+1) 必召回。"""
    def __init__(self, k):
        self.k = k
        self.c = {}
        self.offset = 0

    def update(self, x):
        if x in self.c:
            self.c[x] += 1
        elif len(self.c) < self.k:
            self.c[x] = self.offset + 1
        else:
            self.offset += 1
            dead = [a for a, v in self.c.items() if v == self.offset]
            for a in dead:
                del self.c[a]
            self.c[x] = self.offset + 1

    def lb(self, x):
        return self.c[x] - self.offset if x in self.c else 0

    def top(self, K):
        return [a for a, _ in sorted(self.c.items(),
                                     key=lambda kv: kv[1] - self.offset,
                                     reverse=True)[:K]]


class SpaceSaving:
    """k 个 (key,count)。保证：过估计 ≤ min_count ≤ N/k；f > N/k 必召回。"""
    def __init__(self, k):
        self.k = k
        self.c = {}
        self.buckets = {}
        self.min_count = 0

    def _add(self, x, cnt):
        self.c[x] = cnt
        self.buckets.setdefault(cnt, set()).add(x)

    def _bump(self, x, old, new):
        s = self.buckets[old]
        s.discard(x)
        if not s:
            del self.buckets[old]
            if old == self.min_count:
                self.min_count = new
        self.buckets.setdefault(new, set()).add(x)
        self.c[x] = new

    def update(self, x):
        if x in self.c:
            old = self.c[x]
            self._bump(x, old, old + 1)
        elif len(self.c) < self.k:
            self._add(x, 1)
            self.min_count = 1
        else:
            mc = self.min_count
            victim = next(iter(self.buckets[mc]))
            self.buckets[mc].discard(victim)
            del self.c[victim]
            if not self.buckets[mc]:
                del self.buckets[mc]
            nv = mc + 1
            self.c[x] = nv
            self.buckets.setdefault(nv, set()).add(x)
            self.min_count = nv

    def ub(self, x):
        return self.c[x] if x in self.c else self.min_count

    def est(self, x):
        return self.c[x] if x in self.c else self.min_count

    def top(self, K):
        return [a for a, _ in sorted(self.c.items(), key=lambda kv: -kv[1])[:K]]


def main():
    t0 = time.perf_counter()
    R = 5
    N = 100_000
    KS = [100, 400]
    ALPHAS = [1.1, 1.3]
    DOMAIN = 200_000
    KQ = 50

    print(f"[p4] MG / SS   R={R} N={N} k={KS} alpha={ALPHAS} K={KQ}")
    print()
    print(f"{'alpha':>6}{'k':>6}{'meth':>5}{'P@K':>8}{'R@K':>8}{'F1@K':>8}"
          f"{'recall_guar':>13}{'maxerr':>10}{'bound':>10}{'I10_cov':>9}{'I11_ok':>8}")
    print("-" * 92)

    viol10 = 0
    i11_fail = 0
    for alpha in ALPHAS:
        for k in KS:
            streams = []
            for s in range(R):
                g, _, _ = dgp.spawn(s)
                streams.append(np.clip(g.zipf(alpha, size=N), 1, DOMAIN))
            for meth in ("MG", "SS"):
                P, Rc, F1, rg, me, bd, cov = [], [], [], [], [], [], []
                for stream in streams:
                    truth = Counter(stream.tolist())
                    true_top = set(a for a, _ in truth.most_common(KQ))
                    sk = MisraGries(k) if meth == "MG" else SpaceSaving(k)
                    for v in stream.tolist():
                        sk.update(v)
                    pred = set(sk.top(KQ))
                    inter = len(pred & true_top)
                    p = inter / KQ
                    r = inter / len(true_top)
                    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                    P.append(p); Rc.append(r); F1.append(f1)

                    # 保证式召回：阈值 N/(k+1) for MG, N/k for SS
                    thr = N / (k + 1) if meth == "MG" else N / k
                    guaranteed = set(a for a, c in truth.items() if c > thr)
                    tracked = set(sk.c.keys())
                    rg.append(len(guaranteed & tracked) / len(guaranteed)
                              if guaranteed else float("nan"))

                    # 误差界 + I-10 覆盖（对全 distinct 集 + 零频项）
                    keys = list(truth.keys()) + list(range(DOMAIN + 1, DOMAIN + 201))
                    if meth == "MG":
                        lb = np.array([sk.lb(a) for a in keys])
                        ub = None
                    else:
                        lb = None
                        ub = np.array([sk.ub(a) for a in keys])
                    fs = np.array([truth.get(a, 0) for a in keys])
                    if meth == "MG":
                        errs = fs - lb
                        cov.append(float(np.mean(errs >= 0)))
                        bd.append(N / (k + 1))
                    else:
                        errs = ub - fs
                        cov.append(float(np.mean(errs >= 0)))
                        bd.append(N / k)
                    me.append(float(errs.max()))
                print(f"{alpha:>6}{k:>6}{meth:>5}{np.mean(P):>8.4f}"
                      f"{np.mean(Rc):>8.4f}{np.mean(F1):>8.4f}"
                      f"{np.nanmean(rg):>13.4f}{np.mean(me):>10.1f}"
                      f"{bd[0]:>10.1f}{np.mean(cov):>9.4f}"
                      f"{'PASS' if np.mean(me) <= bd[0] else 'FAIL':>8}")
                if np.mean(cov) < 1.0:
                    viol10 += 1
                if np.mean(me) > bd[0]:
                    i11_fail += 1

    print()
    print("[p4] --- I-10 夹逼覆盖率（要求 == 1.0，零容差） ---")
    print("  区间 [mg_lb, ss_ub] 覆盖真值的比例（对所有 distinct key + 零频 key）")
    print(f"  越界的方法-配置数 = {viol10}  -> {'PASS' if viol10 == 0 else 'FAIL'}")
    print("[p4] --- I-9 保证式召回（要求 == 1.0）与 I-11 误差界 ---")
    print(f"  I-11 越界配置数 = {i11_fail} -> {'PASS' if i11_fail == 0 else 'FAIL'}")

    print()
    print("[p4] --- 补充：MG 与 SS 联合夹逼（两个 sketch 同时跑）---")
    print(f"{'alpha':>6}{'k':>6}{'interval_cov':>14}{'mean_width':>12}{'N/k':>10}")
    for alpha in ALPHAS:
        for k in KS:
            covs, widths = [], []
            for s in range(R):
                g, _, _ = dgp.spawn(s)
                stream = np.clip(g.zipf(alpha, size=N), 1, DOMAIN)
                truth = Counter(stream.tolist())
                mg, ss = MisraGries(k), SpaceSaving(k)
                for v in stream.tolist():
                    mg.update(v); ss.update(v)
                keys = list(truth.keys())
                lo = np.array([mg.lb(a) for a in keys])
                hi = np.array([ss.ub(a) for a in keys])
                f = np.array([truth[a] for a in keys])
                covs.append(float(np.mean((lo <= f) & (f <= hi))))
                widths.append(float(np.mean(hi - lo)))
            print(f"{alpha:>6}{k:>6}{np.mean(covs):>14.6f}"
                  f"{np.mean(widths):>12.3f}{N/k:>10.1f}")

    print()
    print(f"[p4] done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
