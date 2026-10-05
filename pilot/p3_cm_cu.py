"""pilot §B-3: CM vs CU（保守更新）在 Zipf 流上的误差比 + 不变量 I-1/I-2 验证。

自研实现（DataSketches 的 count_min_sketch 是标准 CM，不含保守更新），
哈希用 splitmix64，seed 由独立 PRNG 流派生（L2）。
"""
import sys
import time

import dgp
import numpy as np

M64 = np.uint64
MASK = (1 << 64) - 1


def splitmix64(x, seed):
    """向量化 splitmix64：输出 (N,) 的 uint64。"""
    z = (np.asarray(x, dtype=np.uint64) + np.uint64(seed)) & np.uint64(MASK)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9) & np.uint64(MASK)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB) & np.uint64(MASK)
    return (z ^ (z >> np.uint64(31))) & np.uint64(MASK)


class CountMin:
    def __init__(self, w, d, seed):
        self.w, self.d, self.seed = w, d, seed
        self.C = np.zeros((d, w), dtype=np.int64)

    def build_idx(self, keys):
        h1 = splitmix64(keys, self.seed).astype(object) % self.w
        h2 = splitmix64(keys, self.seed ^ 0x5DEECE66D).astype(object) % self.w
        h1 = np.asarray(splitmix64(keys, self.seed), dtype=np.uint64) % np.uint64(self.w)
        h2 = np.asarray(splitmix64(keys, self.seed ^ 0x5DEECE66D), dtype=np.uint64) % np.uint64(self.w)
        h1 = h1.astype(np.int64)
        h2 = h2.astype(np.int64)
        r = np.arange(self.d, dtype=np.int64)
        return (h1[:, None] + r[None, :] * h2[:, None]) % self.w     # (N, d)

    def update_cm(self, idx):
        C = self.C
        for i in range(idx.shape[0]):
            for rr in range(self.d):
                C[rr, idx[i, rr]] += 1

    def update_cu(self, idx):
        C = self.C
        d = self.d
        for i in range(idx.shape[0]):
            row = idx[i]
            cur = C[np.arange(d), row]
            v = cur.min() + 1
            for rr in range(d):
                j = row[rr]
                if C[rr, j] < v:
                    C[rr, j] = v

    def query(self, keys):
        idx = self.build_idx(keys)
        r = np.arange(self.d, dtype=np.int64)
        return self.C[r[None, :], idx].min(axis=1)


def main():
    t0 = time.perf_counter()
    R = 10
    N = 100_000
    D = 4
    WS = [512, 2048, 8192]
    ALPHAS = [1.1, 1.3]
    DOMAIN = 200_000

    print(f"[p3] CM vs CU   R={R} N={N} d={D} w={WS} alpha={ALPHAS} domain={DOMAIN}")
    print()
    print(f"{'alpha':>6}{'w':>7}{'err_CM':>10}{'err_CU':>10}{'CU/CM':>9}"
          f"{'I1_viol':>9}{'I2_lo':>7}{'I2_hi':>7}{'overest_CU':>12}{'fbar_Q':>9}")
    print("-" * 88)

    rows = []
    for alpha in ALPHAS:
        for w in WS:
            agg = dict(err_cm=[], err_cu=[], v1=0, v2lo=0, v2hi=0,
                       over_cu=[], fbar=[])
            for seed in range(R):
                g, gh, _ = dgp.spawn(seed)
                stream = np.clip(g.zipf(alpha, size=N), 1, DOMAIN)
                uniq, cnt = np.unique(stream, return_counts=True)
                order = np.argsort(-cnt)
                uniq, cnt = uniq[order], cnt[order]
                truth = dict(zip(uniq.tolist(), cnt.tolist()))

                # 查询集：高频 top-100 + 中频 rank 1000..1100 + 零频 100 个
                q_hi = uniq[:100]
                q_mid = uniq[1000:1100] if len(uniq) > 1100 else uniq[-100:]
                q_zero = np.arange(DOMAIN + 1, DOMAIN + 101)     # 未出现
                Q = np.concatenate([q_hi, q_mid, q_zero])
                f_true = np.array([truth.get(int(x), 0) for x in Q])

                seed_h = int(gh.integers(1, 2**31 - 1))
                idx_all = None
                cm = CountMin(w, D, seed_h)
                cu = CountMin(w, D, seed_h)
                idx = cm.build_idx(stream)
                cm.update_cm(idx)
                cu.update_cu(idx)

                e_cm = cm.query(Q)
                e_cu = cu.query(Q)
                denom = np.maximum(f_true, 1)
                rel_cm = np.abs(e_cm - f_true) / denom
                rel_cu = np.abs(e_cu - f_true) / denom
                agg["err_cm"].append(rel_cm.mean())
                agg["err_cu"].append(rel_cu.mean())
                agg["fbar"].append(f_true.mean())
                agg["over_cu"].append(float(np.mean(e_cu >= f_true)))
                agg["v1"] += int(np.sum(e_cm < f_true))            # I-1: CM ≥ f
                agg["v2lo"] += int(np.sum(e_cu < f_true))          # I-2 下界
                agg["v2hi"] += int(np.sum(e_cu > e_cm))            # I-2 上界 CU ≤ CM
            ecm = float(np.mean(agg["err_cm"]))
            ecu = float(np.mean(agg["err_cu"]))
            rows.append((alpha, w, ecm, ecu, agg["v1"], agg["v2lo"], agg["v2hi"],
                         float(np.mean(agg["fbar"]))))
            print(f"{alpha:>6}{w:>7}{ecm:>10.5f}{ecu:>10.5f}{ecu/ecm:>9.4f}"
                  f"{agg['v1']:>9}{agg['v2lo']:>7}{agg['v2hi']:>7}"
                  f"{float(np.mean(agg['over_cu'])):>12.4f}"
                  f"{float(np.mean(agg['fbar'])):>9.2f}")

    print()
    print("[p3] --- I-1 / I-2 判定（要求 violations == 0，零容差） ---")
    tot = sum(r[4] + r[5] + r[6] for r in rows)
    print(f"  I-1  (CM ≥ f)          违反总数 = {sum(r[4] for r in rows)}")
    print(f"  I-2a (CU ≥ f)          违反总数 = {sum(r[5] for r in rows)}")
    print(f"  I-2b (CU ≤ CM)         违反总数 = {sum(r[6] for r in rows)}")
    print(f"  => {'PASS' if tot == 0 else 'FAIL'}")

    print()
    print("[p3] --- c_CU 标定：err_3 ≈ c_CU * (N/w) / fbar_Q ---")
    print(f"{'alpha':>6}{'w':>7}{'N/w/fbar':>11}{'err_CU':>10}{'c_CU':>9}")
    for (a, w, ecm, ecu, _, _, _, fbar) in rows:
        pred = (N / w) / fbar
        print(f"{a:>6}{w:>7}{pred:>11.4f}{ecu:>10.5f}{ecu/pred:>9.5f}")

    print()
    print("[p3] --- I-18 标度律（频率族要求 expo ∈ [-1.10,-0.90]，对 w） ---")
    for a in ALPHAS:
        sub = [r for r in rows if r[0] == a]
        ws = np.array([r[1] for r in sub], dtype=float)
        for tag, col in (("CM", 2), ("CU", 3)):
            errs = np.array([r[col] for r in sub])
            b, aa = np.polyfit(np.log(ws), np.log(errs), 1)
            pred = b * np.log(ws) + aa
            r2 = 1 - np.sum((np.log(errs) - pred) ** 2) / np.sum(
                (np.log(errs) - np.log(errs).mean()) ** 2)
            ok = (-1.10 <= b <= -0.90) and r2 >= 0.95
            print(f"  alpha={a} {tag:<3} expo={b:+.4f} R2={r2:.4f} -> {'PASS' if ok else 'FAIL'}")

    print()
    print(f"[p3] done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
