"""SketchForge pilot — DGP 生成器（与规格 §3 一致）。

纪律：所有 DGP 用 np.random.Generator(PCG64(seed))；sketch 的 hash seed 必须由
独立 PRNG 流派生（泄漏防线 L2），不得与 DGP 流同源。
"""

import numpy as np

EPS_FLOOR = 1e-9


def spawn(seed):
    """返回三条独立 PRNG 流：(dgp, hash, sampling)。L2 防线。"""
    ss = np.random.SeedSequence(seed)
    a, b, c = ss.spawn(3)
    return (
        np.random.Generator(np.random.PCG64(a)),
        np.random.Generator(np.random.PCG64(b)),
        np.random.Generator(np.random.PCG64(c)),
    )


def perm_unique(n, g):
    """恰好 n 个 distinct（真值精确已知 = n）。"""
    return g.permutation(n)


def zipf(n, alpha, domain, g):
    """D2: Zipf 离散，p_i ∝ i^{-alpha}，域大小 domain。"""
    x = g.zipf(alpha, size=n)
    return np.clip(x, 1, domain)


def uniform(n, domain, g):
    """D1: 离散均匀。"""
    return g.integers(0, domain, size=n)


def pareto(n, alpha, g):
    """D3: Pareto 连续，x_m=1。"""
    u = g.random(n)
    return np.power(1.0 - u, -1.0 / alpha)


def gauss_mix(n, n_comp, sigma, g):
    """D4: 高斯混合，成分均值间隔 5*sigma。"""
    w = g.random(n_comp)
    w = w / w.sum()
    comp = g.choice(n_comp, size=n, p=w)
    mu = np.arange(n_comp) * 5.0 * sigma
    return g.normal(mu[comp], sigma)


def elephant_mice(n_elephant=5, n_mice=500_000, g=None):
    """D9: 5 个 elephant 各 1e5 次（50% 流量）+ 5e5 个 mice 各 1 次（50% 流量）。

    返回 (stream, elephant_keys)。
    """
    per = n_mice // n_elephant  # 每个 elephant 出现次数 = mice 数 / 5 => 50/50
    ele = np.repeat(np.arange(1, n_elephant + 1), per)
    mice = np.arange(10_000_000, 10_000_000 + n_mice)
    stream = np.concatenate([ele, mice])
    g.shuffle(stream)
    return stream, np.arange(1, n_elephant + 1)


def drift_abrupt(n, g):
    """D5: 分段突变漂移。段1 Zipf(1.0)/域A；段2 均匀/1e4；段3 Zipf(1.5)/域B（与A不交）。"""
    a = n // 3
    b = n // 3
    c = n - a - b
    s1 = np.clip(g.zipf(1.0, size=a), 1, 100_000)
    s2 = g.integers(1, 10_000, size=b)
    s3 = np.clip(g.zipf(1.5, size=c), 1, 100_000) + 100_000  # 域 B，与 A 不交
    return np.concatenate([s1, s2, s3])


def true_counts(stream):
    """暴力真值频数（离线 oracle，L3：只允许出现在 evaluator）。"""
    return np.unique(stream, return_counts=True)


def true_sorted(stream):
    return np.sort(stream)


def normalized_rank_error(sorted_arr, qhat):
    """|rank(qhat)/N - p| 的构件：返回 rank/N。"""
    n = sorted_arr.size
    return np.searchsorted(sorted_arr, qhat, side="left") / n


P_GRID = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
