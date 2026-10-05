"""Deterministic synthetic data-stream generators.

Author: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sketchforge.core.seed import SeedState
from sketchforge.data.specs import StreamSpec


@dataclass(frozen=True)
class Stream:
    """Materialized stream keys and correlated numeric values."""

    keys: np.ndarray
    values: np.ndarray
    spec: StreamSpec

    def __post_init__(self) -> None:
        if self.keys.dtype != np.int64:
            raise TypeError("stream keys must use dtype int64")
        if self.values.dtype != np.float64:
            raise TypeError("stream values must use dtype float64")
        if self.keys.ndim != 1 or self.values.ndim != 1:
            raise ValueError("stream arrays must be one-dimensional")
        if len(self.keys) != len(self.values):
            raise ValueError("stream key and value lengths must match")


def _truncated_zipf(rng: np.random.Generator, n: int, domain: int, alpha: float) -> np.ndarray:
    ranks = np.arange(1, domain + 1, dtype=np.float64)
    weights = np.power(ranks, -alpha)
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.searchsorted(cdf, rng.random(n), side="right").astype(np.int64)


def _values_from_keys(keys: np.ndarray) -> np.ndarray:
    numeric = keys.astype(np.float64)
    values = np.log1p(np.abs(numeric)) + 0.25 * np.sin(numeric * 0.017)
    return values.astype(np.float64, copy=False)


def _three_part_sizes(n: int) -> tuple[int, int, int]:
    first = n // 3
    second = n // 3
    return first, second, n - first - second


def _uniform(spec: StreamSpec, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, spec.domain, size=spec.n, dtype=np.int64)


def _zipf(spec: StreamSpec, rng: np.random.Generator) -> np.ndarray:
    return _truncated_zipf(rng, spec.n, spec.domain, spec.zipf_alpha)


def _pareto(spec: StreamSpec, rng: np.random.Generator) -> np.ndarray:
    uniform = rng.random(spec.n)
    samples = np.power(1.0 - uniform, -1.0 / spec.zipf_alpha) - 1.0
    return np.clip(np.floor(samples), 0, spec.domain - 1).astype(np.int64)


def _gmm(spec: StreamSpec, rng: np.random.Generator) -> np.ndarray:
    components = int(rng.integers(2, 6))
    weights = rng.dirichlet(np.ones(components, dtype=np.float64))
    centers = rng.uniform(0, max(1, spec.domain - 1), size=components)
    deviations = rng.uniform(
        max(0.5, spec.domain / 100.0), max(1.0, spec.domain / 8.0), size=components
    )
    selected = rng.choice(components, size=spec.n, p=weights)
    samples = rng.normal(centers[selected], deviations[selected])
    return np.clip(np.rint(samples), 0, spec.domain - 1).astype(np.int64)


def _drift_abrupt(spec: StreamSpec, rng: np.random.Generator) -> np.ndarray:
    first_n, second_n, third_n = _three_part_sizes(spec.n)
    first = _truncated_zipf(rng, first_n, spec.domain, 1.0)
    second = rng.integers(0, 10_000, size=second_n, dtype=np.int64)
    new_domain = _truncated_zipf(rng, third_n, spec.domain, 1.5)
    third = new_domain + np.int64(spec.domain + 10_000)
    return np.concatenate((first, second, third)).astype(np.int64, copy=False)


def _drift_gradual(spec: StreamSpec, rng: np.random.Generator) -> np.ndarray:
    source = _truncated_zipf(rng, spec.n, spec.domain, max(1.0, spec.zipf_alpha))
    target = rng.integers(0, spec.domain, size=spec.n, dtype=np.int64) + np.int64(spec.domain)
    mixture = np.arange(spec.n, dtype=np.float64) / max(1, spec.n - 1)
    choose_target = rng.random(spec.n) < mixture
    return np.where(choose_target, target, source).astype(np.int64, copy=False)


def _low_card(spec: StreamSpec, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 10, size=spec.n, dtype=np.int64)


def _high_card(spec: StreamSpec, rng: np.random.Generator) -> np.ndarray:
    del rng
    return np.arange(spec.n, dtype=np.int64)


def _elephant_mice(spec: StreamSpec, rng: np.random.Generator) -> np.ndarray:
    elephant_count = spec.n // 2
    mouse_count = spec.n - elephant_count
    elephants = rng.integers(0, 5, size=elephant_count, dtype=np.int64)
    mice = np.arange(5, 5 + mouse_count, dtype=np.int64)
    keys = np.concatenate((elephants, mice))
    rng.shuffle(keys)
    return keys


_GENERATORS = {
    "uniform": _uniform,
    "zipf": _zipf,
    "pareto": _pareto,
    "gmm": _gmm,
    "drift_abrupt": _drift_abrupt,
    "drift_gradual": _drift_gradual,
    "low_card": _low_card,
    "high_card": _high_card,
    "elephant_mice": _elephant_mice,
}


def generate(spec: StreamSpec) -> Stream:
    """Generate a deterministic stream from a validated specification."""
    data_seed = SeedState(spec.seed).derive("data")
    rng = np.random.default_rng(data_seed)
    keys = _GENERATORS[spec.kind](spec, rng)
    values = _values_from_keys(keys)
    return Stream(keys=keys, values=values, spec=spec)
