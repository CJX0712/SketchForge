"""Synthetic stream determinism and semantic tests.

Author: 晨星
"""

import numpy as np
import pytest

from sketchforge.core.errors import InvalidSpecError
from sketchforge.data.specs import StreamSpec
from sketchforge.data.streams import generate

_KINDS = (
    "uniform",
    "zipf",
    "pareto",
    "gmm",
    "drift_abrupt",
    "drift_gradual",
    "low_card",
    "high_card",
    "elephant_mice",
)


@pytest.mark.parametrize("kind", _KINDS)
def test_all_dgps_are_reproducible_and_have_requested_length(kind: str) -> None:
    spec = StreamSpec(kind=kind, n=301, domain=53, seed=2026)  # type: ignore[arg-type]
    first = generate(spec)
    second = generate(spec)
    np.testing.assert_array_equal(first.keys, second.keys)
    np.testing.assert_array_equal(first.values, second.values)
    assert len(first.keys) == spec.n
    assert first.keys.dtype == np.int64
    assert first.values.dtype == np.float64


def test_different_seeds_change_stochastic_stream() -> None:
    first = generate(StreamSpec(kind="uniform", n=500, domain=101, seed=1))
    second = generate(StreamSpec(kind="uniform", n=500, domain=101, seed=2))
    assert not np.array_equal(first.keys, second.keys)


def test_dgp_semantics_cover_cardinality_and_drift_extremes() -> None:
    high = generate(StreamSpec(kind="high_card", n=100, domain=2, seed=1))
    low = generate(StreamSpec(kind="low_card", n=500, domain=1000, seed=1))
    abrupt = generate(StreamSpec(kind="drift_abrupt", n=300, domain=40, seed=1))
    elephants = generate(StreamSpec(kind="elephant_mice", n=1000, domain=1000, seed=1))

    assert np.unique(high.keys).size == 100
    assert np.unique(low.keys).size <= 10
    assert np.all(abrupt.keys[200:] >= 10_040)
    elephant_updates = sum(np.count_nonzero(elephants.keys == key) for key in range(5))
    assert elephant_updates == 500


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "unknown", "n": 10, "domain": 10, "seed": 1},
        {"kind": "uniform", "n": 0, "domain": 10, "seed": 1},
        {"kind": "uniform", "n": 10, "domain": 0, "seed": 1},
        {"kind": "zipf", "n": 10, "domain": 10, "seed": 1, "zipf_alpha": 0},
        {"kind": "uniform", "n": 10, "domain": 10, "seed": 1, "drift_rate": 2},
    ],
)
def test_invalid_stream_specs_raise_e201(kwargs: dict[str, object]) -> None:
    with pytest.raises(InvalidSpecError) as captured:
        StreamSpec(**kwargs)  # type: ignore[arg-type]
    assert captured.value.code == "E201"
