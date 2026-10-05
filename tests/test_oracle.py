"""Exact-oracle cross-checks against independent brute force.

Author: 晨星
"""

from collections import Counter

import numpy as np
import pytest

from sketchforge.core.errors import EmptyStreamError, InvalidSpecError
from sketchforge.data.oracle import ExactOracle
from sketchforge.data.specs import StreamSpec
from sketchforge.data.streams import Stream, generate


def test_oracle_matches_independent_brute_force() -> None:
    stream = generate(StreamSpec(kind="uniform", n=501, domain=37, seed=41))
    oracle = ExactOracle(stream)
    counts = Counter(int(key) for key in stream.keys)

    assert oracle.total() == len(stream.keys)
    assert oracle.cardinality() == len(set(int(key) for key in stream.keys))
    assert oracle.frequency(7) == counts[7]
    assert oracle.quantile(0.73) == pytest.approx(float(np.quantile(stream.values, 0.73)))
    assert oracle.rank(oracle.quantile(0.5)) == pytest.approx(
        float(np.count_nonzero(stream.values <= oracle.quantile(0.5)) / len(stream.values))
    )
    expected_top = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:8]
    assert oracle.top_k(8) == expected_top


def test_oracle_owns_copies_not_stream_arrays() -> None:
    stream = generate(StreamSpec(kind="low_card", n=50, domain=100, seed=7))
    oracle = ExactOracle(stream)
    before = oracle.frequency(int(stream.keys[0]))
    stream.keys[:] = 999
    assert oracle.frequency(999) == 0
    assert before > 0


def test_oracle_guards_reject_empty_and_invalid_queries() -> None:
    empty_spec = StreamSpec(kind="uniform", n=1, domain=1, seed=1)
    empty = Stream(np.array([], dtype=np.int64), np.array([], dtype=np.float64), empty_spec)
    with pytest.raises(EmptyStreamError) as captured:
        ExactOracle(empty)
    assert captured.value.code == "E203"

    oracle = ExactOracle(generate(empty_spec))
    with pytest.raises(InvalidSpecError) as quantile_error:
        oracle.quantile(1.1)
    assert quantile_error.value.code == "E201"
    with pytest.raises(InvalidSpecError, match="positive"):
        oracle.top_k(0)
