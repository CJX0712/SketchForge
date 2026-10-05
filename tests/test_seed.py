"""Seed-management tests.

Author: 晨星
"""

import random

import numpy as np
import pytest

import sketchforge.core.seed as seed_module
from sketchforge.core.errors import InvalidConfigError, MissingSeedError
from sketchforge.core.seed import SeedState, require, set_all


def test_set_all_replays_python_and_numpy_sequences() -> None:
    set_all(2026)
    first = (random.random(), float(np.random.random()))
    set_all(2026)
    second = (random.random(), float(np.random.random()))
    assert first == second


def test_derive_and_spawn_are_stable_and_independent() -> None:
    state = SeedState(42)
    assert state.derive("data") == state.derive("data")
    assert state.derive("data") != state.derive("hash")
    first = state.spawn_generators(3)
    second = state.spawn_generators(3)
    assert [generator.integers(0, 2**31) for generator in first] == [
        generator.integers(0, 2**31) for generator in second
    ]
    assert (
        len({state_value.derive(str(index)) for index, state_value in enumerate([state] * 3)}) == 3
    )


def test_require_without_set_all_raises_e102(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(seed_module, "_STATE", None)
    with pytest.raises(MissingSeedError) as captured:
        require()
    assert captured.value.code == "E102"


def test_invalid_seed_and_spawn_count_raise_e101() -> None:
    with pytest.raises(InvalidConfigError, match="seed must be"):
        set_all("not-an-int")  # type: ignore[arg-type]
    with pytest.raises(InvalidConfigError, match="non-negative"):
        SeedState(1).spawn_generators(-1)
