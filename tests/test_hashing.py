"""Stable hash tests.

Author: 晨星
"""

import struct

import numpy as np
import pytest

from sketchforge.core.hashing import (
    hash_array_u64,
    hash_index,
    hash_sign,
    hash_u64,
    stable_key,
)


def test_splitmix64_fixed_constant() -> None:
    assert int(hash_u64(stable_key(42), 1337)) == 9657053958485203079


def test_zero_and_nan_encodings_are_canonical() -> None:
    assert stable_key(-0.0) == stable_key(0.0)
    alternate_nan = struct.unpack("<d", struct.pack("<Q", 0x7FF0000000000001))[0]
    assert stable_key(float("nan")) == stable_key(alternate_nan)


def test_vectorized_integer_hash_matches_scalar_bit_for_bit() -> None:
    keys = np.array([0, 1, -1, 42, np.iinfo(np.int64).max], dtype=np.int64)
    vectorized = hash_array_u64(keys, 1337)
    scalar = np.array([hash_u64(stable_key(item), 1337) for item in keys], dtype=np.uint64)
    np.testing.assert_array_equal(vectorized, scalar)


def test_vectorized_float_hash_matches_scalar_bit_for_bit() -> None:
    keys = np.array([-0.0, 0.0, 1.5, np.nan, np.inf], dtype=np.float64)
    vectorized = hash_array_u64(keys, 99)
    scalar = np.array([hash_u64(stable_key(item), 99) for item in keys], dtype=np.uint64)
    np.testing.assert_array_equal(vectorized, scalar)


def test_hash_boundaries_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        hash_index(b"key", 1, 0)
    with pytest.raises(TypeError, match="numeric"):
        hash_array_u64(np.array(["key"], dtype="U3"), 1)
    with pytest.raises(TypeError, match="unsupported"):
        stable_key(object())
    assert hash_sign(b"key", 1) in {-1, 1}
