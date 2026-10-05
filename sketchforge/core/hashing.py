"""Stable, seed-injectable hashing primitives.

``hash_u64(stable_key(42), 1337)`` is always ``9657053958485203079``.

Author: 晨星
"""

from __future__ import annotations

import math
import struct

import numpy as np

_MASK = (1 << 64) - 1
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX_A = np.uint64(0xBF58476D1CE4E5B9)
_MIX_B = np.uint64(0x94D049BB133111EB)
_CANONICAL_NAN = 0x7FF8000000000000


def stable_key(item: object) -> bytes:
    """Encode supported scalar keys into a process-independent byte string."""
    if isinstance(item, (bool, int, np.integer)):
        value = int(item) & _MASK
        return value.to_bytes(8, byteorder="little", signed=False)
    if isinstance(item, (float, np.floating)):
        value = float(item)
        if math.isnan(value):
            return struct.pack("<Q", _CANONICAL_NAN)
        if value == 0.0:
            value = 0.0
        return struct.pack("<d", value)
    if isinstance(item, str):
        return item.encode("utf-8")
    if isinstance(item, bytes):
        return item
    raise TypeError(f"unsupported stable key type: {type(item).__name__}")


def _splitmix64(values: np.ndarray | np.uint64) -> np.ndarray | np.uint64:
    with np.errstate(over="ignore"):
        mixed = values + _GOLDEN
        mixed = (mixed ^ (mixed >> np.uint64(30))) * _MIX_A
        mixed = (mixed ^ (mixed >> np.uint64(27))) * _MIX_B
        return mixed ^ (mixed >> np.uint64(31))


def _fold_bytes(key_bytes: bytes) -> np.uint64:
    if len(key_bytes) <= 8:
        return np.uint64(int.from_bytes(key_bytes.ljust(8, b"\0"), "little"))
    state = np.uint64(len(key_bytes))
    for offset in range(0, len(key_bytes), 8):
        chunk = key_bytes[offset : offset + 8]
        word = np.uint64(int.from_bytes(chunk.ljust(8, b"\0"), "little"))
        state = _splitmix64(state ^ word)
    return state


def hash_u64(key_bytes: bytes, seed: int) -> np.uint64:
    """Hash bytes to a deterministic unsigned 64-bit SplitMix64 value."""
    if not isinstance(key_bytes, bytes):
        raise TypeError("key_bytes must be bytes")
    raw = _fold_bytes(key_bytes)
    return np.uint64(_splitmix64(raw ^ np.uint64(seed & _MASK)))


def hash_index(key_bytes: bytes, seed: int, width: int) -> int:
    """Map a key into ``[0, width)``."""
    if width <= 0:
        raise ValueError("width must be positive")
    return int(hash_u64(key_bytes, seed)) % width


def hash_sign(key_bytes: bytes, seed: int) -> int:
    """Map a key to a deterministic Count-Sketch sign."""
    return 1 if int(hash_u64(key_bytes, seed)) & 1 else -1


def hash_array_u64(keys: np.ndarray, seed: int) -> np.ndarray:
    """Vectorize SplitMix64 over numeric keys without a Python item loop."""
    array = np.asarray(keys)
    if np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raw = array.astype(np.uint64, copy=False)
    elif np.issubdtype(array.dtype, np.floating):
        floats = array.astype(np.float64, copy=False)
        bits = floats.view(np.uint64)
        canonical = np.full(bits.shape, _CANONICAL_NAN, dtype=np.uint64)
        zero = np.zeros(bits.shape, dtype=np.uint64)
        raw = np.where(np.isnan(floats), canonical, np.where(floats == 0.0, zero, bits))
    else:
        raise TypeError("hash_array_u64 supports only numeric NumPy arrays")
    seeded = raw ^ np.uint64(seed & _MASK)
    return np.asarray(_splitmix64(seeded), dtype=np.uint64)
