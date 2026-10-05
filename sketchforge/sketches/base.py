"""Abstract base class and serialization codec shared by every SketchForge sketch.

Design notes
------------
* Byte currency is *serialized size*: :meth:`AbstractSketch.memory_bytes` defaults to
  ``len(self.serialize())`` so a declared budget can always be checked against the
  payload that would actually travel over the wire.
* :meth:`AbstractSketch.memory_slots` is *declarative only*: it reports the logical
  number of slots and must never be compared across methods with different
  slot widths (1000 HLL registers are 1000 bytes, 1000 t-digest centroids are not).
* Determinism: Tier-1 sketches own their NumPy generator (``default_rng`` seeded from
  :class:`~sketchforge.core.seed.SeedState`) and never touch the global RNG.

Author: 晨星
"""

from __future__ import annotations

import json
import struct
from typing import Any, ClassVar

import numpy as np

from sketchforge.core.errors import (
    EstimateUnavailableError,
    MergeTypeError,
    MergeUnsupportedError,
    SerializationError,
    SketchForgeError,
)
from sketchforge.core.hashing import hash_array_u64, hash_index, hash_sign, hash_u64, stable_key
from sketchforge.core.seed import SeedState
from sketchforge.core.seed import require as require_seed
from sketchforge.core.types import Estimate, MethodSpec, Query, QueryType

_MAGIC = b"SF1"
_HEADER = struct.Struct("<3sB")
_JSON_LEN = struct.Struct("<I")
_ARRAY_COUNT = struct.Struct("<H")
_ARRAY_NAME = struct.Struct("<B")
_ARRAY_META = struct.Struct("<BB")
_ARRAY_DIM = struct.Struct("<Q")
_ARRAY_NBYTES = struct.Struct("<Q")

_DTYPE_BY_CODE: dict[int, Any] = {
    0: np.dtype(np.int8),
    1: np.dtype(np.uint8),
    2: np.dtype(np.int16),
    3: np.dtype(np.uint16),
    4: np.dtype(np.int32),
    5: np.dtype(np.uint32),
    6: np.dtype(np.int64),
    7: np.dtype(np.uint64),
    8: np.dtype(np.float32),
    9: np.dtype(np.float64),
    10: np.dtype(np.bool_),
}
_CODE_BY_DTYPE: dict[Any, int] = {dtype: code for code, dtype in _DTYPE_BY_CODE.items()}

_SKIP_STATE_KEYS = frozenset({"_seed", "_params", "_rng", "_gen"})


def _pack_state(name: str, meta: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    """Encode ``name``/``meta``/``arrays`` into the SketchForge wire format."""
    name_bytes = name.encode("utf-8")
    meta_bytes = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    out = bytearray()
    out += _HEADER.pack(_MAGIC, len(name_bytes))
    out += name_bytes
    out += _JSON_LEN.pack(len(meta_bytes))
    out += meta_bytes
    out += _ARRAY_COUNT.pack(len(arrays))
    for key, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        code = _CODE_BY_DTYPE.get(array.dtype)
        if code is None:
            raise SerializationError(
                "unsupported array dtype in sketch state", key=key, dtype=str(array.dtype)
            )
        key_bytes = key.encode("utf-8")
        out += _ARRAY_NAME.pack(len(key_bytes))
        out += key_bytes
        out += _ARRAY_META.pack(code, array.ndim)
        for dim in array.shape:
            out += _ARRAY_DIM.pack(int(dim))
        raw = array.tobytes()
        out += _ARRAY_NBYTES.pack(len(raw))
        out += raw
    return bytes(out)


def _unpack_state(payload: bytes) -> tuple[str, dict[str, Any], dict[str, np.ndarray]]:
    """Reverse :func:`_pack_state` and validate the magic header."""
    if not isinstance(payload, (bytes, bytearray)):
        raise SerializationError("payload must be bytes", payload_type=type(payload).__name__)
    view = memoryview(bytes(payload))
    offset = 0
    magic, name_len = _HEADER.unpack_from(view, offset)
    offset += _HEADER.size
    if magic != _MAGIC:
        raise SerializationError("payload is not a SketchForge serialization", magic=bytes(magic))
    name = bytes(view[offset : offset + name_len]).decode("utf-8")
    offset += name_len
    (meta_len,) = _JSON_LEN.unpack_from(view, offset)
    offset += _JSON_LEN.size
    meta = json.loads(bytes(view[offset : offset + meta_len]).decode("utf-8"))
    offset += meta_len
    (array_count,) = _ARRAY_COUNT.unpack_from(view, offset)
    offset += _ARRAY_COUNT.size
    arrays: dict[str, np.ndarray] = {}
    for _ in range(array_count):
        (key_len,) = _ARRAY_NAME.unpack_from(view, offset)
        offset += _ARRAY_NAME.size
        key = bytes(view[offset : offset + key_len]).decode("utf-8")
        offset += key_len
        code, ndim = _ARRAY_META.unpack_from(view, offset)
        offset += _ARRAY_META.size
        shape: list[int] = []
        for _ in range(ndim):
            (dim,) = _ARRAY_DIM.unpack_from(view, offset)
            offset += _ARRAY_DIM.size
            shape.append(int(dim))
        (nbytes,) = _ARRAY_NBYTES.unpack_from(view, offset)
        offset += _ARRAY_NBYTES.size
        dtype = _DTYPE_BY_CODE.get(code)
        if dtype is None:
            raise SerializationError("unknown dtype code in payload", code=code)
        arrays[key] = np.frombuffer(
            bytes(view[offset : offset + nbytes]), dtype=dtype
        ).reshape(tuple(shape))
        offset += nbytes
    return name, meta, arrays


def _default_seed(seed: int | None) -> int:
    """Resolve an explicit seed, falling back to the configured root seed."""
    if seed is not None:
        return int(seed)
    try:
        return int(require_seed().root)
    except SketchForgeError:
        return 0


class AbstractSketch:
    """Common plumbing for every sketch: seeding, state codec, budget hooks."""

    SUPPORTS_MERGE: ClassVar[bool] = False
    DETERMINISTIC: ClassVar[bool] = True
    HASH_SEED_INJECTABLE: ClassVar[bool] = True
    TIER: ClassVar[int] = 1
    BACKEND: ClassVar[str] = "numpy"
    PARAM_AXIS: ClassVar[str] = "k"
    PARAM_RANGE: ClassVar[tuple[int, int]] = (1, 16)
    QUERY_TYPES: ClassVar[tuple[QueryType, ...]] = ()

    def __init__(self, *, seed: int | None = None, **params: Any) -> None:
        self._seed = _default_seed(seed)
        self._params: dict[str, Any] = dict(params)
        self._seed_state = SeedState(self._seed)

    # ------------------------------------------------------------------ introspection
    @property
    def seed(self) -> int:
        """Root seed used for every derived sub-stream."""
        return self._seed

    @property
    def params(self) -> dict[str, Any]:
        """Copy of the construction parameters."""
        return dict(self._params)

    def describe(self) -> MethodSpec:
        """Advertise the static capabilities of this implementation."""
        return MethodSpec(
            name=type(self).__name__,
            tier=self.TIER,
            backend=self.BACKEND,
            query_types=tuple(self.QUERY_TYPES),
            supports_merge=self.SUPPORTS_MERGE,
            deterministic=self.DETERMINISTIC,
            hash_seed_injectable=self.HASH_SEED_INJECTABLE,
        )

    # ------------------------------------------------------------------ random stream
    def _generator(self, tag: str) -> np.random.Generator:
        """Return a lazily created, seed-derived NumPy generator."""
        gen = getattr(self, "_gen", None)
        if gen is None:
            gen = np.random.default_rng(self._seed_state.derive(f"{type(self).__name__}:{tag}"))
            self._gen = gen
        return gen

    # ------------------------------------------------------------------ ingestion
    def update(self, item: object) -> None:
        """Fold a single item into the sketch."""
        raise NotImplementedError

    def update_batch(self, items: np.ndarray) -> None:
        """Fold an array of items in. Sub-classes must provide a vectorized version."""
        for item in np.asarray(items):
            self.update(item)

    # ------------------------------------------------------------------ queries
    def estimate(self, query: Query) -> Estimate:
        """Answer a normalized query."""
        raise NotImplementedError

    def _reject_query(self, query: Query, expected: QueryType) -> Estimate:
        if query.kind is not expected:
            raise EstimateUnavailableError(
                "query kind not supported by this sketch",
                sketch=type(self).__name__,
                kind=str(query.kind),
                expected=str(expected),
            )
        raise EstimateUnavailableError(
            "sketch cannot answer this query", sketch=type(self).__name__, kind=str(query.kind)
        )

    # ------------------------------------------------------------------ algebra
    def merge(self, other: AbstractSketch) -> AbstractSketch:
        if not self.SUPPORTS_MERGE:
            raise MergeUnsupportedError(
                "sketch does not support merge", sketch=type(self).__name__
            )
        if type(self) is not type(other):
            raise MergeTypeError(
                "cannot merge sketches of different types",
                left=type(self).__name__,
                right=type(other).__name__,
            )
        self._merge_same_type(other)
        return self

    def _merge_same_type(self, other: AbstractSketch) -> None:
        raise MergeUnsupportedError(
            "merge declared but not implemented", sketch=type(self).__name__
        )

    def _require_mergeable(self, other: AbstractSketch) -> None:
        if type(self) is not type(other):
            raise MergeTypeError(
                "cannot merge sketches of different types",
                left=type(self).__name__,
                right=type(other).__name__,
            )

    # ------------------------------------------------------------------ budget
    def memory_slots(self) -> int:
        """Declared number of logical slots (annotation only, never cross-method)."""
        raise NotImplementedError

    def memory_bytes(self) -> int:
        """Serialized byte footprint: the budget currency."""
        return len(self.serialize())

    # ------------------------------------------------------------------ state codec
    def _state(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Split instance state into NumPy arrays and JSON-able scalars."""
        arrays: dict[str, np.ndarray] = {}
        scalars: dict[str, Any] = {}
        for key, value in vars(self).items():
            if key in _SKIP_STATE_KEYS:
                continue
            if isinstance(value, np.ndarray):
                arrays[key] = value
            elif isinstance(value, (bool, int, float, str)):
                scalars[key] = value
            elif isinstance(value, np.generic):
                scalars[key] = value.item()
            elif value is None:
                scalars[key] = None
        return arrays, scalars

    def _restore(self, arrays: dict[str, np.ndarray], scalars: dict[str, Any]) -> None:
        """Re-install instance state produced by :meth:`_state`."""
        for key, value in arrays.items():
            setattr(self, key, value)
        for key, value in scalars.items():
            setattr(self, key, value)

    def serialize(self) -> bytes:
        """Serialize the full sketch state to bytes."""
        arrays, scalars = self._state()
        return _pack_state(
            type(self).__name__,
            {"params": self._params, "seed": self._seed, "scalars": scalars},
            arrays,
        )

    @classmethod
    def deserialize(cls, payload: bytes) -> AbstractSketch:
        """Rebuild a sketch from :meth:`serialize` output."""
        name, meta, arrays = _unpack_state(payload)
        if name != cls.__name__:
            raise SerializationError(
                "payload belongs to another sketch type", expected=cls.__name__, found=name
            )
        obj = cls(seed=int(meta.get("seed", 0)), **dict(meta.get("params", {})))
        scalars = dict(meta.get("scalars", {}))
        for key in ("seed", "params"):
            scalars.pop(key, None)
        obj._restore(arrays, scalars)
        return obj

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        """Map a single :attr:`PARAM_AXIS` value to a full parameter dict."""
        return {cls.PARAM_AXIS: int(value)}

    @classmethod
    def params_for_budget(cls, budget_bytes: int, **kwargs: Any) -> dict[str, Any]:
        """Largest parameters whose *post-ingest* serialized size fits the budget."""
        from sketchforge.sketches.budget import default_probe, fit_param_under_budget

        lo, hi = cls.PARAM_RANGE
        probe = kwargs.pop("probe", None)
        seed = kwargs.pop("seed", 0)
        params, _achieved = fit_param_under_budget(
            cls,
            budget_bytes,
            lo=lo,
            hi=hi,
            probe=default_probe() if probe is None else probe,
            seed=seed,
        )
        return params

    # ------------------------------------------------------------------ hashing helpers
    def _hash_u64(self, item: object) -> int:
        """Hash a scalar item through the single core hashing exit."""
        return int(hash_u64(stable_key(item), self._seed))

    def _index(self, item: object, width: int) -> int:
        return hash_index(stable_key(item), self._seed, width)

    def _sign(self, item: object) -> int:
        return hash_sign(stable_key(item), self._seed)

    def _array_u64(self, items: np.ndarray, tag: str) -> np.ndarray:
        """Vectorized hash of a numeric array using a derived sub-seed."""
        return hash_array_u64(np.asarray(items), self._seed_state.derive(tag))


def bit_length_u64(values: np.ndarray) -> np.ndarray:
    """Vectorized ``int.bit_length`` for unsigned 64-bit arrays (no Python loop)."""
    array = np.asarray(values, dtype=np.uint64)
    remaining = array.copy()
    length = np.zeros(array.shape, dtype=np.uint8)
    for shift in (32, 16, 8, 4, 2, 1):
        move = (remaining >> np.uint64(shift)) != np.uint64(0)
        length += move.astype(np.uint8) * np.uint8(shift)
        remaining = np.where(move, remaining >> np.uint64(shift), remaining)
    length += (remaining != np.uint64(0)).astype(np.uint8)
    return length.astype(np.uint8)
