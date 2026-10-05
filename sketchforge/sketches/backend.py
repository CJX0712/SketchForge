"""Tier-0 backends: thin, deterministic-by-contract wrappers over ``datasketches``.

Importing this module never fails: if ``datasketches`` is absent the adapters still
register (so discovery works) but raise :class:`BackendMissingError` on construction.
Each wrapper overrides ``serialize``/``memory_bytes`` with the library's own bytes so
the *serialized-size currency* stays honest.

Author: 晨星
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sketchforge.core.errors import (
    BackendMissingError,
    EstimateUnavailableError,
)
from sketchforge.core.types import Estimate, Query, QueryType
from sketchforge.sketches.base import AbstractSketch

try:  # pragma: no cover - exercised only when the optional dep is present
    import datasketches as _ds
except Exception as exc:
    _ds = None
    _DS_IMPORT_ERROR = exc


def backend_available() -> bool:
    """Return ``True`` when the optional ``datasketches`` backend can be imported."""
    return _ds is not None


def require_backend() -> Any:
    """Return the ``datasketches`` module or raise :class:`BackendMissingError`."""
    if _ds is None:
        raise BackendMissingError(
            "datasketches backend is not installed",
            detail=str(_DS_IMPORT_ERROR) if _DS_IMPORT_ERROR else None,
        )
    return _ds


def _nominal_bytes(payload: bytes) -> int:
    return len(payload)


class HllSketch(AbstractSketch):
    """HyperLogLog via ``datasketches.hll_sketch`` (no merge, non-deterministic)."""

    TIER = 0
    BACKEND = "datasketches"
    DETERMINISTIC = False
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.CARDINALITY,)
    PARAM_AXIS = "lg_k"
    PARAM_RANGE = (4, 21)

    def __init__(self, *, lg_k: int = 12, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._lg_k = int(lg_k)
        if _ds is None:
            raise BackendMissingError("datasketches backend is not installed")
        self._sk = _ds.hll_sketch(int(lg_k))

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"lg_k": int(value)}

    def update(self, item: object) -> None:
        self._sk.update(int(item))

    def update_batch(self, items: np.ndarray) -> None:
        for item in np.asarray(items):
            self._sk.update(int(item))

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.CARDINALITY:
            raise EstimateUnavailableError("HllSketch answers cardinality only")
        return Estimate(value=float(self._sk.get_estimate()), n_updates=0)

    def memory_slots(self) -> int:
        return 1 << self._lg_k

    def serialize(self) -> bytes:
        return bytes(self._sk.serialize_compact())

    def memory_bytes(self) -> int:
        return _nominal_bytes(self.serialize())

    @classmethod
    def deserialize(cls, payload: bytes) -> HllSketch:
        if _ds is None:
            raise BackendMissingError("datasketches backend is not installed")
        obj = cls(lg_k=12)
        obj._sk = _ds.hll_sketch.deserialize(payload)
        return obj


class CpcSketch(AbstractSketch):
    """Compressed Probabilistic Counting via ``datasketches.cpc_sketch`` (no merge)."""

    TIER = 0
    BACKEND = "datasketches"
    DETERMINISTIC = False
    SUPPORTS_MERGE = False
    HASH_SEED_INJECTABLE = True
    QUERY_TYPES = (QueryType.CARDINALITY,)
    PARAM_AXIS = "lg_k"
    PARAM_RANGE = (4, 24)

    def __init__(self, *, lg_k: int = 11, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._lg_k = int(lg_k)
        if _ds is None:
            raise BackendMissingError("datasketches backend is not installed")
        self._sk = _ds.cpc_sketch(int(lg_k), int(self._seed))

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"lg_k": int(value)}

    def update(self, item: object) -> None:
        self._sk.update(int(item))

    def update_batch(self, items: np.ndarray) -> None:
        for item in np.asarray(items):
            self._sk.update(int(item))

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.CARDINALITY:
            raise EstimateUnavailableError("CpcSketch answers cardinality only")
        return Estimate(value=float(self._sk.get_estimate()), n_updates=0)

    def memory_slots(self) -> int:
        return 1 << self._lg_k

    def serialize(self) -> bytes:
        return bytes(self._sk.serialize())

    def memory_bytes(self) -> int:
        return _nominal_bytes(self.serialize())

    @classmethod
    def deserialize(cls, payload: bytes) -> CpcSketch:
        if _ds is None:
            raise BackendMissingError("datasketches backend is not installed")
        obj = cls(lg_k=11, seed=0)
        obj._sk = _ds.cpc_sketch.deserialize(payload)
        return obj


class KllSketch(AbstractSketch):
    """KLL floats sketch via ``datasketches.kll_floats_sketch`` (non-deterministic)."""

    TIER = 0
    BACKEND = "datasketches"
    DETERMINISTIC = False
    SUPPORTS_MERGE = True
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.QUANTILE,)
    PARAM_AXIS = "k"
    PARAM_RANGE = (8, 2048)

    def __init__(self, *, k: int = 200, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._k = int(k)
        if _ds is None:
            raise BackendMissingError("datasketches backend is not installed")
        self._sk = _ds.kll_floats_sketch(int(k))

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"k": int(value)}

    def update(self, item: object) -> None:
        self._sk.update(float(item))

    def update_batch(self, items: np.ndarray) -> None:
        self._sk.update(np.asarray(items, dtype=np.float64))

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.QUANTILE:
            raise EstimateUnavailableError("KllSketch answers quantile only")
        q = 0.5 if query.q is None else float(query.q)
        value = float(self._sk.get_quantile(q))
        return Estimate(value=value, n_updates=0)

    @classmethod
    def declared_error(cls, k: int) -> float:
        """Normalized rank error bound declared by the library (used by gate G3)."""
        return float(_ds.kll_floats_sketch.get_normalized_rank_error(int(k), False)) if _ds else 0.0

    def memory_slots(self) -> int:
        return self._k

    def serialize(self) -> bytes:
        return bytes(self._sk.serialize())

    def memory_bytes(self) -> int:
        return _nominal_bytes(self.serialize())

    @classmethod
    def deserialize(cls, payload: bytes) -> KllSketch:
        if _ds is None:
            raise BackendMissingError("datasketches backend is not installed")
        obj = cls(k=200)
        obj._sk = _ds.kll_floats_sketch.deserialize(payload)
        return obj


class ReqSketch(AbstractSketch):
    """Relative-Error Quantile sketch via ``datasketches.req_floats_sketch`` (expensive)."""

    TIER = 0
    BACKEND = "datasketches"
    DETERMINISTIC = False
    SUPPORTS_MERGE = True
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.QUANTILE,)
    PARAM_AXIS = "k"
    PARAM_RANGE = (4, 64)

    def __init__(self, *, k: int = 12, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._k = int(k)
        if _ds is None:
            raise BackendMissingError("datasketches backend is not installed")
        self._sk = _ds.req_floats_sketch(int(k))

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"k": int(value)}

    def update(self, item: object) -> None:
        self._sk.update(float(item))

    def update_batch(self, items: np.ndarray) -> None:
        self._sk.update(np.asarray(items, dtype=np.float64))

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.QUANTILE:
            raise EstimateUnavailableError("ReqSketch answers quantile only")
        q = 0.5 if query.q is None else float(query.q)
        return Estimate(value=float(self._sk.get_quantile(q)), n_updates=0)

    def memory_slots(self) -> int:
        return self._k

    def serialize(self) -> bytes:
        return bytes(self._sk.serialize())

    def memory_bytes(self) -> int:
        return _nominal_bytes(self.serialize())


class CountMinSketch(AbstractSketch):
    """Count-Min sketch via ``datasketches.count_min_sketch`` (Tier-0 reference)."""

    TIER = 0
    BACKEND = "datasketches"
    DETERMINISTIC = True
    SUPPORTS_MERGE = True
    HASH_SEED_INJECTABLE = True
    QUERY_TYPES = (QueryType.FREQUENCY,)
    PARAM_AXIS = "lg_width"
    PARAM_RANGE = (4, 18)

    def __init__(self, *, lg_width: int = 10, depth: int = 4, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._lg_width = int(lg_width)
        self._depth = int(depth)
        if _ds is None:
            raise BackendMissingError("datasketches backend is not installed")
        self._sk = _ds.count_min_sketch(int(depth), 1 << int(lg_width), int(self._seed))

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"lg_width": int(value), "depth": 4}

    def update(self, item: object) -> None:
        self._sk.update(int(item))

    def update_batch(self, items: np.ndarray) -> None:
        for item in np.asarray(items):
            self._sk.update(int(item))

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.FREQUENCY:
            raise EstimateUnavailableError("CountMinSketch answers frequency only")
        key = int(query.key) if query.key is not None else 0
        return Estimate(value=float(self._sk.get_estimate(key)), n_updates=0)

    def memory_slots(self) -> int:
        return self._depth * (1 << self._lg_width)

    def serialize(self) -> bytes:
        return bytes(self._sk.serialize())

    def memory_bytes(self) -> int:
        return _nominal_bytes(self.serialize())


class FrequentStringsSketch(AbstractSketch):
    """Heavy-hitters via ``datasketches.frequent_strings_sketch`` (str-encoded keys)."""

    TIER = 0
    BACKEND = "datasketches"
    DETERMINISTIC = True
    SUPPORTS_MERGE = True
    HASH_SEED_INJECTABLE = False
    QUERY_TYPES = (QueryType.HEAVY_HITTERS,)
    PARAM_AXIS = "lg_max_k"
    PARAM_RANGE = (4, 18)

    def __init__(self, *, lg_max_k: int = 10, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._lg_max_k = int(lg_max_k)
        if _ds is None:
            raise BackendMissingError("datasketches backend is not installed")
        self._sk = _ds.frequent_strings_sketch(int(lg_max_k))

    @classmethod
    def params_from_axis(cls, value: int) -> dict[str, Any]:
        return {"lg_max_k": int(value)}

    def update(self, item: object) -> None:
        self._sk.update(str(item))

    def update_batch(self, items: np.ndarray) -> None:
        for item in np.asarray(items):
            self._sk.update(str(int(item)))

    def estimate(self, query: Query) -> Estimate:
        if query.kind is not QueryType.HEAVY_HITTERS:
            raise EstimateUnavailableError("FrequentStringsSketch answers heavy hitters only")
        top_k = query.top_k or 10
        rows = self._sk.get_frequent_items(self._sk.NO_FALSE_NEGATIVES)
        items = [(int(row[0]), int(row[1])) for row in rows[:top_k]]
        return Estimate(value=items, n_updates=0)

    def memory_slots(self) -> int:
        return 1 << self._lg_max_k

    def serialize(self) -> bytes:
        return bytes(self._sk.serialize())

    def memory_bytes(self) -> int:
        return _nominal_bytes(self.serialize())
