"""Sketch method registry with capability discovery.

The registry is populated by importing the family modules, which call
:func:`register` at import time. Discovery filters by query family and
determinism so the evaluation gate can select only reproducible methods.

Author: 晨星
"""

from __future__ import annotations

from sketchforge.core.types import MethodSpec, QueryType

_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    """Register a sketch class (idempotent on name)."""
    _REGISTRY[cls.__name__] = cls
    return cls


def spec_of(cls: type) -> MethodSpec:
    """Advertise a registered class's static capabilities without instantiating it."""
    return MethodSpec(
        name=cls.__name__,
        tier=int(getattr(cls, "TIER", 1)),
        backend=str(getattr(cls, "BACKEND", "numpy")),
        query_types=tuple(getattr(cls, "QUERY_TYPES", ())),
        supports_merge=bool(getattr(cls, "SUPPORTS_MERGE", False)),
        deterministic=bool(getattr(cls, "DETERMINISTIC", True)),
        hash_seed_injectable=bool(getattr(cls, "HASH_SEED_INJECTABLE", True)),
    )


def discover(
    query_type: QueryType | str | None = None,
    *,
    deterministic_only: bool = False,
    tier: int | None = None,
    backend: str | None = None,
) -> list[type]:
    """Return registered sketch classes matching the given filters."""
    key = query_type.value if isinstance(query_type, QueryType) else query_type
    out: list[type] = []
    for cls in _REGISTRY.values():
        types = {t.value if isinstance(t, QueryType) else str(t) for t in getattr(cls, "QUERY_TYPES", ())}
        if key is not None and key not in types:
            continue
        if deterministic_only and not getattr(cls, "DETERMINISTIC", True):
            continue
        if tier is not None and int(getattr(cls, "TIER", 1)) != tier:
            continue
        if backend is not None and str(getattr(cls, "BACKEND", "numpy")) != backend:
            continue
        out.append(cls)
    return out


def specs(query_type: QueryType | str | None = None, **filters: object) -> list[MethodSpec]:
    """Return :class:`MethodSpec` for every matching registered class."""
    deterministic = filters.pop("deterministic_only", False)
    classes = discover(query_type, deterministic_only=bool(deterministic), **filters)  # type: ignore[arg-type]
    return [spec_of(c) for c in classes]


def all_names() -> list[str]:
    """Return all registered sketch class names."""
    return sorted(_REGISTRY.keys())
