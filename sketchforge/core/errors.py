"""Stable SketchForge error taxonomy with collision detection.

Author: 晨星
"""

from __future__ import annotations

from collections.abc import Iterable

_ERROR_DEFINITIONS: list[type[SketchForgeError]] = []


class SketchForgeError(Exception):
    """Base class for errors with machine-readable codes and context."""

    code = "E000"

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _ERROR_DEFINITIONS.append(cls)

    def __init__(self, message: str, **ctx: object) -> None:
        super().__init__(message)
        self.code = type(self).code
        self.message = message
        self.ctx = ctx


class InvalidConfigError(SketchForgeError):
    code = "E101"


class MissingSeedError(SketchForgeError):
    code = "E102"


class InvalidBackendModeError(SketchForgeError):
    code = "E103"


class BudgetConservationError(SketchForgeError):
    code = "E104"


class InvalidSpecError(SketchForgeError):
    code = "E201"


class DataLoadError(SketchForgeError):
    code = "E202"


class EmptyStreamError(SketchForgeError):
    code = "E203"


class OracleUnavailableError(SketchForgeError):
    code = "E204"


class InvalidSketchParameterError(SketchForgeError):
    code = "E301"


class IndistinguishableBudgetError(SketchForgeError):
    code = "E302"


class MergeTypeError(SketchForgeError):
    code = "E303"


class MergeUnsupportedError(SketchForgeError):
    code = "E304"


class SerializationError(SketchForgeError):
    code = "E305"


class EstimateUnavailableError(SketchForgeError):
    code = "E306"


class BackendMissingError(SketchForgeError):
    code = "E401"


class BackendImportError(SketchForgeError):
    code = "E402"


class DeterminismViolationError(SketchForgeError):
    code = "E501"


class MetricOutOfBoundsError(SketchForgeError):
    code = "E502"


class IndistinguishableCostError(SketchForgeError):
    code = "E503"


class PipelineTimeoutError(SketchForgeError):
    code = "E505"


_ERROR_REGISTRY: dict[str, type[SketchForgeError]] = {}


def _discover_registry(
    classes: Iterable[type[SketchForgeError]] | None = None,
) -> dict[str, type[SketchForgeError]]:
    """Discover error classes and reject duplicate codes or class names."""
    candidates = tuple(_ERROR_DEFINITIONS if classes is None else classes)
    registry: dict[str, type[SketchForgeError]] = {}
    names: dict[str, type[SketchForgeError]] = {}
    for error_type in candidates:
        if not issubclass(error_type, SketchForgeError):
            raise RuntimeError(f"non-SketchForge error in registry: {error_type!r}")
        code = error_type.code
        previous_code = registry.get(code)
        if previous_code is not None:
            raise RuntimeError(
                f"duplicate error code {code}: {previous_code.__name__}, {error_type.__name__}"
            )
        previous_name = names.get(error_type.__name__)
        if previous_name is not None:
            raise RuntimeError(
                f"duplicate error class name {error_type.__name__}: "
                f"{previous_name.code}, {error_type.code}"
            )
        registry[code] = error_type
        names[error_type.__name__] = error_type
    if classes is None:
        _ERROR_REGISTRY.clear()
        _ERROR_REGISTRY.update(registry)
    return registry


def error_class(code: str) -> type[SketchForgeError]:
    """Return the registered exception class for ``code``."""
    return _ERROR_REGISTRY[code]


def catalog() -> dict[str, type[SketchForgeError]]:
    """Return a copy of the code-to-class error catalog."""
    return dict(_ERROR_REGISTRY)


_discover_registry()
