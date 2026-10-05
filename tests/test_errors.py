"""Error-catalog collision tests.

Author: 晨星
"""

import pytest

import sketchforge.core.errors as errors_module
from sketchforge.core.errors import (
    InvalidBackendModeError,
    InvalidConfigError,
    SketchForgeError,
    catalog,
    error_class,
)


def test_catalog_contains_backend_mode_error() -> None:
    assert error_class("E103") is InvalidBackendModeError
    assert catalog()["E101"] is InvalidConfigError


def test_duplicate_code_guard_really_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(errors_module, "_ERROR_DEFINITIONS", [])
    type("FirstSyntheticError", (SketchForgeError,), {"code": "E990"})
    type("SecondSyntheticError", (SketchForgeError,), {"code": "E990"})
    with pytest.raises(RuntimeError, match="duplicate error code E990"):
        errors_module._discover_registry()


def test_duplicate_name_guard_really_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(errors_module, "_ERROR_DEFINITIONS", [])
    type("SameSyntheticError", (SketchForgeError,), {"code": "E991"})
    type("SameSyntheticError", (SketchForgeError,), {"code": "E992"})
    with pytest.raises(RuntimeError, match="duplicate error class name SameSyntheticError"):
        errors_module._discover_registry()
