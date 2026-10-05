"""Sketch implementations and the discovery registry.

Importing this package registers every sketch class with
:func:`sketchforge.sketches.registry.register`, so callers can filter by query
family and determinism without importing submodules directly.

Author: 晨星
"""

from __future__ import annotations

from sketchforge.sketches import backend, cardinality, frequency, heavyhitters, quantile
from sketchforge.sketches.budget import (
    default_probe,
    fit_param_under_budget,
    largest_remainder,
    make_plan,
)
from sketchforge.sketches.registry import (
    all_names,
    discover,
    register,
    spec_of,
    specs,
)

__all__ = [
    "all_names",
    "backend",
    "budget",
    "cardinality",
    "default_probe",
    "discover",
    "fit_param_under_budget",
    "frequency",
    "heavyhitters",
    "largest_remainder",
    "make_plan",
    "quantile",
    "register",
    "registry",
    "spec_of",
    "specs",
]
