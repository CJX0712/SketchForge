"""Pipeline package: shared evaluation entry point and orchestration.

Author: 晨星
"""

from sketchforge.pipeline.stages import (
    BenchmarkResult,
    build_context,
    evaluate_allocation,
    fit_laws,
    run_benchmark,
)

__all__ = [
    "BenchmarkResult",
    "build_context",
    "evaluate_allocation",
    "fit_laws",
    "run_benchmark",
]
