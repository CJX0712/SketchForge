"""Small runtime and environment probes without mandatory psutil.

Author: 晨星
"""

from __future__ import annotations

import importlib.metadata
import platform
import sys
import time
import tracemalloc
from dataclasses import dataclass, field

import numpy as np
import scipy


def now_sec() -> float:
    """Return a monotonic high-resolution timestamp."""
    return time.perf_counter()


def rss_bytes() -> int | None:
    """Return process RSS, falling back to traced Python allocations when needed."""
    try:
        import psutil
    except ImportError:
        try:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            current, _ = tracemalloc.get_traced_memory()
            return current
        except (RuntimeError, ValueError):
            return None
    return int(psutil.Process().memory_info().rss)


@dataclass
class PeakTracker:
    """Track elapsed time and the highest observed process-memory reading."""

    started_at: float | None = None
    elapsed_sec: float = 0.0
    peak_bytes: int | None = None
    _running: bool = field(default=False, init=False, repr=False)

    def start(self) -> PeakTracker:
        self.started_at = now_sec()
        self.elapsed_sec = 0.0
        self.peak_bytes = rss_bytes()
        self._running = True
        return self

    def sample(self) -> int | None:
        current = rss_bytes()
        if current is not None and (self.peak_bytes is None or current > self.peak_bytes):
            self.peak_bytes = current
        return current

    def stop(self) -> PeakTracker:
        if self._running and self.started_at is not None:
            self.sample()
            self.elapsed_sec = now_sec() - self.started_at
            self._running = False
        return self

    def __enter__(self) -> PeakTracker:
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.stop()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def env_probe() -> dict[str, str | None]:
    """Return reproducibility-relevant runtime versions."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "datasketches": _package_version("datasketches"),
    }
