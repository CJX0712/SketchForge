"""Environment-driven SketchForge configuration.

Author: 晨星
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sketchforge.core.errors import InvalidBackendModeError, InvalidConfigError

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise InvalidConfigError("invalid boolean environment value", name=name, value=value)


@dataclass
class Config:
    """Top-level reproducible runtime configuration."""

    seed: int = 1337
    seeds: tuple[int, ...] = (1337, 1338, 1339)
    backend: str = "auto"
    n: int = 100_000
    datasets: tuple[str, ...] = (
        "zipf_a1.1",
        "drift_abrupt",
        "low_card",
        "elephant_mice",
    )
    budgets_bytes: tuple[int, ...] = (4096, 65536)
    deterministic_only: bool = False
    out_dir: Path = Path("artifacts")
    time_budget_sec: float = 60.0
    n_jobs: int = 1
    safety: float = 3.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        """Build a configuration from supported environment variables."""
        source = os.environ if env is None else env
        kwargs: dict[str, object] = {}
        try:
            if (value := source.get("SKETCHFORGE_SEED")) is not None:
                seed = int(value)
                kwargs["seed"] = seed
                kwargs["seeds"] = (seed, seed + 1, seed + 2)
            if (value := source.get("SKETCHFORGE_BACKEND")) is not None:
                kwargs["backend"] = value.strip().lower()
            if (value := source.get("SKETCHFORGE_N")) is not None:
                kwargs["n"] = int(value)
            if (value := source.get("SKETCHFORGE_BUDGETS")) is not None:
                kwargs["budgets_bytes"] = tuple(
                    int(part.strip()) for part in value.split(",") if part.strip()
                )
            if (value := source.get("SKETCHFORGE_OUT_DIR")) is not None:
                kwargs["out_dir"] = Path(value)
            if (value := source.get("SKETCHFORGE_DETERMINISTIC_ONLY")) is not None:
                kwargs["deterministic_only"] = _parse_bool("SKETCHFORGE_DETERMINISTIC_ONLY", value)
            if (value := source.get("SKETCHFORGE_SAFETY")) is not None:
                kwargs["safety"] = float(value)
            if (value := source.get("SKETCHFORGE_NO_TIER0")) is not None and _parse_bool(
                "SKETCHFORGE_NO_TIER0", value
            ):
                kwargs["backend"] = "tier1"
        except (TypeError, ValueError) as exc:
            raise InvalidConfigError("failed to parse environment configuration") from exc
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        """Reject configurations that break reproducibility or budget invariants."""
        if self.backend not in {"auto", "tier0", "tier1"}:
            raise InvalidBackendModeError(
                "backend must be auto, tier0, or tier1", backend=self.backend
            )
        if self.n <= 0:
            raise InvalidConfigError("n must be positive", n=self.n)
        if self.n_jobs != 1:
            raise InvalidConfigError(
                "n_jobs must remain 1 for deterministic execution", n_jobs=self.n_jobs
            )
        if not self.budgets_bytes or any(value <= 0 for value in self.budgets_bytes):
            raise InvalidConfigError(
                "budgets_bytes must contain only positive values", budgets=self.budgets_bytes
            )
        if not self.seeds:
            raise InvalidConfigError("at least one evaluation seed is required")
        if self.time_budget_sec <= 0:
            raise InvalidConfigError(
                "time_budget_sec must be positive", time_budget_sec=self.time_budget_sec
            )
        if self.safety <= 0:
            raise InvalidConfigError("safety must be positive", safety=self.safety)
