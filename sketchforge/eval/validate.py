"""Determinism gates (G1/G2/G3) and pre-registered superiority gates.

* G1/G2 are *re-run* determinism checks: two independent runs of the same
  configuration must produce an identical canonical digest (after volatile fields
  and stochastic values are masked).
* G3 bounds every non-deterministic record's value by its library-declared error
  bound (no self-reported tolerances).
* The superiority gates (G-1..G-4, ablation, no-harm) are evaluated against
  pre-registered thresholds; they are *not* tuned to the observed numbers.

Author: 晨星
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from sketchforge.core.errors import DeterminismViolationError

# Fields that legitimately differ between runs and must be masked before comparison.
VOLATILE_PATHS = ("generated_at", "records[].elapsed_sec", "aggregates[].elapsed_sec", "environment.hostname", "environment.pid")

STOCHASTIC_SENTINEL = "<stochastic>"
VOLATILE_SENTINEL = "<volatile>"


def _mask_volatile(obj: Any, path: str = "") -> Any:
    """Replace leaves whose dotted ``path`` is registered in :data:`VOLATILE_PATHS`."""
    if isinstance(obj, dict):
        return {k: _mask_volatile(v, f"{path}.{k}") for k, v in obj.items()}
    if isinstance(obj, list):
        return [_mask_volatile(v, f"{path}[]") for v in obj]
    if path.lstrip(".") in VOLATILE_PATHS:
        return VOLATILE_SENTINEL
    return obj


def _mask_stochastic(obj: Any, stochastic_keys: set[str] | None = None) -> Any:
    """Replace ``value``/``score`` of records flagged ``deterministic: false``."""
    if stochastic_keys is None:
        stochastic_keys = set()
    if isinstance(obj, dict):
        is_stochastic = obj.get("deterministic") is False or obj.get("deterministic") == "false"
        out = {}
        for k, v in obj.items():
            if is_stochastic and k in ("value", "score"):
                out[k] = STOCHASTIC_SENTINEL
            else:
                out[k] = _mask_stochastic(v, stochastic_keys)
        return out
    if isinstance(obj, list):
        return [_mask_stochastic(v, stochastic_keys) for v in obj]
    return obj


def canonicalize(record: dict[str, Any], stochastic_keys: set[str] | None = None) -> str:
    """Return a stable JSON string with volatile and stochastic fields masked."""
    masked = _mask_stochastic(_mask_volatile(copy.deepcopy(record)), stochastic_keys)
    return json.dumps(masked, sort_keys=True, ensure_ascii=False)


def digest(record: dict[str, Any], stochastic_keys: set[str] | None = None) -> str:
    """SHA-256 hex digest of the canonical form of ``record``."""
    return hashlib.sha256(canonicalize(record, stochastic_keys).encode("utf-8")).hexdigest()


def assert_rerun_determinism(
    run_a: dict[str, Any], run_b: dict[str, Any], stochastic_keys: set[str] | None = None
) -> None:
    """Raise :class:`DeterminismViolationError` if two runs disagree after masking."""
    if digest(run_a, stochastic_keys) != digest(run_b, stochastic_keys):
        raise DeterminismViolationError(
            "two runs produced different canonical digests",
            digest_a=digest(run_a, stochastic_keys)[:16],
            digest_b=digest(run_b, stochastic_keys)[:16],
        )


def check_g3(record: dict[str, Any], safety: float) -> bool:
    """Return ``True`` when a non-deterministic record's value respects its bound.

    ``record`` must carry ``value``, ``declared_bound``, and ``deterministic``.
    Deterministic records always pass; stochastic records must satisfy
    ``value <= declared_bound * safety``.
    """
    if record.get("deterministic") is not False:
        return True
    declared = record.get("declared_bound")
    value = record.get("value")
    if declared is None or value is None:
        return False
    return float(value) <= float(declared) * float(safety)


@dataclass
class GateReport:
    """Outcome of the pre-registered superiority gates."""

    delta: float
    g1_max_lt_1: bool
    g2_mean_le_1_minus_delta: bool
    g3_flagship_vs_oracle_le_1_02: bool
    g4_model_consistency: bool
    ablation_le_0_95: bool
    ratios: list[float]
    predicted_ratio: float
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Primary gate: flagship strictly and robustly beats the frozen baseline."""
        return bool(
            self.g1_max_lt_1
            and self.g2_mean_le_1_minus_delta
            and self.g3_flagship_vs_oracle_le_1_02
            and self.g4_model_consistency
            and self.ablation_le_0_95
        )


def evaluate_gates(
    ratios: list[float],
    per_segment_ratios: list[float],
    ablation_ratios: list[float],
    predicted_ratio: float,
    delta: float = 0.15,
) -> GateReport:
    """Evaluate the pre-registered superiority gates (architecture/model-card §10)."""
    ratios = list(ratios)
    notes: list[str] = []
    if not ratios:
        notes.append("no ratios supplied")
        return GateReport(
            delta=delta,
            g1_max_lt_1=False,
            g2_mean_le_1_minus_delta=False,
            g3_flagship_vs_oracle_le_1_02=False,
            g4_model_consistency=False,
            ablation_le_0_95=False,
            ratios=ratios,
            predicted_ratio=predicted_ratio,
            notes=notes,
        )
    g1 = bool(max(ratios) < 1.0)
    mean_ratio = float(sum(ratios) / len(ratios))
    g2 = bool(mean_ratio <= 1.0 - delta)
    oracle_ratio = float(sum(per_segment_ratios) / len(per_segment_ratios)) if per_segment_ratios else float("nan")
    g3 = bool(oracle_ratio <= 1.02)
    g4 = bool(abs(predicted_ratio - mean_ratio) <= 0.03)
    ablation = float(sum(ablation_ratios) / len(ablation_ratios)) if ablation_ratios else float("nan")
    g_ablation = bool(ablation <= 0.95)
    if not g1:
        notes.append(f"max G_ratio={max(ratios):.4f} not < 1.0")
    if not g2:
        notes.append(f"mean G_ratio={mean_ratio:.4f} > {1.0 - delta:.2f}")
    if not g3:
        notes.append(f"flagship/per-segment={oracle_ratio:.4f} > 1.02")
    if not g4:
        notes.append(f"|predicted-mean|={abs(predicted_ratio - mean_ratio):.4f} > 0.03")
    if not g_ablation:
        notes.append(f"flagship/uniform={ablation:.4f} > 0.95")
    return GateReport(
        delta=delta,
        g1_max_lt_1=g1,
        g2_mean_le_1_minus_delta=g2,
        g3_flagship_vs_oracle_le_1_02=g3,
        g4_model_consistency=g4,
        ablation_le_0_95=g_ablation,
        ratios=ratios,
        predicted_ratio=predicted_ratio,
        notes=notes,
    )
