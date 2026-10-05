"""Per-query error metrics and the shared evaluation entry point.

Every metric returns a *loss* (higher is worse). Q1/Q2/Q3 use relative error; Q4
uses ``1 - F1`` so that all four families point in the same direction and can be
combined by a single geometric mean in :mod:`sketchforge.eval.aggregate`.

Author: 晨星
"""

from __future__ import annotations

import numpy as np

from sketchforge.core.errors import EstimateUnavailableError
from sketchforge.core.types import Estimate, Query, QueryType
from sketchforge.data.oracle import ExactOracle
from sketchforge.data.streams import Stream

# Column the sketch for each query family is fed from.
QUERY_COLUMN: dict[QueryType, str] = {
    QueryType.CARDINALITY: "keys",
    QueryType.QUANTILE: "values",
    QueryType.FREQUENCY: "keys",
    QueryType.HEAVY_HITTERS: "keys",
}

# Normalization constant per query family (see architecture §4). It cancels in any
# strategy-to-strategy ratio, so its absolute value is a documentation choice.
SPEC_TOLERANCE: dict[QueryType, float] = {
    QueryType.CARDINALITY: 0.05,
    QueryType.QUANTILE: 0.02,
    QueryType.FREQUENCY: 0.05,
    QueryType.HEAVY_HITTERS: 0.10,
}

METRIC_NAMES: dict[QueryType, str] = {
    QueryType.CARDINALITY: "cardinality_rel_err",
    QueryType.QUANTILE: "quantile_norm_rank_err",
    QueryType.FREQUENCY: "frequency_rel_err",
    QueryType.HEAVY_HITTERS: "heavy_hitter_one_minus_f1",
}


def cardinality_rel_err(estimate: float, true_cardinality: int) -> float:
    if true_cardinality <= 0:
        return 0.0
    return abs(estimate - true_cardinality) / true_cardinality


def quantile_norm_rank_err(estimate: float, oracle: ExactOracle, q: float, n: int) -> float:
    if n <= 0:
        return 0.0
    rank_est = oracle.rank(float(estimate))
    return abs(rank_est - q)


def frequency_rel_err(estimate: float, true_frequency: int) -> float:
    return abs(estimate - true_frequency) / max(true_frequency, 1)


def heavy_hitter_one_minus_f1(estimate: list[tuple[int, int]], oracle: ExactOracle, top_k: int) -> float:
    predicted = {(int(k)) for k, _ in estimate}
    truth = {int(k) for k, _ in oracle.top_k(top_k)}
    if not predicted and not truth:
        return 0.0
    if not predicted or not truth:
        return 1.0
    tp = len(predicted & truth)
    precision = tp / len(predicted)
    recall = tp / len(truth)
    if precision + recall <= 0:
        return 1.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return max(0.0, 1.0 - f1)


def query_loss(
    query_type: QueryType, estimate: Estimate, oracle: ExactOracle, query: Query, n: int
) -> float:
    """Return the loss for a single answered query (higher is worse)."""
    if query_type is QueryType.CARDINALITY:
        return cardinality_rel_err(float(estimate.value), oracle.cardinality())
    if query_type is QueryType.QUANTILE:
        q = 0.5 if query.q is None else float(query.q)
        return quantile_norm_rank_err(float(estimate.value), oracle, q, n)
    if query_type is QueryType.FREQUENCY:
        return frequency_rel_err(float(estimate.value), oracle.frequency(query.key))
    if query_type is QueryType.HEAVY_HITTERS:
        top_k = query.top_k or 10
        if not isinstance(estimate.value, list):
            raise EstimateUnavailableError("heavy hitter estimate must be a list of (key, count)")
        return heavy_hitter_one_minus_f1(estimate.value, oracle, top_k)
    raise EstimateUnavailableError("unknown query type", kind=str(query_type))


def column_for(query_type: QueryType, stream: Stream) -> np.ndarray:
    """Return the stream column a sketch for ``query_type`` should ingest."""
    return getattr(stream, QUERY_COLUMN[query_type])
