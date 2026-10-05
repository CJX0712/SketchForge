"""Load local CSV, JSONL, and NPY streams with explicit failure semantics.

Author: 晨星
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from sketchforge.core.errors import DataLoadError, EmptyStreamError, SketchForgeError
from sketchforge.core.io_utils import open_text
from sketchforge.data.specs import StreamSpec
from sketchforge.data.streams import Stream


def _loaded_stream(keys: object, values: object) -> Stream:
    key_array = np.asarray(keys, dtype=np.int64)
    value_array = np.asarray(values, dtype=np.float64)
    if key_array.ndim != 1 or value_array.ndim != 1 or len(key_array) != len(value_array):
        raise DataLoadError("loaded key and value columns must be one-dimensional and equal length")
    if len(key_array) == 0:
        raise EmptyStreamError("loaded stream is empty")
    spec = StreamSpec(
        kind="uniform",
        n=len(key_array),
        domain=max(1, int(np.unique(key_array).size)),
        seed=0,
        value_dist="loaded",
    )
    return Stream(keys=key_array, values=value_array, spec=spec)


def _load_csv(path: Path, key_col: str | int | None, value_col: str | int | None) -> Stream:
    key_name = "key" if key_col is None else str(key_col)
    value_name = "value" if value_col is None else str(value_col)
    with open_text(path, "r") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = {key_name, value_name} - fields
        if missing:
            raise DataLoadError("CSV is missing required columns", missing=sorted(missing))
        keys: list[int] = []
        values: list[float] = []
        for row in reader:
            keys.append(int(row[key_name]))
            values.append(float(row[value_name]))
    return _loaded_stream(keys, values)


def _load_jsonl(path: Path, key_col: str | int | None, value_col: str | int | None) -> Stream:
    key_name = "key" if key_col is None else str(key_col)
    value_name = "value" if value_col is None else str(value_col)
    keys: list[int] = []
    values: list[float] = []
    with open_text(path, "r") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise DataLoadError("JSONL row must be an object", line=line_number)
            missing = {key_name, value_name} - record.keys()
            if missing:
                raise DataLoadError(
                    "JSONL row is missing required columns",
                    line=line_number,
                    missing=sorted(missing),
                )
            keys.append(int(record[key_name]))
            values.append(float(record[value_name]))
    return _loaded_stream(keys, values)


def _load_npy(path: Path, key_col: str | int | None, value_col: str | int | None) -> Stream:
    array = np.load(path, allow_pickle=False)
    if array.dtype.names:
        key_name = "key" if key_col is None else str(key_col)
        value_name = "value" if value_col is None else str(value_col)
        fields = set(array.dtype.names)
        missing = {key_name, value_name} - fields
        if missing:
            raise DataLoadError("NPY is missing required fields", missing=sorted(missing))
        return _loaded_stream(array[key_name], array[value_name])
    if array.ndim != 2:
        raise DataLoadError("plain NPY input must be a two-dimensional array")
    key_index = 0 if key_col is None else int(key_col)
    value_index = 1 if value_col is None else int(value_col)
    if min(key_index, value_index) < 0 or max(key_index, value_index) >= array.shape[1]:
        raise DataLoadError(
            "NPY column index is out of range",
            columns=array.shape[1],
            key_col=key_index,
            value_col=value_index,
        )
    return _loaded_stream(array[:, key_index], array[:, value_index])


def load(
    path: str | os.PathLike[str],
    *,
    fmt: str | None = None,
    key_col: str | int | None = None,
    value_col: str | int | None = None,
) -> Stream:
    """Load a stream from CSV, JSONL, or NPY data."""
    source = Path(path)
    format_name = (fmt or source.suffix.removeprefix(".")).lower()
    loaders = {"csv": _load_csv, "jsonl": _load_jsonl, "npy": _load_npy}
    if format_name not in loaders:
        raise DataLoadError("unsupported stream format", fmt=format_name)
    if not source.is_file():
        raise DataLoadError("stream file does not exist", path=str(source))
    try:
        return loaders[format_name](source, key_col, value_col)
    except SketchForgeError:
        raise
    except (OSError, TypeError, ValueError, json.JSONDecodeError, csv.Error) as exc:
        raise DataLoadError("failed to load stream", path=str(source), fmt=format_name) from exc
