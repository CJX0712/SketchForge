"""Data-loader success and failure tests.

Author: 晨星
"""

import json

import numpy as np
import pytest

from sketchforge.core.errors import DataLoadError
from sketchforge.data.loaders import load


def test_load_csv_jsonl_and_npy(tmp_path) -> None:
    csv_path = tmp_path / "stream.csv"
    csv_path.write_text("key,value\n1,1.5\n2,2.5\n", encoding="utf-8")
    csv_stream = load(csv_path)
    np.testing.assert_array_equal(csv_stream.keys, np.array([1, 2], dtype=np.int64))

    jsonl_path = tmp_path / "stream.jsonl"
    records = [{"key": 3, "value": 3.5}, {"key": 4, "value": 4.5}]
    jsonl_path.write_text(
        "".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8"
    )
    jsonl_stream = load(jsonl_path)
    np.testing.assert_allclose(jsonl_stream.values, [3.5, 4.5])

    npy_path = tmp_path / "stream.npy"
    np.save(npy_path, np.array([[5, 5.5], [6, 6.5]], dtype=np.float64))
    npy_stream = load(npy_path)
    np.testing.assert_array_equal(npy_stream.keys, np.array([5, 6], dtype=np.int64))


def test_missing_columns_and_unknown_formats_raise_e202(tmp_path) -> None:
    missing = tmp_path / "missing.csv"
    missing.write_text("only\n1\n", encoding="utf-8")
    with pytest.raises(DataLoadError) as missing_error:
        load(missing)
    assert missing_error.value.code == "E202"

    unknown = tmp_path / "stream.txt"
    unknown.write_text("data", encoding="utf-8")
    with pytest.raises(DataLoadError, match="unsupported"):
        load(unknown)

    with pytest.raises(DataLoadError, match="does not exist"):
        load(tmp_path / "absent.csv")
