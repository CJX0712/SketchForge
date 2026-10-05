"""Encoding-safe and atomic local file helpers.

Author: 晨星
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

JsonValue = None | bool | int | float | str | Sequence["JsonValue"] | Mapping[str, "JsonValue"]


def open_text(path: str | os.PathLike[str], mode: str, encoding: str = "utf-8") -> TextIO:
    """Open a text file with an explicit encoding on every platform."""
    if "b" in mode:
        raise ValueError("open_text does not accept binary modes")
    return Path(path).open(mode=mode, encoding=encoding, newline="")


def dump_json(
    path: str | os.PathLike[str],
    obj: Any,
    sort_keys: bool = True,
    indent: int = 2,
) -> None:
    """Atomically replace a UTF-8 JSON document in the destination directory."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        with open_text(temporary, "w") as stream:
            json.dump(obj, stream, ensure_ascii=False, sort_keys=sort_keys, indent=indent)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
