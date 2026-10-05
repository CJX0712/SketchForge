"""Static architectural boundary tests.

Author: 晨星
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "sketchforge"
_FORBIDDEN_CORE_PREFIXES = (
    "sketchforge.data",
    "sketchforge.sketches",
    "sketchforge.pipeline",
)
_BARE_HASH_CALL = re.compile(r"(?<![\w.])hash\s*\(")


def _python_files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*.py") if ".venv" not in path.parts)


def test_core_never_imports_upper_layers() -> None:
    offenders: list[str] = []
    for path in sorted((PACKAGE / "core").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if module.startswith(_FORBIDDEN_CORE_PREFIXES):
                    offenders.append(f"{path.relative_to(ROOT)} imports {module}")
    assert offenders == []


def test_no_cross_top_level_parent_relative_imports() -> None:
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level >= 2:
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_builtin_hash_calls_are_forbidden_repository_wide() -> None:
    offenders: list[str] = []
    for path in _python_files():
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _BARE_HASH_CALL.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{line_number}")
    assert offenders == []
