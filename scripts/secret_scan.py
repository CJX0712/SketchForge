"""Zero-dependency secret-pattern scanner.

Author: 晨星
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_PATTERN_SOURCES = (
    r"sk-[A-Za-z0-9]{20,}",
    "gh" + r"p_",
    "A" + r"KIA",
    "-----BE" + r"GIN",
)
_PATTERNS = tuple(re.compile(source) for source in _PATTERN_SOURCES)
_IGNORED_PARTS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "artifacts", "dist"}


def scan(root: Path) -> list[tuple[Path, int, str]]:
    """Return redacted locations matching common credential signatures."""
    findings: list[tuple[Path, int, str]] = []
    for suffix in ("*.py", "*.md"):
        for path in root.rglob(suffix):
            if any(part in _IGNORED_PARTS for part in path.parts):
                continue
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                for pattern in _PATTERNS:
                    if pattern.search(line):
                        findings.append((path.relative_to(root), line_number, pattern.pattern))
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = scan(root)
    for path, line_number, pattern in findings:
        print(f"{path}:{line_number}: possible secret matching {pattern}")
    if findings:
        print(f"secret scan failed: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print("secret scan passed: no credential patterns found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
