"""Single-root deterministic random seed management.

Author: 晨星
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np

from sketchforge.core.errors import InvalidConfigError, MissingSeedError


@dataclass(frozen=True)
class SeedState:
    """A reproducible root seed with independent child-stream derivation."""

    root: int

    def derive(self, tag: str) -> int:
        """Derive a stable unsigned 64-bit seed for a named subsystem."""
        payload = f"{self.root}:{tag}".encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def spawn_generators(self, n: int) -> list[np.random.Generator]:
        """Spawn independent NumPy generators from the root seed."""
        if n < 0:
            raise InvalidConfigError("generator count must be non-negative", n=n)
        children = np.random.SeedSequence(self.root).spawn(n)
        return [np.random.default_rng(child) for child in children]


_STATE: SeedState | None = None


def set_all(seed: int) -> SeedState:
    """Seed Python and NumPy legacy global RNGs and store the root state."""
    global _STATE
    if not isinstance(seed, int):
        raise InvalidConfigError("seed must be an integer", seed_type=type(seed).__name__)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    _STATE = SeedState(seed)
    return _STATE


def require() -> SeedState:
    """Return the configured root seed or fail closed."""
    if _STATE is None:
        raise MissingSeedError("set_all(seed) must be called before requiring global seed state")
    return _STATE
