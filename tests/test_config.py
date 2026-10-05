"""Configuration parsing and invariant tests.

Author: 晨星
"""

from pathlib import Path

import pytest

from sketchforge.core.config import Config
from sketchforge.core.errors import InvalidBackendModeError, InvalidConfigError


def test_from_env_parses_all_supported_overrides() -> None:
    config = Config.from_env(
        {
            "SKETCHFORGE_SEED": "21",
            "SKETCHFORGE_BACKEND": "tier0",
            "SKETCHFORGE_N": "250",
            "SKETCHFORGE_BUDGETS": "1024, 4096",
            "SKETCHFORGE_OUT_DIR": "reports",
            "SKETCHFORGE_DETERMINISTIC_ONLY": "yes",
            "SKETCHFORGE_SAFETY": "2.5",
        }
    )
    assert config.seed == 21
    assert config.seeds == (21, 22, 23)
    assert config.backend == "tier0"
    assert config.n == 250
    assert config.budgets_bytes == (1024, 4096)
    assert config.out_dir == Path("reports")
    assert config.deterministic_only is True
    assert config.safety == 2.5


def test_no_tier0_forces_tier1_backend() -> None:
    config = Config.from_env({"SKETCHFORGE_BACKEND": "tier0", "SKETCHFORGE_NO_TIER0": "true"})
    assert config.backend == "tier1"


@pytest.mark.parametrize(
    "config",
    [
        Config(n=0),
        Config(n_jobs=2),
        Config(budgets_bytes=()),
        Config(budgets_bytes=(4096, 0)),
        Config(seeds=()),
        Config(time_budget_sec=0),
        Config(safety=0),
    ],
)
def test_invalid_config_invariants_raise_e101(config: Config) -> None:
    with pytest.raises(InvalidConfigError) as captured:
        config.validate()
    assert captured.value.code == "E101"


def test_invalid_backend_raises_e103() -> None:
    with pytest.raises(InvalidBackendModeError) as direct_error:
        Config(backend="gpu").validate()
    assert direct_error.value.code == "E103"

    with pytest.raises(InvalidBackendModeError) as environment_error:
        Config.from_env({"SKETCHFORGE_BACKEND": ""})
    assert environment_error.value.code == "E103"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SKETCHFORGE_SEED", "bad"),
        ("SKETCHFORGE_SEED", ""),
        ("SKETCHFORGE_N", "bad"),
        ("SKETCHFORGE_BUDGETS", ""),
        ("SKETCHFORGE_BUDGETS", "1,bad"),
        ("SKETCHFORGE_DETERMINISTIC_ONLY", "maybe"),
        ("SKETCHFORGE_NO_TIER0", ""),
        ("SKETCHFORGE_SAFETY", "bad"),
    ],
)
def test_invalid_environment_values_raise_e101(name: str, value: str) -> None:
    with pytest.raises(InvalidConfigError) as captured:
        Config.from_env({name: value})
    assert captured.value.code == "E101"
