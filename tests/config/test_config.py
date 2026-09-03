import importlib
from pathlib import Path

import pytest
import yaml


INSTANCE = Path(__file__).parents[2] / "instances/didactic/HOSP-DIDACT-03-01.yaml"


def _document(instance_path: str | Path = INSTANCE) -> dict:
    return {
        "instance": {"path": str(instance_path)},
        "experiment": {
            "num_simulations": 3,
            "n_jobs": 1,
            "output_dirs": {"csv": "tmp/csv", "plots": "tmp/plots"},
        },
        "algorithms": {
            "dmshoa": {
                "max_iterations": 7,
                "population_size": 4,
                "k": 0.3,
                "lower_bound": -5.0,
                "upper_bound": 5.0,
            }
        },
        "logging": {"verbose_mode": True},
    }


@pytest.fixture(autouse=True)
def restore_default_configuration(monkeypatch):
    yield
    monkeypatch.delenv("HOSPITAL_CONFIG_PATH", raising=False)
    monkeypatch.delenv("HOSPITAL_INSTANCE_PATH", raising=False)
    import config.config as config

    importlib.reload(config)


def test_default_configuration_points_to_a_supported_instance():
    import config.config as config

    assert Path(config.INSTANCE_PATH).exists()
    assert config.NUM_SIMULATIONS > 0
    assert config.MSHOA_POP_SIZE > 0
    assert config.MAX_ITERATIONS_MSHOA > 0
    assert set(config.OUTPUT_DIRS) == {"csv", "plots"}


def test_custom_configuration_loads_compatibility_resources_from_selected_instance(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_document(), sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setenv("HOSPITAL_CONFIG_PATH", str(config_path))

    import config.config as config

    config = importlib.reload(config)
    assert config.NUM_SIMULATIONS == 3
    assert config.N_JOBS == 1
    assert config.MSHOA_POP_SIZE == 4
    assert config.MAX_ITERATIONS_MSHOA == 7
    assert config.VERBOSE_MODE is True
    assert config.ALL_ROOMS == ["OR-1", "OR-2"]
    assert config.PERSONNEL_BY_OPERATION[1] == ["AN-1", "AN-2", "AN-3"]
    assert config.PERSONNEL_BY_OPERATION[2] == ["SU-1", "SU-2", "SU-3", "SU-4"]


def test_configuration_requires_the_supported_sections(monkeypatch, tmp_path):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("instance: {}\n", encoding="utf-8")
    monkeypatch.setenv("HOSPITAL_CONFIG_PATH", str(config_path))

    import config.config as config

    with pytest.raises(ValueError, match="missing sections"):
        importlib.reload(config)
