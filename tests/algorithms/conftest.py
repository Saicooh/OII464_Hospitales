"""
Conftest for tests/algorithms.

Provides a session-scoped minimal config path so that tests importing
algorithm modules without their own env setup get a consistent baseline
rather than accidentally loading the real project config.yaml.
"""

import os
import sys
import yaml
import pytest


_MINIMAL_CONFIG = {
    "experiment": {
        "num_simulations": 1,
        "std_factor_times": 0.0,
        "alpha_test": 0.05,
        "num_procedures": 5,
        "output_dirs": {"plots": "results/plots", "csv": "results/csv"},
    },
    "real_data": {"enabled": False},
    "logging": {"verbose_mode": False},
    "times": {
        "setup": {"1": 10, "2": 10, "3": 10},
        "cleanup": {"1": 5, "2": 5, "3": 5},
        "max_wait": {"1": 100, "2": 100},
    },
    "jobs": {"types": {"1": 1, "2": 2, "3": 3, "4": 1, "5": 2}},
    "resources": {"num_pabellones": 2},
    "personnel": {
        "num_anesthesiologists": 1,
        "num_surgeons": 1,
    },
    "algorithms": {
        "alpha": 1e-6,
        "beta": 0.7,
        "gamma": 1.4,
        "delta": 100.0,
        "ga": {
            "enabled": True,
            "population_size": 2,
            "max_generations": 3,
            "crossover_probability": 0.8,
            "mutation_probability": 0.3,
            "elitism_count": 1,
        },
        "dpso": {
            "enabled": True,
            "swarm_size": 2,
            "max_iterations": 3,
            "w": 0.7,
            "c1": 1.5,
            "c2": 1.5,
            "vel_high": 4.0,
            "vel_low": -4.0,
        },
        "sboa": {
            "enabled": True,
            "population_size": 3,
            "max_iterations": 3,
            "lower_bound": -5.0,
            "upper_bound": 5.0,
        },
        "dmshoa": {
            "enabled": True,
            "population_size": 2,
            "max_iterations": 3,
            "k": 0.3,
            "lower_bound": -5.0,
            "upper_bound": 5.0,
        },
    },
}


@pytest.fixture(autouse=True)
def _reset_algorithm_env(tmp_path):
    """Reset HOSPITAL_CONFIG_PATH and algorithm/config module cache before each test.

    This fixture runs automatically for every test in tests/algorithms/.
    Tests that need a custom config call their own setup AFTER this fixture
    runs (fixture order: autouse fixtures first, then test body).

    This prevents cross-test contamination when tests import algorithm modules
    without setting up a config themselves.
    """
    cfg_file = tmp_path / "_baseline_config.yaml"
    cfg_file.write_text(yaml.safe_dump(_MINIMAL_CONFIG), encoding="utf-8")

    # Set baseline config
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)

    # Clear cached modules so each test starts from a known state
    _PREFIXES = ("config", "algorithms", "simulation")
    for mod_name in list(sys.modules.keys()):
        if any(
            mod_name == p or mod_name.startswith(p + ".")
            for p in _PREFIXES
        ):
            del sys.modules[mod_name]

    yield

    # Teardown: clear modules again so the next test is not contaminated
    for mod_name in list(sys.modules.keys()):
        if any(
            mod_name == p or mod_name.startswith(p + ".")
            for p in _PREFIXES
        ):
            del sys.modules[mod_name]
