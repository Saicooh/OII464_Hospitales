"""Configuration for synthetic-instance runs and legacy-compatible reports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


_ROOT = Path(__file__).resolve().parents[1]


def _load_config() -> dict[str, Any]:
    configured_path = os.environ.get("HOSPITAL_CONFIG_PATH")
    path = Path(configured_path) if configured_path else Path(__file__).with_name("config.yaml")
    if not path.is_absolute():
        path = _ROOT / path
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(f"cannot load configuration {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("configuration must be a mapping")
    required = {"instance", "experiment", "algorithms"}
    missing = required - document.keys()
    if missing:
        raise ValueError(f"configuration is missing sections: {sorted(missing)}")
    return document


_CONFIG = _load_config()
ALG_CONFIG = dict(_CONFIG.get("algorithms", {}))
EXP_CONFIG = dict(_CONFIG.get("experiment", {}))

_instance_config = dict(_CONFIG.get("instance", {}))
_configured_instance = Path(
    os.environ.get("HOSPITAL_INSTANCE_PATH", str(_instance_config.get("path", "")))
)
INSTANCE_PATH = str(
    _configured_instance
    if _configured_instance.is_absolute()
    else _ROOT / _configured_instance
)

NUM_SIMULATIONS = int(EXP_CONFIG.get("num_simulations", 1))
STD_FACTOR = float(EXP_CONFIG.get("std_factor_times", 0.15))
ALPHA_TEST = float(EXP_CONFIG.get("alpha_test", 0.05))
OUTPUT_DIRS = {
    str(key): str(value)
    for key, value in dict(EXP_CONFIG.get("output_dirs", {})).items()
}
OUTPUT_DIRS.setdefault("csv", "results/elective/csv")
OUTPUT_DIRS.setdefault("plots", "results/elective/plots")
N_JOBS = int(EXP_CONFIG.get("n_jobs", -1))
VERBOSE_MODE = bool(_CONFIG.get("logging", {}).get("verbose_mode", False))
EMERGENCY_JOBS: list[int] = []


def _selected_resources() -> tuple[list[str], dict[int, list[str]]]:
    """Load resource IDs from the selected synthetic instance when available."""
    fallback = dict(_CONFIG.get("resources", {}))
    rooms = list(fallback.get("rooms", []))
    personnel = {
        int(key): list(value)
        for key, value in dict(fallback.get("personnel", {})).items()
    }
    try:
        document = yaml.safe_load(Path(INSTANCE_PATH).read_text(encoding="utf-8"))
        resources = document.get("resources", {}) if isinstance(document, dict) else {}
        if resources.get("rooms"):
            rooms = list(resources["rooms"])
        if resources.get("personnel"):
            personnel = {
                int(key): list(value)
                for key, value in resources["personnel"].items()
            }
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        pass
    if not rooms:
        rooms = [f"OR-{index}" for index in range(1, 13)]
    if not personnel:
        personnel = {
            1: [f"AN-{index}" for index in range(1, 12)],
            2: [f"SU-{index}" for index in range(1, 30)],
        }
    return rooms, personnel


ALL_ROOMS, PERSONNEL_BY_OPERATION = _selected_resources()
PABELLONES = list(ALL_ROOMS)
ALL_PERSONNEL = [person for group in PERSONNEL_BY_OPERATION.values() for person in group]
NUM_PABELLONES = len(PABELLONES)

_times = dict(_CONFIG.get("times", {}))
SETUP_TIMES = {int(key): float(value) for key, value in dict(_times.get("setup", {})).items()}
CLEANUP_TIMES = {
    int(key): float(value) for key, value in dict(_times.get("cleanup", {})).items()
}
MAX_WAIT_TIMES = {
    int(key): float(value) for key, value in dict(_times.get("max_wait", {})).items()
}
SETUP_TIMES.setdefault(1, 0.0)
CLEANUP_TIMES.setdefault(2, 0.0)
MAX_WAIT_TIMES.setdefault(1, 0.0)
MAX_WAIT_TIMES.setdefault(2, 10_000.0)

JOB_TYPES = {
    int(key): int(value)
    for key, value in dict(_CONFIG.get("jobs", {}).get("types", {})).items()
}
_CYCLIC_TYPES = (1, 2, 3)


def get_job_type(job_id: int) -> int:
    """Return the configured or cyclic legacy procedure type."""
    if job_id in JOB_TYPES:
        return JOB_TYPES[job_id]
    return _CYCLIC_TYPES[(int(job_id) - 1) % len(_CYCLIC_TYPES)]


ALPHA = float(ALG_CONFIG.get("alpha", 1.0e-6))
BETA = float(ALG_CONFIG.get("beta", 0.5))
GAMMA = float(ALG_CONFIG.get("gamma", 1.4))
DELTA = float(ALG_CONFIG.get("delta", 100.0))

GA_CONFIG = dict(ALG_CONFIG.get("ga", {}))
GA_ENABLED = bool(GA_CONFIG.get("enabled", False))
POPULATION_SIZE_GA = int(GA_CONFIG.get("population_size", 30))
MAX_GENERATIONS = int(GA_CONFIG.get("max_generations", 1000))
CROSSOVER_PROBABILITY = float(GA_CONFIG.get("crossover_probability", 0.8))
MUTATION_PROBABILITY = float(GA_CONFIG.get("mutation_probability", 0.3))
ELITISM_COUNT = int(GA_CONFIG.get("elitism_count", 2))

DPSO_CONFIG = dict(ALG_CONFIG.get("dpso", {}))
DPSO_ENABLED = bool(DPSO_CONFIG.get("enabled", False))
SWARM_SIZE_DPSO = int(DPSO_CONFIG.get("swarm_size", 30))
MAX_ITERATIONS_DPSO = int(DPSO_CONFIG.get("max_iterations", 1000))
W_DPSO = float(DPSO_CONFIG.get("w", 0.7))
C1_DPSO = float(DPSO_CONFIG.get("c1", 1.5))
C2_DPSO = float(DPSO_CONFIG.get("c2", 1.5))
VEL_HIGH_DPSO = float(DPSO_CONFIG.get("vel_high", 4.0))
VEL_LOW_DPSO = float(DPSO_CONFIG.get("vel_low", -4.0))

SBOA_CONFIG = dict(ALG_CONFIG.get("sboa", {}))
SBOA_ENABLED = bool(SBOA_CONFIG.get("enabled", False))
SBOA_POP_SIZE = int(SBOA_CONFIG.get("population_size", 30))
SBOA_MAX_ITER = int(SBOA_CONFIG.get("max_iterations", 1000))
SBOA_LOWER_BOUND = float(SBOA_CONFIG.get("lower_bound", -5.0))
SBOA_UPPER_BOUND = float(SBOA_CONFIG.get("upper_bound", 5.0))

MSHOA_CONFIG = dict(ALG_CONFIG.get("dmshoa", {}))
MSHOA_ENABLED = bool(MSHOA_CONFIG.get("enabled", True))
MSHOA_POP_SIZE = int(MSHOA_CONFIG.get("population_size", 30))
MAX_ITERATIONS_MSHOA = int(MSHOA_CONFIG.get("max_iterations", 1000))
MSHOA_K = float(MSHOA_CONFIG.get("k", 0.3))
MSHOA_LOWER_BOUND = float(MSHOA_CONFIG.get("lower_bound", -5.0))
MSHOA_UPPER_BOUND = float(MSHOA_CONFIG.get("upper_bound", 5.0))

MH_CONFIG = dict(ALG_CONFIG.get("mh", {}))
MH_ENABLED = bool(MH_CONFIG.get("enabled", False))
MH_POP_SIZE = int(MH_CONFIG.get("population_size", 20))
MH_MAX_ITERATIONS = int(MH_CONFIG.get("max_iterations", 100))

_ALGORITHMS_CACHE = None


def get_algorithms():
    """Return enabled algorithm specifications in the historical order."""
    global _ALGORITHMS_CACHE
    if _ALGORITHMS_CACHE is None:
        from config.algorithms_loader import load_algorithms

        _ALGORITHMS_CACHE = load_algorithms(
            ga_enabled=GA_ENABLED,
            dpso_enabled=DPSO_ENABLED,
            sboa_enabled=SBOA_ENABLED,
            mshoa_enabled=MSHOA_ENABLED,
            max_generations=MAX_GENERATIONS,
            max_iterations_dpso=MAX_ITERATIONS_DPSO,
            sboa_max_iter=SBOA_MAX_ITER,
            max_iterations_mshoa=MAX_ITERATIONS_MSHOA,
            all_rooms=ALL_ROOMS,
            mh_enabled=MH_ENABLED,
            mh_max_iterations=MH_MAX_ITERATIONS,
        )
    return _ALGORITHMS_CACHE


class _AlgorithmsProxy:
    def __getitem__(self, key):
        return get_algorithms()[key]

    def __iter__(self):
        return iter(get_algorithms())

    def __len__(self):
        return len(get_algorithms())

    def __repr__(self):
        return repr(get_algorithms())


ALGORITHMS = _AlgorithmsProxy()


__all__ = [
    "ALGORITHMS", "ALL_PERSONNEL", "ALL_ROOMS", "ALPHA", "ALPHA_TEST", "ALG_CONFIG",
    "BETA", "C1_DPSO", "C2_DPSO", "CLEANUP_TIMES", "CROSSOVER_PROBABILITY",
    "DELTA", "DPSO_CONFIG", "DPSO_ENABLED", "ELITISM_COUNT", "EMERGENCY_JOBS", "EXP_CONFIG",
    "GA_CONFIG", "GA_ENABLED", "GAMMA", "INSTANCE_PATH", "JOB_TYPES",
    "MAX_GENERATIONS", "MAX_ITERATIONS_DPSO", "MAX_ITERATIONS_MSHOA",
    "MAX_WAIT_TIMES", "MH_CONFIG", "MH_ENABLED", "MH_MAX_ITERATIONS", "MH_POP_SIZE",
    "MSHOA_CONFIG", "MSHOA_ENABLED", "MSHOA_K",
    "MSHOA_POP_SIZE", "MSHOA_LOWER_BOUND", "MSHOA_UPPER_BOUND", "N_JOBS",
    "NUM_PABELLONES", "NUM_SIMULATIONS", "OUTPUT_DIRS", "PABELLONES",
    "PERSONNEL_BY_OPERATION", "POPULATION_SIZE_GA", "SBOA_CONFIG", "SBOA_ENABLED",
    "SBOA_LOWER_BOUND", "SBOA_MAX_ITER", "SBOA_POP_SIZE", "SBOA_UPPER_BOUND",
    "SETUP_TIMES", "STD_FACTOR", "SWARM_SIZE_DPSO", "VEL_HIGH_DPSO", "VEL_LOW_DPSO",
    "VERBOSE_MODE", "get_algorithms", "get_job_type",
]
