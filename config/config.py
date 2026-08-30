import os
import yaml
import warnings
from pathlib import Path

def _load_config() -> dict:
    override_path = os.environ.get("HOSPITAL_CONFIG_PATH")
    if override_path:
        candidate = Path(override_path).expanduser()
        if not candidate.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            candidate = (project_root / candidate).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Config override file not found: {candidate}")
        config_path = candidate
    else:
        config_path = Path(__file__).with_name("config.yaml")
    with config_path.open(encoding="utf-8") as src:
        return yaml.safe_load(src) or {}

_CONFIG = _load_config()

ALG_CONFIG = _CONFIG["algorithms"]
EXP_CONFIG = _CONFIG["experiment"]

# --- Real Data Integration ---
REAL_DATA_CONFIG = _CONFIG.get("real_data", {})
USE_REAL_DATA = REAL_DATA_CONFIG.get("enabled", False)
TRACE_CSV_PATH = REAL_DATA_CONFIG.get(
    "trace_csv_path", "results/csv/raw_batch_trace.csv"
)
RESET_TRACE_ON_START = REAL_DATA_CONFIG.get("reset_trace_on_start", True)

# --- 1. Experiment Parameters ---
NUM_SIMULATIONS = EXP_CONFIG["num_simulations"]
STD_FACTOR = EXP_CONFIG["std_factor_times"]
ALPHA_TEST = EXP_CONFIG["alpha_test"]
OUTPUT_DIRS = EXP_CONFIG["output_dirs"]
N_JOBS = EXP_CONFIG.get("n_jobs", min(10, os.cpu_count() - 2))

# --- Logging Configuration ---
LOGGING_CONFIG = _CONFIG.get("logging", {})
VERBOSE_MODE = LOGGING_CONFIG.get("verbose_mode", True)  # Default: verbose

# --- 2. Problem Parameters ---
TIMES_CONFIG = _CONFIG["times"]

SETUP_TIMES = {int(k): v for k, v in TIMES_CONFIG["setup"].items()}
CLEANUP_TIMES = {int(k): v for k, v in TIMES_CONFIG["cleanup"].items()}
MAX_WAIT_TIMES = {int(k): v for k, v in TIMES_CONFIG["max_wait"].items()}

JOBS_CONFIG = _CONFIG["jobs"]
JOB_TYPES = {int(k): v for k, v in JOBS_CONFIG["types"].items()}

# --- NUM_PROCEDURES: cantidad de jobs/cirugías por lote de simulación ---
_raw_num_proc = EXP_CONFIG.get("num_procedures", len(JOB_TYPES))
if not isinstance(_raw_num_proc, int) or _raw_num_proc < 1:
    _raw_num_proc = len(JOB_TYPES)
if _raw_num_proc < 2:
    warnings.warn(
        f"num_procedures={_raw_num_proc} < 2: la regla 70/80 top20/otros "
        f"requiere al menos 2 jobs.",
        UserWarning,
        stacklevel=2,
    )
NUM_PROCEDURES: int = _raw_num_proc

_CYCLIC_TYPES = [1, 2, 3]


def get_job_type(job_id) -> int:
    """Return the catalogued or cyclic procedure type for an integer job ID."""
    if job_id in JOB_TYPES:
        return JOB_TYPES[job_id]
    return _CYCLIC_TYPES[(job_id - 1) % len(_CYCLIC_TYPES)]


# --- Resources ---
RESOURCES_CONFIG = _CONFIG["resources"]
NUM_PABELLONES = RESOURCES_CONFIG["num_pabellones"]

PABELLONES = [f"Pabellon_{i + 1}" for i in range(NUM_PABELLONES)]
ALL_ROOMS = PABELLONES

# --- Personnel Configuration (Dynamic Assignment) ---
# Reads numeric quantities and generates legacy ID lists internally.
PERSONNEL_CONFIG = _CONFIG["personnel"]


def _validate_personnel_count(value, name: str) -> int:
    """Valida que el valor sea int (no bool), no negativo. Retorna el valor."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"personnel.{name} debe ser un entero, "
            f"pero se recibió {type(value).__name__!r}: {value!r}"
        )
    if value < 0:
        raise ValueError(f"personnel.{name} debe ser >= 0, pero se recibió {value!r}")
    return value


_num_anest = _validate_personnel_count(
    PERSONNEL_CONFIG["num_anesthesiologists"], "num_anesthesiologists"
)
_num_surg = _validate_personnel_count(PERSONNEL_CONFIG["num_surgeons"], "num_surgeons")

if _num_anest == 0 and _num_surg == 0:
    raise ValueError(
        "personnel: al menos un rol debe tener personal > 0 "
        "(num_anesthesiologists y num_surgeons no pueden ser ambos 0)"
    )

_ANESTHESIOLOGISTS: list[str] = [f"A{i + 1}" for i in range(_num_anest)]
_SURGEONS: list[str] = [f"S{i + 1}" for i in range(_num_surg)]

PERSONNEL_BY_OPERATION = {
    1: _ANESTHESIOLOGISTS,  # APR: Anesthesiologists
    2: _SURGEONS,  # OR: Surgeons
}

# Complete list of all personnel (for initialization)
ALL_PERSONNEL = _ANESTHESIOLOGISTS + _SURGEONS

# --- 3. Algorithm Parameters ---
ALPHA = ALG_CONFIG["alpha"]
BETA = ALG_CONFIG.get("beta", 0.5)  # Penalty for total inter-operation waiting time
GAMMA = ALG_CONFIG.get("gamma", 1.0)  # Penalty for maximum inter-operation waiting time
DELTA = ALG_CONFIG.get("delta", 50.0)  # Penalty for room usage imbalance

GA_CONFIG = ALG_CONFIG["ga"]
GA_ENABLED = GA_CONFIG.get("enabled", True)
POPULATION_SIZE_GA = GA_CONFIG["population_size"]
MAX_GENERATIONS = GA_CONFIG["max_generations"]
CROSSOVER_PROBABILITY = GA_CONFIG["crossover_probability"]
MUTATION_PROBABILITY = GA_CONFIG["mutation_probability"]
ELITISM_COUNT = GA_CONFIG["elitism_count"]

DPSO_CONFIG = ALG_CONFIG["dpso"]
DPSO_ENABLED = DPSO_CONFIG.get("enabled", True)
SWARM_SIZE_DPSO = DPSO_CONFIG["swarm_size"]
MAX_ITERATIONS_DPSO = DPSO_CONFIG["max_iterations"]
W_DPSO = DPSO_CONFIG["w"]
C1_DPSO = DPSO_CONFIG["c1"]
C2_DPSO = DPSO_CONFIG["c2"]
VEL_HIGH_DPSO = DPSO_CONFIG["vel_high"]
VEL_LOW_DPSO = DPSO_CONFIG["vel_low"]

SBOA_CONFIG = ALG_CONFIG["sboa"]
SBOA_ENABLED = SBOA_CONFIG.get("enabled", True)
SBOA_POP_SIZE = SBOA_CONFIG["population_size"]
SBOA_MAX_ITER = SBOA_CONFIG["max_iterations"]
SBOA_LOWER_BOUND = SBOA_CONFIG["lower_bound"]
SBOA_UPPER_BOUND = SBOA_CONFIG["upper_bound"]

MSHOA_CONFIG = ALG_CONFIG["dmshoa"]
MSHOA_ENABLED = MSHOA_CONFIG.get("enabled", True)
MSHOA_POP_SIZE = MSHOA_CONFIG["population_size"]
MAX_ITERATIONS_MSHOA = MSHOA_CONFIG["max_iterations"]
MSHOA_K = MSHOA_CONFIG["k"]
MSHOA_LOWER_BOUND = MSHOA_CONFIG["lower_bound"]
MSHOA_UPPER_BOUND = MSHOA_CONFIG["upper_bound"]

# dMShOA Old (variante legacy): opt-in, deshabilitada por defecto
MSHOA_OLD_CONFIG = ALG_CONFIG.get("dmshoa_old", {})
MSHOA_OLD_ENABLED = MSHOA_OLD_CONFIG.get("enabled", False)

# --- Analysis Mode Configuration (opt-in, disabled by default) ---
_ANALYSIS_CONFIG = _CONFIG.get("analysis_mode", {})
ANALYSIS_MODE_ENABLED: bool = bool(_ANALYSIS_CONFIG.get("enabled", False))
ANALYSIS_NUM_RUNS: int = int(_ANALYSIS_CONFIG.get("num_runs", 4))
ANALYSIS_SIMS_PER_RUN: int = int(_ANALYSIS_CONFIG.get("sims_per_run", 300))
ANALYSIS_CHECKPOINT_INTERVAL: int = int(
    _ANALYSIS_CONFIG.get("checkpoint_interval_seconds", 300)
)
ANALYSIS_SQLITE_PATH: str = str(
    _ANALYSIS_CONFIG.get("sqlite_path", "results/analysis.db")
)
ANALYSIS_TEMPORAL_ENABLED: bool = bool(_ANALYSIS_CONFIG.get("temporal_enabled", True))
ANALYSIS_SWEEP_ENABLED: bool = bool(_ANALYSIS_CONFIG.get("sweep_enabled", False))
ANALYSIS_SWEEP_VALUES: list = list(_ANALYSIS_CONFIG.get("sweep_num_procedures", []))
ANALYSIS_SWEEP_SIMS: int = int(_ANALYSIS_CONFIG.get("sweep_sims_per_x", 20))
ANALYSIS_EXPORT_CSV: bool = bool(_ANALYSIS_CONFIG.get("export_csv_after_run", True))
ANALYSIS_CHECKPOINTS_CSV_PATH: str = str(
    _ANALYSIS_CONFIG.get("checkpoints_csv_path", "results/analysis_checkpoints.csv")
)
ANALYSIS_BREAKDOWN_CSV_PATH: str = str(
    _ANALYSIS_CONFIG.get("breakdown_csv_path", "results/analysis_breakdown.csv")
)
ANALYSIS_ITERATIONS_CSV_PATH: str = str(
    _ANALYSIS_CONFIG.get(
        "iterations_csv_path", "results/analysis_algorithm_iterations.csv"
    )
)

# When True, all simulations within a single run share the exact same
# generated patient dataset (data seed = run_idx). Algorithm stochasticity
# (seed = sim_i) is preserved. Useful for isolating solver variance.
ANALYSIS_FIXED_POOL: bool = bool(
    _ANALYSIS_CONFIG.get("fixed_pool_per_run", False)
)

# Artifact persistence policy: 'best_only' | 'all' | 'sampled'
# Controls which algorithm iteration snapshots are saved to disk.
# Default 'best_only' avoids disk saturation during long analysis runs.
ANALYSIS_ARTIFACT_SAVE_MODE: str = str(
    _ANALYSIS_CONFIG.get("artifact_save_mode", "best_only")
)

# Whether to generate full result folders (plots, CSVs) at each checkpoint.
# Opt-in: False by default to keep analysis runs fast unless explicitly requested.
ANALYSIS_FULL_REPORTS_ENABLED: bool = bool(
    _ANALYSIS_CONFIG.get("full_reports_enabled", False)
)

# =============================================================================
# ALGORITHM CONFIGURATIONS FOR PARALLEL EXECUTION
# =============================================================================

_ALGORITHMS_CACHE = None


def get_algorithms():
    """
    Returns the list of enabled algorithms.
    Uses caching to avoid rebuilding the list multiple times.
    """
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
            mshoa_old_enabled=MSHOA_OLD_ENABLED,
        )
    return _ALGORITHMS_CACHE


# For backward compatibility, expose as ALGORITHMS
class _AlgorithmsProxy:
    """Proxy object that lazily initializes ALGORITHMS"""

    def __getitem__(self, key):
        return get_algorithms()[key]

    def __iter__(self):
        return iter(get_algorithms())

    def __len__(self):
        return len(get_algorithms())

    def __repr__(self):
        return repr(get_algorithms())


ALGORITHMS = _AlgorithmsProxy()
