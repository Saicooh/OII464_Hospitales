"""Read-only reconstruction of a campaign instance from persisted results.

Analysis-time code must never re-sample the benchmark instance: the sampler
depends on the numpy/pandas stack, so a re-sample under a different
environment silently desynchronises the reported numerator from the
denominator it is compared against. This module recovers the instance from
``analysis.db`` instead, which is the only record of what the campaign
actually ran.

The database is opened through a read-only URI (``file:<path>?mode=ro``) and
is never written.

Physics recovered per job (all values in minutes):

``chain = max(transition, setup) + anesthesia + surgery + cleanup``

matching ``simulation/scheduler.py``: the room is held from the start of the
transition/setup overlap until cleanup completes.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

#: Provenance records are emitted here rather than beside the database:
#: `.gitignore` excludes `results/` as a directory, and a negation pattern
#: cannot re-include descendants of an excluded directory, so anything written
#: under `results/` is unreachable by version control.
DEFAULT_INSTANCE_JSON_DIR: Path = PROJECT_ROOT / "reproducibility" / "output"

INSTANCE_JSON_SCHEMA_VERSION: int = 1

#: numpy and pandas discriminate between the two interpreters available in this
#: project; both report the same Python version, so the Python version is
#: reported for diagnosis but never tested.
CANONICAL_ENVIRONMENT: dict[str, str] = {"numpy": "2.4.1", "pandas": "2.3.3"}

# `proc_time_min` is derived by subtraction in core/simulation_runner.py, so it
# carries float noise (max spread observed on the campaign: 2.27e-13). Exact
# equality is therefore the wrong predicate for the fixed-instance contract.
DRIFT_TOLERANCE_MIN: float = 1e-9

# Agreement tolerance between the per-algorithm anesthesia estimators.
ALGO_AGREEMENT_TOLERANCE_MIN: float = 1e-6

# Slack allowed when checking a lower bound against an achieved makespan.
LOWER_BOUND_TOLERANCE_MIN: float = 1e-6

BINDING_CRITICAL_PATH = "critical-path"
BINDING_ROOM_LOAD = "room-load"
BINDING_SURGEON_LOAD = "surgeon-load"
BINDING_ANESTHETIST_LOAD = "anesth-load"


class InstanceReconstructionError(RuntimeError):
    """The persisted data does not describe a valid fixed instance."""


class AnesthesiaIdentityError(InstanceReconstructionError):
    """The precondition of the anesthesia identity is not satisfied."""


class LowerBoundViolationError(InstanceReconstructionError):
    """A computed lower bound exceeds an achieved makespan."""


@dataclass(frozen=True)
class ResourceCounts:
    """Room and personnel counts backing the load relaxations."""

    rooms: int
    anesthetists: int
    surgeons: int


def resource_counts() -> ResourceCounts:
    """Resolve resource counts from the active configuration profile.

    Read through the ``config.config`` module at call time rather than bound at
    import time, because alternate profiles declare different personnel counts
    and a hardcoded literal would be silently wrong under them.
    """
    from config import config as project_config

    return ResourceCounts(
        rooms=len(project_config.ALL_ROOMS),
        anesthetists=len(project_config.PERSONNEL_BY_OPERATION[1]),
        surgeons=len(project_config.PERSONNEL_BY_OPERATION[2]),
    )


@dataclass(frozen=True)
class JobInstance:
    """The five-term decomposition of one job, in minutes."""

    job_id: int
    transition_min: float
    setup_min: float
    anesthesia_min: float
    surgery_min: float
    cleanup_min: float
    min_op1_finish_min: float

    @property
    def room_entry_min(self) -> float:
        """Room time consumed before anesthesia starts."""
        return max(self.transition_min, self.setup_min)

    @property
    def chain_min(self) -> float:
        """Total time the room is held by this job."""
        return (
            self.room_entry_min
            + self.anesthesia_min
            + self.surgery_min
            + self.cleanup_min
        )


@dataclass(frozen=True)
class InstanceBounds:
    """The four relaxations and the bound they imply."""

    lb_cp: float
    lb_room: float
    lb_surgeon: float
    lb_anesthetist: float

    @property
    def lb(self) -> float:
        return max(self.lb_cp, self.lb_room, self.lb_surgeon, self.lb_anesthetist)

    @property
    def binding(self) -> str:
        candidates = (
            (self.lb_cp, BINDING_CRITICAL_PATH),
            (self.lb_room, BINDING_ROOM_LOAD),
            (self.lb_surgeon, BINDING_SURGEON_LOAD),
            (self.lb_anesthetist, BINDING_ANESTHETIST_LOAD),
        )
        return max(candidates, key=lambda pair: pair[0])[1]


@dataclass(frozen=True)
class ReconstructedInstance:
    """One campaign instance recovered from persisted simulation results."""

    run_id: int
    num_jobs: int
    jobs: dict[int, JobInstance]
    bounds: InstanceBounds
    best_makespan_min: float
    db_path: str
    resources: ResourceCounts = field(
        default_factory=lambda: ResourceCounts(0, 0, 0)
    )

    @property
    def total_room_work_min(self) -> float:
        return sum(job.chain_min for job in self.jobs.values())

    @property
    def gap_pct(self) -> float:
        lb = self.bounds.lb
        if lb <= 0.0:
            return 0.0
        return (self.best_makespan_min - lb) / lb * 100.0


# --------------------------------------------------------------------------
# SQL
# --------------------------------------------------------------------------

# Per-job constant times plus their spread. `cie10_breakdown` keys on `sim_id`,
# so filtering by run requires the join through `simulations`.
_Q1_BREAKDOWN = """
SELECT b.job_id,
       MIN(b.setup_min),     MAX(b.setup_min),
       MIN(b.proc_time_min), MAX(b.proc_time_min),
       MIN(b.cleanup_min),   MAX(b.cleanup_min)
FROM cie10_breakdown b
JOIN simulations s ON s.sim_id = b.sim_id
WHERE s.run_id = ?
GROUP BY b.job_id
ORDER BY b.job_id
"""

# Transition plus the anesthesia estimator, grouped per algorithm so the
# algorithms form disjoint samples of the same physical floor.
_Q2_WAIT = """
SELECT p.job_id, s.algo_name,
       MIN(p.transition_used), MAX(p.transition_used),
       MIN(p.op1_finish)
FROM patient_wait_metrics p
JOIN simulations s ON s.sim_id = p.sim_id
WHERE s.run_id = ?
GROUP BY p.job_id, s.algo_name
ORDER BY p.job_id, s.algo_name
"""

_Q3_BEST = "SELECT MIN(final_makespan) FROM simulations WHERE run_id = ?"


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)


def _assert_no_drift(
    run_id: int, job_id: int, column: str, low: float, high: float
) -> None:
    """Assertion A1: a fixed instance keeps every per-job time constant.

    The tolerance is not zero: `proc_time_min` is derived by subtraction on the
    write path, so it carries float noise on the order of 1e-13.
    """
    spread = high - low
    if spread > DRIFT_TOLERANCE_MIN:
        raise InstanceReconstructionError(
            f"run_id={run_id}: job {job_id} column {column!r} varies across "
            f"simulations. min={low!r} max={high!r} spread={spread!r} exceeds "
            f"tolerance {DRIFT_TOLERANCE_MIN:g}. The run is not a fixed-pool "
            "campaign."
        )


def _assert_algorithms_agree(
    run_id: int, job_id: int, per_algo_min_op1_finish: dict[str, float]
) -> None:
    """Assertion A2: every algorithm must reach the same physical floor.

    Each algorithm is an independent sample. If ``setup_start = 0`` is
    attainable for this job, all of them must find it; disagreement means at
    least one sample never scheduled the job first, so the anesthesia estimate
    would carry queueing delay.
    """
    values = list(per_algo_min_op1_finish.values())
    spread = max(values) - min(values)
    if spread > ALGO_AGREEMENT_TOLERANCE_MIN:
        raise AnesthesiaIdentityError(
            f"run_id={run_id}: job {job_id} has disagreeing per-algorithm "
            f"MIN(op1_finish). values={per_algo_min_op1_finish!r} "
            f"spread={spread!r} exceeds tolerance "
            f"{ALGO_AGREEMENT_TOLERANCE_MIN:g}. The anesthesia identity "
            "precondition is not satisfied."
        )


def _assert_anesthesia_positive(run_id: int, job_id: int, anesthesia: float) -> None:
    """Assertion A3: a non-positive anesthesia inverts the identity."""
    if anesthesia <= 0.0:
        raise AnesthesiaIdentityError(
            f"run_id={run_id}: job {job_id} yields non-positive anesthesia "
            f"{anesthesia!r}. The derivation "
            "MIN(op1_finish) - max(transition, setup) is invalid here."
        )


def _assert_contiguous_job_ids(job_ids: list[int], run_id: int) -> None:
    """Assertion A5: the persisted job set must be exactly {1..N}."""
    expected = set(range(1, len(job_ids) + 1))
    if set(job_ids) != expected:
        raise InstanceReconstructionError(
            f"run_id={run_id}: persisted job ids are not contiguous from 1. "
            f"got={sorted(job_ids)} expected={sorted(expected)}"
        )


def _compute_bounds(
    jobs: dict[int, JobInstance], resources: ResourceCounts
) -> InstanceBounds:
    chains = [job.chain_min for job in jobs.values()]
    return InstanceBounds(
        lb_cp=max(chains) if chains else 0.0,
        lb_room=sum(chains) / resources.rooms if resources.rooms else 0.0,
        lb_surgeon=(
            sum(job.surgery_min for job in jobs.values()) / resources.surgeons
            if resources.surgeons
            else 0.0
        ),
        lb_anesthetist=(
            sum(job.anesthesia_min for job in jobs.values()) / resources.anesthetists
            if resources.anesthetists
            else 0.0
        ),
    )


def _assert_bound_below_best(
    bounds: InstanceBounds, best_makespan_min: float, run_id: int
) -> None:
    """Assertion A4: no relaxation may exceed an achieved schedule."""
    if bounds.lb > best_makespan_min + LOWER_BOUND_TOLERANCE_MIN:
        raise LowerBoundViolationError(
            f"run_id={run_id}: lower bound exceeds the best achieved makespan. "
            f"lb={bounds.lb:.4f} ({bounds.binding}) "
            f"best_makespan={best_makespan_min:.4f}. "
            "A bound above an achieved schedule is physically impossible."
        )


def reconstruct_instance(db_path: str | Path, run_id: int) -> ReconstructedInstance:
    """Recover the instance executed by ``run_id`` from persisted results.

    Raises
    ------
    FileNotFoundError
        The database file does not exist.
    InstanceReconstructionError
        The run has no persisted rows, or the persisted data violates the
        fixed-instance contract.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"analysis database not found: {path}")

    conn = _connect_read_only(path)
    try:
        breakdown_rows = conn.execute(_Q1_BREAKDOWN, (run_id,)).fetchall()
        wait_rows = conn.execute(_Q2_WAIT, (run_id,)).fetchall()
        best_row = conn.execute(_Q3_BEST, (run_id,)).fetchone()
    finally:
        conn.close()

    if not breakdown_rows or not wait_rows:
        raise InstanceReconstructionError(
            f"run_id={run_id} has no persisted breakdown rows"
        )

    best_makespan_min = best_row[0] if best_row else None
    if best_makespan_min is None:
        raise InstanceReconstructionError(
            f"run_id={run_id} has no persisted simulations"
        )

    job_ids = [int(row[0]) for row in breakdown_rows]
    _assert_contiguous_job_ids(job_ids, run_id)

    # Collapse the per-algorithm rows into per-job aggregates.
    transition_range_by_job: dict[int, list[float]] = {}
    per_algo_min_op1_finish: dict[int, dict[str, float]] = {}
    for job_id, algo_name, trans_min, trans_max, min_op1_finish in wait_rows:
        job_id = int(job_id)
        current = transition_range_by_job.get(job_id)
        if current is None:
            transition_range_by_job[job_id] = [trans_min, trans_max]
        else:
            current[0] = min(current[0], trans_min)
            current[1] = max(current[1], trans_max)
        per_algo_min_op1_finish.setdefault(job_id, {})[algo_name] = min_op1_finish

    jobs: dict[int, JobInstance] = {}
    for row in breakdown_rows:
        (
            raw_job_id,
            setup_low,
            setup_high,
            surgery_low,
            surgery_high,
            cleanup_low,
            cleanup_high,
        ) = row
        job_id = int(raw_job_id)
        transition_low, transition_high = transition_range_by_job[job_id]

        _assert_no_drift(run_id, job_id, "setup_min", setup_low, setup_high)
        _assert_no_drift(run_id, job_id, "proc_time_min", surgery_low, surgery_high)
        _assert_no_drift(run_id, job_id, "cleanup_min", cleanup_low, cleanup_high)
        _assert_no_drift(
            run_id, job_id, "transition_used", transition_low, transition_high
        )

        setup_min = setup_low
        surgery_min = surgery_low
        cleanup_min = cleanup_low

        _assert_algorithms_agree(run_id, job_id, per_algo_min_op1_finish[job_id])

        transition_min = transition_low
        min_op1_finish = min(per_algo_min_op1_finish[job_id].values())
        # op1_finish = setup_start + max(transition, setup) + anesthesia, and
        # setup_start reaches 0 whenever some simulation schedules this job
        # first, which the caller-facing assertions verify separately.
        anesthesia_min = min_op1_finish - max(transition_min, setup_min)
        _assert_anesthesia_positive(run_id, job_id, anesthesia_min)
        jobs[job_id] = JobInstance(
            job_id=job_id,
            transition_min=transition_min,
            setup_min=setup_min,
            anesthesia_min=anesthesia_min,
            surgery_min=surgery_min,
            cleanup_min=cleanup_min,
            min_op1_finish_min=min_op1_finish,
        )

    resources = resource_counts()
    bounds = _compute_bounds(jobs, resources)
    _assert_bound_below_best(bounds, best_makespan_min, run_id)

    return ReconstructedInstance(
        run_id=run_id,
        num_jobs=len(jobs),
        jobs={job_id: jobs[job_id] for job_id in sorted(jobs)},
        bounds=bounds,
        best_makespan_min=best_makespan_min,
        db_path=str(path),
        resources=resources,
    )


# --------------------------------------------------------------------------
# Environment guard
# --------------------------------------------------------------------------


def assert_canonical_environment() -> None:
    """Assert the numpy/pandas versions the campaign artifacts depend on.

    Called by every path that regenerates a published artifact. It compares
    numpy and pandas only; the two interpreters available in this project
    report the same Python version, which is included in the message for
    diagnosis but is not part of the test.
    """
    import numpy
    import pandas

    actual = {"numpy": numpy.__version__, "pandas": pandas.__version__}
    if actual != CANONICAL_ENVIRONMENT:
        raise EnvironmentError(
            "Non-canonical environment. Regenerating campaign artifacts here "
            "produces DIFFERENT numbers.\n"
            f"  expected: {CANONICAL_ENVIRONMENT}\n"
            f"  actual:   {actual}\n"
            f"  interpreter: {sys.executable} "
            f"(python {sys.version.split()[0]})\n"
            "  Install the pinned dependencies from requirements.txt in the "
            "repository's active Python environment."
        )
    return None


# --------------------------------------------------------------------------
# Adapter for the dispatching harness
# --------------------------------------------------------------------------


def to_surgeries_data(inst: ReconstructedInstance) -> dict[int, dict]:
    """Project the instance onto the five keys the dispatch harness reads.

    The sampler additionally produces `prep`, `cleanup` and
    `transition_after_op1`. Those are not reconstructible from persisted
    results and are not consumed by the harness, so they are omitted rather
    than defaulted.
    """
    return {
        job_id: {
            1: job.anesthesia_min,
            2: job.surgery_min,
            "setup_by_op": {1: job.setup_min, 2: 0.0},
            "transition_by_op": {1: job.transition_min, 2: 0.0},
            "cleanup_by_op": {1: 0.0, 2: job.cleanup_min},
        }
        for job_id, job in inst.jobs.items()
    }


# --------------------------------------------------------------------------
# Provenance artifact
# --------------------------------------------------------------------------


def _canonical_digest(payload: dict) -> str:
    """SHA-256 over every key except the digest itself."""
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _instance_payload(inst: ReconstructedInstance) -> dict:
    import numpy
    import pandas

    db_path = Path(inst.db_path)
    payload = {
        "schema_version": INSTANCE_JSON_SCHEMA_VERSION,
        "run_id": inst.run_id,
        "num_jobs": inst.num_jobs,
        "units": "minutes",
        "generated_by": "reproducibility/instance_reconstruction.py",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
        },
        "source_db": db_path.as_posix(),
        "source_db_size_bytes": (
            db_path.stat().st_size if db_path.exists() else None
        ),
        "resources": {
            "rooms": inst.resources.rooms,
            "anesthetists": inst.resources.anesthetists,
            "surgeons": inst.resources.surgeons,
        },
        "jobs": [
            {
                "job_id": job.job_id,
                "transition_min": float(job.transition_min),
                "setup_min": float(job.setup_min),
                "anesthesia_min": float(job.anesthesia_min),
                "surgery_min": float(job.surgery_min),
                "cleanup_min": float(job.cleanup_min),
                "chain_min": float(job.chain_min),
                "min_op1_finish_min": float(job.min_op1_finish_min),
            }
            for _job_id, job in sorted(inst.jobs.items())
        ],
        "bounds": {
            "lb_cp": float(inst.bounds.lb_cp),
            "lb_room": float(inst.bounds.lb_room),
            "lb_surgeon": float(inst.bounds.lb_surgeon),
            "lb_anesthetist": float(inst.bounds.lb_anesthetist),
            "lb": float(inst.bounds.lb),
            "binding": inst.bounds.binding,
        },
        "best_makespan_min": float(inst.best_makespan_min),
        "gap_pct": float(inst.gap_pct),
    }
    payload["content_sha256"] = _canonical_digest(payload)
    return payload


def emit_instance_json(
    inst: ReconstructedInstance, out_dir: str | Path | None = None
) -> Path:
    """Write the checksummed provenance record and return its path.

    Values are serialised through Python's shortest round-trip float
    representation. Rounding belongs to the CSV and report presentation
    layers, never to this record.
    """
    assert_canonical_environment()

    directory = Path(out_dir) if out_dir is not None else DEFAULT_INSTANCE_JSON_DIR
    directory.mkdir(parents=True, exist_ok=True)
    out_path = directory / f"instance_run{inst.run_id}.json"
    out_path.write_text(
        json.dumps(_instance_payload(inst), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def verify_instance_json(path: str | Path) -> bool:
    """Recompute the digest of an emitted record and compare it."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stored = payload.get("content_sha256")
    return bool(stored) and stored == _canonical_digest(payload)
