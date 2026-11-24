"""Invariants of the single scheduling model (dispatch with blocking)."""

import glob
import random
from dataclasses import replace
from pathlib import Path

import pytest

import config.config as config
from data.instance_loader import load_instance
from simulation.scheduler import (
    calculate_schedule_fitness,
    evaluate_solution,
    objective_from,
    schedule_instance_solution,
)

ROOT = Path(__file__).resolve().parents[2]
INSTANCES = sorted(glob.glob(str(ROOT / "instances" / "*" / "*.yaml")))


def _random_solution(context, rng):
    sequence = [job.job_id for job in context.jobs]
    rng.shuffle(sequence)
    return {
        "job_sequence_base": sequence,
        "room_assignment": {
            job.job_id: {op.operation_id: rng.choice(op.eligible_rooms) for op in job.operations}
            for job in context.jobs
        },
    }


def _overlaps(intervals):
    intervals = sorted(intervals)
    return any(b[0] < a[1] - 1e-9 for a, b in zip(intervals, intervals[1:]))


@pytest.mark.parametrize("path", INSTANCES[::4])
def test_random_schedules_respect_blocking_precedence_and_capacity(path):
    context = load_instance(path)
    rng = random.Random(7)
    for _ in range(15):
        decoded = evaluate_solution(context, _random_solution(context, rng))
        assert decoded is not None, "the decoder must not deadlock"
        ops = {(d["Job"], d["Operation"]): d for d in decoded.details}
        rooms, people = {}, {}
        for job in context.jobs:
            anesthesia, surgery = ops[(job.job_id, 1)], ops[(job.job_id, 2)]
            surgery_start = surgery["Start"] + max(surgery["SetupUsed"], surgery["TransitionUsed"])
            assert surgery_start >= anesthesia["ProcessingEnd"] - 1e-9
            assert decoded.waits[job.job_id] == pytest.approx(
                surgery_start - anesthesia["ProcessingEnd"]
            )
            if anesthesia["Resource"] == surgery["Resource"]:
                rooms.setdefault(anesthesia["Resource"], []).append(
                    (anesthesia["Start"], surgery["Finish"])
                )
            else:
                # Blocking: the anesthesia room is held until the surgery starts.
                assert anesthesia["Finish"] >= surgery["Start"] - 1e-9
                rooms.setdefault(anesthesia["Resource"], []).append(
                    (anesthesia["Start"], anesthesia["Finish"])
                )
                rooms.setdefault(surgery["Resource"], []).append(
                    (surgery["Start"], surgery["Finish"])
                )
            hold = max(anesthesia["ProcessingEnd"] + anesthesia["CleanupUsed"], surgery["Start"])
            people.setdefault(anesthesia["Personnel"], []).append((anesthesia["Start"], hold))
            people.setdefault(surgery["Personnel"], []).append((surgery["Start"], surgery["Finish"]))
        assert not any(_overlaps(v) for v in rooms.values())
        assert not any(_overlaps(v) for v in people.values())


def test_typed_scheduler_and_fitness_share_the_model():
    context = load_instance(ROOT / "instances/standard/HOSP-STD-20-02.yaml")
    solution = _random_solution(context, random.Random(3))
    makespan, schedule = schedule_instance_solution(context, solution)
    objective, fitness_makespan, _ = calculate_schedule_fitness(
        solution, type("Data", (), {"context": context})(), return_details=True
    )
    assert makespan == fitness_makespan
    assert objective == pytest.approx(objective_from(evaluate_solution(context, solution)))
    assert len(schedule) == 2 * len(context.jobs)


def test_wait_above_the_limit_is_penalized(monkeypatch):
    loaded = load_instance(ROOT / "instances/standard/HOSP-STD-20-02.yaml")
    solution = _random_solution(loaded, random.Random(11))
    tight = replace(
        loaded,
        jobs=tuple(
            replace(job, operations=(job.operations[0], replace(job.operations[1], max_wait=0.0)))
            for job in loaded.jobs
        ),
    )
    decoded = evaluate_solution(tight, solution)
    assert decoded.excess_wait == pytest.approx(decoded.total_wait)
    monkeypatch.setattr(config, "OMEGA", 0.0)
    without = objective_from(decoded)
    monkeypatch.setattr(config, "OMEGA", 10.0)
    assert objective_from(decoded) == pytest.approx(without + 10.0 * decoded.excess_wait)
