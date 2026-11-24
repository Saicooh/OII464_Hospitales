"""Regression: the compatibility evaluator must refresh stale ready times.

Before the fix, every anesthesia entered the queue with ready time 0 and was
dispatched before any surgery, which inflated inter-operation waits.
"""

from pathlib import Path

from data.instance_loader import load_instance
from simulation.scheduler import calculate_schedule_fitness

ROOT = Path(__file__).resolve().parents[2]


class _Data(dict):
    def __init__(self, context):
        super().__init__({job.job_id: {} for job in context.jobs})
        self.context = context


def _solution(context):
    return {
        "job_sequence_base": [job.job_id for job in context.jobs],
        "room_assignment": {
            job.job_id: {op.operation_id: op.eligible_rooms[0] for op in job.operations}
            for job in context.jobs
        },
    }


def test_surgeries_are_interleaved_with_anesthesias():
    context = load_instance(ROOT / "instances/standard/HOSP-STD-20-01.yaml")
    _, _, details = calculate_schedule_fitness(_solution(context), _Data(context), return_details=True)
    order = [item["Operation"] for item in details]
    first_surgery = order.index(2)
    last_anesthesia = len(order) - 1 - order[::-1].index(1)
    assert first_surgery < last_anesthesia


def test_rooms_are_never_double_booked():
    context = load_instance(ROOT / "instances/hospital_12rooms/HOSP-12R-40-01.yaml")
    _, _, details = calculate_schedule_fitness(_solution(context), _Data(context), return_details=True)
    by_room = {}
    for item in details:
        by_room.setdefault(item["Resource"], []).append((item["Start"], item["Finish"]))
    for intervals in by_room.values():
        intervals.sort()
        assert all(b[0] >= a[1] - 1e-9 for a, b in zip(intervals, intervals[1:]))
