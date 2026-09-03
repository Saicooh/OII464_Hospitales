from dataclasses import replace
from pathlib import Path

import pytest

from data.instance_loader import load_instance
from simulation.scheduler import IneligibleAssignmentError, schedule_instance_solution


INSTANCE = Path(__file__).parents[2] / "instances/didactic/HOSP-DIDACT-03-01.yaml"


def _solution(context):
    return {
        "job_sequence_base": [job.job_id for job in context.jobs],
        "room_assignment": {
            job.job_id: {
                operation.operation_id: operation.eligible_rooms[0]
                for operation in job.operations
            }
            for job in context.jobs
        },
        "personnel_assignment": {
            job.job_id: {
                operation.operation_id: operation.eligible_personnel[0]
                for operation in job.operations
            }
            for job in context.jobs
        },
    }


def test_selected_context_owns_holding_chain_and_resources():
    loaded = load_instance(INSTANCE)
    context = replace(loaded, jobs=loaded.jobs[:1])

    makespan, schedule = schedule_instance_solution(context, _solution(context))

    assert makespan == pytest.approx(30.0)
    assert [entry.processing_end for entry in schedule] == [10.0, 28.0]
    assert schedule[-1].finish == 30.0
    assert {entry.room for entry in schedule} <= set(context.rooms)
    assert schedule[0].personnel == "AN-1"
    assert schedule[1].personnel == "SU-1"


@pytest.mark.parametrize(
    ("field", "value"),
    (("room_assignment", "NOT-A-ROOM"), ("personnel_assignment", "AN-3")),
)
def test_ineligible_candidate_assignment_is_rejected(field, value):
    loaded = load_instance(INSTANCE)
    context = replace(loaded, jobs=loaded.jobs[:1])
    solution = _solution(context)
    solution[field][1][1] = value

    with pytest.raises(IneligibleAssignmentError, match="ineligible"):
        schedule_instance_solution(context, solution)
