"""
Regression tests for the fitness cache invalidation in simulation/scheduler.py.

The module-level `_FITNESS_CACHE` is keyed only by (job sequence, room
assignment); the processing times live in `surgeries_data`, which is NOT part of
the key. Correctness therefore depends entirely on the cache being cleared
whenever a different `surgeries_data` object is supplied.

Invalidation is detected with `id(surgeries_data)`. CPython recycles id() values
once an object is garbage collected, so a freed dataset could be replaced by a
different dataset at the same address, the identity check would report "same
data", and a lookup would silently return another dataset's fitness. The module
guards against this by keeping a strong reference (`_LAST_SURGERIES_DATA_REF`)
to the dataset that currently owns the cache.
"""

import gc

import pytest

from simulation import scheduler
from simulation.scheduler import calculate_schedule_fitness


JOB_IDS = [1, 2, 3]


def _make_surgeries_data(op1_duration, op2_duration):
    """Builds a fresh surgeries_data object with uniform durations."""
    return {
        job: {1: float(op1_duration), 2: float(op2_duration)} for job in JOB_IDS
    }


def _make_solution():
    """Builds a fixed solution so the cache key is identical across datasets."""
    return {
        "job_sequence_base": list(JOB_IDS),
        "room_assignment": {
            job: {1: "Pabellon_1", 2: "Pabellon_2"} for job in JOB_IDS
        },
    }


@pytest.fixture(autouse=True)
def _reset_fitness_cache():
    """Isolates each test from cache state left behind by other tests."""
    scheduler._FITNESS_CACHE.clear()
    scheduler._LAST_SURGERIES_DATA_ID = None
    scheduler._LAST_SURGERIES_DATA_REF = None
    yield
    scheduler._FITNESS_CACHE.clear()
    scheduler._LAST_SURGERIES_DATA_ID = None
    scheduler._LAST_SURGERIES_DATA_REF = None


class TestFitnessCacheInvalidation:
    def test_distinct_dataset_object_invalidates_cache(self):
        """A new surgeries_data object must not reuse the previous fitness."""
        solution = _make_solution()

        data_a = _make_surgeries_data(30, 60)
        fitness_a = calculate_schedule_fitness(solution, data_a)

        data_b = _make_surgeries_data(90, 180)
        fitness_b = calculate_schedule_fitness(solution, data_b)

        assert fitness_a != fitness_b
        # Ground truth: return_details bypasses the cache entirely.
        expected_b, _, _ = calculate_schedule_fitness(
            solution, data_b, return_details=True
        )
        assert fitness_b == pytest.approx(expected_b)

    def test_cache_hit_is_reused_for_the_same_dataset_object(self):
        """Same object plus same solution must still hit the cache."""
        solution = _make_solution()
        data = _make_surgeries_data(30, 60)

        first = calculate_schedule_fitness(solution, data)
        cache_size_after_first = len(scheduler._FITNESS_CACHE)
        second = calculate_schedule_fitness(solution, data)

        assert first == second
        assert len(scheduler._FITNESS_CACHE) == cache_size_after_first

    def test_owning_dataset_is_pinned_so_its_id_cannot_be_recycled(self):
        """The retained reference must always match the recorded id()."""
        solution = _make_solution()

        data_a = _make_surgeries_data(30, 60)
        calculate_schedule_fitness(solution, data_a)
        assert scheduler._LAST_SURGERIES_DATA_REF is data_a
        assert scheduler._LAST_SURGERIES_DATA_ID == id(data_a)

        data_b = _make_surgeries_data(90, 180)
        calculate_schedule_fitness(solution, data_b)
        assert scheduler._LAST_SURGERIES_DATA_REF is data_b
        assert scheduler._LAST_SURGERIES_DATA_ID == id(data_b)

    def test_no_stale_fitness_across_many_short_lived_datasets(self):
        """
        Exercises the id-recycling hazard directly.

        Each dataset is dropped immediately after use, so CPython is free to
        reuse its address for the next one. Every fitness must still match the
        uncached ground truth for its own dataset. Without the retained
        reference this loop produces stale cache hits (measured: a handful per
        few thousand iterations); with it, staleness is impossible.
        """
        solution = _make_solution()

        for step in range(2000):
            data = _make_surgeries_data(20.0 + step, 40.0 + 2 * step)

            cached = calculate_schedule_fitness(solution, data)
            expected, _, _ = calculate_schedule_fitness(
                solution, data, return_details=True
            )

            assert cached == pytest.approx(expected), (
                f"stale cached fitness at step {step}: {cached} != {expected}"
            )

            del data
            gc.collect()

    def test_forced_id_collision_does_not_produce_a_stale_cache_hit(self):
        """
        Reproduces the exact failure mode with a real recycled address.

        A dataset is evaluated, dropped, and new datasets are allocated until
        one lands on the freed address. Under the old identity-only check that
        collision was read as "same dataset" and returned the previous
        dataset's fitness. The retained reference prevents the address from
        ever being freed, so the collision cannot happen at all; the assertion
        below therefore holds either way and pins the invariant.
        """
        solution = _make_solution()

        data_a = _make_surgeries_data(30, 60)
        fitness_a = calculate_schedule_fitness(solution, data_a)
        recycled_id = id(data_a)
        del data_a
        gc.collect()

        collided = None
        for _ in range(200000):
            candidate = _make_surgeries_data(300, 600)
            if id(candidate) == recycled_id:
                collided = candidate
                break
            del candidate

        if collided is None:
            pytest.skip("CPython did not recycle the address within the budget")

        cached = calculate_schedule_fitness(solution, collided)
        expected, _, _ = calculate_schedule_fitness(
            solution, collided, return_details=True
        )

        assert cached != fitness_a
        assert cached == pytest.approx(expected)
