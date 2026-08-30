"""Regression tests for the shared statistical helpers."""

import numpy as np
import pytest

from utils import statistics


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        ([1, 2, 3], [1, 2, 3], 0.0),
        ([1, 2, 4], [0, 3, 2], 0.5),
        ([1, 1, 3, 3], [1, 2, 2, 3], 0.0),
        ([], [], 0.0),
    ],
)
def test_rank_biserial_preserves_zero_ties_and_empty_inputs(x, y, expected):
    assert statistics.rank_biserial(x, y) == pytest.approx(expected)


def test_holm_preserves_original_order_and_empty_input():
    expected = np.array([0.06, 0.08, 0.5])
    np.testing.assert_allclose(statistics.holm([0.02, 0.04, 0.5]), expected)
    np.testing.assert_array_equal(statistics.holm([]), np.array([]))


def test_bootstrap_ci_is_reproducible_for_paired_data():
    x = np.array([0, 1, 2, 4, 7])
    y = np.array([0, 2, 1, 5, 6])
    def mean_difference(left, right):
        return float(np.mean(left - right))

    actual = statistics.bootstrap_ci(
        x, y, stat_fn=mean_difference, n_boot=200, seed=123
    )

    assert actual == pytest.approx((-0.8, 0.8))
    assert actual[0] <= actual[1]
