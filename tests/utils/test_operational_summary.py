"""Tests for the representative PAIRED operational aggregates.

Covers:
    (a) perform_paired_statistical_test with metric='patients_with_extra_wait'
        on a small synthetic paired dict (generalized metric parameter).
    (b) compute_operational_summary win-rate / mean-rank correctness on a tiny
        hand-built example with a known answer.
    (c) export_operational_paired_summary + reused export_statistical_analysis
        write the expected headers/rows.

Strategy: pure functions + tmp_path CSV round-trip, no DB required.
"""

import csv

import numpy as np
import pytest

from utils import reporting, statistics


# ---------------------------------------------------------------------------
# (a) Generalized paired test — metric parameter
# ---------------------------------------------------------------------------


class TestPairedTestMetricParameter:
    def test_default_metric_is_makespan(self):
        """Default behavior must read the 'makespan' key (backward compatible)."""
        all_results = {
            "A": {"makespan": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]},
            "B": {"makespan": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]},
        }
        out = statistics.perform_paired_statistical_test(all_results, 0.05, verbose=False)
        assert out["n_valid"] == 7
        assert len(out["pairwise"]) == 1
        # B is uniformly worse (higher makespan) -> A better when significant.
        pair = out["pairwise"][0]
        assert pair["mean_a"] == pytest.approx(13.0)
        assert pair["mean_b"] == pytest.approx(23.0)

    def test_metric_patients_with_extra_wait(self):
        """metric='patients_with_extra_wait' reads that series instead of makespan."""
        all_results = {
            "GA": {"patients_with_extra_wait": [15, 16, 14, 17, 15, 16, 14, 15]},
            "SBOA": {"patients_with_extra_wait": [10, 9, 11, 10, 12, 9, 10, 11]},
            "dPSO": {"patients_with_extra_wait": [20, 21, 19, 22, 20, 21, 19, 20]},
        }
        out = statistics.perform_paired_statistical_test(
            all_results, 0.05, verbose=False, metric="patients_with_extra_wait"
        )
        assert out["n_valid"] == 8
        assert out["friedman"] is not None
        # 3 algos -> 3 pairwise comparisons
        assert len(out["pairwise"]) == 3
        # Means computed on the waiting series, not makespan.
        by_pair = {(p["algo_a"], p["algo_b"]): p for p in out["pairwise"]}
        assert by_pair[("GA", "SBOA")]["mean_a"] == pytest.approx(15.25)
        assert by_pair[("GA", "SBOA")]["mean_b"] == pytest.approx(10.25)

    def test_metric_missing_key_raises(self):
        """A metric absent from the algo dict must raise (no silent fallback)."""
        all_results = {"A": {"makespan": [1.0] * 8}, "B": {"makespan": [2.0] * 8}}
        with pytest.raises(KeyError):
            statistics.perform_paired_statistical_test(
                all_results, 0.05, verbose=False, metric="patients_with_extra_wait"
            )


# ---------------------------------------------------------------------------
# (b) compute_operational_summary — win-rate / mean-rank known answer
# ---------------------------------------------------------------------------


class TestComputeOperationalSummary:
    def test_win_rate_and_rank_known_answer(self):
        """Hand-built 4-sim example with a fully determined ranking.

        Makespans (rows = sim index):
            sim0: A=10 B=20 C=30 D=40  -> ranks A1 B2 C3 D4
            sim1: A=15 B=10 C=30 D=40  -> ranks A2 B1 C3 D4
            sim2: A=10 B=20 C=05 D=40  -> ranks A2 B3 C1 D4
            sim3: A=10 B=10 C=30 D=40  -> ranks A1.5 B1.5 C3 D4 (average tie rank)

        Wins (average rank==1): A=sim0 -> 1/4=25%; B=sim1 -> 1/4=25%;
                         C=sim2 -> 1/4=25%; D=0 -> 0%.
        Mean ranks: A=(1+2+2+1.5)/4=1.625; B=(2+1+3+1.5)/4=1.875;
                     C=(3+3+1+3)/4=2.5; D=4.0.
        """
        all_results = {
            "A": {"final_makespan": [10, 15, 10, 10]},
            "B": {"final_makespan": [20, 10, 20, 10]},
            "C": {"final_makespan": [30, 30, 5, 30]},
            "D": {"final_makespan": [40, 40, 40, 40]},
        }
        summary = statistics.compute_operational_summary(all_results)

        assert summary["A"]["win_rate_pct"] == pytest.approx(25.0)
        assert summary["B"]["win_rate_pct"] == pytest.approx(25.0)
        assert summary["C"]["win_rate_pct"] == pytest.approx(25.0)
        assert summary["D"]["win_rate_pct"] == pytest.approx(0.0)

        assert summary["A"]["mean_rank"] == pytest.approx(1.625)
        assert summary["B"]["mean_rank"] == pytest.approx(1.875)
        assert summary["C"]["mean_rank"] == pytest.approx(2.5)
        assert summary["D"]["mean_rank"] == pytest.approx(4.0)

    def test_mean_and_sd_and_n(self):
        """Mean/sd (ddof=1) and n reported per metric."""
        all_results = {
            "A": {
                "patients_with_extra_wait": [10.0, 20.0],
                "avg_extra_wait_min": [5.0, 15.0],
                "final_makespan": [100.0, 200.0],
            },
            "B": {
                "patients_with_extra_wait": [30.0, 30.0],
                "avg_extra_wait_min": [1.0, 3.0],
                "final_makespan": [50.0, 60.0],
            },
        }
        summary = statistics.compute_operational_summary(all_results)
        assert summary["A"]["patients_with_extra_wait_mean"] == pytest.approx(15.0)
        # sample std of [10, 20] = sqrt(50) ~ 7.0710678
        assert summary["A"]["patients_with_extra_wait_sd"] == pytest.approx(
            np.std([10.0, 20.0], ddof=1)
        )
        assert summary["A"]["n"] == 2
        assert summary["B"]["avg_extra_wait_min_mean"] == pytest.approx(2.0)

    def test_single_sim_sd_zero(self):
        """A single finite value yields sd=0.0 (no ddof=1 nan)."""
        all_results = {
            "A": {
                "patients_with_extra_wait": [12.0],
                "avg_extra_wait_min": [4.0],
                "final_makespan": [99.0],
            },
        }
        summary = statistics.compute_operational_summary(all_results)
        assert summary["A"]["patients_with_extra_wait_sd"] == 0.0
        assert summary["A"]["win_rate_pct"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# (c) CSV export shape
# ---------------------------------------------------------------------------


class TestOperationalPairedExport:
    def test_export_operational_paired_summary_headers_and_rows(self, tmp_path):
        all_results = {
            "GA": {
                "patients_with_extra_wait": [10.0, 20.0],
                "avg_extra_wait_min": [5.0, 15.0],
                "final_makespan": [100.0, 200.0],
            },
            "SBOA": {
                "patients_with_extra_wait": [30.0, 30.0],
                "avg_extra_wait_min": [1.0, 3.0],
                "final_makespan": [50.0, 60.0],
            },
        }
        summary = statistics.compute_operational_summary(all_results)
        out = tmp_path / "operational_paired_run1.csv"
        result = reporting.export_operational_paired_summary(summary, str(out))
        assert result == str(out)

        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        assert rows[0] == [
            "algo_name",
            "n",
            "patients_with_extra_wait_mean",
            "patients_with_extra_wait_sd",
            "avg_extra_wait_min_mean",
            "avg_extra_wait_min_sd",
            "makespan_win_rate_pct",
            "makespan_mean_rank",
        ]
        # Two algorithms, sorted alphabetically -> GA before SBOA.
        assert [r[0] for r in rows[1:]] == ["GA", "SBOA"]
        ga = rows[1]
        assert ga[1] == "2"  # n
        assert ga[2] == "15.00"  # patients_with_extra_wait_mean

    def test_export_empty_summary_returns_none(self, tmp_path):
        out = tmp_path / "empty.csv"
        assert reporting.export_operational_paired_summary({}, str(out)) is None

    def test_waiting_significance_reuses_statistical_analysis_shape(self, tmp_path):
        """The waiting significance CSV reuses export_statistical_analysis."""
        all_results = {
            "GA": {"patients_with_extra_wait": [15, 16, 14, 17, 15, 16, 14, 15]},
            "SBOA": {"patients_with_extra_wait": [10, 9, 11, 10, 12, 9, 10, 11]},
            "dPSO": {"patients_with_extra_wait": [20, 21, 19, 22, 20, 21, 19, 20]},
        }
        test = statistics.perform_paired_statistical_test(
            all_results, 0.05, verbose=False, metric="patients_with_extra_wait"
        )
        out = tmp_path / "waiting_significance_run1.csv"
        reporting.export_statistical_analysis(test["pairwise"], str(out))

        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        header = rows[0]
        # Same columns as the makespan statistical_analysis export.
        for col in ("algo_a", "algo_b", "p_value", "p_adjusted", "better_algo"):
            assert col in header
        assert len(rows) == 1 + 3  # header + 3 pairwise comparisons
