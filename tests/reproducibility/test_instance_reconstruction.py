"""Tests for reproducibility/instance_reconstruction.py.

The unit layer builds its own campaign-shaped SQLite databases (see
``conftest.build_synthetic_db``) so it runs on any machine.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess

import pytest

from tests.reproducibility.conftest import HAND_COMPUTED_JOBS, build_synthetic_db


class TestReconstructSynthetic:
    def test_reconstructs_hand_computed_five_term_decomposition(self, synthetic_db):
        """The five per-job times and the room-occupancy chain are recovered."""
        from reproducibility.instance_reconstruction import reconstruct_instance

        inst = reconstruct_instance(synthetic_db, 1)
        job = inst.jobs[1]

        assert job.transition_min == pytest.approx(20.0)
        assert job.setup_min == pytest.approx(5.0)
        assert job.anesthesia_min == pytest.approx(30.0)
        assert job.surgery_min == pytest.approx(100.0)
        assert job.cleanup_min == pytest.approx(10.0)
        assert job.chain_min == pytest.approx(160.0)

    def test_reconstructs_second_job_with_setup_dominating_transition(
        self, synthetic_db
    ):
        """Triangulation: job 2 has setup > transition, so `max` picks setup."""
        from reproducibility.instance_reconstruction import reconstruct_instance

        job = reconstruct_instance(synthetic_db, 1).jobs[2]

        assert job.transition_min == pytest.approx(4.0)
        assert job.setup_min == pytest.approx(9.0)
        assert job.anesthesia_min == pytest.approx(12.0)
        assert job.surgery_min == pytest.approx(40.0)
        assert job.cleanup_min == pytest.approx(6.0)
        # max(4, 9) + 12 + 40 + 6 == 67.0, not 4 + 9 + 12 + 40 + 6 == 71.0
        assert job.chain_min == pytest.approx(67.0)

    def test_exposes_run_identity_and_job_set(self, synthetic_db):
        from reproducibility.instance_reconstruction import reconstruct_instance

        inst = reconstruct_instance(synthetic_db, 1)

        assert inst.run_id == 1
        assert inst.num_jobs == 2
        assert sorted(inst.jobs) == [1, 2]
        assert inst.total_room_work_min == pytest.approx(160.0 + 67.0)


class TestMissingSource:
    def test_missing_db_raises_file_not_found(self, tmp_path):
        """A nonexistent database must raise before any connection is opened."""
        from reproducibility.instance_reconstruction import reconstruct_instance

        with pytest.raises(FileNotFoundError):
            reconstruct_instance(tmp_path / "does_not_exist.db", 1)

    def test_unknown_run_id_raises_reconstruction_error(self, synthetic_db):
        """An existing database with no rows for the run must raise, not default."""
        from reproducibility.instance_reconstruction import (
            InstanceReconstructionError,
            reconstruct_instance,
        )

        with pytest.raises(InstanceReconstructionError) as excinfo:
            reconstruct_instance(synthetic_db, 99)

        assert "99" in str(excinfo.value)

    def test_non_contiguous_job_ids_raise(self, tmp_path):
        """Assertion A5: the job set must be exactly {1..N}."""
        from reproducibility.instance_reconstruction import (
            InstanceReconstructionError,
            reconstruct_instance,
        )

        jobs = {1: HAND_COMPUTED_JOBS[1], 3: HAND_COMPUTED_JOBS[2]}
        db = build_synthetic_db(tmp_path / "gap.db", jobs)

        with pytest.raises(InstanceReconstructionError) as excinfo:
            reconstruct_instance(db, 1)

        assert "3" in str(excinfo.value)


#: 13 identical jobs, each with chain max(0, 10) + 10 + 20 + 10 = 50.0.
#: With 12 rooms the room-load relaxation 13*50/12 = 54.1667 exceeds the
#: critical-path relaxation 50.0, so the binding bound changes.
ROOM_BOUND_JOBS: dict[int, dict[str, float]] = {
    job_id: {
        "transition": 0.0,
        "setup": 10.0,
        "anesthesia": 10.0,
        "surgery": 20.0,
        "cleanup": 10.0,
    }
    for job_id in range(1, 14)
}


class TestBounds:
    def test_critical_path_bound_binds_on_the_hand_computed_instance(
        self, synthetic_db
    ):
        """LB is the max of the four relaxations; here the job chain wins."""
        from reproducibility.instance_reconstruction import reconstruct_instance

        bounds = reconstruct_instance(synthetic_db, 1).bounds

        assert bounds.lb_cp == pytest.approx(160.0)
        assert bounds.lb_room == pytest.approx(227.0 / 12.0)
        assert bounds.lb_surgeon == pytest.approx(140.0 / 29.0)
        assert bounds.lb_anesthetist == pytest.approx(42.0 / 11.0)
        assert bounds.lb == pytest.approx(160.0)
        assert bounds.binding == "critical-path"

    def test_room_load_bound_binds_when_jobs_outnumber_rooms(self, tmp_path):
        """Triangulation: a different relaxation must be able to bind."""
        from reproducibility.instance_reconstruction import reconstruct_instance

        db = build_synthetic_db(
            tmp_path / "room.db", ROOM_BOUND_JOBS, best_makespan=60.0
        )
        bounds = reconstruct_instance(db, 1).bounds

        assert bounds.lb_cp == pytest.approx(50.0)
        assert bounds.lb_room == pytest.approx(13 * 50.0 / 12.0)
        assert bounds.lb == pytest.approx(13 * 50.0 / 12.0)
        assert bounds.binding == "room-load"

    def test_lower_bound_above_best_makespan_raises(self, tmp_path):
        """Assertion A4: a bound above an achieved schedule is impossible."""
        from reproducibility.instance_reconstruction import (
            LowerBoundViolationError,
            reconstruct_instance,
        )

        db = build_synthetic_db(
            tmp_path / "violation.db", HAND_COMPUTED_JOBS, best_makespan=100.0
        )

        with pytest.raises(LowerBoundViolationError) as excinfo:
            reconstruct_instance(db, 1)

        message = str(excinfo.value)
        assert "160" in message and "100" in message


class TestDriftAssertion:
    def test_float_noise_within_tolerance_passes(self, tmp_path):
        """`proc_time_min` is derived by subtraction and carries float noise.

        The largest observed per-job spread is 2.27e-13, so the
        predicate must be a tolerance and must not require exact equality.
        """
        from reproducibility.instance_reconstruction import reconstruct_instance

        db = build_synthetic_db(
            tmp_path / "noise.db", HAND_COMPUTED_JOBS, proc_spread={1: 5.684e-14}
        )
        job = reconstruct_instance(db, 1).jobs[1]

        assert job.surgery_min == pytest.approx(100.0, abs=1e-12)

    @pytest.mark.parametrize(
        "knob", ["setup_spread", "proc_spread", "cleanup_spread", "transition_spread"]
    )
    def test_real_drift_is_rejected(self, tmp_path, knob):
        """Assertion A1: a spread above the tolerance breaks the fixed instance."""
        from reproducibility.instance_reconstruction import (
            InstanceReconstructionError,
            reconstruct_instance,
        )

        db = build_synthetic_db(
            tmp_path / f"drift_{knob}.db", HAND_COMPUTED_JOBS, **{knob: {2: 1e-6}}
        )

        with pytest.raises(InstanceReconstructionError) as excinfo:
            reconstruct_instance(db, 1)

        message = str(excinfo.value)
        assert "job 2" in message
        assert "1e-09" in message or "1e-9" in message


class TestAnesthesiaIdentity:
    def test_per_algorithm_min_op1_finish_must_agree(self, tmp_path):
        """Assertion A2: the four algorithms must reach the same physical floor.

        Disagreement means no simulation of at least one algorithm scheduled the
        job first, so the estimator would be inflated by queueing delay.
        """
        from reproducibility.instance_reconstruction import (
            AnesthesiaIdentityError,
            reconstruct_instance,
        )

        db = build_synthetic_db(
            tmp_path / "disagree.db",
            HAND_COMPUTED_JOBS,
            algo_op1_offset={"dPSO": 3.0},
        )

        with pytest.raises(AnesthesiaIdentityError) as excinfo:
            reconstruct_instance(db, 1)

        message = str(excinfo.value)
        assert "job 1" in message
        assert "dPSO" in message
        assert "53.0" in message  # the inflated per-algorithm floor

    def test_agreeing_algorithms_pass(self, synthetic_db):
        """Triangulation: identical per-algorithm floors must not raise."""
        from reproducibility.instance_reconstruction import reconstruct_instance

        assert reconstruct_instance(synthetic_db, 1).jobs[1].min_op1_finish_min == (
            pytest.approx(50.0)
        )

    def test_anesthesia_must_be_positive(self, tmp_path):
        """Assertion A3: a non-positive anesthesia means the identity inverted."""
        from reproducibility.instance_reconstruction import (
            AnesthesiaIdentityError,
            reconstruct_instance,
        )

        jobs = {
            1: {
                "transition": 20.0,
                "setup": 5.0,
                "anesthesia": 0.0,
                "surgery": 100.0,
                "cleanup": 10.0,
            }
        }
        db = build_synthetic_db(tmp_path / "zero_anes.db", jobs)

        with pytest.raises(AnesthesiaIdentityError) as excinfo:
            reconstruct_instance(db, 1)

        assert "job 1" in str(excinfo.value)


class TestResourceCounts:
    def test_default_profile_counts_come_from_config(self):
        """The default profile declares 12 rooms, 11 anesthetists, 29 surgeons."""
        from reproducibility.instance_reconstruction import resource_counts

        counts = resource_counts()

        assert counts.rooms == 12
        assert counts.anesthetists == 11
        assert counts.surgeons == 29

    def test_alternate_profile_does_not_silently_reuse_defaults(self, monkeypatch):
        """A profile declaring 6/6 personnel must yield 6 and 6, not 11 and 29.

        `config.quick.yaml` and `config.validation.yaml` declare 6/6, so a
        hardcoded literal would be silently wrong under those profiles.
        """
        from config import config as project_config
        from reproducibility.instance_reconstruction import resource_counts

        monkeypatch.setattr(
            project_config,
            "PERSONNEL_BY_OPERATION",
            {1: [f"A{i}" for i in range(6)], 2: [f"S{i}" for i in range(6)]},
        )
        counts = resource_counts()

        assert counts.anesthetists == 6
        assert counts.surgeons == 6

    def test_bounds_follow_the_configured_room_count(self, tmp_path, monkeypatch):
        """The room relaxation divides by the configured room count."""
        from config import config as project_config
        from reproducibility.instance_reconstruction import reconstruct_instance

        monkeypatch.setattr(
            project_config, "ALL_ROOMS", [f"Pabellon_{i}" for i in range(4)]
        )
        db = build_synthetic_db(
            tmp_path / "four_rooms.db", HAND_COMPUTED_JOBS, best_makespan=160.0
        )

        bounds = reconstruct_instance(db, 1).bounds

        assert bounds.lb_room == pytest.approx(227.0 / 4.0)


class TestGapInvariant:
    def test_gap_is_zero_when_best_equals_lower_bound(self, synthetic_db):
        from reproducibility.instance_reconstruction import reconstruct_instance

        inst = reconstruct_instance(synthetic_db, 1)

        assert inst.best_makespan_min == pytest.approx(160.0)
        assert inst.gap_pct == pytest.approx(0.0)

    def test_gap_is_positive_and_matches_the_definition(self, tmp_path):
        """Triangulation: a strictly positive gap must compute correctly."""
        from reproducibility.instance_reconstruction import reconstruct_instance

        db = build_synthetic_db(
            tmp_path / "positive_gap.db", HAND_COMPUTED_JOBS, best_makespan=200.0
        )
        inst = reconstruct_instance(db, 1)

        expected = (200.0 - 160.0) / 160.0 * 100.0
        assert inst.gap_pct == pytest.approx(expected)
        assert inst.gap_pct == pytest.approx(25.0)

    @pytest.mark.parametrize("best", [160.0, 175.5, 200.0, 480.0])
    def test_gap_is_never_negative_on_a_synthetic_twin(self, tmp_path, best):
        """The invariant whose absence permitted a negative published gap."""
        from reproducibility.instance_reconstruction import reconstruct_instance

        db = build_synthetic_db(
            tmp_path / f"twin_{best}.db", HAND_COMPUTED_JOBS, best_makespan=best
        )
        inst = reconstruct_instance(db, 1)

        assert inst.bounds.lb <= inst.best_makespan_min
        assert inst.gap_pct >= 0.0


class TestSurgeriesAdapter:
    def test_emits_only_the_keys_the_dispatch_harness_consumes(self, synthetic_db):
        """`simulate_dispatching_rule` reads exactly five keys per job."""
        from reproducibility.instance_reconstruction import (
            reconstruct_instance,
            to_surgeries_data,
        )

        data = to_surgeries_data(reconstruct_instance(synthetic_db, 1))

        assert sorted(data) == [1, 2]
        assert set(data[1]) == {
            1,
            2,
            "setup_by_op",
            "transition_by_op",
            "cleanup_by_op",
        }
        # Keys present in the sampler output but not reconstructible.
        assert "prep" not in data[1]
        assert "cleanup" not in data[1]
        assert "transition_after_op1" not in data[1]

    def test_maps_each_reconstructed_time_to_its_operation(self, synthetic_db):
        from reproducibility.instance_reconstruction import (
            reconstruct_instance,
            to_surgeries_data,
        )

        job = to_surgeries_data(reconstruct_instance(synthetic_db, 1))[1]

        assert job[1] == pytest.approx(30.0)  # anesthesia
        assert job[2] == pytest.approx(100.0)  # surgery
        assert job["setup_by_op"] == {1: pytest.approx(5.0), 2: 0.0}
        assert job["transition_by_op"] == {1: pytest.approx(20.0), 2: 0.0}
        assert job["cleanup_by_op"] == {1: 0.0, 2: pytest.approx(10.0)}


#: Values chosen so that rounding to any fixed number of decimals would be
#: detectable on the JSON round-trip.
PRECISION_JOBS: dict[int, dict[str, float]] = {
    1: {
        "transition": 23.590512345678901,
        "setup": 10.084598765432109,
        "anesthesia": 70.395712345678903,
        "surgery": 348.25141234567891,
        "cleanup": 38.157712345678904,
    }
}


class TestProvenanceArtifact:
    def test_emitted_payload_carries_the_documented_schema(
        self, synthetic_db, tmp_path
    ):
        from reproducibility.instance_reconstruction import (
            emit_instance_json,
            reconstruct_instance,
        )

        inst = reconstruct_instance(synthetic_db, 1)
        out = emit_instance_json(inst, tmp_path)

        assert out.name == "instance_run1.json"
        payload = json.loads(out.read_text(encoding="utf-8"))

        assert payload["schema_version"] == 1
        assert payload["run_id"] == 1
        assert payload["num_jobs"] == 2
        assert payload["units"] == "minutes"
        assert payload["resources"] == {
            "rooms": 12,
            "anesthetists": 11,
            "surgeons": 29,
        }
        assert [job["job_id"] for job in payload["jobs"]] == [1, 2]
        assert payload["jobs"][0]["chain_min"] == pytest.approx(160.0)
        assert payload["bounds"]["binding"] == "critical-path"
        assert payload["best_makespan_min"] == pytest.approx(160.0)
        assert payload["gap_pct"] == pytest.approx(0.0)
        assert set(payload["environment"]) == {"python", "numpy", "pandas"}
        assert len(payload["content_sha256"]) == 64

    def test_checksum_is_identical_across_two_emissions(self, synthetic_db, tmp_path):
        from reproducibility.instance_reconstruction import (
            emit_instance_json,
            reconstruct_instance,
        )

        inst = reconstruct_instance(synthetic_db, 1)
        first = json.loads(
            emit_instance_json(inst, tmp_path / "a").read_text(encoding="utf-8")
        )
        second = json.loads(
            emit_instance_json(inst, tmp_path / "b").read_text(encoding="utf-8")
        )

        assert first["content_sha256"] == second["content_sha256"]

    def test_checksum_changes_when_one_job_value_changes(self, synthetic_db, tmp_path):
        from reproducibility.instance_reconstruction import (
            emit_instance_json,
            reconstruct_instance,
        )

        inst = reconstruct_instance(synthetic_db, 1)
        baseline = json.loads(
            emit_instance_json(inst, tmp_path / "base").read_text(encoding="utf-8")
        )

        mutated_job = dataclasses.replace(inst.jobs[1], surgery_min=100.5)
        mutated = dataclasses.replace(inst, jobs={**inst.jobs, 1: mutated_job})
        changed = json.loads(
            emit_instance_json(mutated, tmp_path / "changed").read_text(
                encoding="utf-8"
            )
        )

        assert changed["content_sha256"] != baseline["content_sha256"]

    def test_verify_detects_a_tampered_environment_field(self, synthetic_db, tmp_path):
        """The digest covers `environment`, so wrong-interpreter output differs."""
        from reproducibility.instance_reconstruction import (
            emit_instance_json,
            reconstruct_instance,
            verify_instance_json,
        )

        out = emit_instance_json(reconstruct_instance(synthetic_db, 1), tmp_path)
        assert verify_instance_json(out) is True

        payload = json.loads(out.read_text(encoding="utf-8"))
        payload["environment"]["numpy"] = "2.2.6"
        out.write_text(json.dumps(payload), encoding="utf-8")

        assert verify_instance_json(out) is False

    def test_floats_round_trip_without_rounding(self, tmp_path):
        from reproducibility.instance_reconstruction import (
            emit_instance_json,
            reconstruct_instance,
        )

        db = build_synthetic_db(tmp_path / "precision.db", PRECISION_JOBS)
        inst = reconstruct_instance(db, 1)
        payload = json.loads(
            emit_instance_json(inst, tmp_path / "out").read_text(encoding="utf-8")
        )

        emitted = payload["jobs"][0]
        assert emitted["surgery_min"] == inst.jobs[1].surgery_min
        assert emitted["transition_min"] == 23.590512345678901
        assert repr(emitted["cleanup_min"]) == repr(inst.jobs[1].cleanup_min)

    def test_default_emission_directory_is_not_git_ignored(self):
        """`.gitignore` excludes `results/` as a directory, so the provenance
        record is emitted where version control can actually reach it."""
        from reproducibility.instance_reconstruction import (
            DEFAULT_INSTANCE_JSON_DIR,
            PROJECT_ROOT,
        )

        candidate = DEFAULT_INSTANCE_JSON_DIR / "instance_run1.json"
        relative = candidate.relative_to(PROJECT_ROOT).as_posix()

        completed = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=PROJECT_ROOT,
            capture_output=True,
        )

        # `git check-ignore -q` exits 1 when the path is NOT ignored.
        assert completed.returncode == 1, (
            f"{relative} is git-ignored; the provenance record would be "
            "uncommittable"
        )


class TestEnvironmentGuard:
    def test_canonical_environment_passes(self):
        from reproducibility.instance_reconstruction import (
            assert_canonical_environment,
        )

        assert assert_canonical_environment() is None

    @pytest.mark.parametrize(
        "module_name, bad_version", [("numpy", "2.2.6"), ("pandas", "2.3.2")]
    )
    def test_non_canonical_versions_are_refused(
        self, monkeypatch, module_name, bad_version
    ):
        """numpy and pandas are the discriminating fields between the two
        available interpreters, which report the same Python version."""
        import importlib
        import sys

        from reproducibility.instance_reconstruction import (
            assert_canonical_environment,
        )

        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "__version__", bad_version)

        with pytest.raises(EnvironmentError) as excinfo:
            assert_canonical_environment()

        message = str(excinfo.value)
        assert bad_version in message
        assert module_name in message
        assert sys.executable in message
        assert "requirements.txt" in message
        assert "E:\\Hospitales" not in message

    def test_python_version_alone_is_not_the_discriminator(self, monkeypatch):
        """Changing Python alone must not affect a canonical dependency set."""
        import importlib
        import sys

        from reproducibility.instance_reconstruction import (
            CANONICAL_ENVIRONMENT,
            assert_canonical_environment,
        )

        for module_name, version in CANONICAL_ENVIRONMENT.items():
            module = importlib.import_module(module_name)
            monkeypatch.setattr(module, "__version__", version)
        monkeypatch.setattr(sys, "version", "3.9.0 (fake build)")

        assert assert_canonical_environment() is None
