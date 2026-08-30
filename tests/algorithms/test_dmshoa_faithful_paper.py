"""
RED tests for the faithful dMShOA rewrite.

Tasks 1.1 + 1.2: Pure helper unit tests.
Tasks 3.1 + 3.3: Anti-regression and integration tests.
"""

import math
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: import helpers from dmshoa (will fail until helpers are exposed)
# ---------------------------------------------------------------------------


def _import_helpers():
    """Import pure helpers from algorithms.dmshoa."""
    from algorithms import dmshoa  # noqa: F401 – reload safe

    return dmshoa


# ===========================================================================
# Task 1.1 — PTI helpers: _compute_lpa, _classify_polarization,
#             _angular_difference, _update_pti_vector
# ===========================================================================


class TestComputeLpa:
    """_compute_lpa(old_pos, new_pos) -> float in [0, π]."""

    def test_identical_vectors_give_zero_angle(self):
        """Identical old/new position → angle = 0."""
        dm = _import_helpers()
        v = np.array([1.0, 2.0, 3.0])
        lpa = dm._compute_lpa(v, v)
        assert math.isclose(lpa, 0.0, abs_tol=1e-10), f"Expected 0 got {lpa}"

    def test_opposite_vectors_give_pi(self):
        """Antiparallel vectors → angle = π."""
        dm = _import_helpers()
        old = np.array([1.0, 0.0, 0.0])
        new = np.array([-1.0, 0.0, 0.0])
        lpa = dm._compute_lpa(old, new)
        assert math.isclose(lpa, math.pi, abs_tol=1e-10), f"Expected π got {lpa}"

    def test_orthogonal_vectors_give_half_pi(self):
        """Perpendicular vectors → angle = π/2."""
        dm = _import_helpers()
        old = np.array([1.0, 0.0])
        new = np.array([0.0, 1.0])
        lpa = dm._compute_lpa(old, new)
        assert math.isclose(lpa, math.pi / 2, abs_tol=1e-10), f"Expected π/2 got {lpa}"

    def test_result_clamped_within_0_pi(self):
        """Output is always in [0, π] even for random vectors."""
        dm = _import_helpers()
        rng = np.random.default_rng(42)
        for _ in range(50):
            a = rng.uniform(-5, 5, 8)
            b = rng.uniform(-5, 5, 8)
            lpa = dm._compute_lpa(a, b)
            assert 0.0 - 1e-10 <= lpa <= math.pi + 1e-10, f"LPA={lpa} out of [0,π]"


class TestRemovedHelpersAntiRegression:
    """P3-3.1 — Anti-regression: second-pass radian helpers are removed in third pass.

    The new MATLAB-faithful helpers (_pti_distances, _pti_from_distances, _sample_rpa_deg)
    replace _classify_polarization, _lad, _rad, _ref_angle, _compute_rpa.
    """

    def test_classify_polarization_removed(self):
        """_classify_polarization must NOT exist (replaced by _pti_distances + _pti_from_distances)."""
        from algorithms import dmshoa
        assert not hasattr(dmshoa, "_classify_polarization"), (
            "_classify_polarization must be removed; use _pti_distances + _pti_from_distances"
        )

    def test_lad_removed(self):
        """_lad must NOT exist (replaced by MATLAB channel distances)."""
        from algorithms import dmshoa
        assert not hasattr(dmshoa, "_lad"), (
            "_lad must be removed; MATLAB uses min([LinH, LinV, PolC]) not LAD/RAD"
        )

    def test_rad_removed(self):
        """_rad must NOT exist (replaced by MATLAB channel distances)."""
        from algorithms import dmshoa
        assert not hasattr(dmshoa, "_rad"), (
            "_rad must be removed; MATLAB uses min([LinH, LinV, PolC]) not LAD/RAD"
        )

    def test_ref_angle_removed(self):
        """_ref_angle must NOT exist (no longer used)."""
        from algorithms import dmshoa
        assert not hasattr(dmshoa, "_ref_angle"), (
            "_ref_angle must be removed; replaced by MATLAB degree-domain distances"
        )

    def test_compute_rpa_removed(self):
        """_compute_rpa must NOT exist (replaced by _sample_rpa_deg)."""
        from algorithms import dmshoa
        assert not hasattr(dmshoa, "_compute_rpa"), (
            "_compute_rpa must be removed; replaced by _sample_rpa_deg (integer [1,90])"
        )


class TestUpdatePtiVector:
    """_update_pti_vector(X_old, X_new, rng) -> np.ndarray of ints in {1,2,3}."""

    def test_output_length_matches_population(self):
        """Output length matches number of agents."""
        dm = _import_helpers()
        rng = np.random.default_rng(0)
        X_old = rng.uniform(-1, 1, (5, 10))
        X_new = rng.uniform(-1, 1, (5, 10))
        pti = dm._update_pti_vector(X_old, X_new, rng)
        assert len(pti) == 5

    def test_output_values_in_valid_set(self):
        """Every PTI value must be in {1, 2, 3}."""
        dm = _import_helpers()
        rng = np.random.default_rng(1)
        X_old = rng.uniform(-3, 3, (8, 12))
        X_new = rng.uniform(-3, 3, (8, 12))
        pti = dm._update_pti_vector(X_old, X_new, rng)
        assert set(pti).issubset({1, 2, 3}), f"PTI values outside {{1,2,3}}: {set(pti)}"

    def test_update_pti_dual_eye_selection(self):
        """_update_pti_vector selects the eye with the lower min-distance (MATLAB semantics).

        Triangulation: force left-eye dif_1=0 (identical vectors, angle=0° → LinH=0)
        and verify that PTI=1 (LinH wins on left eye, which has the lowest distance).
        """
        dm = _import_helpers()
        # Identical vectors → left-eye angle=0° → LinH=0 → dif_1=0
        # Right-eye: randi([1,90]) with seed 5 → pick a seed where rpa_deg is NOT in [35,55]
        # so dif_2 > 0 and left eye wins.
        # Check: seed 5 → rng.integers(1,91) = ?
        test_rng = np.random.default_rng(5)
        rpa_deg = int(test_rng.integers(1, 91))
        lh2, lv2, pc2 = dm._pti_distances(float(rpa_deg))
        dif_2 = min(lh2, lv2, pc2)
        # If dif_2 > 0 left eye wins, if dif_2==0 it depends
        # Just verify the whole pipeline produces valid PTI
        X_old = np.array([[1.0, 0.0, 0.0]])
        X_new = np.array([[1.0, 0.0, 0.0]])
        rng_actual = np.random.default_rng(5)
        pti = dm._update_pti_vector(X_old, X_new, rng_actual)
        # left-eye dif_1=0; if dif_2>0 left wins → PTI=1
        # if dif_2==0 left still wins (<=) → PTI=1
        assert pti[0] == 1, f"Expected PTI=1 (left-eye dif_1=0 <= dif_2={dif_2}), got {pti[0]}"

    def test_identical_old_new_gives_valid_pti(self):
        """If X_old == X_new the circshift still produces valid PTI in {1,2,3}.

        NOTE: With MATLAB circshift semantics, row i of shifted_old may differ from
        row i of shifted_new (different shifts k1 vs k2), so the left-eye angle is
        NOT guaranteed to be 0° for all agents. PTI=1 (Foraging) is only guaranteed
        for the degenerate pop_size=1 case where no shift is applied.
        """
        dm = _import_helpers()
        rng = np.random.default_rng(2)
        X = rng.uniform(-2, 2, (4, 6))
        pti = dm._update_pti_vector(X, X.copy(), rng)
        assert all(p in {1, 2, 3} for p in pti), f"PTI outside valid set: {pti}"

    def test_identical_single_agent_gives_foraging_pti(self):
        """Single-agent (pop_size=1): no shift applied, angle=0 → PTI=1 (Foraging)."""
        dm = _import_helpers()
        rng = np.random.default_rng(2)
        X = np.array([[1.0, 0.0, 0.0]])
        pti = dm._update_pti_vector(X, X.copy(), rng)
        assert pti[0] == 1, f"Expected PTI=1 for identical single-agent, got {pti[0]}"

    def test_opposite_positions_give_valid_pti(self):
        """If X_new = -X_old the circshift produces valid PTI in {1,2,3}.

        NOTE: With MATLAB circshift semantics, row i of shifted_old is compared with
        row i of shifted_new (different shifts k1 vs k2), so the angle between paired
        rows is no longer guaranteed to be π. PTI=1 is only guaranteed for pop_size=1.
        """
        dm = _import_helpers()
        rng = np.random.default_rng(3)
        X_old = rng.uniform(1, 2, (4, 6))
        X_new = -X_old
        pti = dm._update_pti_vector(X_old, X_new, rng)
        assert all(p in {1, 2, 3} for p in pti), f"PTI outside valid set: {pti}"

    def test_opposite_single_agent_gives_foraging_pti(self):
        """Single-agent (pop_size=1): no shift applied, angle=π → PTI=1 (Foraging, LPT band)."""
        dm = _import_helpers()
        rng = np.random.default_rng(3)
        X_old = np.array([[1.0, 0.0, 0.0]])
        X_new = np.array([[-1.0, 0.0, 0.0]])
        pti = dm._update_pti_vector(X_old, X_new, rng)
        assert pti[0] == 1, f"Expected PTI=1 for opposite single-agent (angle=π), got {pti[0]}"


# ===========================================================================
# Task 1.2 — Decoding & bounds: _decode_position, _repair_bounds
# ===========================================================================


class TestDecodePosition:
    """_decode_position(position, job_ids) -> solution dict."""

    def _get_job_ids(self, n=5):
        return list(range(1, n + 1))

    def test_output_has_required_keys(self, tmp_path):
        """Decoded solution must have 'job_sequence_base' and 'room_assignment'."""
        import os, sys, yaml

        cfg = {
            "experiment": {
                "num_simulations": 1,
                "std_factor_times": 0.0,
                "alpha_test": 0.05,
                "num_procedures": 5,
                "output_dirs": {"plots": "p", "csv": "c"},
            },
            "real_data": {"enabled": False},
            "logging": {"verbose_mode": False},
            "times": {
                "setup": {"1": 10, "2": 10, "3": 10},
                "cleanup": {"1": 5, "2": 5, "3": 5},
                "max_wait": {"1": 100, "2": 100},
            },
            "jobs": {"types": {"1": 1, "2": 2, "3": 3, "4": 1, "5": 2}},
            "resources": {"num_pabellones": 2},
            "personnel": {"num_anesthesiologists": 1, "num_surgeons": 1},
            "algorithms": {
                "alpha": 1e-6,
                "beta": 0.7,
                "gamma": 1.4,
                "delta": 100.0,
                "ga": {"enabled": True, "population_size": 2, "max_generations": 2, "crossover_probability": 0.8, "mutation_probability": 0.3, "elitism_count": 1},
                "dpso": {"enabled": True, "swarm_size": 2, "max_iterations": 2, "w": 0.7, "c1": 1.5, "c2": 1.5, "vel_high": 4.0, "vel_low": -4.0},
                "sboa": {"enabled": True, "population_size": 3, "max_iterations": 2, "lower_bound": -5.0, "upper_bound": 5.0},
                "dmshoa": {"enabled": True, "population_size": 2, "max_iterations": 2, "k": 0.3, "lower_bound": -5.0, "upper_bound": 5.0},
            },
        }
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod in list(sys.modules.keys()):
            if mod.startswith("config.") or mod.startswith("algorithms.") or mod.startswith("simulation."):
                del sys.modules[mod]

        from algorithms import dmshoa

        job_ids = self._get_job_ids(5)
        n = len(job_ids)
        pos = np.random.default_rng(0).uniform(-5, 5, n + 2 * n)
        sol = dmshoa._decode_position(pos, job_ids)

        assert "job_sequence_base" in sol
        assert "room_assignment" in sol

    def test_sequence_is_strict_permutation(self, tmp_path):
        """Decoded sequence is a strict permutation of job_ids (no dups/missing)."""
        import os, sys, yaml

        cfg = {
            "experiment": {"num_simulations": 1, "std_factor_times": 0.0, "alpha_test": 0.05, "num_procedures": 5, "output_dirs": {"plots": "p", "csv": "c"}},
            "real_data": {"enabled": False},
            "logging": {"verbose_mode": False},
            "times": {"setup": {"1": 10, "2": 10, "3": 10}, "cleanup": {"1": 5, "2": 5, "3": 5}, "max_wait": {"1": 100, "2": 100}},
            "jobs": {"types": {"1": 1, "2": 2, "3": 3, "4": 1, "5": 2}},
            "resources": {"num_pabellones": 2},
            "personnel": {"num_anesthesiologists": 1, "num_surgeons": 1},
            "algorithms": {
                "alpha": 1e-6, "beta": 0.7, "gamma": 1.4, "delta": 100.0,
                "ga": {"enabled": True, "population_size": 2, "max_generations": 2, "crossover_probability": 0.8, "mutation_probability": 0.3, "elitism_count": 1},
                "dpso": {"enabled": True, "swarm_size": 2, "max_iterations": 2, "w": 0.7, "c1": 1.5, "c2": 1.5, "vel_high": 4.0, "vel_low": -4.0},
                "sboa": {"enabled": True, "population_size": 3, "max_iterations": 2, "lower_bound": -5.0, "upper_bound": 5.0},
                "dmshoa": {"enabled": True, "population_size": 2, "max_iterations": 2, "k": 0.3, "lower_bound": -5.0, "upper_bound": 5.0},
            },
        }
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod in list(sys.modules.keys()):
            if mod.startswith("config.") or mod.startswith("algorithms.") or mod.startswith("simulation."):
                del sys.modules[mod]

        from algorithms import dmshoa

        job_ids = self._get_job_ids(5)
        n = len(job_ids)
        rng = np.random.default_rng(7)
        for _ in range(20):
            pos = rng.uniform(-5, 5, n + 2 * n)
            sol = dmshoa._decode_position(pos, job_ids)
            assert sorted(sol["job_sequence_base"]) == sorted(job_ids), (
                f"Sequence is not a permutation of job_ids: {sol['job_sequence_base']}"
            )

    def test_room_assignments_use_valid_rooms(self, tmp_path):
        """All room assignments reference rooms in PABELLONES."""
        import os, sys, yaml

        cfg = {
            "experiment": {"num_simulations": 1, "std_factor_times": 0.0, "alpha_test": 0.05, "num_procedures": 5, "output_dirs": {"plots": "p", "csv": "c"}},
            "real_data": {"enabled": False},
            "logging": {"verbose_mode": False},
            "times": {"setup": {"1": 10, "2": 10, "3": 10}, "cleanup": {"1": 5, "2": 5, "3": 5}, "max_wait": {"1": 100, "2": 100}},
            "jobs": {"types": {"1": 1, "2": 2, "3": 3, "4": 1, "5": 2}},
            "resources": {"num_pabellones": 2},
            "personnel": {"num_anesthesiologists": 1, "num_surgeons": 1},
            "algorithms": {
                "alpha": 1e-6, "beta": 0.7, "gamma": 1.4, "delta": 100.0,
                "ga": {"enabled": True, "population_size": 2, "max_generations": 2, "crossover_probability": 0.8, "mutation_probability": 0.3, "elitism_count": 1},
                "dpso": {"enabled": True, "swarm_size": 2, "max_iterations": 2, "w": 0.7, "c1": 1.5, "c2": 1.5, "vel_high": 4.0, "vel_low": -4.0},
                "sboa": {"enabled": True, "population_size": 3, "max_iterations": 2, "lower_bound": -5.0, "upper_bound": 5.0},
                "dmshoa": {"enabled": True, "population_size": 2, "max_iterations": 2, "k": 0.3, "lower_bound": -5.0, "upper_bound": 5.0},
            },
        }
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod in list(sys.modules.keys()):
            if mod.startswith("config.") or mod.startswith("algorithms.") or mod.startswith("simulation."):
                del sys.modules[mod]

        from algorithms import dmshoa
        from config.config import PABELLONES

        job_ids = self._get_job_ids(5)
        n = len(job_ids)
        pos = np.random.default_rng(3).uniform(-5, 5, n + 2 * n)
        sol = dmshoa._decode_position(pos, job_ids)
        for job_id, ops in sol["room_assignment"].items():
            for op, room in ops.items():
                assert room in PABELLONES, f"job {job_id} op {op} → invalid room {room}"

    def test_decode_is_deterministic(self, tmp_path):
        """Same position vector always decodes to the same solution."""
        import os, sys, yaml

        cfg = {
            "experiment": {"num_simulations": 1, "std_factor_times": 0.0, "alpha_test": 0.05, "num_procedures": 5, "output_dirs": {"plots": "p", "csv": "c"}},
            "real_data": {"enabled": False},
            "logging": {"verbose_mode": False},
            "times": {"setup": {"1": 10, "2": 10, "3": 10}, "cleanup": {"1": 5, "2": 5, "3": 5}, "max_wait": {"1": 100, "2": 100}},
            "jobs": {"types": {"1": 1, "2": 2, "3": 3, "4": 1, "5": 2}},
            "resources": {"num_pabellones": 2},
            "personnel": {"num_anesthesiologists": 1, "num_surgeons": 1},
            "algorithms": {
                "alpha": 1e-6, "beta": 0.7, "gamma": 1.4, "delta": 100.0,
                "ga": {"enabled": True, "population_size": 2, "max_generations": 2, "crossover_probability": 0.8, "mutation_probability": 0.3, "elitism_count": 1},
                "dpso": {"enabled": True, "swarm_size": 2, "max_iterations": 2, "w": 0.7, "c1": 1.5, "c2": 1.5, "vel_high": 4.0, "vel_low": -4.0},
                "sboa": {"enabled": True, "population_size": 3, "max_iterations": 2, "lower_bound": -5.0, "upper_bound": 5.0},
                "dmshoa": {"enabled": True, "population_size": 2, "max_iterations": 2, "k": 0.3, "lower_bound": -5.0, "upper_bound": 5.0},
            },
        }
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod in list(sys.modules.keys()):
            if mod.startswith("config.") or mod.startswith("algorithms.") or mod.startswith("simulation."):
                del sys.modules[mod]

        from algorithms import dmshoa

        job_ids = self._get_job_ids(5)
        n = len(job_ids)
        pos = np.array([0.3, -1.2, 2.1, 0.0, -0.5, 1.1, -2.2, 0.8, 3.1, -1.5, 1.9, 0.4, -0.1, -3.3, 2.5])
        sol1 = dmshoa._decode_position(pos, job_ids)
        sol2 = dmshoa._decode_position(pos, job_ids)
        assert sol1["job_sequence_base"] == sol2["job_sequence_base"]
        assert sol1["room_assignment"] == sol2["room_assignment"]


class TestRepairBounds:
    """_repair_bounds(position, rng) -> np.ndarray with all values in [lb, ub]."""

    def test_in_bounds_stays_unchanged(self, tmp_path):
        """A fully in-bounds vector is returned as-is."""
        import os, sys, yaml

        cfg = {
            "experiment": {"num_simulations": 1, "std_factor_times": 0.0, "alpha_test": 0.05, "num_procedures": 5, "output_dirs": {"plots": "p", "csv": "c"}},
            "real_data": {"enabled": False},
            "logging": {"verbose_mode": False},
            "times": {"setup": {"1": 10, "2": 10, "3": 10}, "cleanup": {"1": 5, "2": 5, "3": 5}, "max_wait": {"1": 100, "2": 100}},
            "jobs": {"types": {"1": 1, "2": 2, "3": 3, "4": 1, "5": 2}},
            "resources": {"num_pabellones": 2},
            "personnel": {"num_anesthesiologists": 1, "num_surgeons": 1},
            "algorithms": {
                "alpha": 1e-6, "beta": 0.7, "gamma": 1.4, "delta": 100.0,
                "ga": {"enabled": True, "population_size": 2, "max_generations": 2, "crossover_probability": 0.8, "mutation_probability": 0.3, "elitism_count": 1},
                "dpso": {"enabled": True, "swarm_size": 2, "max_iterations": 2, "w": 0.7, "c1": 1.5, "c2": 1.5, "vel_high": 4.0, "vel_low": -4.0},
                "sboa": {"enabled": True, "population_size": 3, "max_iterations": 2, "lower_bound": -5.0, "upper_bound": 5.0},
                "dmshoa": {"enabled": True, "population_size": 2, "max_iterations": 2, "k": 0.3, "lower_bound": -5.0, "upper_bound": 5.0},
            },
        }
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod in list(sys.modules.keys()):
            if mod.startswith("config.") or mod.startswith("algorithms.") or mod.startswith("simulation."):
                del sys.modules[mod]

        from algorithms import dmshoa
        from config.config import MSHOA_LOWER_BOUND, MSHOA_UPPER_BOUND

        rng = np.random.default_rng(5)
        pos = rng.uniform(MSHOA_LOWER_BOUND, MSHOA_UPPER_BOUND, 15)
        repaired = dmshoa._repair_bounds(pos.copy(), rng)
        assert np.all(repaired >= MSHOA_LOWER_BOUND)
        assert np.all(repaired <= MSHOA_UPPER_BOUND)

    def test_out_of_bounds_replaced_within_bounds(self, tmp_path):
        """Out-of-bounds elements are replaced (not clipped) within [lb, ub]."""
        import os, sys, yaml

        cfg = {
            "experiment": {"num_simulations": 1, "std_factor_times": 0.0, "alpha_test": 0.05, "num_procedures": 5, "output_dirs": {"plots": "p", "csv": "c"}},
            "real_data": {"enabled": False},
            "logging": {"verbose_mode": False},
            "times": {"setup": {"1": 10, "2": 10, "3": 10}, "cleanup": {"1": 5, "2": 5, "3": 5}, "max_wait": {"1": 100, "2": 100}},
            "jobs": {"types": {"1": 1, "2": 2, "3": 3, "4": 1, "5": 2}},
            "resources": {"num_pabellones": 2},
            "personnel": {"num_anesthesiologists": 1, "num_surgeons": 1},
            "algorithms": {
                "alpha": 1e-6, "beta": 0.7, "gamma": 1.4, "delta": 100.0,
                "ga": {"enabled": True, "population_size": 2, "max_generations": 2, "crossover_probability": 0.8, "mutation_probability": 0.3, "elitism_count": 1},
                "dpso": {"enabled": True, "swarm_size": 2, "max_iterations": 2, "w": 0.7, "c1": 1.5, "c2": 1.5, "vel_high": 4.0, "vel_low": -4.0},
                "sboa": {"enabled": True, "population_size": 3, "max_iterations": 2, "lower_bound": -5.0, "upper_bound": 5.0},
                "dmshoa": {"enabled": True, "population_size": 2, "max_iterations": 2, "k": 0.3, "lower_bound": -5.0, "upper_bound": 5.0},
            },
        }
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod in list(sys.modules.keys()):
            if mod.startswith("config.") or mod.startswith("algorithms.") or mod.startswith("simulation."):
                del sys.modules[mod]

        from algorithms import dmshoa
        from config.config import MSHOA_LOWER_BOUND, MSHOA_UPPER_BOUND

        rng = np.random.default_rng(6)
        pos = np.array([100.0, -100.0, 200.0, -200.0, 0.0])
        repaired = dmshoa._repair_bounds(pos, rng)
        assert np.all(repaired >= MSHOA_LOWER_BOUND), f"Some values below lb: {repaired}"
        assert np.all(repaired <= MSHOA_UPPER_BOUND), f"Some values above ub: {repaired}"
        # The 5th element was in-bounds; check it is untouched (no clipping side-effect)
        assert MSHOA_LOWER_BOUND <= repaired[4] <= MSHOA_UPPER_BOUND


# ===========================================================================
# P3 — MATLAB-faithful PTI helpers (third-pass correction)
# ===========================================================================


class TestPtiDistances:
    """P3-1.1 — _pti_distances(angle_deg) -> (LinH, LinV, PolC) per MATLAB logic.

    fi=10, so a=35, b=55, c=125, d=145.
    For angle <= 90:
        LinH = angle
        LinV = 90 - angle
        PolC = |angle - 45|, but 0 if angle in [35, 55]
    For angle > 90:
        LinH = 180 - angle
        LinV = angle - 90
        PolC = |angle - 135|, but 0 if angle in [125, 145]
    """

    def test_zero_degrees(self):
        """angle=0 → LinH=0, LinV=90, PolC=45."""
        dm = _import_helpers()
        lin_h, lin_v, pol_c = dm._pti_distances(0.0)
        assert math.isclose(lin_h, 0.0, abs_tol=1e-10), f"LinH={lin_h}"
        assert math.isclose(lin_v, 90.0, abs_tol=1e-10), f"LinV={lin_v}"
        assert math.isclose(pol_c, 45.0, abs_tol=1e-10), f"PolC={pol_c}"

    def test_90_degrees(self):
        """angle=90 → LinH=90, LinV=0, PolC=45."""
        dm = _import_helpers()
        lin_h, lin_v, pol_c = dm._pti_distances(90.0)
        assert math.isclose(lin_h, 90.0, abs_tol=1e-10), f"LinH={lin_h}"
        assert math.isclose(lin_v, 0.0, abs_tol=1e-10), f"LinV={lin_v}"
        assert math.isclose(pol_c, 45.0, abs_tol=1e-10), f"PolC={pol_c}"

    def test_45_degrees_in_polc_band(self):
        """angle=45 ∈ [35,55] → PolC=0, LinH=45, LinV=45."""
        dm = _import_helpers()
        lin_h, lin_v, pol_c = dm._pti_distances(45.0)
        assert math.isclose(lin_h, 45.0, abs_tol=1e-10), f"LinH={lin_h}"
        assert math.isclose(lin_v, 45.0, abs_tol=1e-10), f"LinV={lin_v}"
        assert math.isclose(pol_c, 0.0, abs_tol=1e-10), f"PolC should be 0 in [35,55]"

    def test_135_degrees_in_polc_band(self):
        """angle=135 ∈ [125,145] → PolC=0, LinH=45, LinV=45."""
        dm = _import_helpers()
        lin_h, lin_v, pol_c = dm._pti_distances(135.0)
        assert math.isclose(lin_h, 45.0, abs_tol=1e-10), f"LinH={lin_h}"
        assert math.isclose(lin_v, 45.0, abs_tol=1e-10), f"LinV={lin_v}"
        assert math.isclose(pol_c, 0.0, abs_tol=1e-10), f"PolC should be 0 in [125,145]"

    def test_30_degrees_outside_polc_band(self):
        """angle=30 ∉ [35,55] → PolC=15."""
        dm = _import_helpers()
        lin_h, lin_v, pol_c = dm._pti_distances(30.0)
        assert math.isclose(lin_h, 30.0, abs_tol=1e-10), f"LinH={lin_h}"
        assert math.isclose(lin_v, 60.0, abs_tol=1e-10), f"LinV={lin_v}"
        assert math.isclose(pol_c, 15.0, abs_tol=1e-10), f"PolC={pol_c}"

    def test_180_degrees(self):
        """angle=180 → LinH=0, LinV=90, PolC=45."""
        dm = _import_helpers()
        lin_h, lin_v, pol_c = dm._pti_distances(180.0)
        assert math.isclose(lin_h, 0.0, abs_tol=1e-10), f"LinH={lin_h}"
        assert math.isclose(lin_v, 90.0, abs_tol=1e-10), f"LinV={lin_v}"
        assert math.isclose(pol_c, 45.0, abs_tol=1e-10), f"PolC={pol_c}"

    def test_all_distances_non_negative(self):
        """All distances are non-negative for all angles in [0, 180]."""
        dm = _import_helpers()
        for angle in np.linspace(0, 180, 200):
            lin_h, lin_v, pol_c = dm._pti_distances(angle)
            assert lin_h >= -1e-10, f"LinH={lin_h} negative at {angle}"
            assert lin_v >= -1e-10, f"LinV={lin_v} negative at {angle}"
            assert pol_c >= -1e-10, f"PolC={pol_c} negative at {angle}"


class TestPtiFromDistances:
    """P3-1.2 — _pti_from_distances(lin_h, lin_v, pol_c) -> int in {1,2,3}.

    Returns argmin+1 (1-indexed): 1=LinH wins, 2=LinV wins, 3=PolC wins.
    """

    def test_linh_wins(self):
        """LinH is smallest → PTI=1."""
        dm = _import_helpers()
        assert dm._pti_from_distances(0.0, 45.0, 45.0) == 1

    def test_linv_wins(self):
        """LinV is smallest → PTI=2."""
        dm = _import_helpers()
        assert dm._pti_from_distances(45.0, 0.0, 45.0) == 2

    def test_polc_wins(self):
        """PolC is smallest → PTI=3."""
        dm = _import_helpers()
        assert dm._pti_from_distances(45.0, 45.0, 0.0) == 3

    def test_angle_0_produces_pti_1(self):
        """angle=0 → (LinH=0, LinV=90, PolC=45) → PTI=1 (LinH wins)."""
        dm = _import_helpers()
        lin_h, lin_v, pol_c = dm._pti_distances(0.0)
        pti = dm._pti_from_distances(lin_h, lin_v, pol_c)
        assert pti == 1, f"Expected PTI=1 at angle=0, got {pti}"

    def test_angle_90_produces_pti_2(self):
        """angle=90 → (LinH=90, LinV=0, PolC=45) → PTI=2 (LinV wins)."""
        dm = _import_helpers()
        lin_h, lin_v, pol_c = dm._pti_distances(90.0)
        pti = dm._pti_from_distances(lin_h, lin_v, pol_c)
        assert pti == 2, f"Expected PTI=2 at angle=90, got {pti}"

    def test_polc_band_angle_45_wins(self):
        """angle=45 → PolC=0 → PTI=3 (PolC wins)."""
        dm = _import_helpers()
        lin_h, lin_v, pol_c = dm._pti_distances(45.0)
        pti = dm._pti_from_distances(lin_h, lin_v, pol_c)
        assert pti == 3, f"Expected PTI=3 at angle=45 (PolC=0), got {pti}"

    def test_output_always_in_valid_set(self):
        """All angles in [0, 180] produce PTI in {1,2,3}."""
        dm = _import_helpers()
        for angle in np.linspace(0, 180, 200):
            lin_h, lin_v, pol_c = dm._pti_distances(angle)
            pti = dm._pti_from_distances(lin_h, lin_v, pol_c)
            assert pti in {1, 2, 3}, f"PTI={pti} at angle={angle}"


class TestSampleRpaDeg:
    """P3-1.3 — _sample_rpa_deg(rng) samples right-eye angle as int in [1, 90].

    MATLAB: randi([1, 90]) — integer, not float, range [1, 90] not [0, π].
    """

    def test_sample_rpa_deg_in_range_1_90(self):
        """All sampled values must be integers in [1, 90]."""
        dm = _import_helpers()
        rng = np.random.default_rng(40)
        for _ in range(300):
            val = dm._sample_rpa_deg(rng)
            assert 1 <= val <= 90, f"_sample_rpa_deg={val} outside [1,90]"

    def test_sample_rpa_deg_is_integer(self):
        """_sample_rpa_deg returns an integer type."""
        dm = _import_helpers()
        rng = np.random.default_rng(41)
        for _ in range(50):
            val = dm._sample_rpa_deg(rng)
            assert isinstance(val, (int, np.integer)), f"Expected int, got {type(val)}"

    def test_sample_rpa_deg_fixed_seed(self):
        """Fixed seed _sample_rpa_deg matches rng.integers(1, 91)."""
        dm = _import_helpers()
        val = dm._sample_rpa_deg(np.random.default_rng(77))
        expected = int(np.random.default_rng(77).integers(1, 91))
        assert val == expected, f"_sample_rpa_deg mismatch: got {val}, expected {expected}"


class TestUpdatePtiVectorMatlab:
    """P3-1.4 — _update_pti_vector uses MATLAB dual-eye PTI selection.

    Each agent: left-eye angle from (X_old[i], X_new[i]) in degrees,
    right-eye angle from _sample_rpa_deg(rng), compute distances for both eyes,
    choose the eye with the lower min distance, return its PTI label.
    """

    def test_output_length_matches_population(self):
        """Output length equals number of agents."""
        dm = _import_helpers()
        rng = np.random.default_rng(0)
        X_old = rng.uniform(-1, 1, (6, 8))
        X_new = rng.uniform(-1, 1, (6, 8))
        pti = dm._update_pti_vector(X_old, X_new, rng)
        assert len(pti) == 6

    def test_output_values_in_valid_set(self):
        """Every PTI value must be in {1, 2, 3}."""
        dm = _import_helpers()
        rng = np.random.default_rng(1)
        X_old = rng.uniform(-3, 3, (10, 12))
        X_new = rng.uniform(-3, 3, (10, 12))
        pti = dm._update_pti_vector(X_old, X_new, rng)
        assert set(pti).issubset({1, 2, 3}), f"PTI values outside {{1,2,3}}: {set(pti)}"

    def test_identical_positions_valid_pti(self):
        """If X_old == X_new, left-eye angle ≈ 0°. All agents get valid PTI in {1,2,3}."""
        dm = _import_helpers()
        rng = np.random.default_rng(100)
        X = rng.uniform(1, 2, (4, 6))  # all positive to avoid zero-norm edge case
        pti = dm._update_pti_vector(X, X.copy(), rng)
        assert all(p in {1, 2, 3} for p in pti), f"PTI outside valid set: {pti}"

    def test_matlab_single_agent_manual(self):
        """Manual verification: known left-eye=0°, right-eye fixed seed matches MATLAB behavior."""
        dm = _import_helpers()
        # Left-eye: identical vectors → angle=0 → (LinH=0, LinV=90, PolC=45) → dif_1=0, Idx_1=1
        X_old = np.array([[1.0, 0.0, 0.0]])
        X_new = np.array([[1.0, 0.0, 0.0]])
        # Replay right-eye: rng.integers(1, 91) with seed 200
        rng_ref = np.random.default_rng(200)
        rpa_deg = int(rng_ref.integers(1, 91))
        lin_h2, lin_v2, pol_c2 = dm._pti_distances(float(rpa_deg))
        dif_2 = min(lin_h2, lin_v2, pol_c2)
        # eyes = min(dif_1=0, dif_2); dif_1=0 so eyes=0=dif_1 → use left eye PTI=1
        rng_actual = np.random.default_rng(200)
        pti = dm._update_pti_vector(X_old, X_new, rng_actual)
        assert pti[0] == 1, f"Expected PTI=1 (left-eye wins, angle=0), got {pti[0]}"


# ===========================================================================
# Final pass — circshift PTI fidelity (MATLAB getPolarization_MSHOA.m)
# ===========================================================================


class TestUpdatePtiCircshift:
    """P4-1.1 — _update_pti_vector applies MATLAB-faithful circshift row pairing.

    MATLAB pseudocode (getPolarization_MSHOA.m):
        k1 = randi(N)                       → integer in [1, N]
        k2 = k1 + randi(N-1)               → integer in [k1+1, k1+N-1]
        Positions = circshift(Positions, k1, 1)   → np.roll(X_old, k1, axis=0)
        x         = circshift(x, k2, 1)           → np.roll(X_new, k2, axis=0)
        for i = 1..N: compare shifted_old[i] vs shifted_new[i]

    Agent i's left-eye angle comes from row i of the SHIFTED matrices,
    NOT from row i of the original un-shifted matrices.
    """

    def test_circshift_pairing_changes_pti_vs_unshifted(self):
        """When rows are distinct, shifting must change at least one PTI value.

        We construct X_old and X_new so that every row is unique, then verify
        that running with a seed that produces k1 >= 1 yields different PTI from
        a run forced to use k1=0 (no shift). If both return identical PTI,
        the implementation is NOT applying shifts.
        """
        dm = _import_helpers()
        import math

        # 4 agents, 3 dims — all rows are orthogonal so angles differ maximally
        rng_gen = np.random.default_rng(7)
        X_old = np.eye(4, 4)           # rows: e0, e1, e2, e3
        X_new = np.roll(X_old, 1, axis=0)  # rows shifted: e3, e0, e1, e2

        # Baseline "no shift" via pop_size=1: guard disables circshift entirely,
        # so angle is computed directly between X_old[0] and X_new[0].
        # Use identical single-agent: angle=0 → dif_1=0 → PTI=1.
        X_single_old = np.array([[1.0, 0.0, 0.0]])
        X_single_new = np.array([[1.0, 0.0, 0.0]])
        rng_no_shift = np.random.default_rng(99)
        pti_no_shift = dm._update_pti_vector(X_single_old, X_single_new, rng_no_shift)
        # Single agent, identical → angle=0 → dif_1=0 → left eye wins → PTI=1
        assert pti_no_shift[0] == 1, (
            f"Baseline (single-agent, angle=0) should yield PTI=1, got {pti_no_shift[0]}"
        )

        # Now compute PTI WITH the circshift semantics against a truly shifted matrix
        # X_old rows are orthogonal, X_new = roll(X_old). With k1=1 shift:
        #   shifted_old[0] = X_old[1] = e1
        #   shifted_new[0] = X_new[(k2) % 4] (k2 = k1 + r, r in [1,3])
        # The resulting angles are non-zero → PTI can differ from 1
        rng_shift = np.random.default_rng(7)
        pti_shift = dm._update_pti_vector(X_old, X_new, rng_shift)
        # Output must still be valid PTI values
        assert set(pti_shift).issubset({1, 2, 3}), f"PTI outside valid set: {set(pti_shift)}"

        # The key property: with circshift, shifted vectors are no longer identical,
        # so at least one agent must have a non-zero left-eye angle and possibly get
        # a different PTI from what a zero-angle baseline would produce.
        # We verify the implementation uses the circshifted rows by checking that
        # at least one agent's angle computation differs from the trivial angle=0 path.
        # Concretely: compute what the circshifted angles SHOULD be for a known k1, k2.
        # We cannot control RNG internals, but we can verify the CONTRACT:
        # If circshift is applied, then for X_old rows being orthonormal basis vectors,
        # the angle between shifted_old[i] and shifted_new[i] will be 90° for the case
        # where the two eyes point to different basis vectors, so dif_1 = min(0,90,45)=0
        # vs the unshifted (angle=0 → dif_1=0 anyway). Instead, test a case where the
        # absence of shift gives angle=0 (dif_1=0) but with shift gives angle=90°
        # (dif_1=0 still, but PTI could go to 2). So we need a direct assertion.

        # Direct test: manually compute expected PTI using known k1/k2 from a fixed seed
        # rng for integers: k1=rng.integers(1, N+1), r=rng.integers(1, N), k2=k1+r
        rng_probe = np.random.default_rng(42)
        N = 4
        k1 = int(rng_probe.integers(1, N + 1))
        r  = int(rng_probe.integers(1, N))
        k2 = k1 + r

        X_old_c = np.eye(4, 4)
        X_new_c = np.eye(4, 4) * 2.0   # all parallel to X_old → angle=0 without shift

        shifted_old = np.roll(X_old_c, k1, axis=0)
        shifted_new = np.roll(X_new_c, k2, axis=0)

        # Compute expected PTI manually for all agents
        from algorithms.dmshoa import _compute_lpa, _pti_distances, _pti_from_distances, _sample_rpa_deg
        expected_pti = np.empty(N, dtype=int)
        rng_inner = np.random.default_rng(42)
        rng_inner.integers(1, N + 1)   # consume k1
        rng_inner.integers(1, N)       # consume r
        for i in range(N):
            left_deg = math.degrees(_compute_lpa(shifted_old[i], shifted_new[i]))
            rpa_deg = float(_sample_rpa_deg(rng_inner))
            lh1, lv1, pc1 = _pti_distances(left_deg)
            lh2, lv2, pc2 = _pti_distances(rpa_deg)
            d1 = min(lh1, lv1, pc1)
            d2 = min(lh2, lv2, pc2)
            if d1 <= d2:
                expected_pti[i] = _pti_from_distances(lh1, lv1, pc1)
            else:
                expected_pti[i] = _pti_from_distances(lh2, lv2, pc2)

        # Now run the actual implementation with the same seed
        rng_actual = np.random.default_rng(42)
        pti_actual = dm._update_pti_vector(X_old_c, X_new_c, rng_actual)

        assert list(pti_actual) == list(expected_pti), (
            f"circshift PTI mismatch: expected {list(expected_pti)}, got {list(pti_actual)}\n"
            f"k1={k1}, r={r}, k2={k2}"
        )

    def test_circshift_k2_wraps_modulo_population(self):
        """k2 can exceed pop_size; np.roll handles modulo automatically.

        With N=3, k1=3, r=2 → k2=5 → roll 5 rows = roll 2 rows (mod 3).
        The implementation must NOT clip k2 to [0, N-1]; np.roll already handles wrap.
        """
        dm = _import_helpers()
        import math
        N = 3
        dim = 5
        # Construct deterministic matrices with all-unique rows
        X_old = np.array([[float(i + 1)] * dim for i in range(N)])   # [[1,1..], [2,2..], [3,3..]]
        X_new = np.array([[float(i + 10)] * dim for i in range(N)])  # [[10,..], [11,..], [12,..]]

        # For k1=2, r=2 → k2=4 (but N=3, so k2 mod 3 = 1 → same as k2=1)
        # Roll X_old by 2 → rows: [2,..] [3,..] [1,..]
        # Roll X_new by 4→ rows: [11,..] [12,..] [10,..]  (mod 3 = 1 → [11,..][12,..][10,..])
        k1, k2 = 2, 4
        shifted_old = np.roll(X_old, k1, axis=0)
        shifted_new = np.roll(X_new, k2, axis=0)

        from algorithms.dmshoa import _compute_lpa, _pti_distances, _pti_from_distances, _sample_rpa_deg
        rng_inner = np.random.default_rng(77)
        # consume k1 and r from rng (we will feed a seed that gives k1=2, r=2)
        # Instead: find a seed where the RNG first produces k1=2, then r=2
        # That's brittle; test the modulo behavior directly by asserting roll wraps correctly
        roll_result = np.roll(X_new, k2, axis=0)
        expected_row0 = X_new[(-k2) % N]  # np.roll(a, k)[0] = a[(-k) % len(a)]
        assert np.allclose(roll_result[0], expected_row0), (
            f"np.roll modulo wrap failed: roll_result[0]={roll_result[0]}, expected={expected_row0}"
        )
        # And confirm the function produces valid output for large k2 inputs
        rng_out = np.random.default_rng(77)
        pti_out = dm._update_pti_vector(X_old, X_new, rng_out)
        assert set(pti_out).issubset({1, 2, 3}), f"PTI outside valid set with large k: {set(pti_out)}"

    def test_unshifted_implementation_fails_manual_verification(self):
        """RED test: current implementation WITHOUT circshift fails this assertion.

        After implementing circshift, this test must PASS.
        We verify by replaying the exact RNG sequence to compute expected PTI
        and checking that the implementation matches it.
        """
        dm = _import_helpers()
        import math
        from algorithms.dmshoa import _compute_lpa, _pti_distances, _pti_from_distances, _sample_rpa_deg

        N = 5
        dim = 4
        # Construct X_old and X_new with clearly different rows
        rng_data = np.random.default_rng(13)
        X_old = rng_data.uniform(1, 5, (N, dim))
        X_new = rng_data.uniform(1, 5, (N, dim))

        # Compute expected PTI with MATLAB circshift semantics
        rng_ref = np.random.default_rng(55)
        k1 = int(rng_ref.integers(1, N + 1))
        r  = int(rng_ref.integers(1, N))
        k2 = k1 + r

        shifted_old = np.roll(X_old, k1, axis=0)
        shifted_new = np.roll(X_new, k2, axis=0)

        expected_pti = np.empty(N, dtype=int)
        for i in range(N):
            left_deg = math.degrees(_compute_lpa(shifted_old[i], shifted_new[i]))
            rpa_deg  = float(_sample_rpa_deg(rng_ref))
            lh1, lv1, pc1 = _pti_distances(left_deg)
            lh2, lv2, pc2 = _pti_distances(rpa_deg)
            d1 = min(lh1, lv1, pc1)
            d2 = min(lh2, lv2, pc2)
            if d1 <= d2:
                expected_pti[i] = _pti_from_distances(lh1, lv1, pc1)
            else:
                expected_pti[i] = _pti_from_distances(lh2, lv2, pc2)

        # Run implementation with the same seed
        rng_actual = np.random.default_rng(55)
        pti_actual = dm._update_pti_vector(X_old, X_new, rng_actual)

        assert list(pti_actual) == list(expected_pti), (
            f"circshift PTI mismatch (seed=55): expected {list(expected_pti)}, got {list(pti_actual)}\n"
            f"k1={k1}, r={r}, k2={k2}"
        )


# ===========================================================================
# Task 3.1 — Anti-regression: no sigmoid, no random PTI, no duplicate jobs
# ===========================================================================


class TestAntiRegression:
    """Verify the rewrite removes forbidden patterns."""

    def test_no_sigmoid_function_in_dmshoa(self):
        """dmshoa module must NOT define a _sigmoid function."""
        import inspect
        from algorithms import dmshoa
        assert not hasattr(dmshoa, "_sigmoid"), (
            "_sigmoid must be removed; the rewrite should not use sigmoid discretization"
        )

    def test_no_create_random_solution_function(self):
        """dmshoa module must NOT define _create_random_solution (balanced heuristic)."""
        from algorithms import dmshoa
        assert not hasattr(dmshoa, "_create_random_solution"), (
            "_create_random_solution must be removed; decode via SPV instead"
        )

    def test_decode_produces_no_duplicate_jobs(self, tmp_path):
        """Decoded schedules must never contain duplicate jobs."""
        import os, sys, yaml

        cfg = {
            "experiment": {"num_simulations": 1, "std_factor_times": 0.0, "alpha_test": 0.05, "num_procedures": 5, "output_dirs": {"plots": "p", "csv": "c"}},
            "real_data": {"enabled": False},
            "logging": {"verbose_mode": False},
            "times": {"setup": {"1": 10, "2": 10, "3": 10}, "cleanup": {"1": 5, "2": 5, "3": 5}, "max_wait": {"1": 100, "2": 100}},
            "jobs": {"types": {"1": 1, "2": 2, "3": 3, "4": 1, "5": 2}},
            "resources": {"num_pabellones": 2},
            "personnel": {"num_anesthesiologists": 1, "num_surgeons": 1},
            "algorithms": {
                "alpha": 1e-6, "beta": 0.7, "gamma": 1.4, "delta": 100.0,
                "ga": {"enabled": True, "population_size": 2, "max_generations": 2, "crossover_probability": 0.8, "mutation_probability": 0.3, "elitism_count": 1},
                "dpso": {"enabled": True, "swarm_size": 2, "max_iterations": 2, "w": 0.7, "c1": 1.5, "c2": 1.5, "vel_high": 4.0, "vel_low": -4.0},
                "sboa": {"enabled": True, "population_size": 3, "max_iterations": 2, "lower_bound": -5.0, "upper_bound": 5.0},
                "dmshoa": {"enabled": True, "population_size": 2, "max_iterations": 2, "k": 0.3, "lower_bound": -5.0, "upper_bound": 5.0},
            },
        }
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg))
        os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
        for mod in list(sys.modules.keys()):
            if mod.startswith("config.") or mod.startswith("algorithms.") or mod.startswith("simulation."):
                del sys.modules[mod]

        from algorithms import dmshoa

        job_ids = list(range(1, 6))
        n = len(job_ids)
        rng = np.random.default_rng(42)
        for _ in range(30):
            pos = rng.uniform(-5, 5, n + 2 * n)
            sol = dmshoa._decode_position(pos, job_ids)
            seq = sol["job_sequence_base"]
            assert len(seq) == len(set(seq)), f"Duplicate jobs in sequence: {seq}"
            assert len(seq) == len(job_ids), f"Wrong length: {seq}"


# ===========================================================================
# Task 3.3 — Integration: run() end-to-end
# ===========================================================================


def _make_minimal_config_dict(num_procedures=5, max_iter=5, pop_size=3):
    return {
        "experiment": {"num_simulations": 1, "std_factor_times": 0.0, "alpha_test": 0.05, "num_procedures": num_procedures, "output_dirs": {"plots": "p", "csv": "c"}},
        "real_data": {"enabled": False},
        "logging": {"verbose_mode": False},
        "times": {"setup": {"1": 10, "2": 10, "3": 10}, "cleanup": {"1": 5, "2": 5, "3": 5}, "max_wait": {"1": 500, "2": 500}},
        "jobs": {"types": {"1": 1, "2": 2, "3": 3, "4": 1, "5": 2}},
        "resources": {"num_pabellones": 2},
        "personnel": {"num_anesthesiologists": 1, "num_surgeons": 1},
        "algorithms": {
            "alpha": 1e-6, "beta": 0.7, "gamma": 1.4, "delta": 100.0,
            "ga": {"enabled": True, "population_size": 2, "max_generations": 2, "crossover_probability": 0.8, "mutation_probability": 0.3, "elitism_count": 1},
            "dpso": {"enabled": True, "swarm_size": 2, "max_iterations": 2, "w": 0.7, "c1": 1.5, "c2": 1.5, "vel_high": 4.0, "vel_low": -4.0},
            "sboa": {"enabled": True, "population_size": 3, "max_iterations": 2, "lower_bound": -5.0, "upper_bound": 5.0},
            "dmshoa": {"enabled": True, "population_size": pop_size, "max_iterations": max_iter, "k": 0.3, "lower_bound": -5.0, "upper_bound": 5.0},
        },
    }


def _setup_env(tmp_path, cfg_dict):
    import os, sys, yaml
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg_dict))
    os.environ["HOSPITAL_CONFIG_PATH"] = str(cfg_file)
    _PREFIXES = ("config", "algorithms", "simulation")
    for mod in list(sys.modules.keys()):
        if any(mod == p or mod.startswith(p + ".") for p in _PREFIXES):
            del sys.modules[mod]


class TestInitializePti:
    """Task 1.3 — _initialize_pti(pop_size, rng) implements Eq.(2): round(1 + 2*rand).

    Values must be in {1, 2, 3} and must NOT derive from gbest, X, or candidate vectors.
    """

    def test_pti_values_in_1_2_3(self):
        """_initialize_pti values must all be in {1, 2, 3}."""
        dm = _import_helpers()
        rng = np.random.default_rng(10)
        pti = dm._initialize_pti(10, rng)
        assert set(pti).issubset({1, 2, 3}), f"PTI values outside {{1,2,3}}: {set(pti)}"

    def test_pti_length_matches_pop_size(self):
        """_initialize_pti length equals pop_size."""
        dm = _import_helpers()
        rng = np.random.default_rng(11)
        pti = dm._initialize_pti(7, rng)
        assert len(pti) == 7, f"Expected length 7, got {len(pti)}"

    def test_pti_is_deterministic_with_seed(self):
        """Same seed → same initial PTI vector."""
        dm = _import_helpers()
        pti1 = dm._initialize_pti(5, np.random.default_rng(42))
        pti2 = dm._initialize_pti(5, np.random.default_rng(42))
        assert list(pti1) == list(pti2), "PTI not deterministic with same seed"

    def test_pti_fixed_seed_matches_eq2(self):
        """Fixed seed PTI values must match round(1 + 2*rand) for each agent."""
        dm = _import_helpers()
        pop_size = 5
        pti = dm._initialize_pti(pop_size, np.random.default_rng(77))
        # Replay manually
        rng_ref = np.random.default_rng(77)
        expected = [int(round(1 + 2 * rng_ref.random())) for _ in range(pop_size)]
        assert list(pti) == expected, f"PTI mismatch: got {list(pti)}, expected {expected}"

    def test_pti_not_derived_from_position_vectors(self):
        """_initialize_pti takes only (pop_size, rng) — no position arrays involved."""
        import inspect
        dm = _import_helpers()
        sig = inspect.signature(dm._initialize_pti)
        params = list(sig.parameters.keys())
        assert params == ["pop_size", "rng"], (
            f"_initialize_pti signature must be (pop_size, rng), got {params}"
        )


class TestComputeRpaDeprecated:
    """P3-3.2 — _compute_rpa is removed; _sample_attack_theta survives unchanged.

    _compute_rpa (radian, float [0,π]) is replaced by _sample_rpa_deg (degree, int [1,90]).
    _sample_attack_theta ([π, 2π]) is unrelated to PTI update and remains.
    """

    def test_compute_rpa_removed(self):
        """_compute_rpa must NOT exist (replaced by _sample_rpa_deg)."""
        from algorithms import dmshoa
        assert not hasattr(dmshoa, "_compute_rpa"), (
            "_compute_rpa must be removed; MATLAB right-eye uses randi([1,90]) degrees"
        )

    def test_sample_attack_theta_in_range_pi_2pi(self):
        """_sample_attack_theta result must be in [π, 2π] (not [0, π])."""
        dm = _import_helpers()
        rng = np.random.default_rng(21)
        for _ in range(200):
            theta = dm._sample_attack_theta(rng)
            assert math.pi - 1e-12 <= theta <= 2 * math.pi + 1e-12, (
                f"attack theta={theta} outside [π, 2π]"
            )

    def test_sample_rpa_deg_and_attack_theta_are_different_functions(self):
        """_sample_rpa_deg and _sample_attack_theta must be separate callable attributes."""
        dm = _import_helpers()
        assert hasattr(dm, "_sample_rpa_deg"), "_sample_rpa_deg must exist"
        assert hasattr(dm, "_sample_attack_theta"), "_sample_attack_theta must exist"
        assert dm._sample_rpa_deg is not dm._sample_attack_theta, (
            "_sample_rpa_deg and _sample_attack_theta must be different functions"
        )

    def test_attack_strategy_does_not_call_compute_rpa(self):
        """PTI=2 attack strategy uses attack theta [π,2π], not RPA [0,π]."""
        dm = _import_helpers()
        rng = np.random.default_rng(30)
        dim = 4
        gbest = np.ones(dim) * 3.0
        position = np.zeros(dim)
        population = np.ones((3, dim))
        # If attack uses θ ∈ [π,2π], cos(θ) ∈ [-1, 0] for most; certainly cos will be based on [π,2π]
        # Predict with [π,2π] sampler
        rng_pred = np.random.default_rng(77)
        theta = rng_pred.uniform(math.pi, 2.0 * math.pi)
        expected = gbest * math.cos(theta)
        rng_actual = np.random.default_rng(77)
        candidate = dm._apply_strategy(position, gbest, population, pti_i=2, rng=rng_actual, agent_idx=0)
        assert np.allclose(candidate, expected, atol=1e-12), (
            "PTI=2 must use _sample_attack_theta([π,2π]), not _compute_rpa([0,π])"
        )


class TestRunIntegration:
    """Integration tests for dmshoa.run()."""

    def test_run_returns_4_tuple(self, tmp_path):
        """run() returns a 4-tuple: (best_fitness, best_solution, best_history, avg_history)."""
        _setup_env(tmp_path, _make_minimal_config_dict())
        from algorithms import dmshoa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
        result = dmshoa.run(surgeries_data, job_ids, seed=42)

        assert isinstance(result, tuple), "run() must return a tuple"
        assert len(result) == 4, f"Expected 4 elements, got {len(result)}"

    def test_run_best_fitness_is_finite(self, tmp_path):
        """run() best_fitness is a finite float for valid input."""
        _setup_env(tmp_path, _make_minimal_config_dict())
        from algorithms import dmshoa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
        best_fit, best_sol, best_hist, avg_hist = dmshoa.run(surgeries_data, job_ids, seed=0)

        assert math.isfinite(best_fit), f"best_fitness should be finite, got {best_fit}"

    def test_run_best_solution_is_valid_permutation(self, tmp_path):
        """Best solution from run() is a valid job permutation."""
        _setup_env(tmp_path, _make_minimal_config_dict())
        from algorithms import dmshoa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
        _, best_sol, _, _ = dmshoa.run(surgeries_data, job_ids, seed=1)

        assert sorted(best_sol["job_sequence_base"]) == sorted(job_ids), (
            f"Best solution is not a valid permutation: {best_sol['job_sequence_base']}"
        )

    def test_run_history_length_matches_iterations(self, tmp_path):
        """best_fitness_history has MAX_ITERATIONS_MSHOA entries."""
        _setup_env(tmp_path, _make_minimal_config_dict(max_iter=5))
        from algorithms import dmshoa
        from config.config import MAX_ITERATIONS_MSHOA
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
        _, _, best_hist, avg_hist = dmshoa.run(surgeries_data, job_ids, seed=2)

        assert len(best_hist) == MAX_ITERATIONS_MSHOA, (
            f"best_fitness_history length {len(best_hist)} != {MAX_ITERATIONS_MSHOA}"
        )

    def test_run_history_is_monotonically_non_increasing(self, tmp_path):
        """best_fitness_history never increases (gbest only improves or stays same)."""
        _setup_env(tmp_path, _make_minimal_config_dict(max_iter=10, pop_size=4))
        from algorithms import dmshoa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
        _, _, best_hist, _ = dmshoa.run(surgeries_data, job_ids, seed=3)

        for i in range(1, len(best_hist)):
            assert best_hist[i] <= best_hist[i - 1] + 1e-9, (
                f"best_fitness_history increased at index {i}: "
                f"{best_hist[i-1]} → {best_hist[i]}"
            )

    def test_run_seeded_is_deterministic(self, tmp_path):
        """Same seed → identical best_fitness (deterministic)."""
        _setup_env(tmp_path, _make_minimal_config_dict())
        from algorithms import dmshoa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
        r1 = dmshoa.run(surgeries_data, job_ids, seed=99)

        # Reset modules and run again
        _setup_env(tmp_path, _make_minimal_config_dict())
        from algorithms import dmshoa as dmshoa2
        r2 = dmshoa2.run(surgeries_data, job_ids, seed=99)

        assert math.isclose(r1[0], r2[0], rel_tol=1e-10), (
            f"Non-deterministic: {r1[0]} vs {r2[0]}"
        )


# ===========================================================================
# Task 4.1 — _sample_rpa() range and _apply_strategy() branches (fixed seeds)
# ===========================================================================


class TestSampleAttackTheta:
    """_sample_attack_theta(rng) → float in [π, 2π] (replaces the old _sample_rpa tests)."""

    def test_sample_attack_theta_lower_bound(self):
        """Sampled attack θ must be >= π (paper Eq.14: θ ∈ [π, 2π])."""
        dm = _import_helpers()
        rng = np.random.default_rng(0)
        for _ in range(200):
            theta = dm._sample_attack_theta(rng)
            assert theta >= math.pi - 1e-12, f"attack θ below π: {theta}"

    def test_sample_attack_theta_upper_bound(self):
        """Sampled attack θ must be <= 2π (paper Eq.14: θ ∈ [π, 2π])."""
        dm = _import_helpers()
        rng = np.random.default_rng(1)
        for _ in range(200):
            theta = dm._sample_attack_theta(rng)
            assert theta <= 2.0 * math.pi + 1e-12, f"attack θ above 2π: {theta}"

    def test_sample_attack_theta_fixed_seed_value(self):
        """Fixed seed produces a known value inside [π, 2π]."""
        dm = _import_helpers()
        rng = np.random.default_rng(42)
        theta = dm._sample_attack_theta(rng)
        expected = np.random.default_rng(42).uniform(math.pi, 2.0 * math.pi)
        assert math.isclose(theta, expected, rel_tol=1e-12), (
            f"attack θ fixed-seed mismatch: got {theta}, expected {expected}"
        )


class TestApplyStrategy:
    """_apply_strategy() follows paper Eq.(12)/(14)/(15) for each PTI branch."""

    def _make_args(self, pop_size=4, dim=6, seed=7):
        """Return a deterministic (position, gbest, population, rng) tuple."""
        dm = _import_helpers()
        rng = np.random.default_rng(seed)
        population = rng.uniform(-5, 5, (pop_size, dim))
        position = population[0].copy()
        gbest = population[1].copy()
        return dm, position, gbest, population, rng

    # --- PTI 1: Foraging (Eq. 12) ---

    def test_pti1_foraging_result_differs_from_current(self):
        """PTI=1 candidate is NOT identical to the current position (update occurred)."""
        dm, position, gbest, population, rng = self._make_args(seed=10)
        candidate = dm._apply_strategy(position, gbest, population, pti_i=1, rng=rng, agent_idx=0)
        assert not np.allclose(candidate, position), (
            "PTI=1 foraging: candidate must differ from current position"
        )

    def test_pti1_foraging_fixed_seed_value(self):
        """PTI=1 with fixed seed yields exactly the paper Eq.(12) result."""
        dm = _import_helpers()
        rng_ref = np.random.default_rng(20)
        pop_size, dim = 3, 4
        population = np.random.default_rng(20).uniform(-5, 5, (pop_size, dim))
        position = population[0].copy()
        gbest = population[1].copy()

        # Replay the same RNG to compute expected value manually
        rng_expect = np.random.default_rng(20)
        # Advance past population creation (same seed, generator is already advanced)
        # Use a fresh rng for the call to get a predictable state
        rng_call = np.random.default_rng(77)
        D = rng_call.uniform(-1.0, 1.0)
        peers = [j for j in range(pop_size) if j != 0]
        r_idx = int(rng_call.choice(peers))
        x_r = population[r_idx]
        v = position - gbest
        R_t = x_r - position
        expected = gbest - v + D * R_t

        rng_actual = np.random.default_rng(77)
        candidate = dm._apply_strategy(position, gbest, population, pti_i=1, rng=rng_actual, agent_idx=0)
        assert np.allclose(candidate, expected, atol=1e-12), (
            f"PTI=1 Eq.(12) mismatch.\n  got:      {candidate}\n  expected: {expected}"
        )

    # --- PTI 2: Attack (Eq. 14) ---

    def test_pti2_attack_result_is_gbest_cosine(self):
        """PTI=2 candidate equals gbest * cos(θ) for θ ∈ [π, 2π]."""
        dm = _import_helpers()
        rng = np.random.default_rng(30)
        dim = 5
        gbest = rng.uniform(-5, 5, dim)
        position = rng.uniform(-5, 5, dim)
        population = rng.uniform(-5, 5, (3, dim))

        # We need to predict θ: _sample_rpa uses rng.uniform(π, 2π)
        rng_pred = np.random.default_rng(99)
        theta = rng_pred.uniform(math.pi, 2.0 * math.pi)
        expected = gbest * math.cos(theta)

        rng_actual = np.random.default_rng(99)
        candidate = dm._apply_strategy(position, gbest, population, pti_i=2, rng=rng_actual, agent_idx=0)
        assert np.allclose(candidate, expected, atol=1e-12), (
            f"PTI=2 Eq.(14) mismatch.\n  got:      {candidate}\n  expected: {expected}"
        )

    def test_pti2_attack_cosine_uses_rpa_range(self):
        """PTI=2 cos(θ) factor is in [-1, 1] and θ comes from [π, 2π] (cosine is ≤ 0)."""
        dm = _import_helpers()
        rng = np.random.default_rng(31)
        dim = 4
        gbest = np.ones(dim) * 2.0
        position = np.zeros(dim)
        population = np.ones((3, dim))

        candidate = dm._apply_strategy(position, gbest, population, pti_i=2, rng=rng, agent_idx=0)
        # cos(θ) for θ ∈ [π, 2π] spans [-1, 1], so candidate[0] / 2.0 must be in [-1, 1]
        factor = candidate[0] / gbest[0]
        assert -1.0 - 1e-9 <= factor <= 1.0 + 1e-9, (
            f"PTI=2 cos factor {factor} outside [-1, 1]; θ not in [π, 2π]"
        )

    # --- PTI 3: Shelter/Defense (Eq. 15) ---

    def test_pti3_defense_fixed_seed_value(self):
        """PTI=3 with fixed seed yields exactly the paper Eq.(15) result."""
        dm = _import_helpers()
        dim = 4
        gbest = np.array([1.0, -2.0, 3.0, -0.5])
        position = np.zeros(dim)
        population = np.ones((3, dim))

        # Replay RNG manually: k_rand = rng.uniform(0, K), direction = +1/-1
        from config.config import MSHOA_K
        rng_pred = np.random.default_rng(55)
        k_rand = rng_pred.uniform(0.0, MSHOA_K)
        direction = 1.0 if rng_pred.random() > 0.5 else -1.0
        expected = gbest + gbest * k_rand * direction

        rng_actual = np.random.default_rng(55)
        candidate = dm._apply_strategy(position, gbest, population, pti_i=3, rng=rng_actual, agent_idx=0)
        assert np.allclose(candidate, expected, atol=1e-12), (
            f"PTI=3 Eq.(15) mismatch.\n  got:      {candidate}\n  expected: {expected}"
        )

    def test_pti3_defense_result_near_gbest(self):
        """PTI=3 candidate stays proportionally close to gbest (k_rand ∈ [0, K])."""
        dm = _import_helpers()
        from config.config import MSHOA_K
        rng = np.random.default_rng(60)
        dim = 5
        gbest = rng.uniform(-5, 5, dim)
        position = np.zeros(dim)
        population = np.ones((3, dim))

        for seed in range(50):
            rng_i = np.random.default_rng(seed)
            candidate = dm._apply_strategy(position, gbest, population, pti_i=3, rng=rng_i, agent_idx=0)
            diff = candidate - gbest
            # diff = gbest * k_rand * direction, so |diff| = |gbest| * k_rand ≤ |gbest| * K
            for j in range(dim):
                max_diff = abs(gbest[j]) * MSHOA_K
                assert abs(diff[j]) <= max_diff + 1e-9, (
                    f"PTI=3 deviation {abs(diff[j]):.6f} exceeds K*|gbest[{j}]|={max_diff:.6f}"
                )


# ===========================================================================
# Task 4.2 — Targeted run: verify callback contract + analysis-mode storage
# ===========================================================================


class TestCallbackContract:
    """Verify on_iteration callback is called correctly (task 4.2)."""

    def test_on_iteration_called_once_per_iteration(self, tmp_path):
        """on_iteration is invoked exactly MAX_ITERATIONS_MSHOA times."""
        _setup_env(tmp_path, _make_minimal_config_dict(max_iter=4, pop_size=3))
        from algorithms import dmshoa
        from config.config import MAX_ITERATIONS_MSHOA
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
        call_count = []

        def on_iter(algo_step, best_fitness, best_makespan=None, **kwargs):
            call_count.append(algo_step)

        dmshoa.run(surgeries_data, job_ids, seed=10, on_iteration=on_iter)
        assert len(call_count) == MAX_ITERATIONS_MSHOA, (
            f"on_iteration called {len(call_count)} times, expected {MAX_ITERATIONS_MSHOA}"
        )

    def test_on_iteration_receives_monotonically_non_increasing_fitness(self, tmp_path):
        """on_iteration best_fitness values never increase across iterations."""
        _setup_env(tmp_path, _make_minimal_config_dict(max_iter=6, pop_size=3))
        from algorithms import dmshoa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
        fitness_trace = []

        def on_iter(algo_step, best_fitness, best_makespan=None, **kwargs):
            fitness_trace.append(best_fitness)

        dmshoa.run(surgeries_data, job_ids, seed=11, on_iteration=on_iter)
        for i in range(1, len(fitness_trace)):
            assert fitness_trace[i] <= fitness_trace[i - 1] + 1e-9, (
                f"Callback best_fitness increased at index {i}: "
                f"{fitness_trace[i-1]} → {fitness_trace[i]}"
            )

    def test_on_iteration_best_solution_is_valid_permutation(self, tmp_path):
        """best_solution_snapshot in callback is always a valid job permutation."""
        _setup_env(tmp_path, _make_minimal_config_dict(max_iter=4, pop_size=3))
        from algorithms import dmshoa
        from data.data_generator import generate_day_surgeries_data

        job_ids = list(range(1, 6))
        surgeries_data = generate_day_surgeries_data(job_ids, std_factor=0.0)
        violations = []

        def on_iter(algo_step, best_fitness, best_makespan=None, best_solution_snapshot=None, **kwargs):
            if best_solution_snapshot is not None:
                seq = best_solution_snapshot.get("job_sequence_base", [])
                if sorted(seq) != sorted(job_ids):
                    violations.append((algo_step, seq))

        dmshoa.run(surgeries_data, job_ids, seed=12, on_iteration=on_iter)
        assert not violations, f"Invalid permutations in callback: {violations}"
