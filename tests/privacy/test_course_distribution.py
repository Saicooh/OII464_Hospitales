from pathlib import Path

import yaml


ROOT = Path(__file__).parents[2]
CONFIG_FILES = (
    ROOT / "config/config.yaml",
    ROOT / "config/config.quick.yaml",
    ROOT / "config/config.validation.yaml",
)
REMOVED_PATHS = (
    "algorithms/dmshoa_adaptado.py",
    "data/data_generator.py",
    "data/real_batch_generator.py",
    "data/pkl_loader.py",
    "data/raw_trace_writer.py",
    "replay_run.py",
    "core/analysis_persistence.py",
    "utils/results_locator.py",
    "datasets/2_dataset_procesado_actualizado.pkl",
    "datasets/replay_days/replay_days.pkl",
)


def test_removed_paths_and_private_artifacts_are_absent():
    assert not any((ROOT / relative_path).exists() for relative_path in REMOVED_PATHS)
    generated = (
        path
        for path in ROOT.rglob("*")
        if "__pycache__" not in path.parts
    )
    assert not [path for path in generated if path.suffix.lower() == ".pkl"]
    assert not [path for path in generated if "replay" in path.name.lower()]


def test_configuration_is_instance_driven_and_historical_comparison_enabled():
    for path in CONFIG_FILES:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert set(document) == {"instance", "experiment", "algorithms", "logging"}
        assert {"ga", "dpso", "sboa", "dmshoa"} <= set(
            document["algorithms"]
        )
        assert {"num_simulations", "n_jobs", "output_dirs"} <= set(
            document["experiment"]
        )
        assert set(document["experiment"]["output_dirs"]) == {"csv", "plots"}


def test_supported_materials_contain_no_retired_runtime_markers():
    excluded_directories = {
        ".git", ".pytest_cache", "__pycache__", "openspec", "results"
    }
    excluded_files = {
        Path(__file__).resolve(),
        ROOT / "tests/e2e/test_run_outputs.py",
    }
    markers = ("pkl", "raw_trace", "real_data", "replay")
    violations: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.resolve() in excluded_files:
            continue
        if any(part in excluded_directories for part in path.parts):
            continue
        relative = path.relative_to(ROOT).as_posix().lower()
        if any(marker.lower() in relative for marker in markers):
            violations.append(relative)
            continue
        if path.suffix.lower() not in {".py", ".yaml", ".yml", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8").lower()
        if any(marker.lower() in text for marker in markers):
            violations.append(relative)
    assert violations == []
