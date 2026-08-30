"""Tests for the real-day replay input builder entry point."""

import os
import subprocess
import sys
from pathlib import Path


def test_importing_replay_builder_has_no_campaign_side_effects():
    repo = Path(__file__).resolve().parents[2]
    replay_dir = repo / "datasets" / "replay_days"
    output_paths = (
        replay_dir / "replay_days.pkl",
        replay_dir / "replay_days_summary.csv",
    )
    before = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in output_paths
        if path.exists()
    }

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo), env.get("PYTHONPATH")) if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", "import reproducibility.build_replay_days"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    after = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in output_paths
        if path.exists()
    }
    assert after == before
