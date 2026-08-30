"""
utils/results_locator.py — Utility to dynamically locate the active results directory.

Resolution order:
1. Environment variable RESULTS_DIR if specified and existing.
2. The latest timestamped directory in `results/` matching YYYYMMDD_HHMMSS
   that contains an `analysis.db` with complete campaign simulations (>= 100).
3. The latest timestamped directory in `results/` containing `analysis.db`.
4. Fallback to `results/` root directory.
"""

import os
import re
import sqlite3
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_results_dir() -> Path:
    """Returns Path object pointing to the active results directory."""
    env_dir = os.environ.get("RESULTS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.exists():
            return p

    results_base = _PROJECT_ROOT / "results"
    if not results_base.exists():
        results_base.mkdir(parents=True, exist_ok=True)
        return results_base

    timestamp_pattern = re.compile(r"^\d{8}_\d{6}$")
    candidates = []

    for item in results_base.iterdir():
        if item.is_dir() and timestamp_pattern.match(item.name):
            db_path = item / "analysis.db"
            if db_path.exists():
                candidates.append(item)

    if not candidates:
        return results_base

    full_candidates = []
    for cand in candidates:
        db_path = cand / "analysis.db"
        try:
            conn = sqlite3.connect(db_path)
            cnt = conn.execute("SELECT COUNT(*) FROM simulations").fetchone()[0]
            conn.close()
            if cnt >= 100:
                full_candidates.append(cand)
        except Exception:
            pass

    if full_candidates:
        full_candidates.sort(key=lambda p: p.name, reverse=True)
        return full_candidates[0]

    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def get_analysis_db_path() -> Path:
    """Returns Path object pointing to analysis.db within the active results directory."""
    return get_results_dir() / "analysis.db"
