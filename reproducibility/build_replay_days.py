"""
Phase A of the real-day replay experiment.

Reconstructs real elective surgical days from the raw hospital PKL using the
SAME duration formulas and cleaning filters the model uses (data.pkl_loader),
then draws a stratified sample of 30 days per size bucket (15/20/25/30 +/-1)
and stores them for the replay runner (../replay_run.py, Phase B).

For every day it records: the explicit per-surgery ``surgeries_data`` dict in the
scheduler contract, the actual number of operating rooms opened that day, and the
realized (manual) makespan = max(Salida Quirofano) - min(Inicio Anestesia).

Outputs (all inside the repo, replicable):
    datasets/replay_days/replay_days.pkl
    datasets/replay_days/replay_days_summary.csv

Run:  python reproducibility/build_replay_days.py
"""
from pathlib import Path
import pickle
import sys
from collections import Counter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import pandas as pd, numpy as np

PKL = REPO / "datasets" / "2_dataset_procesado_actualizado.pkl"
OUTDIR = REPO / "datasets" / "replay_days"
SEED = 42  # deterministic day selection

def _find_column(raw: pd.DataFrame, substring: str) -> str:
    for c in raw.columns:
        if substring.lower() in str(c).lower():
            return c
    raise KeyError(substring)


def build_day(g: pd.DataFrame, anesthesia_column: str, exit_column: str):
    surgeries = {}
    for jid, (idx, r) in enumerate(g.iterrows(), start=1):
        surgeries[jid] = {
            1: float(r["tiempo_anestesia"]),
            2: float(r["tiempo_cirugia"]),
            "setup_by_op": {1: float(r["setup_qx_anestesia"]), 2: 0.0},
            "transition_by_op": {1: float(r["tiempo_transicion"]), 2: 0.0},
            "transition_after_op1": float(r["tiempo_transicion"]),
            "cleanup_by_op": {1: 0.0, 2: float(r["tiempo_limpieza"])},
            "prep": float(r["tiempo_preparacion"]),
            "cleanup": float(r["tiempo_limpieza"]),
            "codigo_cie10": str(r["codigo_cie10"]),
            "source_record_id": int(idx),
        }
    realized = (
        pd.to_datetime(g[exit_column]).max()
        - pd.to_datetime(g[anesthesia_column]).min()
    ).total_seconds() / 60.0
    return surgeries, int(g["Pavilion_Number"].nunique()), float(realized)


def build_replay_days(
    pkl_path: str | Path = PKL,
    output_dir: str | Path = OUTDIR,
    seed: int = SEED,
) -> list[dict]:
    """Build and persist the deterministic real-day replay inputs."""
    from data.pkl_loader import load_and_prepare

    pkl_path = Path(pkl_path)
    output_dir = Path(output_dir)

    # Durations via the same loader the model uses.
    df_clean, _meta = load_and_prepare(str(pkl_path))
    print(
        f"df_clean valid rows: {len(df_clean)} "
        f"(record_id unique: {df_clean.record_id.nunique()})"
    )

    # Raw day/room/type/timestamps, restricted to surviving records.
    raw = pd.read_pickle(pkl_path)
    c_day = _find_column(raw, "fecha de pab")
    c_tipo = _find_column(raw, "tipo atenci")
    c_anes = _find_column(raw, "inicio anestesia")
    c_salida = _find_column(raw, "salida quir")

    df_clean = df_clean.set_index("record_id")
    sub = raw.loc[
        df_clean.index, [c_day, c_tipo, "Pavilion_Number", c_anes, c_salida]
    ].copy()
    sub["day"] = pd.to_datetime(sub[c_day]).dt.date
    sub["elective"] = sub[c_tipo].astype(str).str.upper().str.startswith("PROG")
    for col_dur in [
        "tiempo_anestesia",
        "tiempo_cirugia",
        "setup_qx_anestesia",
        "tiempo_transicion",
        "tiempo_limpieza",
        "tiempo_preparacion",
        "codigo_cie10",
    ]:
        sub[col_dur] = df_clean[col_dur]

    elective = sub[sub["elective"]].copy()
    print(f"elective valid rows: {len(elective)}")

    days = []
    for day, group in elective.groupby("day"):
        if group[
            [
                "tiempo_anestesia",
                "tiempo_cirugia",
                "setup_qx_anestesia",
                "tiempo_transicion",
                "tiempo_limpieza",
            ]
        ].isna().any().any():
            continue
        surgeries, rooms, realized = build_day(group, c_anes, c_salida)
        if realized <= 0 or rooms < 1:
            continue
        days.append(
            {
                "day": str(day),
                "n": len(group),
                "room_count": min(rooms, 12),
                "rooms_raw": rooms,
                "realized_makespan": realized,
                "surgeries_data": surgeries,
            }
        )

    summary = pd.DataFrame(
        [{k: v for k, v in day.items() if k != "surgeries_data"} for day in days]
    )
    print(f"\nTotal elective days built: {len(days)}")
    print(
        "size distribution:",
        summary.n.describe()[["min", "25%", "50%", "75%", "max"]].to_dict(),
    )

    rng = np.random.default_rng(seed)
    selected = []
    for size in [15, 20, 25, 30]:
        pool = [day for day in days if size - 1 <= day["n"] <= size + 1]
        indices = rng.permutation(len(pool))[:30]
        picked = [pool[i] for i in indices]
        selected.extend([{**day, "bucket": size} for day in picked])
        realized = [day["realized_makespan"] for day in picked]
        room_counts = [day["room_count"] for day in picked]
        print(
            f"  bucket {size}: pool={len(pool)} picked={len(picked)} | "
            f"realized median={np.median(realized):.1f} | "
            f"rooms median={np.median(room_counts):.0f}"
        )

    print(f"\nSelected total: {len(selected)} days")
    print(
        "room_count distribution:",
        dict(sorted(Counter(day["room_count"] for day in selected).items())),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "replay_days.pkl", "wb") as output_file:
        pickle.dump(selected, output_file)
    pd.DataFrame(
        [{k: v for k, v in day.items() if k != "surgeries_data"} for day in selected]
    ).to_csv(output_dir / "replay_days_summary.csv", index=False)
    print(f"Saved -> {output_dir / 'replay_days.pkl'} (+ summary csv)")
    return selected


def main() -> None:
    build_replay_days()


if __name__ == "__main__":
    main()
