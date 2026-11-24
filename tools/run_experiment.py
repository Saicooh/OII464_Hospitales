"""Ejecuta las variantes de ``algorithms/mh.py`` con presupuesto fijo.

Ejemplos (desde la raíz del repositorio):

    python tools/run_experiment.py --split desarrollo --seeds 10 --out results/piloto
    python tools/run_experiment.py --split prueba --seeds 30 --out results/final

Cada corrida escribe una fila en ``runs.csv``; los registros por bloque y las
curvas de convergencia se guardan en ``blocks.csv`` y ``curves.csv``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from joblib import Parallel, delayed  # noqa: E402

from algorithms.mh import METHODS, MOVES, HybridParams, run_hybrid  # noqa: E402
from data.instance_loader import load_instance  # noqa: E402


def _ids(family: str, sizes: tuple[int, ...], replicas: tuple[int, ...]) -> list[str]:
    folder = "standard" if family == "STD" else "hospital_12rooms"
    return [
        f"instances/{folder}/HOSP-{family}-{size}-{replica:02d}.yaml"
        for size in sizes
        for replica in replicas
    ]


SPLITS: dict[str, list[str]] = {
    "verificacion": [
        f"instances/didactic/HOSP-DIDACT-{size:02d}-01.yaml" for size in (3, 5, 8, 10)
    ],
    "desarrollo": _ids("STD", (15, 20, 25, 30), (1,)) + _ids("12R", (15, 20, 25, 30), (1,)),
    "prueba": _ids("STD", (15, 20, 25, 30), (2, 3))
    + _ids("12R", (15, 20, 25, 30), (2, 3))
    + _ids("12R", (40, 50, 60), (1, 2, 3)),
}


def _one(path: str, method: str, seed: int, params: HybridParams, label: str) -> dict:
    context = load_instance(ROOT / path)
    result = run_hybrid(context, seed, method=method, params=params)
    row = {
        "config": label,
        "instance": context.instance_id,
        "jobs": len(context.jobs),
        "method": method,
        "seed": seed,
        "fitness": result.best_value,
        "makespan": result.makespan,
        "total_wait": result.total_wait,
        "max_wait": result.max_wait,
        "evaluations": result.evaluations,
        "seconds": result.elapsed,
        "mutations": result.mutations,
        "decisions": sum(1 for block in result.block_log if block["complete"]),
        "wait_violations": result.wait_violations,
    }
    for name in MOVES:
        row[f"n_{name}"] = result.move_counts.get(name, 0)
        row[f"q_{name}"] = result.move_values.get(name, 0.0)
    blocks = [
        {
            "config": label, "instance": context.instance_id, "method": method,
            "seed": seed, "evaluations": block["evaluations"], "p": block["p"],
            "reward": block["reward"], "best": block["best"],
            "q": json.dumps(block["q"]), "q_move": json.dumps(block["q_move"]),
        }
        for block in result.block_log
    ]
    curve = [
        {
            "config": label, "instance": context.instance_id, "method": method,
            "seed": seed, "evaluations": evals, "best": best,
        }
        for index, (evals, best) in enumerate(zip(result.eval_history, result.best_history))
        if index % params.block == 0 or index == len(result.eval_history) - 1
    ]
    return {"row": row, "blocks": blocks, "curve": curve}


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), default="desarrollo")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--out", default="results/experimento")
    parser.add_argument("--label", default="principal")
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--curves", action="store_true", help="guardar curvas por iteración")
    parser.add_argument(
        "--instance-index", type=int, default=None,
        help="ejecutar solo la instancia con este índice dentro del split (p. ej. SLURM_ARRAY_TASK_ID)",
    )
    for item in dataclasses.fields(HybridParams):
        if item.name == "probabilities":
            parser.add_argument("--probabilities", nargs="+", type=float)
        else:
            parser.add_argument(f"--{item.name.replace('_', '-')}", type=type(item.default))
    args = parser.parse_args()

    overrides = {
        item.name: getattr(args, item.name)
        for item in dataclasses.fields(HybridParams)
        if getattr(args, item.name, None) is not None
    }
    if "probabilities" in overrides:
        overrides["probabilities"] = tuple(overrides["probabilities"])
    params = HybridParams(**overrides)

    paths = SPLITS[args.split]
    if args.instance_index is not None:
        paths = [paths[args.instance_index]]
    tasks = [
        (path, method, seed)
        for path in paths
        for method in args.methods
        for seed in range(args.first_seed, args.first_seed + args.seeds)
    ]
    print(f"{len(tasks)} corridas | split={args.split} | {params}", flush=True)
    outputs = Parallel(n_jobs=args.jobs, verbose=1)(
        delayed(_one)(path, method, seed, params, args.label) for path, method, seed in tasks
    )

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "runs.csv", [item["row"] for item in outputs])
    _write(out / "blocks.csv", [block for item in outputs for block in item["blocks"]])
    if args.curves:
        _write(out / "curves.csv", [point for item in outputs for point in item["curve"]])
    (out / "config.json").write_text(
        json.dumps(
            {
                "split": args.split, "instances": paths,
                "methods": args.methods, "seeds": args.seeds,
                "first_seed": args.first_seed, "params": dataclasses.asdict(params),
                "python": sys.version, "platform": platform.platform(),
                "processor": platform.processor(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Resultados en {out}", flush=True)


if __name__ == "__main__":
    main()
