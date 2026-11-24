"""Une los resultados parciales de un arreglo SLURM en una sola carpeta.

Uso:  python tools/merge_results.py results/final/parts results/final
"""

import json
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parts, out = Path(sys.argv[1]), Path(sys.argv[2])
    folders = sorted(p for p in parts.iterdir() if (p / "runs.csv").exists())
    if not folders:
        raise SystemExit(f"no hay resultados en {parts}")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("runs.csv", "blocks.csv", "curves.csv"):
        frames = [pd.read_csv(f / name) for f in folders if (f / name).exists()]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out / name, index=False)
    config = json.loads((folders[0] / "config.json").read_text(encoding="utf-8"))
    config["instances"] = [i for f in folders for i in json.loads((f / "config.json").read_text(encoding="utf-8"))["instances"]]
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    runs = pd.read_csv(out / "runs.csv")
    print(f"{len(folders)} partes, {len(runs)} corridas, {runs['instance'].nunique()} instancias -> {out}")


if __name__ == "__main__":
    main()
