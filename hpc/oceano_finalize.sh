#!/bin/bash
# Une la prueba final, verifica los conteos y empaqueta resultados_hpc.tar.gz.
# Lo ejecuta automáticamente la última tarea de hpc/oceano.slurm; también se puede
# correr a mano, por ejemplo para reempaquetar con los logs completos:
#   bash hpc/oceano_finalize.sh
set -euo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."

.venv/bin/python tools/merge_results.py results/final/parts results/final
.venv/bin/python - <<'EOF'
import sys

import numpy as np
import pandas as pd


def check(path, rows, instances, methods, seeds):
    runs = pd.read_csv(path)
    problems = []
    if len(runs) != rows:
        problems.append(f"{len(runs)} filas, se esperaban {rows}")
    if not np.isfinite(runs["fitness"]).all():
        problems.append(f"{int((~np.isfinite(runs['fitness'])).sum())} fitness no finitos")
    if not (runs["evaluations"] == 10000).all():
        problems.append(f"{int((runs['evaluations'] != 10000).sum())} filas con evaluations != 10000")
    if runs["instance"].nunique() != instances:
        problems.append(f"{runs['instance'].nunique()} instancias, se esperaban {instances}")
    if runs["method"].nunique() != methods:
        problems.append(f"{runs['method'].nunique()} variantes, se esperaban {methods}")
    if set(runs["seed"]) != set(range(1, seeds + 1)):
        problems.append("semillas distintas de 1..%d" % seeds)
    if runs.duplicated(["instance", "method", "seed"]).any():
        problems.append("corridas duplicadas")
    tag = "OK   " if not problems else "ERROR"
    print(f"{tag} {path}: {len(runs)} filas, {runs['instance'].nunique()} instancias, "
          f"{runs['method'].nunique()} variantes" + ("" if not problems else " | " + "; ".join(problems)))
    return not problems


ok = check("results/final/runs.csv", 5250, 25, 7, 30)
for name in ("L3", "L10", "e005", "e02"):
    ok = check(f"results/sensibilidad/{name}/runs.csv", 160, 8, 1, 20) and ok
sys.exit(0 if ok else 1)
EOF

# Dentro del trabajo el log de la propia tarea sigue creciendo: tar devuelve 1 ("file changed").
status=0
tar --warning=no-file-changed -czf resultados_hpc.tar.gz results/final results/sensibilidad results/slurm || status=$?
(( status <= 1 )) || exit "$status"
echo "[OK] $(realpath resultados_hpc.tar.gz) ($(du -h resultados_hpc.tar.gz | cut -f1))"
