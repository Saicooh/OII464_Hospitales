#!/bin/bash
# Lanza la prueba final (25 instancias x 7 variantes x 30 semillas) y la sensibilidad
# (dos_niveles con L=3, L=10, epsilon=0,05 y epsilon=0,2) en el HPC Océano PUCV con el
# esquema de FS_MH_pipeline_: una ola de 3 tareas x 40 CPU (CPU/normal) que comparten
# una cola de 29 unidades, de la más larga a la más corta. Los comandos de cada unidad son
# los mismos de hpc/final.slurm y hpc/sensibilidad.slurm; solo cambia --jobs.
#
# Sin --submit solo muestra el plan. Las unidades terminadas (results/oceano/done/) se
# omiten, así que relanzar después de un fallo repite solo lo pendiente.
#   bash hpc/submit_oceano.sh                       # plan
#   bash hpc/submit_oceano.sh --submit              # enviar
#   bash hpc/submit_oceano.sh --submit --shards 1   # si solo queda un cupo libre
set -euo pipefail
usage='Uso: bash hpc/submit_oceano.sh [--submit] [--shards 1-3] [--lanes N]'
submit=0 shards=3 lanes=2
while (( $# )); do
  case "$1" in
    --submit) submit=1; shift ;;
    --shards) shards="${2:?}"; shift 2 ;;
    --lanes) lanes="${2:?}"; shift 2 ;;
    -h|--help) echo "$usage"; exit 0 ;;
    *) echo "$usage" >&2; exit 2 ;;
  esac
done
[[ "$shards" =~ ^[1-3]$ ]] || { echo "--shards debe ser 1, 2 o 3 (MaxJobs=3)" >&2; exit 2; }
[[ "$lanes" =~ ^[1-9][0-9]*$ ]] && (( lanes <= 40 )) || { echo "--lanes debe estar entre 1 y 40" >&2; exit 2; }

cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo "Falta .venv. Crear con Python >= 3.11 accesible desde los nodos (NFS), p. ej.:" >&2
  echo "  ~/miniconda3/envs/mi_tesis/bin/python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 2
fi
.venv/bin/python -c 'import sys; assert sys.version_info >= (3, 11), sys.version'

# Cola: id, log, costo relativo (corridas x trabajos de la instancia), comando.
queue="$(.venv/bin/python - <<'EOF'
import re
import sys

sys.path.insert(0, "tools")
from run_experiment import METHODS, SPLITS

def size(path):
    return int(re.search(r"-(\d+)-\d+\.yaml$", path).group(1))

units = []
for i, path in enumerate(SPLITS["prueba"]):
    units.append((
        f"final_{i}", f"results/slurm/final_{i}.out", len(METHODS) * 30 * size(path),
        f"tools/run_experiment.py --split prueba --seeds 30 --fixed-p 0.15 --curves "
        f"--instance-index {i} --out results/final/parts/{i:02d}",
    ))
dev = sum(size(path) for path in SPLITS["desarrollo"])
for k, (name, flag) in enumerate(
    [("L3", "--block 3"), ("L10", "--block 10"), ("e005", "--epsilon 0.05"), ("e02", "--epsilon 0.2")]
):
    units.append((
        f"sens_{k}", f"results/slurm/sens_{k}.out", 20 * dev,
        f"tools/run_experiment.py --split desarrollo --seeds 20 --methods dos_niveles {flag} "
        f"--out results/sensibilidad/{name}",
    ))
for unit in sorted(units, key=lambda u: -u[2]):
    print(*unit, sep="\t")
EOF
)"

total=0 pending=0 cost=0
printf '%-9s %-10s %8s  %s\n' unidad estado costo salida
while IFS=$'\t' read -r id _log c cmd; do
  total=$((total + 1))
  if [[ -e "results/oceano/done/$id" ]]; then status=hecha
  else status=pendiente; pending=$((pending + 1)); cost=$((cost + c)); fi
  printf '%-9s %-10s %8s  %s\n' "$id" "$status" "$c" "${cmd##*--out }"
done <<< "$queue"
# ~0,25 s por corrida y trabajo de la instancia, medido en Xeon Gold 6148 con un hilo.
echo "Pendientes: $pending de $total | estimación ideal ≈ $(( cost / 4 / (shards * 40) / 60 + 1 )) min con $shards x 40 CPU ($lanes carriles de $((40 / lanes)) workers por tarea)"
(( pending > 0 )) || { echo "Nada pendiente. Para unir, verificar y empaquetar: bash hpc/oceano_finalize.sh"; exit 0; }

if [[ -n "$(squeue -u "$USER" -h -n bandido-oceano -o %i)" ]]; then
  echo "[ERROR] Ya hay trabajos bandido-oceano en cola; esperar a que terminen o cancelarlos." >&2
  exit 1
fi
max_submit="$(sacctmgr -n -P show assoc user="$USER" format=MaxSubmit 2>/dev/null || true)"
max_submit="$(grep -E '^[0-9]+$' <<< "$max_submit" | head -n1 || true)"
max_submit="${max_submit:-4}"
active="$(squeue -u "$USER" -h -r | wc -l)"
if (( active + shards > max_submit )); then
  echo "[ERROR] Tienes $active trabajos en cola y MaxSubmit=$max_submit; no caben $shards tareas más." >&2
  echo "        Usar --shards $(( max_submit - active > 0 ? max_submit - active : 1 )) o esperar." >&2
  exit 1
fi

if (( ! submit )); then
  echo "Plan solamente. Enviar con: bash hpc/submit_oceano.sh --submit"
  exit 0
fi

state="results/oceano/runs/$(date +%Y%m%dT%H%M%S)"
mkdir -p "$state/claims" results/oceano/done results/slurm
printf '%s\n' "$queue" > "$state/queue.tsv"
jobid="$(sbatch --parsable --array=1-"$shards" \
  --export=ALL,OCEANO_STATE="$state",OCEANO_LANES="$lanes" hpc/oceano.slurm)"
echo "$jobid" > "$state/jobid"
echo "Enviado: arreglo $jobid ($shards x 40 CPU) | estado en $state"
echo "Seguimiento: squeue -u \$USER --array   |   sacct -j $jobid --format=JobID,State,Elapsed,NodeList,AllocCPUS,MaxRSS,ExitCode"
