#!/bin/bash
# Prueba final y sensibilidad en el HPC Océano PUCV (oceano-01) con el esquema medido en
# FS_MH_pipeline_ (hpc/README.md): una ola de 3 tareas simultáneas de 40 CPU en CPU/normal,
# un hilo BLAS por worker y una cola compartida de unidades, de la más larga a la más corta,
# con reclamos atómicos (mkdir). Las 3 tareas caben en MaxJobs=3 y MaxSubmit=4; el arreglo
# de 25 tareas de hpc/final.slurm se rechaza con AssocMaxSubmitJobLimit.
# No enviar directamente:  bash hpc/submit_oceano.sh --submit
#SBATCH --job-name=bandido-oceano
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=1-3
#SBATCH --output=results/slurm/oceano_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
state="${OCEANO_STATE:?Enviar con hpc/submit_oceano.sh}"
lanes="${OCEANO_LANES:-2}"
shard="${SLURM_ARRAY_TASK_ID:-1}"
done_dir=results/oceano/done
workers=$(( SLURM_CPUS_PER_TASK / lanes ))
(( workers >= 1 )) || { echo "[ERROR] OCEANO_LANES=$lanes supera las CPUs asignadas" >&2; exit 2; }

source .venv/bin/activate
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
export JOBLIB_TEMP_FOLDER="${SLURM_TMPDIR:-/tmp}/bandido-${SLURM_JOB_ID}"
mkdir -p "$JOBLIB_TEMP_FOLDER" "$state/claims" "$done_dir"
trap 'rm -rf -- "$JOBLIB_TEMP_FOLDER"' EXIT

echo "[INICIO] $(date -Is) job=${SLURM_ARRAY_JOB_ID}_${shard} nodo=$(hostname) cpus=${SLURM_CPUS_PER_TASK} carriles=${lanes} workers/carril=${workers}"
python -c 'import sys, numpy; print("python", sys.version.split()[0], "| numpy", numpy.__version__)'

# Cada carril toma la siguiente unidad libre de la cola y la corre con --jobs $workers.
# Varios carriles por nodo solapan la cola final de una unidad con el inicio de otra.
lane() {
  local lane_id="$1" id log cmd start
  while IFS=$'\t' read -r id log _cost cmd; do
    [[ -e "$done_dir/$id" ]] && continue
    mkdir "$state/claims/$id" 2>/dev/null || continue
    echo "[$(date +%T)] carril $lane_id: $id"
    echo "=== $id | job ${SLURM_ARRAY_JOB_ID}_${shard} | $(hostname) | $(date -Is) | --jobs $workers" >> "$log"
    start=$SECONDS
    # $cmd se separa por espacios a propósito: submit_oceano.sh lo genera sin espacios internos.
    # shellcheck disable=SC2086
    if python $cmd --jobs "$workers" >> "$log" 2>&1; then
      printf 'job=%s_%s nodo=%s segundos=%s\n' "$SLURM_ARRAY_JOB_ID" "$shard" "$(hostname)" \
        "$((SECONDS - start))" > "$done_dir/$id"
      echo "[$(date +%T)] carril $lane_id: $id OK ($((SECONDS - start)) s)"
    else
      echo "$id" >> "$state/failed.$shard"
      echo "[$(date +%T)] carril $lane_id: $id FALLÓ; ver $log" >&2
    fi
  done < "$state/queue.tsv"
}

pids=()
for l in $(seq "$lanes"); do lane "$l" & pids+=("$!"); done
for p in "${pids[@]}"; do wait "$p"; done

pending=0
while IFS=$'\t' read -r id _; do [[ -e "$done_dir/$id" ]] || pending=$((pending + 1)); done < "$state/queue.tsv"
echo "[FIN] $(date -Is) unidades sin terminar en toda la cola: $pending"

# La última tarea en terminar une, verifica y empaqueta; mkdir garantiza que sea solo una.
if (( pending == 0 )) && mkdir "$state/finalize.claim" 2>/dev/null; then
  bash hpc/oceano_finalize.sh
fi

if [[ -s "$state/failed.$shard" ]]; then
  echo "[ERROR] Unidades fallidas en esta tarea: $(tr '\n' ' ' < "$state/failed.$shard")" >&2
  echo "        Relanzar solo lo pendiente con: bash hpc/submit_oceano.sh --submit" >&2
  exit 1
fi
