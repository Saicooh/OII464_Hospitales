#!/bin/bash
# Prueba final: un trabajo del arreglo por instancia (25 instancias x 7 variantes x 30 semillas).
# Uso, desde la raíz del repositorio:  sbatch hpc/final.slurm
# Preparar una vez (Python >= 3.11):  python -m venv .venv && .venv/bin/pip install -r requirements.txt
# Al terminar:  python tools/merge_results.py results/final/parts results/final
#SBATCH --job-name=bandido-final
#SBATCH --array=0-24
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/final_%a.out

set -euo pipefail
mkdir -p results/slurm
# Ajustar a cómo se activa Python en el clúster (module load, conda, venv).
source .venv/bin/activate
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python tools/run_experiment.py --split prueba --seeds 30 --fixed-p 0.15 --curves \
  --instance-index "${SLURM_ARRAY_TASK_ID}" --jobs "${SLURM_CPUS_PER_TASK}" \
  --out "results/final/parts/$(printf '%02d' "${SLURM_ARRAY_TASK_ID}")"
