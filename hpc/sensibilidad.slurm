#!/bin/bash
# Sensibilidad de A7 en desarrollo: L en {3, 10} y epsilon en {0.05, 0.2}.
# Uso:  sbatch hpc/sensibilidad.slurm
#SBATCH --job-name=bandido-sens
#SBATCH --array=0-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm/sens_%a.out

set -euo pipefail
mkdir -p results/slurm
source .venv/bin/activate
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

NAMES=(L3 L10 e005 e02)
FLAGS=("--block 3" "--block 10" "--epsilon 0.05" "--epsilon 0.2")
i="${SLURM_ARRAY_TASK_ID}"
python tools/run_experiment.py --split desarrollo --seeds 20 --methods dos_niveles \
  ${FLAGS[$i]} --jobs "${SLURM_CPUS_PER_TASK}" --out "results/sensibilidad/${NAMES[$i]}"
