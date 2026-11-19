#!/bin/bash
#SBATCH --job-name=opsis-slide_timing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
# run on the same GPU type as the field timing of final_fov_s0 (H200), a few minutes
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"; source env/opsis_env.sh          # submit from the repository root
PY="${PYTHON:-python}"; export OPSIS_TOOL_DEVICE=cuda
cd $OPSIS_ROOT; nvidia-smi -L; hostname; date
$PY scripts/opsis_s/111_slide_timing.py --slides 20 --reps 10
nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv
date
