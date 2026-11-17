#!/bin/bash
#SBATCH --job-name=opsis-camelyon_dense
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source env/opsis_env.sh          # submit from the repository root
PY="${PYTHON:-python}"
cd $OPSIS_ROOT; nvidia-smi -L; hostname; date
: "${ARGS:?set ARGS, e.g. ARGS='--name jev_all_s0 --protocol all'}"
export OPSIS_TOOL_DEVICE=cuda; $PY scripts/91_camelyon_dense.py $ARGS
date
