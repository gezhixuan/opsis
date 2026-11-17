#!/bin/bash
#SBATCH --job-name=opsis-tuned
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source env/opsis_env.sh          # submit from the repository root
PY="${PYTHON:-python}"
cd $OPSIS_ROOT; nvidia-smi -L; hostname; date
: "${ARGS:?set ARGS, e.g. ARGS='--name jev_all_s0 --protocol all'}"
$PY scripts/21_train_tuned_state.py $ARGS
date
