#!/bin/bash
#SBATCH --job-name=opsis-pntokens
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source env/opsis_env.sh          # submit from the repository root
PY="${PYTHON:-python}"; export HF_HUB_OFFLINE=1
cd $OPSIS_ROOT; nvidia-smi -L; hostname; date
$PY scripts/81_pannuke_tokens.py
date
