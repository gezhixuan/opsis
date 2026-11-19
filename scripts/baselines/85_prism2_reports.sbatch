#!/bin/bash
#SBATCH --job-name=opsis-prism2_reports
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source env/opsis_env.sh          # submit from the repository root
PY="${PYTHON_PRISM:-python}"   # the environment of env/requirements-prism.txt (PRISM2 remote code needs transformers 4.48)
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $OPSIS_ROOT; nvidia-smi -L; hostname; date
: "${ARGS:?set ARGS, e.g. ARGS='--name vlm_all_s0 --protocol all'}"
$PY scripts/baselines/85_prism2_reports.py $ARGS
date
