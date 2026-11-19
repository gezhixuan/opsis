#!/bin/bash
#SBATCH --job-name=opsis-c16-fm-zs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# TITAN and PRISM2 zero shot on the CAMELYON16 test set (scripts/baselines/101_camelyon_fm_zeroshot.py); same venv as scripts/baselines/58_prism2_zeroshot.py (PRISM2 remote code needs transformers 4.48).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source env/opsis_env.sh          # submit from the repository root
PY="${PYTHON_PRISM:-python}"   # the environment of env/requirements-prism.txt (PRISM2 remote code needs transformers 4.48)
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $OPSIS_ROOT; nvidia-smi -L; hostname; date
$PY scripts/baselines/101_camelyon_fm_zeroshot.py ${ARGS:-}
date
