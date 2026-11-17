#!/bin/bash
#SBATCH --job-name=opsis-vlmstate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source env/opsis_env.sh          # submit from the repository root
PY="${PYTHON_VLM:-python}"   # the environment of env/requirements-vlm.txt (Qwen3-VL, peft, flash attention)
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $OPSIS_ROOT; nvidia-smi -L; hostname; date
: "${ARGS:?set ARGS, e.g. ARGS='--name vlm_all_s0 --protocol all'}"
$PY scripts/41_train_vlm_state.py $ARGS
date
