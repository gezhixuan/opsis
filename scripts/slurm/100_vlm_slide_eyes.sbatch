#!/bin/bash
#SBATCH --job-name=opsis-eyes
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
# Local VLMs looking at the slide on slide level questions (scripts/100). ARGS='--name eyes_qwen3vl8b_pan --model ... --qset pan --shard 0/3'
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source env/opsis_env.sh          # submit from the repository root

PY="${PYTHON_VLM:-python}"   # the environment of env/requirements-vlm.txt (Qwen3-VL, peft, flash attention)
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OPSIS_TOOL_DEVICE=cpu
cd $OPSIS_ROOT; nvidia-smi -L; hostname; date
: "${ARGS:?set ARGS}"
$PY scripts/100_vlm_slide_eyes.py $ARGS --resume
date
