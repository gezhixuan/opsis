#!/bin/bash
#SBATCH --job-name=opsis-cellvit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# CellViT++ NuCLS super class head on our split: train the head, then run the repository's NuCLS inference on test/.
#   sbatch --export=ALL,BACKBONE=SAM-H scripts/baselines/59_cellvit_nucls.sbatch     (SAM-H | Virchow | ViT256)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source env/opsis_env.sh          # submit from the repository root
PYC="${PYTHON_CELLVIT:-python}"   # the environment of the CellViT++ repository
export WANDB_MODE=offline; export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1   # torch >= 2.6 refuses the numpy scalars in the CellViT checkpoints otherwise
REPO="${OPSIS_THIRD_PARTY:-$OPSIS_ROOT/third_party}/CellViT-plus-plus"; DS=$OPSIS_DATA/derived/cellvit_nucls
: "${BACKBONE:?set BACKBONE=SAM-H|Virchow|ViT256}"
# LEVEL=main (the seven NuCLS main classes, config <BACKBONE>_main.yaml, runs/cellvit_nucls_main) or super (default, four super classes)
LEVEL=${LEVEL:-super}; CFG=$BACKBONE; RUNS=cellvit_nucls; [ "$LEVEL" = main ] && { CFG=${BACKBONE}_main; RUNS=cellvit_nucls_main; }
cd $REPO; nvidia-smi -L; hostname; date
if [ -z "${INFER_ONLY:-}" ]; then $PYC cellvit/train_cell_classifier_head.py --config $DS/train_configs/$CFG.yaml --gpu 0; else echo "INFER_ONLY: skipping training"; fi
LOGDIR=$(ls -td $OPSIS_DATA/runs/$RUNS/$BACKBONE/*/ | head -1); echo "trained head in $LOGDIR"
CKPT=$(grep "^cellvit_path:" $DS/train_configs/$CFG.yaml | awk '{print $2}')
$PYC cellvit/training/evaluate/inference_cellvit_experiment_nucls.py --logdir $LOGDIR --dataset_path $DS --classification_level ${LEVEL}_classification --cellvit_path $CKPT --gpu 0
date; echo CELLVIT_DONE $BACKBONE
