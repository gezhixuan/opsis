#!/bin/bash
#SBATCH --job-name=opsis-abmil
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# ABMIL or CLAM-SB (scripts/57) over a list of tasks in one job: the bags of all tasks are loaded once, then every task and seed
# is trained. TASKS: space separated operators; QSET: pan | v2 | labels (labels = the BRCA slide_labels csv, full bags)
# SEEDS: e.g. 0 or 0+1+2; MODEL: abmil (default) | clam_sb; HELDOUT=1 (pan) also scores the held out projects; RESUME=1 keeps
# finished (task, seed) checkpoints and only re-evaluates them. Override memory with sbatch --mem=300G for pan.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source env/opsis_env.sh          # submit from the repository root
PY="${PYTHON:-python}"; 
cd $OPSIS_ROOT; nvidia-smi -L; hostname; date
: "${TASKS:?set TASKS}"; QSET=${QSET:-pan}; SEEDS=${SEEDS:-0}; MODEL=${MODEL:-abmil}; P=abmil; [ "$MODEL" = clam_sb ] && P=clam
EXTRA=""; [ "${HELDOUT:-0}" = 1 ] && EXTRA="$EXTRA --heldout"; [ "${RESUME:-0}" = 1 ] && EXTRA="$EXTRA --resume"
T=$(echo $TASKS | tr ' ' '+'); echo "=== $T ($MODEL, $QSET, seeds $SEEDS$EXTRA)"
case $QSET in
  pan) $PY scripts/57_clam_baseline.py --model $MODEL --task $T --seeds $SEEDS --qset pan --max-tiles 4096 --prefix ${P}_pan_ $EXTRA ;;
  v2) $PY scripts/57_clam_baseline.py --model $MODEL --task $T --seeds $SEEDS --qset v2 --prefix ${P}_ $EXTRA ;;
  labels) $PY scripts/57_clam_baseline.py --model $MODEL --task $T --seeds $SEEDS --prefix ${P}_ $EXTRA ;;
esac || echo "FAILED $T"
date
