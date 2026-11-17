#!/usr/bin/env bash
# The asking condition (Claude Opus 5.5 with the Opsis tools) on every question of the pan test split (1,393 slides), one phase per question family in order of
# value (composites, then synoptic, then the clinical and molecular table fields), N shards in parallel per phase, each shard its
# own jsonl under results/organiser/<NAME>_<phase>_s<i>.jsonl (resumable). A shard that stops on three failed episodes in a row
# (rate limit) ends the driver; rerun it later and every shard resumes. Stop with the pid files.
#   bash scripts/81b_pan_organiser.sh            (N=4 MODEL=claude-opus-5-5 NAME=opus55_tools_pan by default)
# BACKEND=api MODEL=gpt-6-sol runs the OpenAI Responses API instead of the Claude Code CLI, ARM=vision EXTRA=--blank-image the grey image
# control of the looking condition, and the phase tables_noct is the table fields without cancer_type.
# PHASES=ct_alias EXTRA=--alias reruns cancer_type with an opaque slide name: the TCGA barcode's site code identifies the cancer
# type and a run without the alias cited it in 1,303 of 1,393 cancer_type reasons.
# The gpt-6-sol tools condition of the paper: N=4 BACKEND=api MODEL=gpt-6-sol NAME=sol_tools_pan_alias PHASES="composites synoptic" EXTRA=--alias
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
: "${OPSIS_DATA:?source env/opsis_env.sh first}"; export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
N="${N:-4}"; MODEL="${MODEL:-claude-opus-5-5}"; NAME="${NAME:-opus55_tools_pan}"; OUT=results/organiser; PHASES="${PHASES:-composites synoptic tables}"; EXTRA="${EXTRA:-}"; BACKEND="${BACKEND:-claude}"; ARM="${ARM:-tools}"
for phase in $PHASES; do
  case "$phase" in ct_alias) OPS="cancer_type";; tables) OPS="cancer_type,molecular_subtype,tumor_purity_bin,leukocyte_fraction_bin";; tables_noct) OPS="molecular_subtype,tumor_purity_bin,leukocyte_fraction_bin";; synoptic) OPS="pan_grade,pan_lvi,pan_pni,pan_necrosis,pan_precursor,pan_mitotic,pan_multifocal,pan_infiltrate";; *) OPS="$phase";; esac
  echo "$(date +%T) phase $phase ops $OPS shards $N"; pids=()
  for i in $(seq 0 $((N - 1))); do
    python scripts/81_organiser_eval.py --name "${NAME}_${phase}_s$i" --arm "$ARM" --backend "$BACKEND" --model "$MODEL" --slides 2000 --split test --ops "$OPS" --shard "$i/$N" --resume $EXTRA >> "$OUT/${NAME}_${phase}_s$i.log" 2>&1 &
    pids+=($!); echo $! > "$OUT/${NAME}_${phase}_s$i.pid"
  done
  fail=0; for j in "${!pids[@]}"; do wait "${pids[$j]}" || fail=1; rm -f "$OUT/${NAME}_${phase}_s$j.pid"; done
  echo "$(date +%T) phase $phase done, fail=$fail"; [ "$fail" = 1 ] && { echo "a shard stopped (rate limit?), rerun this script to resume"; exit 1; }
done
echo "$(date +%T) all phases done"
