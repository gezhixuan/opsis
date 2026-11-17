#!/usr/bin/env bash
# Copy to env/opsis_env.sh, edit the paths, and source it before running any script.
export OPSIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OPSIS_DATA="${OPSIS_DATA:-$OPSIS_ROOT/data}"               # raw/ derived/ features/ runs/
export OPSIS_WSI="${OPSIS_WSI:-$OPSIS_DATA/tcga_wsi}"             # TCGA diagnostic slides from the GDC
export OPSIS_ENCODERS="${OPSIS_ENCODERS:-$OPSIS_ROOT/checkpoints/encoders}"   # Virchow2/ (gated, paige-ai/Virchow2)
export PYTHONPATH="$OPSIS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# API keys for the organiser and the report extraction are read from the environment:
#   export OPENAI_API_KEY=...   export ANTHROPIC_API_KEY=...
