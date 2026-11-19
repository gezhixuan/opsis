#!/bin/bash
# Retry the BCSS Google Drive folder download until masks/ holds 151 files (Drive rate limits gdown after ~100 files).
PY="${PYTHON:-python}"
D="${OPSIS_DATA:?source env/opsis_env.sh first}/raw/bcss"
for i in $(seq 1 24); do
  n=$(ls $D/masks 2>/dev/null | wc -l); echo "$(date) attempt $i: masks $n"
  [ "$n" -ge 151 ] && { echo BCSS_MASKS_COMPLETE; break; }
  $PY -m gdown --folder 'https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss' -O $D --continue 2>&1 | grep -v '^\s*[0-9]*%' | tail -3
  sleep 1800
done
