#!/bin/bash
# Pan cancer bulk extraction with gpt-6-luna in batches of 500 (resume safe: existing batch files are skipped), then validation + repair.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; PY="${PYTHON:-python}"
export D="${OPSIS_DATA:?source env/opsis_env.sh first}/derived/report_pan"; export OPSIS_TCGA="${OPSIS_TCGA:-$OPSIS_DATA/tcga}"; mkdir -p $D/batches
[ -f $D/reports_all.jsonl ] || $PY - <<'PY'
import pandas as pd, glob, os, json
PP=os.environ['OPSIS_TCGA']; R=pd.read_csv(f'{PP}/reports/TCGA_Reports.csv'); R['participant']=R['patient_filename'].str[:12]
proj={}
for f in glob.glob(f'{PP}/gigapath_h5/*_trident_virchow2/*_trident.csv'):
    p=os.path.basename(os.path.dirname(f)).split('_')[0].upper()
    for x in pd.read_csv(f)['participant'].unique(): proj.setdefault(x,p)
R['project']=R['participant'].map(proj); R=R[R.project.notna()].drop_duplicates('participant').sort_values(['project','participant'])
with open(os.environ['D']+'/reports_all.jsonl','w') as f:
    for r in R.itertuples(): f.write(json.dumps(dict(participant=r.participant, project=r.project, text=r.text))+'\n')
print('reports_all', len(R))
PY
N=$(wc -l < $D/reports_all.jsonl); B=500; i=0
while [ $((i*B)) -lt $N ]; do
  f=$D/batches/reports_b$(printf %02d $i).jsonl; o=$D/batches/extracted_b$(printf %02d $i).jsonl
  [ -f $f ] || sed -n "$((i*B+1)),$(((i+1)*B))p" $D/reports_all.jsonl > $f
  if [ ! -f $o ]; then $PY $ROOT/scripts/questions/54_extract_openai.py --reports $f --schema $ROOT/configs/reports/report_pan_schema.json --out $o --model gpt-6-luna --workers 6; fi
  $PY $ROOT/scripts/questions/55_validate_pan.py --reports $f --repair $o | grep -v "^    ("
  i=$((i+1))
done
echo PAN_BULK_DONE
