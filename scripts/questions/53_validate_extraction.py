#!/usr/bin/env python
"""Validate the report extraction v2 batches: line counts, schema fields, allowed values, quotes as substrings, claim counts.
    python scripts/questions/53_validate_extraction.py"""
import glob, json, os, re, collections
from opsis import paths
R2 = f"{paths.DERIVED}/report_v2"; S = json.load(open(f"{paths.ROOT}/configs/reports/report_v2_schema.json"))
FIELDS = [k for k in S["fields"] if k != "h_and_e_visible_claims"]
def allowed(k):
    d = S["fields"][k]
    if k in ("mitoses_per_10hpf", "non_neoplastic_findings"): return None
    d = re.sub(r"\([^)]*\)", "", d).replace("one of:", "")                      # drop parenthetical notes
    return {v.strip() for v in re.split(r"[|,]", d) if v.strip()} | {"not reported"}
texts = {}
for f in [f"{R2}/pilot/reports_pilot.jsonl"] + sorted(glob.glob(f"{R2}/batches/reports_b*.jsonl")):
    for l in open(f): j = json.loads(l); texts[j["participant"]] = j["text"]
tot = collections.Counter(); bad = []
files = [f"{R2}/pilot/extracted_sonnet.jsonl"] + sorted(glob.glob(f"{R2}/batches/extracted_b*.jsonl"))
for f in files:
    src = f.replace("extracted_sonnet", "reports_pilot").replace("extracted_", "reports_"); n_in = sum(1 for _ in open(src)) if os.path.exists(src) else -1
    n = 0
    for i, l in enumerate(open(f)):
        try: j = json.loads(l)
        except Exception: bad.append((f, i, "json")); continue
        n += 1; p = j.get("participant"); t = texts.get(p, "")
        for k in FIELDS:
            v = j.get("fields", {}).get(k)
            if v is None: bad.append((f, p, f"missing {k}")); continue
            a = allowed(k)
            if a is not None and str(v.get("value")) not in a: bad.append((f, p, f"value {k}={v.get('value')}"))
            q = v.get("quote", "")
            if q and q not in t: bad.append((f, p, f"quote {k}"))
        c = j.get("h_and_e_visible_claims", [])
        if not (1 <= len(c) <= 8): bad.append((f, p, f"claims {len(c)}"))
    tot[os.path.basename(f)] = (n, n_in)
print("batch: extracted / input"); [print(f"  {k:26s} {v[0]:3d} / {v[1]}") for k, v in tot.items()]
print("records:", sum(v[0] for v in tot.values()), "| problems:", len(bad)); [print("  ", b) for b in bad[:15]]
