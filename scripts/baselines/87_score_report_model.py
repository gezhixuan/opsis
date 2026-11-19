#!/usr/bin/env python
"""Score a report model on the slide questions: its free text reports were extracted with the pan cancer schema (scripts/questions/54_extract_openai.py);
each extracted field is mapped to the operator that the TCGA report label feeds, and compared with the ground truth of the
pan TCGA question set. Reports accuracy over all questions (unstated = wrong), accuracy on the questions the report stated
(with coverage), per operator, test and held out projects. Output in the format of scripts/opsis_s/51_train_slide.py (per_op, per_op_heldout_project).
    python scripts/baselines/87_score_report_model.py --extracted derived/report_models/prism2_extracted.jsonl --name prism2_report_model"""
import argparse, json, os
import numpy as np, pandas as pd
from opsis import paths, metrics as M
from opsis.questions import FAMILY
ap = argparse.ArgumentParser(); ap.add_argument("--extracted", required=True); ap.add_argument("--name", default="prism2_report_model"); a = ap.parse_args()
Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_pan/questions_seed0.parquet")
X = {j["participant"]: j for j in map(json.loads, open(a.extracted)) if "fields" in j}
def val(j, k):
    v = j["fields"].get(k); v = v.get("value") if isinstance(v, dict) else v; return None if v in (None, "", "not reported") else str(v).strip().lower()
MAP = {"pan_grade": lambda j: val(j, "histologic_grade"), "pan_lvi": lambda j: {"present": "yes", "absent": "no"}.get(val(j, "lymphovascular_invasion")), "pan_pni": lambda j: {"present": "yes", "absent": "no"}.get(val(j, "perineural_invasion")),
       "pan_necrosis": lambda j: {"present": "yes", "absent": "no"}.get(val(j, "necrosis")), "pan_precursor": lambda j: {"present": "yes", "absent": "no"}.get(val(j, "precursor_or_in_situ_component")), "pan_mitotic": lambda j: val(j, "mitotic_activity"),
       "pan_multifocal": lambda j: val(j, "multifocal"), "pan_infiltrate": lambda j: val(j, "lymphocytic_infiltrate"),
       "high_grade_and_lvi": lambda j: None if (val(j, "histologic_grade") is None or val(j, "lymphovascular_invasion") is None) else ("yes" if val(j, "histologic_grade") in ("3", "4") and val(j, "lymphovascular_invasion") == "present" else "no"),
       "necrosis_and_high_mitotic": lambda j: None if (val(j, "necrosis") is None or val(j, "mitotic_activity") is None) else ("yes" if val(j, "necrosis") == "present" and val(j, "mitotic_activity") == "high" else "no")}
OUT = f"{paths.RUNS}/{a.name}"; paths.ensure(OUT); res = {"model": a.name, "per_op": {}, "per_op_heldout_project": {}, "per_op_stated": {}, "n_reports": len(X)}
for sp, key in (("test", "per_op"), ("heldout_project", "per_op_heldout_project")):
    for op, f in MAP.items():
        g = Q[(Q["split"] == sp) & (Q["op"] == op) & Q["slide"].isin(X)]
        if g.empty: continue
        K = int(g["candidates"].map(lambda c: len(json.loads(c))).max()); pr = np.full((len(g), K), 1.0 / K); stated = np.zeros(len(g), bool)
        for i, r in enumerate(g.itertuples()):
            c = json.loads(r.candidates); v = f(X[r.slide])
            if v in c: pr[i] = 1e-3; pr[i, c.index(v)] = 1 - 1e-3 * (K - 1); stated[i] = True
        y = g["answer"].to_numpy(); m = M.summarize(pr, y, K, g["atype"].iloc[0], g["patient"].to_numpy()); m["coverage"] = float(stated.mean()); m["acc_all_unstated_wrong"] = float(((pr.argmax(1) == y) & stated).mean())
        res[key][op] = dict(family=FAMILY.get(op, "slide"), **m)
        if stated.sum() >= 5:
            ms = M.summarize(pr[stated], y[stated], K, g["atype"].iloc[0], g["patient"].to_numpy()[stated]); res[("per_op_stated" if sp == "test" else "per_op_stated_heldout")] = res.get("per_op_stated" if sp == "test" else "per_op_stated_heldout", {}); res["per_op_stated" if sp == "test" else "per_op_stated_heldout"][op] = dict(family=FAMILY.get(op, "slide"), **ms)
        print(f"  {sp:16s} {op:26s} n={m['n']:4d} stated={m['coverage']:.2f} acc(unstated wrong)={m['acc_all_unstated_wrong']:.3f} acc(stated)={res.get('per_op_stated' if sp == 'test' else 'per_op_stated_heldout', {}).get(op, {}).get('acc', float('nan')):.3f} maj={m['majority_acc']:.2f}", flush=True)
json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1); paths.ensure(f"{paths.RESULTS}/runs/{a.name}"); os.system(f"cp {OUT}/metrics_test.json {paths.RESULTS}/runs/{a.name}/"); print("RUN_DONE", OUT)
