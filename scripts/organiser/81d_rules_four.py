#!/usr/bin/env python
"""The rules condition of the organiser study with a rule for four of the five composites. The saved runs
results/organiser/structured_<project>30.jsonl answer high grade with LVI and necrosis with high mitotic activity by rule and
count the other three composites as wrong. This script adds the two rules that can be written from the pan cancer primitives
(scripts/organiser/81_organiser_eval.py structured_answer has the same code): grade from its components = the candidate grade with the largest pan_grade
probability, DCIS with comedo necrosis = P(precursor yes) x P(necrosis yes) >= 0.5. Precursor type has no rule and stays wrong.
Every other row is copied unchanged. Runs on a login node (the slide decoder on CPU, no language model).
    python scripts/organiser/81d_rules_four.py
Writes results/organiser/structured4_<project>30.jsonl."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from opsis import paths, tools as T
D = f"{paths.ROOT}/results/organiser"; Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_pan/questions_seed0.parquet", columns=["qid", "candidates"]).set_index("qid")
def P(slide, o): r = T.ask_slide(slide, [o])["answers"][0]; return dict(zip(r["candidates"], r["probabilities"]))
for proj in ("brca", "luad", "blca", "ucec"):
    rows = [json.loads(l) for l in open(f"{D}/structured_{proj}30.jsonl")]; out = []; n = 0
    for r in rows:
        op = r["operator"]; pr = None
        if r["pred"] is None and op == "grade_from_components": g = P(r["slide"], "pan_grade"); pr = [g.get(x, 0.0) for x in json.loads(Q.loc[r["qid"], "candidates"])]
        elif r["pred"] is None and op == "dcis_with_comedo": y = P(r["slide"], "pan_precursor").get("yes", 0) * P(r["slide"], "pan_necrosis").get("yes", 0); pr = [1 - y, y]
        if pr is not None:
            pred = int(np.argmax(pr)); r = dict(r, pred=pred, correct=(pred == r["answer"]), confidence=float(max(pr)), text=json.dumps([round(float(x), 3) for x in pr])); n += 1
        out.append(r)
    open(f"{D}/structured4_{proj}30.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in out))
    df = pd.DataFrame(out); print(proj, len(df), "questions, rules filled on", n, "| accuracy", round(df.correct.mean(), 4), "| per composite", {o: (len(g), round(g.correct.mean(), 3)) for o, g in df.groupby("operator") if o in ("grade_from_components", "dcis_with_comedo", "precursor_type", "high_grade_and_lvi", "necrosis_and_high_mitotic")})
