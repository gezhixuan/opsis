#!/usr/bin/env python
"""Opsis-S with fixed rules on the composite operators of the pan test split: the slide decoder (final_slide_pan_s*, which never
trained the composite family) answers the primitive fields and a rule composes the composite from the probabilities, as in the
rules condition of scripts/81 (--arm structured). Rules exist for high_grade_and_lvi (P(grade 3 or 4) x P(LVI yes)) and
necrosis_and_high_mitotic (P(necrosis yes) x P(mitotic high)); precursor_type, grade_from_components and dcis_with_comedo have
no rule from the pan fields and count as wrong in the "all" row, as in the organiser study. The decoder's own answer to the
untrained operator token (metrics_test.json per_op) is reported beside it for reference. Writes results/organiser/rules_composites.json.
    PYTHONPATH=src OPSIS_DATA=... python scripts/88e_rules_composites.py"""
import json, glob, os, sys, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from opsis import paths
COMPS = ["high_grade_and_lvi", "precursor_type", "necrosis_and_high_mitotic", "grade_from_components", "dcis_with_comedo"]
def probs(P, slide, op):
    r = P[(P.fov == slide) & (P.op == op)]
    if r.empty: return None
    r = r.iloc[0]; c = json.loads(r.candidates); return dict(zip(c, r.probs[:len(c)]))
acc = collections.defaultdict(list); n = {}; untrained = collections.defaultdict(list); later = collections.defaultdict(list); miss = {}
for d in sorted(glob.glob(f"{paths.RUNS}/final_slide_pan_s[0-9]")):
    P = pd.read_parquet(f"{d}/predictions_test.parquet"); m = json.load(open(f"{d}/metrics_test.json"))["per_op"]
    for op in COMPS:
        Q = P[P.op == op]; n[op] = len(Q); untrained[op].append(m[op]["acc"])
        if op == "high_grade_and_lvi": y = [(lambda g, l: None if g is None or l is None else (g.get("3", 0) + g.get("4", 0)) * l.get("yes", 0))(probs(P, s, "pan_grade"), probs(P, s, "pan_lvi")) for s in Q.fov]
        elif op == "necrosis_and_high_mitotic": y = [(lambda a, b: None if a is None or b is None else a.get("yes", 0) * b.get("high", 0))(probs(P, s, "pan_necrosis"), probs(P, s, "pan_mitotic")) for s in Q.fov]
        else: y = [None] * len(Q)
        pred = [None if v is None else int(v >= 0.5) for v in y]
        acc[op].append(np.mean([p == a for p, a in zip(pred, Q.answer)]) if all(v is not None for v in y) else None)
        # two rules written after the study from the calls the organiser makes. They are not part of
        # the rules condition of the organiser study and are reported apart: the grade with the largest pan_grade probability among the
        # candidates, and P(precursor yes) x P(necrosis yes) >= 0.5
        if op == "grade_from_components":
            pr = [(lambda g, c: None if g is None else int(np.argmax([g.get(k, 0) for k in c])))(probs(P, s, "pan_grade"), json.loads(c)) for s, c in zip(Q.fov, Q.candidates)]
        elif op == "dcis_with_comedo":
            pr = [(lambda a, b: None if a is None or b is None else int(a.get("yes", 0) * b.get("yes", 0) >= 0.5))(probs(P, s, "pan_precursor"), probs(P, s, "pan_necrosis")) for s in Q.fov]
        else: pr = None
        if pr is not None: later[op].append(float(np.mean([p == a for p, a in zip(pr, Q.answer)]))); miss[op] = sum(v is None for v in pr)      # a question whose primitive has no saved prediction counts as wrong
out = {}
for op in COMPS:
    a = [v for v in acc[op] if v is not None]
    out[op] = dict(n=n[op], rules=None if not a else round(float(np.mean(a)), 4), rules_sd=None if len(a) < 2 else round(float(np.std(a, ddof=1)), 4), untrained_operator=round(float(np.mean(untrained[op])), 4), seeds=len(acc[op]))
N = sum(n.values()); out["all"] = dict(n=N, rules_no_rule_wrong=round(sum((out[op]["rules"] or 0) * n[op] for op in COMPS) / N, 4), untrained_operator=round(sum(out[op]["untrained_operator"] * n[op] for op in COMPS) / N, 4))
for op in later:
    out[op]["rule_written_later"] = round(float(np.mean(later[op])), 4); out[op]["rule_written_later_missing_primitive"] = miss[op]
W = [op for op in COMPS if out[op]["rules"] is not None or out[op].get("rule_written_later") is not None]
out["all"]["with_rule_or_later_rule"] = dict(ops=W, n=sum(n[op] for op in W), acc=round(sum((out[op]["rules"] if out[op]["rules"] is not None else out[op]["rule_written_later"]) * n[op] for op in W) / sum(n[op] for op in W), 4))
# the rules condition with a rule for four of the five composites (the reference condition of the paper): the two
# first rules and the two written later, precursor type has none and counts as wrong in the total
for op in COMPS: out[op]["rules_four"] = out[op]["rules"] if out[op]["rules"] is not None else out[op].get("rule_written_later")
out["all"]["rules_four_no_rule_wrong"] = round(sum((out[op]["rules_four"] or 0) * n[op] for op in COMPS) / N, 4)
json.dump(out, open(f"{paths.ROOT}/results/organiser/rules_composites.json", "w"), indent=1); print(json.dumps(out, indent=1))
