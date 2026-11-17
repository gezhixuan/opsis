#!/usr/bin/env python
"""Every comparable on the organiser's own question set (the 189 questions about 30 TCGA-BRCA test slides that every organiser
condition answered), so that the composite rows of the main table compare every model on the same slides and questions. TITAN and
PRISM2 linear probes are refitted with the protocol of scripts/84 and 83 (train split, C chosen on val) and scored on these
questions, TITAN and PRISM2 zero shot come from their saved predictions, Opsis-S (final_slide_pan, 3 seeds) from its saved
test predictions, and the organiser conditions from results/organiser. Writes results/collected/RESULTS_organiser_set_comparables.md and
results/organiser/comparables_brca30.json.
    python scripts/88b_organiser_set_comparables.py"""
import json, os, pickle
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from opsis import paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ORG = f"{ROOT}/results/organiser"; R = paths.RUNS
ARMS = {"rules": "structured_brca30", "gpt-6-sol tools": "sol_tools_brca30", "gpt-6-sol eyes": "sol_vision_brca30", "gpt-5.5 tools": "gpt55_tools_brca30", "gpt-5.5 eyes": "gpt55_vision_brca30", "gpt-6-astra tools": "gpt6astra_tools_brca30", "gpt-6-astra eyes": "gpt6astra_vision_brca30", "Opus 5.5 tools": "opus55_tools_brca30", "Opus 5.5 eyes": "opus55_vision_brca30"}
O = {k: pd.DataFrame([json.loads(l) for l in open(f"{ORG}/{v}.jsonl")]) for k, v in ARMS.items()}
qids = set.intersection(*[set(df.qid) for df in O.values()]); print("common organiser questions", len(qids), flush=True)
Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_pan/questions_seed0.parquet")
S = Q[Q.qid.isin(qids)].copy(); S["label"] = S.apply(lambda r: json.loads(r.candidates)[int(r.answer)], axis=1)
SYN = ["pan_grade", "pan_lvi", "pan_pni", "pan_necrosis", "pan_precursor", "pan_mitotic", "pan_multifocal"]; COMP = ["high_grade_and_lvi", "necrosis_and_high_mitotic", "grade_from_components", "dcis_with_comedo", "precursor_type"]
ops = [o for o in SYN + COMP if o in set(S.op)]
print("slides", S.slide.nunique(), "ops", ops, flush=True)


def probe(feats, name):
    """refit one logistic regression per operator with the protocol of scripts/84 and 83, score the organiser questions"""
    out = {}
    for op in ops:
        g = Q[(Q.op == op) & Q.slide.isin(feats)].copy(); g["label"] = g.apply(lambda r: json.loads(r.candidates)[int(r.answer)], axis=1)
        tr, va, te = g[g.split == "train"], g[g.split == "val"], g[g.qid.isin(qids)]
        cands_all = sorted(g.label.unique()); cidx = {c: i for i, c in enumerate(cands_all)}
        Xtr, Xva = [np.stack([feats[s] for s in x.slide]) for x in (tr, va)]; ytr, yva = [x.label.map(cidx).to_numpy() for x in (tr, va)]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd; best = None
        for C in (0.001, 0.01, 0.1, 1.0):
            clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr); pv = clf.predict_proba(Xva); vl = -np.mean(np.log(np.clip(pv[np.arange(len(yva)), np.searchsorted(clf.classes_, yva)], 1e-9, 1)))
            if best is None or vl < best[0]: best = (vl, C, clf)
        clf = best[2]; Xte = (np.stack([feats[s] for s in te.slide]) - mu) / sd; pred = clf.classes_[clf.predict_proba(Xte).argmax(1)]
        out[op] = dict(acc=float(np.mean(pred == te.label.map(cidx).to_numpy())), n=int(len(te)), C=best[1]); print(f"  {name:12s} {op:26s} n={len(te):3d} acc={out[op]['acc']:.3f} C={best[1]}", flush=True)
    return out


d = pickle.load(open(f"{paths.MODELS}/TITAN/TCGA_TITAN_features.pkl", "rb")); ET = {str(n): np.asarray(v, np.float32) for n, v in zip(d["filenames"], d["embeddings"])}
z = np.load(f"{paths.FEATURES}/prism2_pan/embeddings.npz"); EP = dict(zip(map(str, z["slides"]), z["base"]))
titan_p, prism_p = probe(ET, "TITAN probe"), probe(EP, "PRISM2 probe")


def zs_acc(P):
    P = P[P.qid.isin(qids)].copy(); out = {}
    for op, g in P.groupby("op"):
        pred = g.probs.map(lambda p: int(np.argmax(json.loads(p) if isinstance(p, str) else p))); out[op] = dict(acc=float((pred == g.answer.astype(int)).mean()), n=int(len(g)))
    return out


titan_z = zs_acc(pd.read_parquet(f"{R}/titan_pan_zeroshot/predictions_all.parquet")); prism_z = zs_acc(pd.read_parquet(f"{R}/prism2_pan_zeroshot/predictions_test.parquet"))
seeds = []
for s in range(3):
    P = pd.read_parquet(f"{R}/final_slide_pan_s{s}/predictions_test.parquet"); seeds.append(zs_acc(P))
ours = {op: dict(acc=float(np.mean([sd[op]["acc"] for sd in seeds])), sd=float(np.std([sd[op]["acc"] for sd in seeds], ddof=1)), n=seeds[0][op]["n"]) for op in ops if op in seeds[0]}
maj = {op: float(g.label.value_counts().iloc[0] / len(g)) for op, g in S.groupby("op")}
org = {k: {op: float(g.correct.mean()) for op, g in df.groupby("operator")} for k, df in O.items()}; org_all = {k: float(df[df.qid.isin(qids)].correct.mean()) for k, df in O.items()}
maj_all = float(sum(S.groupby("op").label.agg(lambda x: x.value_counts().iloc[0])) / len(S))

cols = ["operator", "n", "majority", "Opsis-S (3 seeds)", "TITAN probe", "PRISM2 probe", "TITAN zero shot", "PRISM2 zero shot"] + list(ARMS)
f3 = lambda v: "" if v is None else f"{v:.3f}"
lines = ["# Every comparable on the organiser question set (generated by scripts/88b_organiser_set_comparables.py)", "",
         f"The {len(qids)} questions about {S.slide.nunique()} TCGA-BRCA test slides that every organiser arm answered, accuracy per operator. The TITAN and PRISM2 probes are refitted on the train split with C chosen on val (the protocol of scripts/84 and 83) and scored on these questions. Zero shot and Opsis-S (the pan slide model, mean over 3 seeds) are read from their saved predictions on the same questions. Rules is Opsis-S with fixed rules and no language model, the composites without a rule counted as wrong. The pan test split numbers of the same models are in results/collected/RESULTS_pan_comparables.md.", "",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
rows_json = {}
for op in ops:
    r = [op, str(ours[op]["n"]), f"{maj[op]:.2f}", f"{ours[op]['acc']:.3f} ± {ours[op]['sd']:.3f}", f3(titan_p[op]["acc"]), f3(prism_p[op]["acc"]), f3(titan_z[op]["acc"]) if op in titan_z else "", f3(prism_z[op]["acc"]) if op in prism_z else ""] + [f3(org[k].get(op)) for k in ARMS]
    lines.append("| " + " | ".join(r) + " |"); rows_json[op] = dict(n=ours[op]["n"], majority=maj[op], opsis_s=ours[op]["acc"], opsis_s_sd=ours[op]["sd"], titan_probe=titan_p[op]["acc"], prism2_probe=prism_p[op]["acc"], titan_zs=titan_z.get(op, {}).get("acc"), prism2_zs=prism_z.get(op, {}).get("acc"), **{k.replace(" ", "_").replace("-", "").replace(".", ""): org[k].get(op) for k in ARMS})
lines.append("| " + " | ".join(["all (common questions)", str(len(qids)), f"{maj_all:.2f}", "", "", "", "", ""] + [f3(org_all[k]) for k in ARMS]) + " |")
open(f"{ROOT}/results/collected/RESULTS_organiser_set_comparables.md", "w").write("\n".join(lines) + "\n"); json.dump(dict(n=len(qids), per_op=rows_json, all=org_all), open(f"{ORG}/comparables_brca30.json", "w"), indent=1)
print("\n".join(lines))
