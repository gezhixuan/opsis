#!/usr/bin/env python
"""TITAN and PRISM2 as comparables on the breast slide question set (slide_questions_v2, slide and molecular families, our
patient split): linear probes on the released TITAN slide embeddings (768 d) and on the PRISM2 base embeddings cached for the
pan set (features/prism2_pan/embeddings.npz, diagnostic embeddings kept under per_op_diag), same protocol as scripts/baselines/84_titan_pan.py and 83
(logistic regression, C chosen on val), and TITAN text prompted zero shot for the morphology fields. CPU only, minutes.
Writes runs/titan_brca_probe, runs/prism2_brca_probe, runs/titan_brca_zeroshot (per_op on the test split).
    python scripts/baselines/88_fm_probes_brca.py [--zeroshot] [--skip-probe]"""
import argparse, json, os, pickle
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from opsis import paths, metrics as M
from opsis.questions import FAMILY
ap = argparse.ArgumentParser(); ap.add_argument("--zeroshot", action="store_true"); ap.add_argument("--skip-probe", action="store_true"); ap.add_argument("--titan", default=f"{paths.MODELS}/TITAN"); a = ap.parse_args()
Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_v2/questions_seed0.parquet"); Q = Q[Q["family"].isin(["slide", "molecular"])]
d = pickle.load(open(f"{a.titan}/TCGA_TITAN_features.pkl", "rb")); ET = {str(n): np.asarray(v, np.float32) for n, v in zip(d["filenames"], d["embeddings"])}
z = np.load(f"{paths.FEATURES}/prism2_pan/embeddings.npz"); EP = {"base": dict(zip(map(str, z["slides"]), z["base"])), "diag": dict(zip(map(str, z["slides"]), z["diag"]))}
Q = Q[Q["slide"].isin(ET) & Q["slide"].isin(EP["base"])]; print("breast slides with both embeddings", Q["slide"].nunique(), "test", Q[Q.split == "test"]["slide"].nunique(), flush=True)


def probe_all(name, feats_by_key):
    """feats_by_key: {"": feats, "_diag": feats} -> per_op and per_op_diag"""
    OUT = f"{paths.RUNS}/{name}"; paths.ensure(OUT); pres = {"model": name, "per_op": {}, "per_op_diag": {}}
    for sfx, feats in feats_by_key.items():
        for op, g in Q.groupby("op"):
            g = g.assign(label=g.apply(lambda r: json.loads(r.candidates)[int(r.answer)], axis=1)); tr, va, te = g[g["split"] == "train"], g[g["split"] == "val"], g[g["split"] == "test"]
            if len(tr) < 20 or tr.label.nunique() < 2 or va.empty or te.empty: continue
            cands_all = sorted(g.label.unique()); cidx = {c: i for i, c in enumerate(cands_all)}
            Xtr, Xva = [np.stack([feats[s] for s in x["slide"]]) for x in (tr, va)]; ytr, yva = [x.label.map(cidx).to_numpy() for x in (tr, va)]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd; best = None
            for C in (0.001, 0.01, 0.1, 1.0):
                clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr); pv = clf.predict_proba(Xva); vl = -np.mean(np.log(np.clip(pv[np.arange(len(yva)), np.searchsorted(clf.classes_, yva)], 1e-9, 1)))
                if best is None or vl < best[0]: best = (vl, C, clf)
            clf = best[2]; Xte = (np.stack([feats[s] for s in te["slide"]]) - mu) / sd; K = len(cands_all); pr = np.full((len(te), K), 1e-6); pr[:, clf.classes_.astype(int)] = clf.predict_proba(Xte); pr /= pr.sum(1, keepdims=True)
            yte = te.label.map(cidx).to_numpy(); m = M.summarize(pr, yte, K, "noul" if K == 2 else g["atype"].iloc[0], te["patient"].to_numpy())
            if K == 2 and len(np.unique(yte)) == 2: m["auroc"] = float(roc_auc_score(yte, pr[:, 1]))
            m["C"] = best[1]; pres["per_op" + sfx][op] = dict(family=FAMILY.get(op, "slide"), **m)
            if sfx == "": print(f"  {name:18s} {op:22s} n={m['n']:4d} acc={m['acc']:.3f} (maj {m['majority_acc']:.3f}) auroc={m.get('auroc', '')} nll={m['nll']:.3f} C={best[1]}", flush=True)
    json.dump(pres, open(f"{OUT}/metrics_test.json", "w"), indent=1); paths.ensure(f"{paths.RESULTS}/runs/{name}"); os.system(f"cp {OUT}/metrics_test.json {paths.RESULTS}/runs/{name}/"); return pres


if not a.skip_probe:
    probe_all("titan_brca_probe", {"": ET})
    probe_all("prism2_brca_probe", {"": EP["base"], "_diag": EP["diag"]})
if a.zeroshot:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(a.titan, trust_remote_code=True).eval(); T = ["a whole slide image of CLASSNAME.", "an H&E whole slide image showing CLASSNAME.", "histopathology of CLASSNAME."]; dev = torch.device("cpu")
    ZS = {"lvi_present": {"no": "breast carcinoma without lymphovascular invasion", "yes": "breast carcinoma with lymphovascular invasion"},
          "necrosis_present": {"no": "breast carcinoma without necrosis", "yes": "breast carcinoma with necrosis"},
          "multifocal": {"no": "a unifocal breast carcinoma", "yes": "a multifocal breast carcinoma"},
          "in_situ_present": {"no": "invasive breast carcinoma without an in situ component", "yes": "invasive breast carcinoma with ductal carcinoma in situ"},
          "lcis_present": {"no": "breast carcinoma without lobular carcinoma in situ", "yes": "breast carcinoma with lobular carcinoma in situ"},
          "calcifications": {"no": "breast carcinoma without calcifications", "yes": "breast carcinoma with calcifications"},
          "dcis_comedo_necrosis": {"no": "ductal carcinoma in situ without comedo necrosis", "yes": "ductal carcinoma in situ with comedo necrosis"},
          "histologic_grade": {"1": "a well differentiated, Nottingham grade 1 breast carcinoma", "2": "a moderately differentiated, Nottingham grade 2 breast carcinoma", "3": "a poorly differentiated, Nottingham grade 3 breast carcinoma"},
          "nuclear_grade": {"1": "breast carcinoma with nuclear grade 1", "2": "breast carcinoma with nuclear grade 2", "3": "breast carcinoma with nuclear grade 3"},
          "dcis_grade": {"low": "low grade ductal carcinoma in situ", "intermediate": "intermediate grade ductal carcinoma in situ", "high": "high grade ductal carcinoma in situ"},
          "tubule_score": {"1": "breast carcinoma with tubule formation score 1", "2": "breast carcinoma with tubule formation score 2", "3": "breast carcinoma with tubule formation score 3"},
          "pleomorphism_score": {"1": "breast carcinoma with nuclear pleomorphism score 1", "2": "breast carcinoma with nuclear pleomorphism score 2", "3": "breast carcinoma with nuclear pleomorphism score 3"},
          "mitotic_score": {"1": "breast carcinoma with mitotic count score 1", "2": "breast carcinoma with mitotic count score 2", "3": "breast carcinoma with mitotic count score 3"},
          "histologic_type": {"ductal": "invasive ductal carcinoma of the breast", "lobular": "invasive lobular carcinoma of the breast", "other": "breast carcinoma of a special histologic type"},
          "er_positive": {"no": "estrogen receptor negative breast carcinoma", "yes": "estrogen receptor positive breast carcinoma"},
          "pr_positive": {"no": "progesterone receptor negative breast carcinoma", "yes": "progesterone receptor positive breast carcinoma"},
          "her2_positive": {"no": "HER2 negative breast carcinoma", "yes": "HER2 positive breast carcinoma"},
          "pam50": {"LumA": "luminal A breast carcinoma", "LumB": "luminal B breast carcinoma", "Basal": "basal-like breast carcinoma", "Her2": "HER2-enriched breast carcinoma", "Normal": "normal-like breast carcinoma"}}   # added later, written once before scoring
    OUT = f"{paths.RUNS}/titan_brca_zeroshot"; paths.ensure(OUT); res = {"model": "TITAN zero shot (text prompts) on the breast question set", "per_op": {}}; rows = []
    with torch.no_grad():
        for op, g in Q[Q["split"] == "test"].groupby("op"):
            if op not in ZS: continue
            for cands_json, gg in g.groupby("candidates"):
                cands = json.loads(cands_json); names = [ZS[op].get(str(c), f"breast carcinoma, {c}") for c in cands]
                clf = model.zero_shot_classifier([[n] for n in names], T, device=dev); pr = np.concatenate([model.zero_shot(torch.from_numpy(ET[s]).float(), clf).float().numpy() for s in gg["slide"]], 0)
                for r, p in zip(gg.itertuples(), pr): rows.append(dict(qid=r.qid, slide=r.slide, patient=r.patient, op=op, atype=r.atype, candidates=cands_json, answer=int(r.answer), probs=json.dumps([float(x) for x in p])))
    P = pd.DataFrame(rows); P.to_parquet(f"{OUT}/predictions_test.parquet", index=False)
    for op, g in P.groupby("op"):
        K = int(g["candidates"].map(lambda c: len(json.loads(c))).max()); pr = np.zeros((len(g), K))
        for i, p in enumerate(g["probs"]): v = json.loads(p); pr[i, :len(v)] = v
        pr = pr / pr.sum(1, keepdims=True); y = g["answer"].to_numpy(); m = M.summarize(pr, y, K, g["atype"].iloc[0], g["patient"].to_numpy())
        if K == 2 and len(np.unique(y)) == 2: m["auroc"] = float(roc_auc_score(y, pr[:, 1]))
        res["per_op"][op] = dict(family=FAMILY.get(op, "slide"), **m); print(f"  titan zs {op:22s} n={m['n']:4d} acc={m['acc']:.3f} (maj {m['majority_acc']:.3f}) auroc={m.get('auroc', '')} nll={m['nll']:.3f}", flush=True)
    json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1); paths.ensure(f"{paths.RESULTS}/runs/titan_brca_zeroshot"); os.system(f"cp {OUT}/metrics_test.json {paths.RESULTS}/runs/titan_brca_zeroshot/")
print("RUN_DONE")
