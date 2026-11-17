#!/usr/bin/env python
"""TITAN (slide foundation model, CONCH v1.5 patch features, released TCGA slide embeddings) as a comparable on the pan TCGA
question set: linear probes on the released 768 d slide embeddings for every operator (same protocol as the PRISM2 probes,
C chosen on val) and, with --zeroshot, TITAN's text prompted zero shot classification for the report operators and the
32 way cancer type. CPU only. Writes runs/titan_pan_probe and runs/titan_pan_zeroshot (per_op, per_op_heldout_project).
    python scripts/84_titan_pan.py [--zeroshot]"""
import argparse, json, os, pickle
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from opsis import paths, metrics as M
from opsis.questions import FAMILY
ap = argparse.ArgumentParser(); ap.add_argument("--zeroshot", action="store_true"); ap.add_argument("--skip-probe", action="store_true"); ap.add_argument("--model", default=f"{paths.MODELS}/TITAN"); a = ap.parse_args()
d = pickle.load(open(f"{a.model}/TCGA_TITAN_features.pkl", "rb")); E = {str(n): np.asarray(v, np.float32) for n, v in zip(d["filenames"], d["embeddings"])}
Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_pan/questions_seed0.parquet"); Q = Q[Q["slide"].isin(E)]; print("slides with TITAN embeddings", Q["slide"].nunique(), flush=True)
def probe_all(name, feats):
    OUT = f"{paths.RUNS}/{name}"; paths.ensure(OUT); pres = {"model": name, "per_op": {}, "per_op_heldout_project": {}}
    for op, g in Q.groupby("op"):
        g = g.assign(label=g.apply(lambda r: json.loads(r.candidates)[int(r.answer)], axis=1)); tr, va = g[g["split"] == "train"], g[g["split"] == "val"]
        if len(tr) < 20 or tr.label.nunique() < 2 or va.empty: continue
        cands_all = sorted(g.label.unique()); cidx = {c: i for i, c in enumerate(cands_all)}
        Xtr, Xva = [np.stack([feats[s] for s in x["slide"]]) for x in (tr, va)]; ytr, yva = [x.label.map(cidx).to_numpy() for x in (tr, va)]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd; best = None
        for C in (0.001, 0.01, 0.1, 1.0):
            clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr); pv = clf.predict_proba(Xva); vl = -np.mean(np.log(np.clip(pv[np.arange(len(yva)), np.searchsorted(clf.classes_, yva)], 1e-9, 1)))
            if best is None or vl < best[0]: best = (vl, C, clf)
        clf = best[2]
        for sp, key in (("test", "per_op"), ("heldout_project", "per_op_heldout_project")):
            te = g[g["split"] == sp]
            if te.empty: continue
            Xte = (np.stack([feats[s] for s in te["slide"]]) - mu) / sd; K = len(cands_all); pr = np.full((len(te), K), 1e-6); pr[:, clf.classes_.astype(int)] = clf.predict_proba(Xte); pr /= pr.sum(1, keepdims=True)
            yte = te.label.map(cidx).to_numpy(); m = M.summarize(pr, yte, K, "noul" if K == 2 else g["atype"].iloc[0], te["patient"].to_numpy())
            if K == 2 and len(np.unique(yte)) == 2: m["auroc"] = float(roc_auc_score(yte, pr[:, 1]))
            m["C"] = best[1]; pres[key][op] = dict(family=FAMILY.get(op, "slide"), **m)
            if sp == "test": print(f"  probe {op:26s} n={m['n']:4d} acc={m['acc']:.3f} (maj {m['majority_acc']:.3f}) auroc={m.get('auroc', '')} nll={m['nll']:.3f} C={best[1]}", flush=True)
    json.dump(pres, open(f"{OUT}/metrics_test.json", "w"), indent=1); paths.ensure(f"{paths.RESULTS}/runs/{name}"); os.system(f"cp {OUT}/metrics_test.json {paths.RESULTS}/runs/{name}/"); return pres
if not a.skip_probe: probe_all("titan_pan_probe", E)
if a.zeroshot:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True).eval(); T = ["a whole slide image of CLASSNAME.", "an H&E whole slide image showing CLASSNAME.", "histopathology of CLASSNAME."]; dev = torch.device("cpu")
    PROJ = {"ACC": "adrenocortical carcinoma", "BLCA": "bladder urothelial carcinoma", "BRCA": "breast invasive carcinoma", "CESC": "cervical squamous cell carcinoma", "CHOL": "cholangiocarcinoma", "COAD": "colon adenocarcinoma", "DLBC": "diffuse large B cell lymphoma", "ESCA": "esophageal carcinoma", "GBM": "glioblastoma", "HNSC": "head and neck squamous cell carcinoma", "KICH": "kidney chromophobe renal cell carcinoma", "KIRC": "kidney clear cell renal cell carcinoma", "KIRP": "kidney papillary renal cell carcinoma", "LGG": "lower grade glioma", "LIHC": "hepatocellular carcinoma", "LUAD": "lung adenocarcinoma", "LUSC": "lung squamous cell carcinoma", "MESO": "mesothelioma", "OV": "ovarian serous carcinoma", "PAAD": "pancreatic adenocarcinoma", "PCPG": "pheochromocytoma or paraganglioma", "PRAD": "prostate adenocarcinoma", "READ": "rectum adenocarcinoma", "SARC": "sarcoma", "SKCM": "cutaneous melanoma", "STAD": "stomach adenocarcinoma", "TGCT": "testicular germ cell tumor", "THCA": "thyroid carcinoma", "THYM": "thymoma", "UCEC": "uterine corpus endometrial carcinoma", "UCS": "uterine carcinosarcoma", "UVM": "uveal melanoma"}
    ZS = {"pan_necrosis": {"no": "tumor without necrosis", "yes": "tumor with necrosis"}, "pan_lvi": {"no": "tumor without lymphovascular invasion", "yes": "tumor with lymphovascular invasion"}, "pan_pni": {"no": "tumor without perineural invasion", "yes": "tumor with perineural invasion"},
          "pan_precursor": {"no": "invasive tumor without an in situ component", "yes": "tumor with a precursor or in situ component"}, "pan_multifocal": {"no": "a unifocal tumor", "yes": "a multifocal tumor"},
          "pan_grade": {"1": "a well differentiated, grade 1 tumor", "2": "a moderately differentiated, grade 2 tumor", "3": "a poorly differentiated, grade 3 tumor"}, "pan_mitotic": {"low": "tumor with low mitotic activity", "moderate": "tumor with moderate mitotic activity", "high": "tumor with high mitotic activity"},
          "high_grade_and_lvi": {"no": "a tumor that is not both high grade and lymphovascular invasive", "yes": "a high grade tumor with lymphovascular invasion"}, "necrosis_and_high_mitotic": {"no": "tumor without both necrosis and high mitotic activity", "yes": "tumor with necrosis and high mitotic activity"}, "dcis_with_comedo": {"no": "ductal carcinoma in situ without comedo necrosis", "yes": "ductal carcinoma in situ with comedo necrosis"},
          # added later, written once before scoring: the operators that had no prompt
          "pan_infiltrate": {"none": "tumor without a lymphocytic infiltrate", "mild": "tumor with a mild lymphocytic infiltrate", "moderate": "tumor with a moderate lymphocytic infiltrate", "marked": "tumor with a marked lymphocytic infiltrate"},
          "precursor_type": {"in_situ_carcinoma": "tumor with carcinoma in situ", "adenoma": "tumor arising in an adenoma", "dysplasia_or_intraepithelial": "tumor with dysplasia or intraepithelial neoplasia", "other": "tumor with another kind of precursor lesion"},
          "grade_from_components": {"1": "a well differentiated, Nottingham grade 1 breast carcinoma", "2": "a moderately differentiated, Nottingham grade 2 breast carcinoma", "3": "a poorly differentiated, Nottingham grade 3 breast carcinoma"},
          "tumor_purity_bin": {"<0.5": "tumor tissue in which fewer than half of the cells are tumor cells", "0.5-0.65": "tumor tissue in which 50 to 65 percent of the cells are tumor cells", "0.65-0.8": "tumor tissue in which 65 to 80 percent of the cells are tumor cells", ">=0.8": "tumor tissue in which more than 80 percent of the cells are tumor cells"},
          "leukocyte_fraction_bin": {"<0.09": "tumor tissue in which fewer than 9 percent of the cells are leukocytes", "0.09-0.17": "tumor tissue in which 9 to 17 percent of the cells are leukocytes", "0.17-0.30": "tumor tissue in which 17 to 30 percent of the cells are leukocytes", ">=0.30": "tumor tissue in which more than 30 percent of the cells are leukocytes"}}
    OUT = f"{paths.RUNS}/titan_pan_zeroshot"; paths.ensure(OUT); res = {"model": "TITAN zero shot (text prompts)", "per_op": {}, "per_op_heldout_project": {}}; rows = []
    with torch.no_grad():
        for op, g in Q[Q["split"].isin(["test", "heldout_project"])].groupby("op"):
            if op not in ZS and op != "cancer_type": continue
            for cands_json, gg in g.groupby("candidates"):
                cands = json.loads(cands_json); names = [PROJ.get(c, c) if op == "cancer_type" else ZS[op].get(c, c) for c in cands]
                clf = model.zero_shot_classifier([[n] for n in names], T, device=dev); pr = np.concatenate([model.zero_shot(torch.from_numpy(E[s]).float(), clf).float().numpy() for s in gg["slide"]], 0)
                for r, p in zip(gg.itertuples(), pr): rows.append(dict(qid=r.qid, slide=r.slide, patient=r.patient, split=r.split, op=op, atype=r.atype, candidates=cands_json, answer=int(r.answer), probs=json.dumps([float(x) for x in p])))
    P = pd.DataFrame(rows); P.to_parquet(f"{OUT}/predictions_all.parquet", index=False)
    for sp, key in (("test", "per_op"), ("heldout_project", "per_op_heldout_project")):
        for op, g in P[P["split"] == sp].groupby("op"):
            K = int(g["candidates"].map(lambda c: len(json.loads(c))).max()); pr = np.zeros((len(g), K))
            for i, p in enumerate(g["probs"]): v = json.loads(p); pr[i, :len(v)] = v
            pr = pr / pr.sum(1, keepdims=True); y = g["answer"].to_numpy(); m = M.summarize(pr, y, K, g["atype"].iloc[0], g["patient"].to_numpy())
            if K == 2 and len(np.unique(y)) == 2: m["auroc"] = float(roc_auc_score(y, pr[:, 1]))
            res[key][op] = dict(family=FAMILY.get(op, "slide"), **m)
            if sp == "test": print(f"  zs {op:26s} n={m['n']:4d} acc={m['acc']:.3f} (maj {m['majority_acc']:.3f}) auroc={m.get('auroc', '')} nll={m['nll']:.3f}", flush=True)
    json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1); paths.ensure(f"{paths.RESULTS}/runs/titan_pan_zeroshot"); os.system(f"cp {OUT}/metrics_test.json {paths.RESULTS}/runs/titan_pan_zeroshot/")
print("RUN_DONE")
