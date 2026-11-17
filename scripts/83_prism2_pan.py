#!/usr/bin/env python
"""PRISM2 as the slide foundation model comparable on the pan TCGA question set (slide_questions_pan): slide embeddings for
every slide (cached, resumable), zero shot yes/no and multiple choice answers on the test and held out project slides for
the report operators and the composite operators, and logistic regression probes (base and diagnostic embeddings, C chosen on
val) for every operator including the 32 way cancer type and the composites. Metrics in the format of scripts/51
(per_op, per_op_heldout_project) under runs/prism2_pan_zeroshot and runs/prism2_pan_probe.
    python scripts/83_prism2_pan.py [--max-tiles 8192] [--smoke]"""
import argparse, json, os, re, sys, time
import numpy as np, pandas as pd, torch, h5py
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from opsis import paths, metrics as M
from opsis.questions import FAMILY
ap = argparse.ArgumentParser(); ap.add_argument("--model", default=f"{paths.MODELS}/PRISM2"); ap.add_argument("--max-tiles", type=int, default=8192); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--skip-zeroshot", action="store_true"); ap.add_argument("--skip-probe", action="store_true"); ap.add_argument("--zs-ops", default=None, help="zero shot only for these operators (joined by +) and merge them into the saved predictions_all.parquet, the other operators keep their saved answers")
a = ap.parse_args(); dev = torch.device("cuda"); sfx = "_smoke" if a.smoke else ""
Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_pan/questions_seed0.parquet"); S = Q.drop_duplicates("slide").set_index("slide")
EMB = f"{paths.FEATURES}/prism2_pan"; paths.ensure(EMB); EMBF = f"{EMB}/embeddings{sfx}.npz"
YESNO = {"pan_lvi": "Is lymphovascular invasion present?", "pan_pni": "Is perineural invasion present?", "pan_necrosis": "Is tumor necrosis present?", "pan_precursor": "Is a precursor lesion or in situ component of the tumor present?", "pan_multifocal": "Is the tumor multifocal?",
         "high_grade_and_lvi": "Is the tumor both high grade and showing lymphovascular invasion?", "necrosis_and_high_mitotic": "Does the tumor show both necrosis and high mitotic activity?", "dcis_with_comedo": "Does the ductal carcinoma in situ show comedo necrosis?"}
MCQ = {"pan_grade": ("What is the histologic grade of the tumor?", {"1": "Grade 1, well differentiated", "2": "Grade 2, moderately differentiated", "3": "Grade 3, poorly differentiated or high grade"}),
       "pan_mitotic": ("How high is the mitotic activity of the tumor?", {"low": "Low", "moderate": "Moderate", "high": "High"}), "pan_infiltrate": ("How dense is the lymphocytic infiltrate?", None),
       "grade_from_components": ("What is the Nottingham histologic grade of the invasive carcinoma?", {"1": "Grade 1", "2": "Grade 2", "3": "Grade 3"}), "precursor_type": ("What type of precursor or in situ lesion is present?", None),
       # added later, written once before scoring: the table fields that had no prompt
       "tumor_purity_bin": ("What fraction of the cells in this tumor are tumor cells?", {"<0.5": "Less than 50%", "0.5-0.65": "50% to 65%", "0.65-0.8": "65% to 80%", ">=0.8": "80% or more"}),
       "leukocyte_fraction_bin": ("What fraction of the cells in this tumor sample are leukocytes?", {"<0.09": "Less than 9%", "0.09-0.17": "9% to 17%", "0.17-0.30": "17% to 30%", ">=0.30": "30% or more"})}
# cancer type has 32 candidates, more than the option letters: one yes/no score per candidate, normalised over the candidates
PROJ = {"ACC": "adrenocortical carcinoma", "BLCA": "bladder urothelial carcinoma", "BRCA": "breast invasive carcinoma", "CESC": "cervical squamous cell carcinoma", "CHOL": "cholangiocarcinoma", "COAD": "colon adenocarcinoma", "DLBC": "diffuse large B cell lymphoma", "ESCA": "esophageal carcinoma", "GBM": "glioblastoma", "HNSC": "head and neck squamous cell carcinoma", "KICH": "kidney chromophobe renal cell carcinoma", "KIRC": "kidney clear cell renal cell carcinoma", "KIRP": "kidney papillary renal cell carcinoma", "LGG": "lower grade glioma", "LIHC": "hepatocellular carcinoma", "LUAD": "lung adenocarcinoma", "LUSC": "lung squamous cell carcinoma", "MESO": "mesothelioma", "OV": "ovarian serous carcinoma", "PAAD": "pancreatic adenocarcinoma", "PCPG": "pheochromocytoma or paraganglioma", "PRAD": "prostate adenocarcinoma", "READ": "rectum adenocarcinoma", "SARC": "sarcoma", "SKCM": "cutaneous melanoma", "STAD": "stomach adenocarcinoma", "TGCT": "testicular germ cell tumor", "THCA": "thyroid carcinoma", "THYM": "thymoma", "UCEC": "uterine corpus endometrial carcinoma", "UCS": "uterine carcinosarcoma", "UVM": "uveal melanoma"}   # the names of scripts/84
PERCLASS = {"cancer_type": ("Is this {name}?", PROJ)}
slides_all = sorted(S.index); 
if a.smoke: slides_all = sorted(S[S["split"] == "train"].index)[:8] + sorted(S[S["split"] == "test"].index)[:4]
def load_partial(path):
    """rows and finished slides of an interrupted zero shot run. A line cut off by the kill is dropped and the file rewritten without it, so the next append starts on a new line"""
    rows, seen, keep = [], set(), []
    if os.path.exists(path):
        for l in open(path):
            try: d = json.loads(l)
            except json.JSONDecodeError: continue
            rows += d["rows"]; seen |= set(d["slides"]); keep.append(json.dumps(d) + "\n")
        tmp = f"{path}.tmp{os.getpid()}"; open(tmp, "w").writelines(keep); os.replace(tmp, path)
    return rows, seen
def tiles(slide, rng):
    with h5py.File(S.loc[slide, "h5_path"], "r") as h:
        n = h["features"].shape[0]
        if n > a.max_tiles: idx = np.sort(rng.choice(n, a.max_tiles, replace=False)); X = h["features"][idx, :1280]
        else: X = h["features"][:, :1280]
    return torch.from_numpy(X.astype(np.float32))
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained(a.model, trust_remote_code=True, torch_dtype="auto").to(dev).eval(); processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True); rng = np.random.default_rng(0)
# ---- 1. embeddings for every slide (resumable) ----
done = {}
if os.path.exists(EMBF):
    z = np.load(EMBF); done = {s: (b, d) for s, b, d in zip(z["slides"], z["base"], z["diag"])}; print("cached embeddings", len(done), flush=True)
todo = [s for s in slides_all if s not in done]; t0 = time.time()
for b0 in range(0, len(todo), a.batch):
    bs = todo[b0:b0 + a.batch]; batch = processor(tile_embeddings=[tiles(s, rng) for s in bs]).to(dev)
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): B = model.get_base_embedding(**batch).float().cpu().numpy(); Dg = model.get_diagnostic_embedding(**batch).float().cpu().numpy()
    for i, s in enumerate(bs): done[s] = (B[i], Dg[i])
    if (b0 // a.batch) % 100 == 0 or b0 + len(bs) == len(todo):
        print(f"  embeddings {b0 + len(bs)}/{len(todo)}, {time.time() - t0:.0f} s", flush=True); ks = sorted(done); np.savez(EMBF, slides=np.array(ks), base=np.stack([done[k][0] for k in ks]), diag=np.stack([done[k][1] for k in ks]))
# ---- 2. zero shot on test + held out project slides ----
if not a.skip_zeroshot:
    OUT = f"{paths.RUNS}/prism2_pan_zeroshot{sfx}"; paths.ensure(OUT); letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ZSOPS = set(a.zs_ops.split("+")) if a.zs_ops else (set(YESNO) | set(MCQ) | set(PERCLASS)); assert ZSOPS <= (set(YESNO) | set(MCQ) | set(PERCLASS)), ZSOPS
    Qz = Q[Q["op"].isin(ZSOPS) & Q["split"].isin(["test", "heldout_project"]) & Q["slide"].isin(slides_all)]; zs = sorted(Qz["slide"].unique()); t0 = time.time()
    # one line per finished batch in PART, so a job that hits its time limit loses one batch and the next job goes on from there
    PART = f"{OUT}/partial_{'+'.join(sorted(ZSOPS)) if a.zs_ops else 'all'}.jsonl"; rows, seen = load_partial(PART)
    if seen: print(f"resuming the zero shot: {len(seen & set(zs))} of {len(zs)} slides are in {PART}", flush=True)
    zs = [s for s in zs if s not in seen]; part = open(PART, "a")
    for b0 in range(0, len(zs), a.batch):
        bs = zs[b0:b0 + a.batch]; batch = processor(tile_embeddings=[tiles(s, rng) for s in bs]).to(dev); qb = Qz[Qz["slide"].isin(bs)]; n0 = len(rows)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for op in sorted(qb["op"].unique()):
                g = qb[qb["op"] == op]
                if op in PERCLASS:
                    qt, names = PERCLASS[op]
                    for cands_json, gg in g.groupby("candidates"):
                        cands = json.loads(cands_json); sub = [bs.index(s) for s in gg["slide"]]; sb = {"tile_embeddings": batch["tile_embeddings"][sub], "attention_mask": batch["attention_mask"][sub]}
                        sc = np.stack([model.yes_no_score(**sb, question=qt.format(name=names.get(c, c))).float().cpu().numpy() for c in cands], 1)      # [slides, candidates]
                        for j, r in enumerate(gg.itertuples()): rows.append(dict(qid=r.qid, slide=r.slide, patient=r.patient, split=r.split, op=op, atype=r.atype, candidates=cands_json, answer=int(r.answer), probs=json.dumps([float(x) for x in sc[j] / max(sc[j].sum(), 1e-9)])))
                elif op in YESNO:
                    p = model.yes_no_score(tile_embeddings=batch["tile_embeddings"], attention_mask=batch["attention_mask"], question=YESNO[op]).float().cpu().numpy()
                    for r in g.itertuples(): rows.append(dict(qid=r.qid, slide=r.slide, patient=r.patient, split=r.split, op=op, atype=r.atype, candidates=r.candidates, answer=int(r.answer), probs=json.dumps([float(1 - p[bs.index(r.slide)]), float(p[bs.index(r.slide)])])))
                else:
                    qtext, names = MCQ[op]
                    for cands_json, gg in g.groupby("candidates"):
                        cands = json.loads(cands_json); nm = names or {c: c.replace("_", " ") for c in cands}; prompt = qtext + "\nOptions:\n" + "\n".join(f"{letters[i]}. {nm.get(c, c)}" for i, c in enumerate(cands))
                        sub = [bs.index(s) for s in gg["slide"]]; sb = {"tile_embeddings": batch["tile_embeddings"][sub], "attention_mask": batch["attention_mask"][sub]}
                        ans = model.get_response(**sb, prompt=prompt, max_new_tokens=20)
                        for j, r in enumerate(gg.itertuples()):
                            txt = str(ans[j]).strip(); m = re.match(r"\s*\(?([A-Z])\b", txt); k = letters.index(m.group(1)) if m else -1
                            if k < 0: hits = [i for i, c in enumerate(cands) if nm.get(c, c).lower() in txt.lower()]; k = hits[0] if hits else -1
                            pr = [0.0] * len(cands)
                            if 0 <= k < len(cands): pr[k] = 1.0
                            rows.append(dict(qid=r.qid, slide=r.slide, patient=r.patient, split=r.split, op=op, atype=r.atype, candidates=cands_json, answer=int(r.answer), probs=json.dumps(pr), raw=txt[:80]))
        part.write(json.dumps(dict(slides=list(bs), rows=rows[n0:]), default=str) + "\n"); part.flush()
        if (b0 // a.batch) % 50 == 0: print(f"  zero shot {b0 + len(bs)}/{len(zs)} slides, {time.time() - t0:.0f} s", flush=True)
    part.close()
    P = pd.DataFrame(rows, columns=["qid", "slide", "patient", "split", "op", "atype", "candidates", "answer", "probs", "raw"])
    if a.zs_ops and os.path.exists(f"{OUT}/predictions_all.parquet"):
        P0 = pd.read_parquet(f"{OUT}/predictions_all.parquet"); P = pd.concat([P0[~P0["op"].isin(ZSOPS)], P], ignore_index=True); print(f"merged {len(P0)} saved rows with the new operators {sorted(ZSOPS)}: {len(P)} rows", flush=True)
    P.to_parquet(f"{OUT}/predictions_all.parquet", index=False); res = {"model": "PRISM2 zero shot (pan TCGA questions)", "max_tiles": a.max_tiles}; EPS = 1e-3
    os.remove(PART)                                                                # the answers are in predictions_all.parquet now
    for sp, key in (("test", "per_op"), ("heldout_project", "per_op_heldout_project")):
        res[key] = {}; Ps = P[P["split"] == sp]; Ps.to_parquet(f"{OUT}/predictions_{sp}.parquet", index=False)
        for op, g in Ps.groupby("op"):
            K = int(g["candidates"].map(lambda c: len(json.loads(c))).max()); probs = np.zeros((len(g), K))
            for i, p in enumerate(g["probs"]): v = json.loads(p); probs[i, :len(v)] = v
            y = g["answer"].to_numpy()
            if op in MCQ: probs = probs * (1 - K * EPS) + EPS
            m = M.summarize(probs, y, K, g["atype"].iloc[0], g["patient"].to_numpy())
            if op in YESNO and len(np.unique(y)) == 2: m["auroc"] = float(roc_auc_score(y, probs[:, 1]))
            res[key][op] = dict(family=FAMILY.get(op, "slide"), **m); print(f"  zs {sp:16s} {op:26s} n={m['n']:4d} acc={m['acc']:.3f} (maj {m['majority_acc']:.3f}) auroc={m.get('auroc', '')} nll={m['nll']:.3f}", flush=True)
    json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1); json.dump(vars(a), open(f"{OUT}/config.json", "w"), indent=1); paths.ensure(f"{paths.RESULTS}/runs/{os.path.basename(OUT)}"); os.system(f"cp {OUT}/*.json {paths.RESULTS}/runs/{os.path.basename(OUT)}/")
# ---- 3. probes on every operator ----
if a.skip_probe: sys.exit(0)
OUT = f"{paths.RUNS}/prism2_pan_probe{sfx}"; paths.ensure(OUT); z = np.load(EMBF); E = {"base": dict(zip(z["slides"], z["base"])), "diag": dict(zip(z["slides"], z["diag"]))}
pres = {"model": "PRISM2 linear probe on the pan TCGA questions (logistic regression, C chosen on val)", "per_op": {}, "per_op_heldout_project": {}, "per_op_diag": {}, "per_op_heldout_project_diag": {}}
Qp = Q[Q["slide"].isin(set(z["slides"]))]
for op, g in Qp.groupby("op"):
    g = g.assign(label=g.apply(lambda r: json.loads(r.candidates)[int(r.answer)], axis=1)); tr, va = g[g["split"] == "train"], g[g["split"] == "val"]
    if len(tr) < (4 if a.smoke else 20) or tr.label.nunique() < 2 or va.empty: continue
    cands_all = sorted(g.label.unique()); cidx = {c: i for i, c in enumerate(cands_all)}
    for emb_name in ("base", "diag"):
        Xtr, Xva = [np.stack([E[emb_name][s] for s in d["slide"]]) for d in (tr, va)]; ytr, yva = [d.label.map(cidx).to_numpy() for d in (tr, va)]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd; best = None
        for C in (0.001, 0.01, 0.1, 1.0):
            clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr); pv = clf.predict_proba(Xva); vl = -np.mean(np.log(np.clip(pv[np.arange(len(yva)), np.searchsorted(clf.classes_, yva)], 1e-9, 1)))
            if best is None or vl < best[0]: best = (vl, C, clf)
        clf = best[2]
        for sp, key in (("test", "per_op"), ("heldout_project", "per_op_heldout_project")):
            te = g[g["split"] == sp]
            if te.empty: continue
            Xte = (np.stack([E[emb_name][s] for s in te["slide"]]) - mu) / sd; K = len(cands_all); pr = np.full((len(te), K), 1e-6); pr[:, clf.classes_.astype(int)] = clf.predict_proba(Xte); pr /= pr.sum(1, keepdims=True)
            # answers are indexed by each row's own candidate list; map back to the global label index
            yte = te.label.map(cidx).to_numpy(); m = M.summarize(pr, yte, K, "noul" if K == 2 else g["atype"].iloc[0], te["patient"].to_numpy())
            if K == 2 and len(np.unique(yte)) == 2: m["auroc"] = float(roc_auc_score(yte, pr[:, 1]))
            m["C"] = best[1]; pres[key + ("" if emb_name == "base" else "_diag")][op] = dict(family=FAMILY.get(op, "slide"), **m)
            if emb_name == "base": print(f"  probe {sp:16s} {op:26s} n={m['n']:4d} acc={m['acc']:.3f} (maj {m['majority_acc']:.3f}) auroc={m.get('auroc', '')} nll={m['nll']:.3f} C={best[1]}", flush=True)
json.dump(pres, open(f"{OUT}/metrics_test.json", "w"), indent=1); json.dump(vars(a), open(f"{OUT}/config.json", "w"), indent=1); paths.ensure(f"{paths.RESULTS}/runs/{os.path.basename(OUT)}"); os.system(f"cp {OUT}/*.json {paths.RESULTS}/runs/{os.path.basename(OUT)}/")
print("RUN_DONE", OUT)
