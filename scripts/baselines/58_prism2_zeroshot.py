#!/usr/bin/env python
"""PRISM2 (Paige, Virchow2 tiles + Perceiver + Phi-3) on our slide level questions: zero shot yes/no scoring, multiple
choice answers, and a linear probe on the slide embeddings, all on our patient split.
Tile input: PRISM2 wants Virchow2 class tokens of 224 px tiles at 0.5 mpp; our cache holds 256 px tiles at 0.5 mpp with
[class token, mean patch token] (2,560 d), so the first 1,280 dims are used (an approximation, noted in the results).
    python scripts/baselines/58_prism2_zeroshot.py [--splits test] [--mismatch] [--probe] [--max-tiles 8192] [--smoke]
Writes runs/prism2_zeroshot[_mismatch]/metrics_test.json (per_op in the format of scripts/opsis_s/51_train_slide.py) and embeddings.npz."""
import argparse, json, os, re, sys, time
import numpy as np, pandas as pd, torch, h5py
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from opsis import paths, metrics as M
from opsis.questions import FAMILY

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=f"{paths.MODELS}/PRISM2"); ap.add_argument("--splits", default="test"); ap.add_argument("--mismatch", action="store_true", help="control: each slide's tiles replaced by another slide's (probe of arXiv 2610.26805)")
ap.add_argument("--probe", action="store_true", help="also extract base/diagnostic embeddings for all slides and fit logistic regression probes on our split")
ap.add_argument("--max-tiles", type=int, default=8192); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--name", default=None)
ap.add_argument("--h5-dir", default=f"{paths.TCGA}/gigapath_h5/brca_trident_virchow2/h5_files")
a = ap.parse_args(); dev = torch.device("cuda")
name = a.name or ("prism2_zeroshot" + ("_mismatch" if a.mismatch else "") + ("_smoke" if a.smoke else "")); OUT = f"{paths.RUNS}/{name}"; paths.ensure(OUT); RES = f"{paths.RESULTS}/runs/{name}"; paths.ensure(RES)
Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_v2/questions_seed0.parquet"); Q = Q[Q["family"].isin(["slide", "molecular"])]
# question phrasing for PRISM2 (yes/no for binary operators, the model card's multiple choice format for the others)
YESNO = {"er_positive": "Is the tumor estrogen receptor (ER) positive?", "pr_positive": "Is the tumor progesterone receptor (PR) positive?", "her2_positive": "Is the tumor HER2 positive?",
         "in_situ_present": "Is ductal carcinoma in situ present?", "lvi_present": "Is lymphovascular invasion present?", "necrosis_present": "Is tumor necrosis present in the invasive carcinoma?",
         "multifocal": "Is the invasive carcinoma multifocal?", "pni_present": "Is perineural invasion present?", "lcis_present": "Is lobular carcinoma in situ present?", "calcifications": "Are calcifications present?",
         "dcis_comedo_necrosis": "Does the ductal carcinoma in situ show comedo necrosis?"}
MCQ = {"histologic_type": ("What is the histologic type of the invasive carcinoma?", {"ductal": "Invasive ductal carcinoma, no special type", "lobular": "Invasive lobular carcinoma", "mixed": "Mixed invasive ductal and lobular carcinoma", "other": "Another special type of carcinoma"}),
       "histologic_grade": ("What is the Nottingham histologic grade of the invasive carcinoma?", {"1": "Grade 1", "2": "Grade 2", "3": "Grade 3"}),
       "nuclear_grade": ("What is the nuclear grade of the invasive carcinoma?", {"1": "Grade 1", "2": "Grade 2", "3": "Grade 3"}),
       "tubule_score": ("What is the Nottingham tubule formation score?", {"1": "Score 1", "2": "Score 2", "3": "Score 3"}),
       "pleomorphism_score": ("What is the Nottingham nuclear pleomorphism score?", {"1": "Score 1", "2": "Score 2", "3": "Score 3"}),
       "mitotic_score": ("What is the Nottingham mitotic count score?", {"1": "Score 1", "2": "Score 2", "3": "Score 3"}),
       "dcis_grade": ("What is the nuclear grade of the ductal carcinoma in situ?", {"low": "Low grade", "intermediate": "Intermediate grade", "high": "High grade"}),
       "pam50": ("What is the PAM50 molecular subtype of the tumor?", {"LumA": "Luminal A", "LumB": "Luminal B", "Basal": "Basal-like", "Her2": "HER2-enriched", "Normal": "Normal-like"})}
keep = set(YESNO) | set(MCQ); Q = Q[Q["op"].isin(keep) & Q["split"].isin(a.splits.split(","))]
slides = sorted(Q["slide"].unique())
if a.smoke: slides = slides[:6]; Q = Q[Q["slide"].isin(slides)]
print(f"{name}: {len(slides)} slides, {len(Q)} questions, ops {sorted(Q['op'].unique())}", flush=True)
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained(a.model, trust_remote_code=True, torch_dtype="auto").to(dev).eval(); processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
def tiles(slide, rng):
    with h5py.File(f"{a.h5_dir}/{slide}.h5", "r") as h:
        n = h["features"].shape[0]
        if n > a.max_tiles: idx = np.sort(rng.choice(n, a.max_tiles, replace=False)); X = h["features"][idx, :1280]
        else: X = h["features"][:, :1280]
    return torch.from_numpy(X.astype(np.float32))
rng = np.random.default_rng(0); src = {s: s for s in slides}
if a.mismatch: perm = list(slides); rng.shuffle(perm); src = {s: p for s, p in zip(slides, perm)}
letters = "ABCDEFGHIJ"; rows = []; t0 = time.time()
for b0 in range(0, len(slides), a.batch):
    bs = slides[b0:b0 + a.batch]; emb = [tiles(src[s], rng) for s in bs]; batch = processor(tile_embeddings=emb).to(dev)
    qb = Q[Q["slide"].isin(bs)]
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for op in sorted(qb["op"].unique()):
            g = qb[qb["op"] == op]
            if op in YESNO:
                p = model.yes_no_score(tile_embeddings=batch["tile_embeddings"], attention_mask=batch["attention_mask"], question=YESNO[op]).float().cpu().numpy()
                for r in g.itertuples(): rows.append(dict(qid=r.qid, slide=r.slide, patient=r.patient, op=op, atype=r.atype, candidates=r.candidates, answer=int(r.answer), probs=json.dumps([float(1 - p[bs.index(r.slide)]), float(p[bs.index(r.slide)])]), raw=""))
            else:
                qtext, names = MCQ[op]; cands = json.loads(g["candidates"].iloc[0]); prompt = qtext + "\nOptions:\n" + "\n".join(f"{letters[i]}. {names[c]}" for i, c in enumerate(cands))
                ans = model.get_response(**batch, prompt=prompt, max_new_tokens=20)
                for r in g.itertuples():
                    txt = str(ans[bs.index(r.slide)]).strip(); m = re.match(r"\s*\(?([A-J])\b", txt); k = letters.index(m.group(1)) if m else -1
                    if k < 0:                                                           # fall back to matching the option text
                        hits = [i for i, c in enumerate(cands) if names[c].lower() in txt.lower()]; k = hits[0] if hits else -1
                    pr = [0.0] * len(cands)
                    if 0 <= k < len(cands): pr[k] = 1.0
                    rows.append(dict(qid=r.qid, slide=r.slide, patient=r.patient, op=op, atype=r.atype, candidates=r.candidates, answer=int(r.answer), probs=json.dumps(pr), raw=txt[:80]))
    print(f"  {b0 + len(bs)}/{len(slides)} slides, {time.time() - t0:.0f} s", flush=True)
P = pd.DataFrame(rows); P.to_parquet(f"{OUT}/predictions_test.parquet", index=False)
res = {"model": "PRISM2", "mismatch": a.mismatch, "tiles": "Virchow2 class token, first 1,280 dims of the 256 px cache (PRISM2 expects 224 px)", "max_tiles": a.max_tiles, "per_op": {}}
EPS = 1e-3
for op, g in P.groupby("op"):
    K = len(json.loads(g["candidates"].iloc[0])); probs = np.stack([np.array(json.loads(p)) for p in g["probs"]]); y = g["answer"].to_numpy()
    if op in MCQ: probs = probs * (1 - K * EPS) + EPS                                    # one hot answers: smooth so that NLL is finite
    m = M.summarize(probs, y, K, g["atype"].iloc[0], g["patient"].to_numpy())
    if op in YESNO and len(np.unique(y)) == 2: m["auroc"] = float(roc_auc_score(y, probs[:, 1])); m["ap"] = float(average_precision_score(y, probs[:, 1]))
    m["unparsed"] = int((probs.max(1) < 0.5).sum()) if op in MCQ else 0
    res["per_op"][op] = dict(family=FAMILY[op], **m)
    print(f"  {op:22s} n={m['n']:4d} acc={m['acc']:.3f} (maj {m['majority_acc']:.3f}) f1={m['macro_f1']:.3f} auroc={m.get('auroc', '')} nll={m['nll']:.3f} unparsed={m['unparsed']}", flush=True)
for fam in ["slide", "molecular"]:
    ops = [o for o in res["per_op"] if FAMILY[o] == fam]
    if ops: res[f"mean_acc_{fam}"] = float(np.mean([res["per_op"][o]["acc"] for o in ops]))
json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1); json.dump(vars(a), open(f"{OUT}/config.json", "w"), indent=1)
for f in ("config.json", "metrics_test.json"): os.system(f"cp {OUT}/{f} {RES}/{f}")

if a.probe:                                                                             # slide embeddings for every labelled slide, then logistic regression on our split
    L = pd.read_csv(f"{paths.DERIVED}/slide_questions_v2/slide_labels_seed0.csv"); allslides = L["slide"].tolist()
    if a.smoke: allslides = allslides[:12]; L = L[L["slide"].isin(allslides)]
    base, diag = {}, {}; t0 = time.time()
    for b0 in range(0, len(allslides), a.batch):
        bs = allslides[b0:b0 + a.batch]; batch = processor(tile_embeddings=[tiles(s, rng) for s in bs]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            B = model.get_base_embedding(**batch).float().cpu().numpy(); Dg = model.get_diagnostic_embedding(**batch).float().cpu().numpy()
        for i, s in enumerate(bs): base[s] = B[i]; diag[s] = Dg[i]
        if (b0 // a.batch) % 25 == 0: print(f"  embeddings {b0 + len(bs)}/{len(allslides)}, {time.time() - t0:.0f} s", flush=True)
    np.savez(f"{OUT}/embeddings.npz", slides=np.array(allslides), base=np.stack([base[s] for s in allslides]), diag=np.stack([diag[s] for s in allslides]))
    pres = {"model": "PRISM2 linear probe (logistic regression, C chosen on val)", "per_op": {}}
    for emb_name, E in (("base", base), ("diag", diag)):
        for op in ["er_positive", "pr_positive", "her2_positive", "pam50", "histologic_grade", "histologic_type"]:
            Lo = L[L[op].notna()]; tr, va, te = [Lo[Lo["split"] == s] for s in ("train", "val", "test")]
            if a.smoke and (len(tr) < 4 or te.empty): continue
            Xtr, Xva, Xte = [np.stack([E[s] for s in d["slide"]]) for d in (tr, va, te)]; ytr, yva, yte = [d[op].astype(int).to_numpy() for d in (tr, va, te)]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
            best = None
            for C in (0.001, 0.01, 0.1, 1.0):
                clf = LogisticRegression(C=C, max_iter=2000, class_weight=None).fit(Xtr, ytr); vl = -np.mean(np.log(np.clip(clf.predict_proba(Xva)[np.arange(len(yva)), np.searchsorted(clf.classes_, yva)], 1e-9, 1)))
                if best is None or vl < best[0]: best = (vl, C, clf)
            clf = best[2]; K = len(json.loads(Q[Q.op == op]["candidates"].iloc[0])) if (Q.op == op).any() else int(L[op].max()) + 1
            pr = np.zeros((len(yte), K)); pr[:, clf.classes_.astype(int)] = clf.predict_proba(Xte)
            m = M.summarize(pr, yte, K, "noul" if K == 2 else "choice", te["patient"].to_numpy())
            if K == 2: m["auroc"] = float(roc_auc_score(yte, pr[:, 1])); m["ap"] = float(average_precision_score(yte, pr[:, 1]))
            else:
                try: m["auroc"] = float(roc_auc_score(yte, pr, multi_class="ovr", average="macro", labels=list(range(K))))
                except ValueError: m["auroc"] = None
            m["C"] = best[1]; pres["per_op"][f"{op}@{emb_name}"] = dict(family=FAMILY.get(op, "slide"), **m)
            print(f"  probe {emb_name:4s} {op:18s} n={m['n']:4d} acc={m['acc']:.3f} (maj {m['majority_acc']:.3f}) f1={m['macro_f1']:.3f} auroc={m.get('auroc')} nll={m['nll']:.3f} C={best[1]}", flush=True)
    pdir = f"{paths.RUNS}/prism2_probe" + ("_smoke" if a.smoke else ""); paths.ensure(pdir); json.dump(pres, open(f"{pdir}/metrics_test.json", "w"), indent=1); paths.ensure(f"{paths.RESULTS}/runs/{os.path.basename(pdir)}"); os.system(f"cp {pdir}/metrics_test.json {paths.RESULTS}/runs/{os.path.basename(pdir)}/")
print("RUN_DONE", OUT)
