#!/usr/bin/env python
"""Virchow2, the frozen encoder under Opsis, read out the usual way: linear probes on its features, one classifier per
operator trained with labels (logistic regression, features standardised on train, C in {0.001, 0.01, 0.1, 1} chosen by val
NLL, the protocol of the TITAN and PRISM2 probes of scripts/baselines/84_titan_pan.py, 83 and 88).
  slides: the mean of the 2,560 d Virchow2 tile vectors over up to --max-tiles random tiles (seed 0) of the slide, for the pan
          TCGA question set (runs/virchow2_pan_probe: per_op on the test split and per_op_heldout_project, composites included)
          and the breast set slide_questions_v2 (runs/virchow2_brca_probe, slide and molecular families, as scripts/baselines/88_fm_probes_brca.py).
  fields: the NuCLS final question set on the cached 448 px tokens (features/virchow2/tokens_448.h5): the question's box pooled
          exactly as the decoder's box_pool does (mean of the FOV and context tokens whose centre lies in the box, the global mean
          when none does), the second box the same or zeros, and the mean FOV token, 3,840 d. One probe per (operator, args)
          group, since a linear classifier has no argument slot; operators with a runtime candidate (cell_is, tissue_is) are
          skipped. runs/virchow2_fov_probe: per_op over the test questions of the groups that could be trained.
Mean pooled slide vectors are cached in features/virchow2_meanpool/embeddings.npz (resumable). CPU only.
    PYTHONPATH=src OPSIS_DATA=... python scripts/baselines/102_virchow2_probes.py [--workers 8] [--max-tiles 8192] [--skip-slides] [--skip-fov]"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, h5py
from multiprocessing import Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from opsis import paths, metrics as M
from opsis.questions import FAMILY
from opsis.dataset import FovGeometry
ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--max-tiles", type=int, default=8192)
ap.add_argument("--brca-h5", default=f"{paths.TCGA}/gigapath_h5/brca_trident_virchow2/h5_files")
ap.add_argument("--skip-slides", action="store_true"); ap.add_argument("--skip-fov", action="store_true"); ap.add_argument("--max-groups", type=int, default=48)
a = ap.parse_args(); MAXT = a.max_tiles; CS = (0.001, 0.01, 0.1, 1.0)


def fit(Xtr, ytr, Xva, yva):
    """standardise on train, pick C by val NLL; returns (clf, mu, sd, C)"""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd; best = None
    for C in CS:
        clf = LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr); pv = clf.predict_proba(Xva)
        vl = -np.mean(np.log(np.clip(pv[np.arange(len(yva)), np.searchsorted(clf.classes_, yva)], 1e-9, 1)))
        if best is None or vl < best[0]: best = (vl, C, clf)
    return best[2], mu, sd, best[1]


def predict(clf, mu, sd, X, K):
    pr = np.full((len(X), K), 1e-6); pr[:, clf.classes_.astype(int)] = clf.predict_proba((X - mu) / sd); return pr / pr.sum(1, keepdims=True)


def summarise(pr, y, K, atype, patients):
    m = M.summarize(pr, y, K, "noul" if K == 2 else atype, patients)
    if K == 2 and len(np.unique(y)) == 2: m["auroc"] = float(roc_auc_score(y, pr[:, 1]))
    return m


def save(name, pres):
    OUT = f"{paths.RUNS}/{name}"; paths.ensure(OUT, f"{paths.RESULTS}/runs/{name}")
    for d in (OUT, f"{paths.RESULTS}/runs/{name}"): json.dump(pres, open(f"{d}/metrics_test.json", "w"), indent=1)


# ---------------- slides ----------------
def meanpool(job):
    slide, path = job
    try:
        with h5py.File(path, "r") as h:
            n = h["features"].shape[0]
            if n > MAXT: idx = np.sort(np.random.default_rng(0).choice(n, MAXT, replace=False)); X = h["features"][idx]
            else: X = h["features"][:]
        return slide, X.astype(np.float32).mean(0)
    except Exception as e:
        print("  failed", slide, e, flush=True); return slide, None


def slide_probes(Q, feats, name, splits):
    pres = {"model": name, "per_op": {}, "per_op_heldout_project": {}}
    for op, g in Q.groupby("op"):
        g = g.assign(label=g.apply(lambda r: json.loads(r.candidates)[int(r.answer)], axis=1)); tr, va = g[g["split"] == "train"], g[g["split"] == "val"]
        if len(tr) < 20 or tr.label.nunique() < 2 or va.empty: continue
        cands_all = sorted(g.label.unique()); cidx = {c: i for i, c in enumerate(cands_all)}; K = len(cands_all)
        Xtr, Xva = [np.stack([feats[s] for s in x["slide"]]) for x in (tr, va)]; ytr, yva = [x.label.map(cidx).to_numpy() for x in (tr, va)]
        clf, mu, sd, C = fit(Xtr, ytr, Xva, yva)
        for sp, key in splits:
            te = g[g["split"] == sp]
            if te.empty: continue
            pr = predict(clf, mu, sd, np.stack([feats[s] for s in te["slide"]]), K); yte = te.label.map(cidx).to_numpy()
            m = summarise(pr, yte, K, g["atype"].iloc[0], te["patient"].to_numpy()); m["C"] = C; pres[key][op] = dict(family=FAMILY.get(op, "slide"), **m)
            if key == "per_op": print(f"  {name:20s} {op:26s} n={m['n']:5d} acc={m['acc']:.3f} (maj {m['majority_acc']:.3f}) auroc={m.get('auroc', '')} nll={m['nll']:.3f} C={C}", flush=True)
    save(name, pres)


if not a.skip_slides:
    QP = pd.read_parquet(f"{paths.DERIVED}/slide_questions_pan/questions_seed0.parquet")
    QB = pd.read_parquet(f"{paths.DERIVED}/slide_questions_v2/questions_seed0.parquet"); QB = QB[QB["family"].isin(["slide", "molecular"])]
    path_of = dict(zip(QP["slide"], QP["h5_path"]))
    for s in QB["slide"].unique(): path_of.setdefault(s, f"{a.brca_h5}/{s}.h5")
    EMB = f"{paths.FEATURES}/virchow2_meanpool"; paths.ensure(EMB); EMBF = f"{EMB}/embeddings.npz"; feats = {}
    if os.path.exists(EMBF): z = np.load(EMBF); feats = dict(zip(map(str, z["slides"]), z["mean"])); print("cached mean pooled slides", len(feats), flush=True)
    todo = [(s, p) for s, p in sorted(path_of.items()) if s not in feats and os.path.exists(p)]; print("slides to pool", len(todo), "of", len(path_of), flush=True); t0 = time.time()
    with Pool(a.workers) as pool:
        for i, (s, v) in enumerate(pool.imap_unordered(meanpool, todo, chunksize=4), 1):
            if v is not None: feats[s] = v
            if i % 500 == 0 or i == len(todo):
                ks = sorted(feats); np.savez(EMBF, slides=np.array(ks), mean=np.stack([feats[k] for k in ks])); print(f"  pooled {i}/{len(todo)}, {time.time() - t0:.0f} s", flush=True)
    slide_probes(QP[QP["slide"].isin(feats)], feats, "virchow2_pan_probe", (("test", "per_op"), ("heldout_project", "per_op_heldout_project")))
    slide_probes(QB[QB["slide"].isin(feats)], feats, "virchow2_brca_probe", (("test", "per_op"),))

# ---------------- fields ----------------
if not a.skip_fov:
    Q = pd.read_parquet(f"{paths.DERIVED}/questions_final/questions_seed0.parquet"); Q = Q[~Q["op"].isin(["cell_is", "tissue_is"])].reset_index(drop=True)
    F = pd.read_csv(f"{paths.DERIVED}/questions_final/fovs.csv").set_index("fov"); G = pd.read_csv(f"{paths.FEATURES}/virchow2/geometry_448.csv").set_index("fov")
    H = h5py.File(f"{paths.FEATURES}/virchow2/tokens_448.h5", "r"); names = {n.decode(): i for i, n in enumerate(H["fov"][:])}; size, ctx_px, grid = int(H.attrs["size"]), int(H.attrs["ctx_px"]), int(H.attrs["size"]) // int(H.attrs["patch"])
    miss = Q[~Q["fov"].isin(names)]; print(f"  {miss.fov.nunique()} FOVs ({len(miss)} questions, splits {miss.split.value_counts().to_dict()}) have no cached tokens and are dropped", flush=True); Q = Q[Q["fov"].isin(names)].reset_index(drop=True)
    X = np.zeros((len(Q), 3 * 1280), np.float32); t0 = time.time()
    for k, (f, idx) in enumerate(Q.groupby("fov").indices.items()):
        i = names[f]; tok = np.concatenate([H["tok_fov"][i], H["tok_ctx"][i]], 0).astype(np.float32); g = FovGeometry(F.loc[f], G.loc[f], size, ctx_px)
        pos = g.token_positions(grid); glob = tok.mean(0); mean_fov = tok[:grid * grid].mean(0)
        def pool(box):
            x0, y0, x1, y1 = g.box_features(json.loads(box))[:4]; inside = (pos[:, 0] >= x0) & (pos[:, 0] < x1) & (pos[:, 1] >= y0) & (pos[:, 1] < y1)
            return tok[inside].mean(0) if inside.any() else glob
        for j in idx:
            r = Q.iloc[j]; X[j, :1280] = pool(r.box); X[j, 1280:2560] = pool(r.box2) if r.box2 else 0.0; X[j, 2560:] = mean_fov
        if k % 200 == 0: print(f"  fields {k}/{Q.fov.nunique()}, {time.time() - t0:.0f} s", flush=True)
    Q["label"] = Q.apply(lambda r: json.loads(r.candidates)[int(r.answer)], axis=1)
    pres = {"model": "virchow2_fov_probe", "per_op": {}}
    for op, g in Q.groupby("op"):
        groups = list(g.groupby("args"))
        if len(groups) > a.max_groups: print(f"  skip {op}: {len(groups)} argument groups", flush=True); continue
        cands_all = sorted(g.label.unique()); cidx = {c: i for i, c in enumerate(cands_all)}; K = len(cands_all); PR, Y, PAT, trained = [], [], [], 0
        for args, gg in groups:
            tr, va, te = gg[gg["split"] == "train"], gg[gg["split"] == "val"], gg[gg["split"] == "test"]
            if len(tr) < 20 or tr.label.nunique() < 2 or va.empty or te.empty: continue
            clf, mu, sd, C = fit(X[tr.index], tr.label.map(cidx).to_numpy(), X[va.index], va.label.map(cidx).to_numpy()); trained += 1
            PR.append(predict(clf, mu, sd, X[te.index], K)); Y.append(te.label.map(cidx).to_numpy()); PAT.append(te["patient"].to_numpy())
        if not PR: continue
        pr, y, pat = np.concatenate(PR), np.concatenate(Y), np.concatenate(PAT); m = summarise(pr, y, K, g["atype"].iloc[0], pat)
        m["groups"] = trained; m["n_test_total"] = int((g["split"] == "test").sum()); pres["per_op"][op] = dict(family=FAMILY.get(op, g["family"].iloc[0]), **m)
        print(f"  fov probe {op:26s} n={m['n']:5d}/{m['n_test_total']} acc={m['acc']:.3f} (maj {m['majority_acc']:.3f}) nll={m['nll']:.3f} groups={trained}", flush=True)
    save("virchow2_fov_probe", pres)
print("done", flush=True)
# ---------------- the results table from whatever runs exist ----------------
L = ["# Virchow2 linear probes (generated by scripts/baselines/102_virchow2_probes.py)", "",
     "Virchow2, the frozen encoder under Opsis, read out the usual way: one logistic regression per operator trained with labels on our split (standardised, C in {0.001, 0.01, 0.1, 1} by val NLL, the protocol of the TITAN and PRISM2 probes). Slides: the mean of the 2,560 d tile vectors over up to 8,192 random tiles. Fields: the question's box pooled from the cached 448 px FOV and context tokens as the decoder's box_pool does, the second box or zeros, and the mean FOV token (3,840 d), one probe per (operator, args) group; n is the test questions of the groups that could be trained, of the n of the row.", ""]
for run, key, title in (("virchow2_fov_probe", "per_op", "NuCLS final question set, test split"), ("virchow2_brca_probe", "per_op", "TCGA-BRCA slide questions v2, test split"), ("virchow2_pan_probe", "per_op", "pan TCGA slide questions, test split"), ("virchow2_pan_probe", "per_op_heldout_project", "pan TCGA slide questions, held out projects")):
    f = f"{paths.RESULTS}/runs/{run}/metrics_test.json"
    if not os.path.exists(f): continue
    P = json.load(open(f))[key]; L += [f"## {title} ({run})", "", "| operator | n | majority | accuracy | NLL | ECE | AUROC | C | groups |", "|---|---|---|---|---|---|---|---|---|"]
    for op, m in sorted(P.items()): L.append(f"| {op} | {m['n']}{'/' + str(m['n_test_total']) if 'n_test_total' in m else ''} | {m['majority_acc']:.3f} | {m['acc']:.3f} | {m['nll']:.3f} | {m.get('ece', float('nan')):.3f} | {m.get('auroc', '')} | {m.get('C', '')} | {m.get('groups', '')} |")
    L.append("")
open(f"{paths.ROOT}/results/collected/RESULTS_virchow2_probes.md", "w").write("\n".join(L) + "\n")
