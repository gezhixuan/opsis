#!/usr/bin/env python
"""TITAN and PRISM2 zero shot on the official CAMELYON16 test set (129 slides, 49 with metastases): the slide foundation
models as comparables of the CAMELYON16 transfer row, with no training on the dataset. Features come from a TRIDENT run
on CAMELYON16 ($OPSIS_CAMELYON): 20x 256 px Virchow2 tiles (2,560 d, first 1,280 = class token, what
PRISM2 reads, as in scripts/baselines/58_prism2_zeroshot.py) and 20x 512 px CONCH v1.5 tiles pooled by the TITAN slide encoder (768 d, pre projection, as
the released TCGA embeddings used in scripts/baselines/84_titan_pan.py). PRISM2 is asked a yes/no question and scored with P(yes); TITAN scores the
slide embedding against text prompts for the negative and the positive class (templates of scripts/baselines/84_titan_pan.py). The first prompt of
each model is the one fixed before looking at the data and is the reported number; the others are listed for transparency.
    python scripts/baselines/101_camelyon_fm_zeroshot.py [--max-tiles 8192] [--smoke]
Writes results/runs/camelyon_prism2_zeroshot/metrics.json, results/runs/camelyon_titan_zeroshot/metrics.json and
results/collected/RESULTS_camelyon_fm_zeroshot.md."""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, h5py
from sklearn.metrics import roc_auc_score, average_precision_score
from opsis import paths
ap = argparse.ArgumentParser(); ap.add_argument("--max-tiles", type=int, default=8192); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--prism2", default=f"{paths.MODELS}/PRISM2"); ap.add_argument("--titan", default=f"{paths.MODELS}/TITAN")
ap.add_argument("--feat", default=f"{paths.CAMELYON}/trident_virchow2"); ap.add_argument("--skip", default="", help="comma list of models to skip: prism2,titan")
a = ap.parse_args(); DD = paths.CAMELYON; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S = pd.read_csv(f"{DD}/splits.csv"); S = S[(S.dataset == "c16") & (S.role == "T1")].copy(); S["y"] = S.label_official.isin(["micro", "macro"]).astype(int)
S = S.sort_values("slide").reset_index(drop=True)
if a.smoke: S = pd.concat([S[S.y == 0].head(3), S[S.y == 1].head(3)])
slides = S.slide.tolist(); y = S.y.to_numpy(); print(f"{len(slides)} slides, {int(y.sum())} positive", flush=True)
def report(scores):
    p = np.asarray(scores); return dict(auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)), acc=float(((p >= 0.5).astype(int) == y).mean()), n=int(len(y)), n_pos=int(y.sum()))
def write(name, res): d = f"{paths.RESULTS}/runs/{name}"; paths.ensure(d); json.dump(res, open(f"{d}/metrics.json", "w"), indent=1); print(name, json.dumps({k: v for k, v in res.items() if k != "per_slide"})[:600], flush=True)
skip = set(a.skip.split(",")) if a.skip else set(); out = {}
# ---------- PRISM2 ----------
PRISM_Q = ["Is metastatic carcinoma present in this lymph node?", "Is there a breast cancer metastasis in this sentinel lymph node?", "Are tumor cells present?"]
if "prism2" not in skip:
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(a.prism2, trust_remote_code=True, torch_dtype="auto").to(dev).eval(); processor = AutoProcessor.from_pretrained(a.prism2, trust_remote_code=True)
    rng = np.random.default_rng(0)
    def tiles(slide):
        with h5py.File(f"{a.feat}/20x_256px_0px_overlap/features_virchow2/{slide}.h5", "r") as h:
            n = h["features"].shape[0]
            if n > a.max_tiles: idx = np.sort(rng.choice(n, a.max_tiles, replace=False)); X = h["features"][idx, :1280]
            else: X = h["features"][:, :1280]
        return torch.from_numpy(X.astype(np.float32)), n
    sc = {q: [] for q in PRISM_Q}; ntiles = []; t0 = time.time()
    for b0 in range(0, len(slides), a.batch):
        bs = slides[b0:b0 + a.batch]; tt = [tiles(s) for s in bs]; ntiles += [t[1] for t in tt]; batch = processor(tile_embeddings=[t[0] for t in tt]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for q in PRISM_Q: sc[q] += model.yes_no_score(tile_embeddings=batch["tile_embeddings"], attention_mask=batch["attention_mask"], question=q).float().cpu().numpy().tolist()
        print(f"  prism2 {b0 + len(bs)}/{len(slides)} {time.time() - t0:.0f} s", flush=True)
    res = {"model": "PRISM2 zero shot, P(yes) of yes_no_score", "tiles": "Virchow2 class token (first 1,280 of 2,560 dims) of the 20x 256 px TRIDENT tiles, at most max_tiles sampled", "max_tiles": a.max_tiles, "mean_tiles": float(np.mean(ntiles)),
           "prompt": PRISM_Q[0], **report(sc[PRISM_Q[0]]), "other_prompts": {q: report(sc[q]) for q in PRISM_Q[1:]}, "per_slide": {s: {q: float(v) for q, v in zip(PRISM_Q, [sc[q][i] for q in PRISM_Q])} for i, s in enumerate(slides)}}
    write("camelyon_prism2_zeroshot", res); out["prism2"] = res
    del model; torch.cuda.empty_cache()
# ---------- TITAN ----------
TEMPL = ["a whole slide image of CLASSNAME.", "an H&E whole slide image showing CLASSNAME.", "histopathology of CLASSNAME."]
TITAN_C = [("lymph node without metastasis / lymph node with metastatic breast carcinoma", [["lymph node without metastasis"], ["lymph node with metastatic breast carcinoma"]]),
           ("normal lymph node / breast cancer metastasis in a lymph node", [["normal lymph node"], ["breast cancer metastasis in a lymph node"]]),
           ("benign lymphoid tissue / metastatic carcinoma", [["benign lymphoid tissue"], ["metastatic carcinoma"]])]
if "titan" not in skip:
    from transformers import AutoModel
    model = AutoModel.from_pretrained(a.titan, trust_remote_code=True).eval(); tdev = torch.device("cpu"); model = model.to(tdev)
    E = {}
    for s in slides:
        with h5py.File(f"{a.feat}/20x_512px_0px_overlap/slide_features_titan/{s}.h5", "r") as h: E[s] = np.asarray(h["features"][:]).reshape(-1)
    sc = {}
    with torch.no_grad():
        for name, cls in TITAN_C:
            clf = model.zero_shot_classifier(cls, TEMPL, device=tdev); sc[name] = [float(model.zero_shot(torch.from_numpy(E[s]).float(), clf)[0, 1]) for s in slides]
    res = {"model": "TITAN zero shot, softmax over the two class prompts (three templates, as scripts/baselines/84_titan_pan.py), P(positive class)", "embedding": "TRIDENT 20x 512 px CONCH v1.5 tiles pooled by the TITAN slide encoder, 768 d pre projection",
           "prompt": TITAN_C[0][0], **report(sc[TITAN_C[0][0]]), "other_prompts": {n: report(sc[n]) for n, _ in TITAN_C[1:]}, "per_slide": {s: {n: sc[n][i] for n, _ in TITAN_C} for i, s in enumerate(slides)}}
    write("camelyon_titan_zeroshot", res); out["titan"] = res
if not a.smoke and len(out) == 2:
    L = ["# TITAN and PRISM2 zero shot on the CAMELYON16 test set (generated by scripts/baselines/101_camelyon_fm_zeroshot.py)", "",
         f"Official test set, {len(slides)} slides, {int(y.sum())} with metastases (micro or macro). No training on CAMELYON16 for either model. Features from TRIDENT on CAMELYON16: PRISM2 reads the Virchow2 class tokens of the 20x 256 px tiles (at most {a.max_tiles} sampled per slide, mean {out['prism2']['mean_tiles']:.0f} tiles), TITAN reads its slide embedding pooled from the 20x 512 px CONCH v1.5 tiles. The first prompt of each model was fixed before scoring and is the reported number.", "",
         "| model | prompt | AUROC | AP | accuracy at 0.5 |", "|---|---|---|---|---|"]
    for key, nm in (("prism2", "PRISM2"), ("titan", "TITAN")):
        r = out[key]; L.append(f"| {nm} | {r['prompt']} | {r['auroc']:.3f} | {r['ap']:.3f} | {r['acc']:.3f} |")
        for q, rr in r["other_prompts"].items(): L.append(f"| {nm}, other prompt | {q} | {rr['auroc']:.3f} | {rr['ap']:.3f} | {rr['acc']:.3f} |")
    L += ["", "For the transfer row of Table 1: the Opsis field decoder's dense read of the same 129 slides gives 0.864 (results/runs/camelyon_dense_T1, top 5 mean of P(tumour cells) over all 80 um fields); the best challenge entry trained on the cohort reaches 0.994."]
    open(f"{paths.ROOT}/results/collected/RESULTS_camelyon_fm_zeroshot.md", "w").write("\n".join(L) + "\n"); print("\n".join(L[4:]))
