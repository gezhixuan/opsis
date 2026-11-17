#!/usr/bin/env python
"""Tile level evaluation of the per tile scores of scripts/69 against the BCSS region masks (TCGA-BRCA, 151 ROIs at 0.25 MPP,
class codes in raw/bcss/meta/gtruth_codes.tsv). For every 20x tile inside a BCSS ROI the fraction of each mask class under the
tile is computed; a tile is necrotic when necrosis_or_debris covers >= --frac of it. Reports the tile level AUROC and average
precision of the necrosis logit (slide trained on report labels only, no tile supervision), split by the slide split of the run,
and the same for the other heads as a specificity check (LVI / PNI / precursor logits should not track necrosis).
    python scripts/71_eval_tile_necrosis.py --run final_tile_noisyor_s0 [--frac 0.5]
Writes runs/<run>/tile_eval_bcss.json and markdown tables to stdout."""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from opsis import paths
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--frac", type=float, default=0.5)
ap.add_argument("--bcss", default=paths.BCSS); a = ap.parse_args()
Image.MAX_IMAGE_PIXELS = None
codes = pd.read_csv(f"{a.bcss}/meta/gtruth_codes.tsv", sep="\t"); code = dict(zip(codes["label"], codes["GT_code"])); NEC = code["necrosis_or_debris"]
B = pd.read_csv(f"{a.bcss}/meta/roiBounds_BaseMagnification.csv"); B = B.rename(columns={B.columns[0]: "roi"}); B["stem"] = B["roi"].str.split("_").str[0]
TS = f"{paths.RUNS}/{a.run}/tile_scores"; files = {re.sub(r"-01Z-00-", "-", os.path.basename(f).split(".")[0]): f for f in glob.glob(f"{TS}/*.npz")}
rows = []
for r in B.itertuples():
    if r.stem not in files: continue
    z = np.load(files[r.stem]); L = z["logits"].astype(np.float32); C = z["coords_level0"]; ps = float(z["patch_size_level0"]); ops = list(z["ops"]); split = str(z["split"])
    mp = glob.glob(f"{a.bcss}/masks/{r.roi}*.png") or glob.glob(f"{a.bcss}/masks/{r.stem}_xmin{r.xmin}_ymin{r.ymin}*.png")
    if not mp: continue
    m = np.array(Image.open(mp[0])); sx = (r.xmax - r.xmin) / m.shape[1]; sy = (r.ymax - r.ymin) / m.shape[0]      # level 0 px per mask px
    inside = (C[:, 0] >= r.xmin) & (C[:, 0] + ps <= r.xmax) & (C[:, 1] >= r.ymin) & (C[:, 1] + ps <= r.ymax)
    for i in np.flatnonzero(inside):
        x0 = int((C[i, 0] - r.xmin) / sx); y0 = int((C[i, 1] - r.ymin) / sy); x1 = int((C[i, 0] + ps - r.xmin) / sx); y1 = int((C[i, 1] + ps - r.ymin) / sy)
        sub = m[y0:y1, x0:x1]
        if sub.size == 0: continue
        fr = {lab: float((sub == c).mean()) for lab, c in code.items() if lab != "outside_roi"}
        rows.append(dict(stem=r.stem, roi=r.roi, split=split, tile=int(i), **{f"frac_{k}": v for k, v in fr.items()}, **{f"logit_{op}": float(L[i, j]) for j, op in enumerate(ops)}))
T = pd.DataFrame(rows); print(f"tiles inside ROIs: {len(T)} on {T.stem.nunique()} slides; split counts {T.groupby('split').stem.nunique().to_dict()}")
T["necrotic"] = T["frac_necrosis_or_debris"] >= a.frac; res = {"frac": a.frac, "n_tiles": len(T), "n_slides": int(T.stem.nunique())}
out = []
for sp, g in [("all", T)] + [(s, g) for s, g in T.groupby("split")] + [("val+test", T[T.split.isin(["val", "test"])])]:
    y = g["necrotic"].to_numpy()
    if len(g) == 0 or y.sum() == 0 or (~y).sum() == 0: continue
    rec = dict(split=sp, n_slides=int(g.stem.nunique()), n_tiles=len(g), pos_rate=float(y.mean()))
    for op in [c[6:] for c in T.columns if c.startswith("logit_")]:
        rec[f"auroc_{op}"] = float(roc_auc_score(y, g[f"logit_{op}"])); rec[f"ap_{op}"] = float(average_precision_score(y, g[f"logit_{op}"]))
    out.append(rec)
R = pd.DataFrame(out); res["table"] = out; print(R.round(3).to_string(index=False))
# what the necrosis head fires on: mean logit by dominant mask class
T["dominant"] = T[[c for c in T.columns if c.startswith("frac_")]].idxmax(1).str[5:]
D = T.groupby("dominant")["logit_pan_necrosis"].agg(["count", "mean"]).sort_values("mean", ascending=False); print(D.round(2)); res["necrosis_logit_by_dominant_class"] = D.round(3).to_dict()
json.dump(res, open(f"{paths.RUNS}/{a.run}/tile_eval_bcss.json", "w"), indent=1); T.to_parquet(f"{paths.RUNS}/{a.run}/tile_eval_bcss.parquet", index=False); print("written", f"{paths.RUNS}/{a.run}/tile_eval_bcss.json")
