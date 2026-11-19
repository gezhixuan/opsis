#!/usr/bin/env python
"""Extra FOV level operators for the final version, appended to the v2 FOV question set:
  stil_score_bin   stromal TILs burden of the FOV: sum of the ellipse areas of sTIL nuclei lying in stroma, divided by the
                   stroma area of the FOV window (BCSS), binned 0 | (0, 1%] | (1%, 10%] | over 10% (TILs Working Group spirit,
                   nucleus boxes as the area proxy); valid when the FOV window has tissue labels and at least 5 % stroma.
  mitotic_present  Noul: at least one NuCLS mitotic figure inside the annotated rectangle.
  mitotic_count_bin Ordinal 0 | 1 | 2 or more mitotic figures in the rectangle.
    python scripts/questions/67_build_fov_extra.py [--seed 0] [--qset questions] [--out questions_final]
Writes $OPSIS_DATA/derived/<out>/questions_seed<k>.parquet (= the source set plus the new operators) and copies the split."""
import argparse, hashlib, json, os, shutil
import numpy as np, pandas as pd
from opsis import paths
from opsis.questions import FAMILY

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--qset", default="questions"); ap.add_argument("--out", default="questions_final"); a = ap.parse_args()
Q = pd.read_parquet(f"{paths.DERIVED}/{a.qset}/questions_seed{a.seed}.parquet"); F = pd.read_csv(f"{paths.DERIVED}/facts/fovs.csv").set_index("fov"); N = pd.read_csv(f"{paths.DERIVED}/facts/nuclei.csv"); R = pd.read_csv(f"{paths.DERIVED}/facts/regions.csv")
W = R[(R.scope == "fov") & (R.region == "whole")].set_index("fov")
meta = Q.drop_duplicates("fov").set_index("fov")[["slide", "patient", "split"]]
MPP = 0.2; rows = []
def emit(fov, op, atype, cands, ans, text, box, derivation, args=None):
    key = f"{fov}|{op}|{json.dumps(args or {}, sort_keys=True)}"; m = meta.loc[fov]
    rows.append(dict(fov=fov, slide=m.slide, patient=m.patient, family=FAMILY[op], op=op, args=json.dumps(args or {"region": "whole"}, sort_keys=True), atype=atype, candidates=json.dumps(cands), answer=int(ans), box=json.dumps(box), box2="", text=text, derivation=derivation, split=m.split, qid=hashlib.sha1(key.encode()).hexdigest()[:16]))
n_stil = n_mit = 0
for fov, g in N[N.in_rect].groupby("fov"):
    if fov not in meta.index: continue
    f = F.loc[fov]; rect = [float(f.rect_x0), float(f.rect_y0), float(f.rect_x1), float(f.rect_y1)]
    mit = int((g["raw"] == "mitotic_figure").sum())
    emit(fov, "mitotic_present", "noul", ["no", "yes"], int(mit >= 1), "Is at least one mitotic figure present in the annotated rectangle of the field?", rect, "nucls_raw")
    emit(fov, "mitotic_count_bin", "ordinal", ["0", "1", ">=2"], 0 if mit == 0 else 1 if mit == 1 else 2, "How many mitotic figures are in the annotated rectangle of the field?", rect, "nucls_raw"); n_mit += 1
    if fov in W.index and W.loc[fov, "valid_frac"] >= 0.8 and W.loc[fov, "frac_stroma"] >= 0.05:
        st = g[(g["super"] == "sTIL") & (g["tissue_class"].isin(["stroma", "inflamed_stroma"]))]
        area = float(((st.xmax - st.xmin) * (st.ymax - st.ymin) * np.pi / 4 * MPP * MPP).sum())            # um2, ellipse inside the box
        stroma = float(W.loc[fov, "area_um2"] * (W.loc[fov, "frac_stroma"] + W.loc[fov].get("frac_inflamed_stroma", 0.0)))
        frac = area / max(stroma, 1.0); b = 0 if frac == 0 else 1 if frac <= 0.01 else 2 if frac <= 0.10 else 3
        emit(fov, "stil_score_bin", "ordinal", ["0", "(0,1%]", "(1%,10%]", ">10%"], b, "What fraction of the stromal area of the field is occupied by lymphocytes or plasma cells (stromal TILs)?", rect, "nucls+bcss"); n_stil += 1
E = pd.DataFrame(rows); out = pd.concat([Q, E[Q.columns]], ignore_index=True); paths.ensure(f"{paths.DERIVED}/{a.out}")
out.to_parquet(f"{paths.DERIVED}/{a.out}/questions_seed{a.seed}.parquet", index=False)
for f in os.listdir(f"{paths.DERIVED}/{a.qset}"):
    if f.endswith(".json"): shutil.copy(f"{paths.DERIVED}/{a.qset}/{f}", f"{paths.DERIVED}/{a.out}/{f}")
print(f"added {len(E)} questions ({n_mit} FOVs with mitotic ops, {n_stil} with stil_score_bin) -> {len(out)} total in {a.out}")
for op, g in E.groupby("op"): print(f"  {op:18s} n={len(g):5d} answer dist {g.answer.value_counts().sort_index().to_dict()} | test n={int((g.split=='test').sum())}")
