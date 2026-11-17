#!/usr/bin/env python
"""Build the fact tables (derived/facts/{slides,fovs,nuclei,regions}.csv) from NuCLS and the BCSS masks.
    python scripts/10_build_facts.py [--limit N] [--tissue bcss|tiger] [--workers 16]
Prints the nucleus class x tissue class cross-tab as an alignment check (tumour nuclei should sit in tumour tissue)."""
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, openslide
from opsis import paths, facts
from opsis.data import nucls, tissue, wsi

ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--tissue", default="bcss"); ap.add_argument("--workers", type=int, default=16)
a = ap.parse_args()
OUT = f"{paths.DERIVED}/facts"; paths.ensure(OUT)
names = nucls.list_fovs()
if a.limit: names = names[:: max(1, len(names) // a.limit)][: a.limit]
idx = wsi.slide_index(); hosp = nucls.hospital_of_slides()
slides = sorted({nucls.parse_name(n)["slide"] for n in names})
srows = []
for s in slides:
    p = idx.get(s)
    mpp = w = h = None
    if p:
        sl = openslide.OpenSlide(p); mpp = wsi.resolve_mpp(sl); w, h = sl.dimensions
    srows.append(dict(slide=s, patient=s[:12], hospital=hosp.get(s, ""), path=p or "", base_mpp=mpp, width=w, height=h))
S = pd.DataFrame(srows)
# a slide without resolution metadata: the NuCLS png scale gives base_mpp = 0.2 * (png px per level-0 px)
for i, r in S.iterrows():
    if r["path"] and not (isinstance(r["base_mpp"], float) and np.isfinite(r["base_mpp"])):
        n0 = next(n for n in names if nucls.parse_name(n)["slide"] == r["slide"]); m = nucls.parse_name(n0); W, H = nucls.image_size(n0)
        S.loc[i, "base_mpp"] = nucls.FOV_MPP * W / float(m["right"] - m["left"]); S.loc[i, "mpp_source"] = "nucls_scale"
S["mpp_source"] = S.get("mpp_source", pd.Series(index=S.index, dtype=object)).fillna("openslide")
S.to_csv(f"{OUT}/slides.csv", index=False); print(f"{len(S)} slides, {S['path'].ne('').sum()} in the store, mpp from NuCLS scale: {int((S['mpp_source'] == 'nucls_scale').sum())}", flush=True)
MPP = dict(zip(S["slide"], S["base_mpp"]))
ROIS = tissue.index_rois() if a.tissue == "tiger" else None
BROIS = tissue.index_bcss() if a.tissue == "bcss" else None
def work(n):
    meta = nucls.parse_name(n)
    return facts.fov_facts(n, ROIS, tissue_source=a.tissue, bcss_rois=BROIS, base_mpp=MPP.get(meta["slide"]))
F, NU, RE = [], [], []
with ProcessPoolExecutor(a.workers) as ex:
    for i, (fr, nu, re_) in enumerate(ex.map(work, names, chunksize=4)):
        F.append(fr); NU.append(nu); RE.append(re_)
        if i % 200 == 0: print(f"  {i}/{len(names)}", flush=True)
F = pd.DataFrame(F); NU = pd.concat(NU, ignore_index=True); RE = pd.concat(RE, ignore_index=True)
F["tissue_source"] = a.tissue
F.to_csv(f"{OUT}/fovs.csv", index=False); NU.to_csv(f"{OUT}/nuclei.csv", index=False); RE.to_csv(f"{OUT}/regions.csv", index=False)
print(f"fovs {len(F)}, nuclei {len(NU)} (in rect {int(NU['in_rect'].sum())}), regions {len(RE)}")
print(f"FOVs with tissue coverage >= {facts.TISSUE_MIN_COVER}: {int((F['tissue_cover'] >= facts.TISSUE_MIN_COVER).sum())} / {len(F)}; whole-region tissue_valid: {int(RE[RE.region == 'whole']['tissue_valid'].sum())}")
print("png size: W median", int(F["W"].median()), "H median", int(F["H"].median()), "| rect size median", int((F["rect_x1"] - F["rect_x0"]).median()), int((F["rect_y1"] - F["rect_y0"]).median()))
ct = pd.crosstab(NU.loc[NU["labelled"] & (NU["tissue_code"] != 0), "super"], NU.loc[NU["labelled"] & (NU["tissue_code"] != 0), "tissue_class"], normalize="index").round(2)
print("alignment check, P(tissue class | nucleus super class):"); print(ct.to_string())
