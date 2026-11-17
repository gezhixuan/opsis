#!/usr/bin/env python
"""Context crops around every FOV from the TCGA slides: derived/context/mpp<m>_<px>/<fov>.png.
    python scripts/12_build_contexts.py --mpp 1.0 --px 448 [--workers 12]"""
import argparse, os
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from opsis import paths, facts
from opsis.data import nucls, wsi

ap = argparse.ArgumentParser(); ap.add_argument("--mpp", type=float, default=facts.CTX_MPP); ap.add_argument("--px", type=int, default=facts.CTX_PX); ap.add_argument("--workers", type=int, default=12); a = ap.parse_args()
OUT = f"{paths.DERIVED}/context/mpp{a.mpp}_{a.px}"; paths.ensure(OUT)
S = pd.read_csv(f"{paths.DERIVED}/facts/slides.csv"); P = dict(zip(S["slide"], S["path"].fillna(""))); MPP = dict(zip(S["slide"], S["base_mpp"]))
names = nucls.list_fovs(); by_slide = {}
for n in names: by_slide.setdefault(nucls.parse_name(n)["slide"], []).append(n)
def work(slide):
    p = P.get(slide, "")
    if not p: return (slide, 0, "no slide")
    r = wsi.SlideReader(p); k = 0
    if not (r.mpp == r.mpp): r.mpp = float(MPP[slide])          # NaN -> value estimated from the NuCLS scale
    if not (r.mpp == r.mpp): return (slide, 0, "no mpp")
    for n in by_slide[slide]:
        o = f"{OUT}/{n}.png"
        if os.path.exists(o): k += 1; continue
        m = nucls.parse_name(n); cx = (m["left"] + m["right"]) / 2.0; cy = (m["top"] + m["bottom"]) / 2.0
        r.read_context(cx, cy, a.mpp, a.px).save(o + ".tmp.png"); os.replace(o + ".tmp.png", o); k += 1
    return (slide, k, "ok")
with ProcessPoolExecutor(a.workers) as ex:
    res = list(ex.map(work, sorted(by_slide)))
print("slides done", sum(1 for r in res if r[2] == "ok"), "fovs", sum(r[1] for r in res), "missing slides", [r[0] for r in res if r[2] != "ok"])
