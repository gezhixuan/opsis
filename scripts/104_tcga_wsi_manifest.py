#!/usr/bin/env python
"""Manifest for downloading the TCGA diagnostic slides from the GDC. Lists every slide of the pan registry (slide_questions_pan, 10,712 slides,
the BRCA v2 set is a subset) plus the BCSS source slides, looks each file name up in the GDC files endpoint (file UUID, size,
md5) and writes $OPSIS_WSI/manifests/all.csv with a download tier per slide:
    1  test split (1,393, the organiser sets are inside it) and the BCSS slides
    2  held out projects (1,145)
    3  val split (1,402)
    4  train split (6,772)
The tile features of every split are extracted with TRIDENT, which needs all four tiers.
and one gdc-client style manifest per tier (manifests/tier<k>.txt). Downloads are done by scripts/105.
    python scripts/104_tcga_wsi_manifest.py [--out $OPSIS_WSI]"""
import argparse, glob, json, os, sys, time, urllib.request
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from opsis import paths
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=paths.WSI); ap.add_argument("--bcss-run", default="final_tile_noisyor_s0"); ap.add_argument("--chunk", type=int, default=200)
a = ap.parse_args(); M = f"{a.out}/manifests"; os.makedirs(M, exist_ok=True)

Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_pan/questions_seed0.parquet")
S = Q.drop_duplicates("slide")[["slide", "project", "split"]].reset_index(drop=True)
bcss = {os.path.basename(f)[:-4] for f in glob.glob(f"{paths.RUNS}/{a.bcss_run}/tile_scores/*.npz")}
extra = sorted(bcss - set(S.slide))
if extra: S = pd.concat([S, pd.DataFrame(dict(slide=extra, project="BRCA", split="bcss_only"))], ignore_index=True)
TIER = {"test": 1, "heldout_project": 2, "val": 3, "train": 4, "bcss_only": 1}
S["tier"] = [1 if s in bcss else TIER[sp] for s, sp in zip(S.slide, S.split)]
S["bcss"] = S.slide.isin(bcss)
print(f"{len(S)} slides: {S.split.value_counts().to_dict()}, BCSS slides {len(bcss)} ({len(extra)} outside the registry)", flush=True)

API = "https://api.gdc.cancer.gov/files"
def lookup(names):
    body = dict(filters=dict(op="in", content=dict(field="file_name", value=[n + ".svs" for n in names])),
                fields="file_id,file_name,md5sum,file_size,state,access,cases.project.project_id", size=len(names) + 10, format="json")
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r: return json.load(r)["data"]["hits"]
        except Exception as e: print("gdc api error", repr(e)[:120], flush=True); time.sleep(5 * (attempt + 1))
    raise RuntimeError("gdc api failed")
rows = {}; t0 = time.time(); names = list(S.slide)
for i in range(0, len(names), a.chunk):
    for h in lookup(names[i:i + a.chunk]):
        rows[h["file_name"][:-4]] = dict(file_id=h["file_id"], size=int(h["file_size"]), md5=h.get("md5sum", ""), state=h.get("state", ""), access=h.get("access", ""),
                                         gdc_project=(h.get("cases") or [{}])[0].get("project", {}).get("project_id", ""))
    print(f"  {min(i + a.chunk, len(names))}/{len(names)} looked up, {len(rows)} found, {time.time() - t0:.0f} s", flush=True)
R = pd.DataFrame.from_dict(rows, orient="index"); S = S.join(R, on="slide")
missing = S[S.file_id.isna()]; print(f"not found in the GDC: {len(missing)}", flush=True)
if len(missing): print(missing[["slide", "project", "split"]].head(20).to_string(index=False))
S.to_csv(f"{M}/all.csv", index=False)
for k, g in S[S.file_id.notna()].groupby("tier"):
    g[["file_id", "slide", "md5", "size"]].assign(filename=g.slide + ".svs", state="released")[["file_id", "filename", "md5", "size", "state"]].rename(columns={"file_id": "id"}).to_csv(f"{M}/tier{k}.txt", sep="\t", index=False)
    print(f"tier {k}: {len(g)} slides, {g['size'].sum() / 1e12:.2f} TB, {g.project.nunique()} projects, controlled access {int((g.access == 'controlled').sum())}", flush=True)
print(f"total {S['size'].sum() / 1e12:.2f} TB; manifests in {M}")
