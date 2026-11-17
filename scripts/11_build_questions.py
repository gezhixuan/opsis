#!/usr/bin/env python
"""Patient split, area quartiles from the training split, then questions for every FOV.
    python scripts/11_build_questions.py [--seed 0] [--n-cells 12]
Writes derived/questions/split_seed<k>.json and derived/questions/questions_seed<k>.parquet (one row per valid question)."""
import argparse, json, os
import numpy as np, pandas as pd
from opsis import paths, splits
from opsis.questions import QuestionBuilder, OPS

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n-cells", type=int, default=12); a = ap.parse_args()
FD = f"{paths.DERIVED}/facts"; OUT = f"{paths.DERIVED}/questions"; paths.ensure(OUT)
F = pd.read_csv(f"{FD}/fovs.csv"); NU = pd.read_csv(f"{FD}/nuclei.csv"); RE = pd.read_csv(f"{FD}/regions.csv")
NU["tissue_class"] = NU["tissue_class"].fillna(""); NU["compartment"] = NU["compartment"].fillna("")
sp = splits.split_patients(F["patient"].unique(), seed=a.seed); splits.write_split(sp, f"{OUT}/split_seed{a.seed}.json")
fold = splits.fold_of(sp); F["split"] = F["patient"].map(fold)
train_fovs = set(F.loc[F["split"] == "train", "fov"])
tr_area = NU[NU["fov"].isin(train_fovs) & NU["polygon"] & NU["in_rect"]]["area_um2"].dropna()
area_q = np.quantile(tr_area, [0.25, 0.5, 0.75]).tolist(); print("area quartiles (um2, train polygons):", np.round(area_q, 1))
json.dump({"area_q": area_q, "seed": a.seed, "n_cells": a.n_cells}, open(f"{OUT}/bins_seed{a.seed}.json", "w"))
rows = []
for fr in F.to_dict("records"):
    nu = NU[NU["fov"] == fr["fov"]].set_index("nucleus_id", drop=False)
    re_ = RE[RE["fov"] == fr["fov"]]
    qs = QuestionBuilder(fr, nu, re_, area_q, n_cells=a.n_cells).build()
    for q in qs: q["split"] = fr["split"]
    rows += qs
Q = pd.DataFrame(rows); Q.to_parquet(f"{OUT}/questions_seed{a.seed}.parquet", index=False)
print(f"{len(Q)} questions over {Q['fov'].nunique()} FOVs; per split:", Q["split"].value_counts().to_dict())
print(Q.groupby(["family", "op"]).size().to_string())
print("answer balance (choice/ordinal majority share):")
for op, g in Q.groupby("op"):
    vc = g["answer"].value_counts(normalize=True); print(f"  {op:26s} n={len(g):6d} majority={vc.iloc[0]:.2f} classes={len(vc)}")
