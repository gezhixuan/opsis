#!/usr/bin/env python
"""NuCLS in the CellViT++ detection dataset layout, on OUR patient split: the repository's own
NuCLS loader (cellvit/training/datasets/nucls.py) reads <root>/<split>/images/<fov>.png and <root>/<split>/labels/<fov>.csv
with columns x, y, raw_classification, main_classification, super_classification, and selects slides by the first "_"
token of the file stem through a filelist csv. Images stay at the NuCLS native 0.2 mpp (as in the CellViT++ experiments), padded with white to 1024 x 1024 so that the loader can batch them.
    python scripts/baselines/59_cellvit_nucls_dataset.py [--seed 0] [--out $OPSIS_DATA/derived/cellvit_nucls]
Also writes train_configs/<backbone>.yaml for the classifier head (hyperparameters copied from the repository's NuCLS
super class logs) and the filelists splits/fold_0/{train,val}.csv (train / val patients); test patients go to test/."""
import argparse, os
import numpy as np, pandas as pd, yaml
from PIL import Image
PAD = 1024                                                                                                 # the Virchow CellViT accepts 256 or 1024 px inputs only; the largest NuCLS FOV is 830 x 798 px
from opsis import paths, splits

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
ap.add_argument("--repo", default=f"{paths.THIRD_PARTY}/CellViT-plus-plus"); ap.add_argument("--ckpt-dir", default=f"{paths.MODELS}/CellViT-plus-plus/checkpoints")
a = ap.parse_args(); OUT = a.out or f"{paths.DERIVED}/cellvit_nucls"
F = pd.read_csv(f"{paths.DERIVED}/facts/fovs.csv"); N = pd.read_csv(f"{paths.DERIVED}/facts/nuclei.csv")
sp = splits.read_split(f"{paths.DERIVED}/questions/split_seed{a.seed}.json"); fold = splits.fold_of(sp); F["split"] = F["patient"].map(fold)
F = F[F["split"].notna()]; print("FOVs per split", F["split"].value_counts().to_dict(), "| slides", F.groupby("split")["slide"].nunique().to_dict())
src = f"{paths.RAW}/nucls/NuCLS_dataset/rgb"; n_cells = {"train": 0, "test": 0}
for sub in ("train", "test"):
    for d in ("images", "labels"): paths.ensure(f"{OUT}/{sub}/{d}")
for r in F.itertuples():
    sub = "test" if r.split == "test" else "train"                                                      # CellViT++ draws train and val from train/ via filelists
    dst = f"{OUT}/{sub}/images/{r.fov}.png"                                                            # the repository stacks a batch of images, so every image is padded (white, bottom / right) to PAD x PAD
    if not os.path.exists(dst):
        im = Image.open(f"{src}/{r.fov}.png").convert("RGB"); canvas = Image.new("RGB", (PAD, PAD), (255, 255, 255)); canvas.paste(im, (0, 0)); canvas.save(dst)
    g = N[N["fov"] == r.fov]; g = g[(g["cx"] >= 0) & (g["cy"] >= 0) & (g["cx"] < r.W) & (g["cy"] < r.H)]
    pd.DataFrame({"x": g["cx"].round().astype(int), "y": g["cy"].round().astype(int), "raw_classification": g["raw"], "main_classification": g["main"], "super_classification": g["super"]}).to_csv(f"{OUT}/{sub}/labels/{r.fov}.csv", index=False)
    n_cells[sub] += len(g)
print("cells written", n_cells)
paths.ensure(f"{OUT}/splits/fold_0")
for k in ("train", "val"): pd.Series(sorted(F[F["split"] == k]["slide"].unique())).to_csv(f"{OUT}/splits/fold_0/{k}.csv", index=False, header=False)
yaml.safe_dump({0: "tumor_any", 1: "nonTIL_stromal", 2: "sTIL", 3: "other_nucleus"}, open(f"{OUT}/label_map.yaml", "w"))
# training configs: the repository's NuCLS super class hyperparameters per backbone (Virchow has no NuCLS log, SAM-H values are used)
logs = f"{a.repo}/logs/Classifiers/NuCLS/Clean-Dataset/Super"; paths.ensure(f"{OUT}/train_configs")
for backbone, ckpt, log in (("SAM-H", "SAM/CellViT-SAM-H-x40-AMP.pth", "SAM-H"), ("Virchow", "Virchow/CellViT-Virchow-x40-AMP.pth", "SAM-H"), ("ViT256", "HIPT-256/CellViT-256-x40-AMP.pth", "ViT256")):
    cfg = yaml.safe_load(open(f"{logs}/{log}/Fold-0/config.yaml"))
    cfg["data"].update(dataset="NuCLS", dataset_path=OUT, train_filelist=f"{OUT}/splits/fold_0/train.csv", val_filelist=f"{OUT}/splits/fold_0/val.csv")
    cfg["cellvit_path"] = f"{a.ckpt_dir}/{ckpt}"; cfg["logging"].update(log_dir=f"{paths.RUNS}/cellvit_nucls/{backbone}", wandb_dir=f"{paths.RUNS}/cellvit_nucls/wandb", mode="offline", log_comment=f"nucls_super_{backbone}")
    yaml.safe_dump(cfg, open(f"{OUT}/train_configs/{backbone}.yaml", "w"), sort_keys=False)
print("CELLVIT_NUCLS_DONE", OUT, "configs:", sorted(os.listdir(f"{OUT}/train_configs")))
