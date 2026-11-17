"""NuCLS corrected single-rater set: FOV images at 0.2 mpp, nucleus csv per FOV, mask with the annotated FOV rectangle.

File name: <slide>_id-<fov id>_left-<L>_top-<T>_bottom-<B>_right-<R>, with L/T/B/R in the level-0 frame of the TCGA slide.
The png has W = (R - L) * base_mpp / 0.2 pixels (verified against the TCGA slides in $OPSIS_WSI, mean abs diff 1.0 grey levels).
Mask channel 0: 253 marks the intended annotation region (a rectangle), nucleus pixels carry class codes, 0 is the margin
outside that rectangle. Nuclei in the margin are annotated only partially, so region facts use the rectangle only.
"""
import os
import re

import numpy as np
import pandas as pd
from PIL import Image

from opsis import paths

FOV_MPP = 0.2
FOV_CODE = 253
NAME_RE = re.compile(r"^(TCGA-\w\w-\w{4})-(DX\d)_id-(\w+)_left-(\d+)_top-(\d+)_bottom-(\d+)_right-(\d+)$")

SUPER = ["tumor_any", "nonTIL_stromal", "sTIL", "other_nucleus"]           # AMBIGUOUS is kept out of class facts
SUPER_SHORT = {"tumor_any": "tumor", "nonTIL_stromal": "stromal", "sTIL": "sTIL", "other_nucleus": "other"}
MAIN = ["tumor_nonMitotic", "tumor_mitotic", "nonTILnonMQ_stromal", "macrophage", "lymphocyte", "plasma_cell", "other_nucleus"]
AMBIGUOUS = "AMBIGUOUS"


def parse_name(name):
    m = NAME_RE.match(name)
    if not m:
        raise ValueError(f"unexpected NuCLS name {name}")
    slide = f"{m.group(1)}-{m.group(2)}"
    return dict(fov=name, slide=slide, patient=m.group(1), fov_group=m.group(3),
                left=int(m.group(4)), top=int(m.group(5)), bottom=int(m.group(6)), right=int(m.group(7)))


def list_fovs(root=paths.NUCLS):
    return sorted(os.path.splitext(f)[0] for f in os.listdir(f"{root}/rgb") if f.endswith(".png"))


def rgb_path(name, root=paths.NUCLS):
    return f"{root}/rgb/{name}.png"


def image_size(name, root=paths.NUCLS):
    with Image.open(rgb_path(name, root)) as im:
        return im.size  # (W, H)


def fov_rect(name, root=paths.NUCLS):
    """Rectangle of the annotated region in png pixels (x0, y0, x1, y1), from the 253 code of mask channel 0."""
    m = np.asarray(Image.open(f"{root}/mask/{name}.png"))[..., 0]
    ys, xs = np.nonzero(m == FOV_CODE)
    if len(ys) == 0:                                   # no explicit region: nuclei cover everything, use the full image
        return 0, 0, m.shape[1], m.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _polygon_area(xs, ys):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) < 3:
        return np.nan
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def load_nuclei(name, root=paths.NUCLS):
    """One row per nucleus: classes, box, centroid (png px), polygon flag and polygon area (um^2)."""
    df = pd.read_csv(f"{root}/csv/{name}.csv")
    df = df.rename(columns={"Unnamed: 0": "nucleus_id", "raw_classification": "raw", "main_classification": "main",
                            "super_classification": "super"})
    if "nucleus_id" not in df.columns:
        df.insert(0, "nucleus_id", np.arange(len(df)))
    df["cx"] = (df["xmin"] + df["xmax"]) / 2.0
    df["cy"] = (df["ymin"] + df["ymax"]) / 2.0
    df["polygon"] = df["type"].eq("polyline")
    areas = []
    for t, xs, ys in zip(df["type"], df["coords_x"], df["coords_y"]):
        if t != "polyline":
            areas.append(np.nan)
            continue
        areas.append(_polygon_area([float(v) for v in str(xs).split(",")], [float(v) for v in str(ys).split(",")]))
    df["area_um2"] = np.asarray(areas) * FOV_MPP * FOV_MPP
    df["labelled"] = df["super"].isin(SUPER)
    return df[["nucleus_id", "raw", "main", "super", "type", "polygon", "xmin", "ymin", "xmax", "ymax", "cx", "cy",
               "area_um2", "labelled"]]


def hospital_of_slides(root=paths.NUCLS):
    """slide -> hospital code (the NuCLS fold files are grouped by source hospital)."""
    out = {}
    d = f"{root}/train_test_splits"
    for f in os.listdir(d):
        t = pd.read_csv(f"{d}/{f}")
        if "slide_name" not in t.columns:
            continue
        hs = t["hospital"] if "hospital" in t.columns else t["slide_name"].str[5:7]   # TCGA tissue source site code
        for s, h in zip(t["slide_name"], hs):
            out[s] = h
    return out
