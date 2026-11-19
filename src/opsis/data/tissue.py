"""Tissue region labels for a FOV from the TIGER roi-level tissue-bcss masks (BCSS annotations re-rendered by the TIGER
organisers at level 1 of the 40x pyramid, about 0.5 mpp). File name: <slide barcode>_[x0, y0, x1, y1].png with the box in
that 0.5 mpp frame, so level-0 coordinate = 2 * tiger coordinate (checked against the BCSS origins, e.g. 3401 -> 6798).

TIGER codes: 0 outside the ROI, 1 invasive tumour, 2 tumour associated stroma, 3 in situ tumour, 4 healthy glands,
5 necrosis not in situ, 6 inflamed stroma, 7 rest. The mapping to BCSS codes is checked by scripts/data/10_build_facts.py --check.
"""
import os
import re

import numpy as np
from PIL import Image

from opsis import paths

TIGER_RE = re.compile(r"^(TCGA-\w\w-\w{4})-01Z-00-(DX\d)\.[\w-]+_\[(\d+), (\d+), (\d+), (\d+)\]\.png$")
TIGER_SCALE = 2                                           # level-0 px per tiger px
TIGER_CODES = {0: "outside", 1: "invasive_tumor", 2: "stroma", 3: "in_situ", 4: "healthy_glands", 5: "necrosis",
               6: "inflamed_stroma", 7: "rest"}
TISSUE_CLASSES = ["tumor", "stroma", "inflamed_stroma", "in_situ", "necrosis", "other"]
CODE_TO_CLASS = {1: "tumor", 2: "stroma", 6: "inflamed_stroma", 3: "in_situ", 5: "necrosis", 4: "other", 7: "other"}
CODE_TO_IDX = {c: TISSUE_CLASSES.index(n) for c, n in CODE_TO_CLASS.items()}
# compartments used by the compositional questions: tumour (invasive) versus stroma (plain plus inflamed)
COMPARTMENT_OF_CODE = {1: "tumor", 2: "stroma", 6: "stroma"}


def index_rois(root=f"{paths.TIGER_ROI}/tissue-bcss/masks"):
    """slide -> list of (x0, y0, x1, y1, path) in the tiger frame."""
    out = {}
    for f in sorted(os.listdir(root)):
        m = TIGER_RE.match(f)
        if not m:
            continue
        slide = f"{m.group(1)}-{m.group(2)}"
        out.setdefault(slide, []).append((int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)), f"{root}/{f}"))
    return out


def tissue_map_for_fov(meta, W, H, rois):
    """TIGER code map at the FOV png resolution (H x W uint8) and the covered fraction, or (None, 0.0).

    meta carries the level-0 box of the FOV; the png has W px over (right - left) level-0 px.
    """
    cands = rois.get(meta["slide"], [])
    if not cands:
        return None, 0.0
    L, T, R, B = meta["left"], meta["top"], meta["right"], meta["bottom"]
    best, best_ov = None, 0
    for x0, y0, x1, y1, p in cands:
        ov = max(0, min(R, TIGER_SCALE * x1) - max(L, TIGER_SCALE * x0)) * max(0, min(B, TIGER_SCALE * y1) - max(T, TIGER_SCALE * y0))
        if ov > best_ov:
            best, best_ov = (x0, y0, x1, y1, p), ov
    if best is None or best_ov == 0:
        return None, 0.0
    x0, y0, x1, y1, p = best
    mask = np.asarray(Image.open(p))
    if mask.ndim == 3:
        mask = mask[..., 0]
    sx = (R - L) / float(W)
    sy = (B - T) / float(H)
    X = L + (np.arange(W) + 0.5) * sx                   # level-0 x of each png column
    Y = T + (np.arange(H) + 0.5) * sy
    j = np.floor(X / TIGER_SCALE).astype(int) - x0
    i = np.floor(Y / TIGER_SCALE).astype(int) - y0
    out = np.zeros((H, W), dtype=np.uint8)
    vi = (i >= 0) & (i < mask.shape[0])
    vj = (j >= 0) & (j < mask.shape[1])
    if vi.any() and vj.any():
        out[np.ix_(vi, vj)] = mask[i[vi]][:, j[vj]]
    return out, float((out != 0).mean())


# ---------------------------------------------------------------------------------------------------------------------
# Original BCSS masks (0.25 mpp, 21 codes, level-0 origin in the file name), used when all masks are present.
# The class mapping below is the one used for the question candidates.
BCSS_RE = re.compile(r"^(TCGA-\w\w-\w{4}-DX\d)_xmin(\d+)_ymin(\d+)_MPP-0\.2500\.png$")
BCSS_MPP = 0.25
BCSS_CODES = {0: "outside_roi", 1: "tumor", 2: "stroma", 3: "lymphocytic_infiltrate", 4: "necrosis_or_debris",
              5: "glandular_secretions", 6: "blood", 7: "exclude", 8: "metaplasia_NOS", 9: "fat", 10: "plasma_cells",
              11: "other_immune_infiltrate", 12: "mucoid_material", 13: "normal_acinus_or_duct", 14: "lymphatics",
              15: "undetermined", 16: "nerve", 17: "skin_adnexa", 18: "blood_vessel", 19: "angioinvasion", 20: "dcis", 21: "other"}
# BCSS code -> TIGER-style code used by the rest of the pipeline (0 = do not care, 7 = rest)
BCSS_TO_TIGER = {0: 0, 7: 0, 15: 0, 1: 1, 2: 2, 3: 6, 10: 6, 11: 6, 4: 5, 20: 3, 13: 4}


def index_bcss(root=f"{paths.BCSS}/masks"):
    out = {}
    if not os.path.isdir(root):
        return out
    for f in sorted(os.listdir(root)):
        m = BCSS_RE.match(f)
        if m:
            out.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3)), f"{root}/{f}"))
    return out


def bcss_map_for_box(L, T, R, B, W, H, rois, base_mpp, slide):
    """TIGER-coded map (H x W) of the level-0 box [L, R) x [T, B) rendered at W x H px from the original BCSS mask."""
    cands = rois.get(slide, []) if rois else []
    if not cands or base_mpp is None or not np.isfinite(base_mpp):
        return None, 0.0
    sb = BCSS_MPP / base_mpp                              # level-0 px per bcss px
    best, best_ov = None, 0
    for xmin, ymin, p in cands:
        with Image.open(p) as im:
            w, h = im.size
        ov = max(0, min(R, xmin + w * sb) - max(L, xmin)) * max(0, min(B, ymin + h * sb) - max(T, ymin))
        if ov > best_ov:
            best, best_ov = (xmin, ymin, p), ov
    if best is None:
        return None, 0.0
    xmin, ymin, p = best
    mask = np.asarray(Image.open(p))
    if mask.ndim == 3:
        mask = mask[..., 0]
    X = L + (np.arange(W) + 0.5) * (R - L) / float(W)
    Y = T + (np.arange(H) + 0.5) * (B - T) / float(H)
    j = np.floor((X - xmin) / sb).astype(int)
    i = np.floor((Y - ymin) / sb).astype(int)
    lut = np.zeros(256, dtype=np.uint8)
    for c in range(22):
        lut[c] = BCSS_TO_TIGER.get(c, 7)
    out = np.zeros((H, W), dtype=np.uint8)
    vi = (i >= 0) & (i < mask.shape[0])
    vj = (j >= 0) & (j < mask.shape[1])
    if vi.any() and vj.any():
        out[np.ix_(vi, vj)] = lut[mask[i[vi]][:, j[vj]]]
    return out, float((out != 0).mean())


def bcss_map_for_fov(meta, W, H, rois, base_mpp):
    """TIGER-coded map at the FOV png resolution from the original BCSS mask (needs the slide's base mpp)."""
    return bcss_map_for_box(meta["left"], meta["top"], meta["right"], meta["bottom"], W, H, rois, base_mpp, meta["slide"])
