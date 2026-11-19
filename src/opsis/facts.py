"""Fact tables computed from annotations only (no model output).

fovs.csv     one row per FOV: ids, png size, level-0 box, scale, annotated rectangle, tissue coverage
nuclei.csv   one row per nucleus: classes, centroid (png px), polygon area, tissue code at the centroid, inside-rectangle flag
regions.csv  one row per (FOV, region): nucleus counts per super class, tissue area fractions, validity flags

Regions are defined inside the annotated rectangle: whole, left, right, top, bottom, q00, q01, q10, q11 (quadrants).
Tissue fractions are over the valid area (code not 0); tissue facts are valid when the valid area covers at least
TISSUE_MIN_COVER of the region.
"""
import numpy as np
import pandas as pd

from opsis.data import nucls, tissue

TISSUE_MIN_COVER = 0.8
REGIONS = ["whole", "left", "right", "top", "bottom", "q00", "q01", "q10", "q11"]
CTX_MPP = 1.0            # context window: CTX_PX pixels at CTX_MPP um/px, centred on the FOV (scripts/data/12_build_contexts.py uses the same)
CTX_PX = 448


def region_boxes(rect):
    x0, y0, x1, y1 = rect
    xm = (x0 + x1) // 2
    ym = (y0 + y1) // 2
    return {"whole": (x0, y0, x1, y1), "left": (x0, y0, xm, y1), "right": (xm, y0, x1, y1), "top": (x0, y0, x1, ym),
            "bottom": (x0, ym, x1, y1), "q00": (x0, y0, xm, ym), "q01": (xm, y0, x1, ym), "q10": (x0, ym, xm, y1),
            "q11": (xm, ym, x1, y1)}


def in_box(cx, cy, box):
    x0, y0, x1, y1 = box
    return (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)


def fov_facts(name, rois, tissue_source="tiger", bcss_rois=None, base_mpp=None):
    meta = nucls.parse_name(name)
    W, H = nucls.image_size(name)
    rect = nucls.fov_rect(name)
    nuc = nucls.load_nuclei(name)
    nuc.insert(0, "fov", name)
    nuc["in_rect"] = in_box(nuc["cx"].values, nuc["cy"].values, rect)
    tmap, cover = tissue.tissue_map_for_fov(meta, W, H, rois) if tissue_source == "tiger" else tissue.bcss_map_for_fov(meta, W, H, bcss_rois, base_mpp)
    if tmap is None:
        nuc["tissue_code"] = 0
    else:
        ix = np.clip(nuc["cx"].values.astype(int), 0, W - 1)
        iy = np.clip(nuc["cy"].values.astype(int), 0, H - 1)
        nuc["tissue_code"] = tmap[iy, ix].astype(int)
    nuc["tissue_class"] = [tissue.CODE_TO_CLASS.get(int(c), "") for c in nuc["tissue_code"]]
    nuc["compartment"] = [tissue.COMPARTMENT_OF_CODE.get(int(c), "") for c in nuc["tissue_code"]]
    scale = W / float(meta["right"] - meta["left"])                       # png px per level-0 px
    # context window in level-0 coordinates and in the FOV frame (png px, may lie outside the image)
    cx0 = (meta["left"] + meta["right"]) / 2.0
    cy0 = (meta["top"] + meta["bottom"]) / 2.0
    side0 = CTX_PX * CTX_MPP / base_mpp if base_mpp and np.isfinite(base_mpp) else None
    if side0 is not None:
        ctx_l0 = (cx0 - side0 / 2.0, cy0 - side0 / 2.0, cx0 + side0 / 2.0, cy0 + side0 / 2.0)
        cmap, ccover = (tissue.bcss_map_for_box(*ctx_l0, CTX_PX, CTX_PX, bcss_rois, base_mpp, meta["slide"]) if tissue_source == "bcss"
                        else (None, 0.0))
    else:
        ctx_l0, cmap, ccover = (np.nan,) * 4, None, 0.0
    fov_row = dict(meta, W=W, H=H, scale=scale, base_mpp=base_mpp, rect_x0=rect[0], rect_y0=rect[1],
                   rect_x1=rect[2], rect_y1=rect[3], tissue_cover=cover, ctx_cover=ccover,
                   ctx_left=ctx_l0[0], ctx_top=ctx_l0[1], ctx_right=ctx_l0[2], ctx_bottom=ctx_l0[3],
                   n_nuclei=int(len(nuc)), n_in_rect=int(nuc["in_rect"].sum()))
    rows = []
    boxes = region_boxes(rect)
    for rname in REGIONS:
        box = boxes[rname]
        sel = nuc[nuc["in_rect"] & in_box(nuc["cx"].values, nuc["cy"].values, box)]
        lab = sel[sel["labelled"]]
        r = dict(fov=name, scope="fov", region=rname, x0=box[0], y0=box[1], x1=box[2], y1=box[3],
                 area_um2=(box[2] - box[0]) * (box[3] - box[1]) * nucls.FOV_MPP ** 2,
                 n_total=int(len(sel)), n_labelled=int(len(lab)))
        for k in nucls.SUPER:
            r[f"n_{nucls.SUPER_SHORT[k]}"] = int((lab["super"] == k).sum())
        for k in nucls.SUPER:
            for comp in ("tumor", "stroma"):
                r[f"n_{nucls.SUPER_SHORT[k]}_in_{comp}"] = int(((lab["super"] == k) & (lab["compartment"] == comp)).sum())
        if tmap is None:
            r["valid_frac"] = 0.0
            for c in tissue.TISSUE_CLASSES:
                r[f"frac_{c}"] = np.nan
            r["frac_comp_tumor"] = np.nan
            r["frac_comp_stroma"] = np.nan
        else:
            sub = tmap[box[1]:box[3], box[0]:box[2]]
            valid = sub != 0
            nv = int(valid.sum())
            r["valid_frac"] = nv / float(sub.size) if sub.size else 0.0
            for c in tissue.TISSUE_CLASSES:
                codes = [k for k, v in tissue.CODE_TO_CLASS.items() if v == c]
                r[f"frac_{c}"] = float(np.isin(sub, codes).sum() / nv) if nv else np.nan
            r["frac_comp_tumor"] = float((sub == 1).sum() / nv) if nv else np.nan
            r["frac_comp_stroma"] = float(np.isin(sub, [2, 6]).sum() / nv) if nv else np.nan
        r["tissue_valid"] = bool(r["valid_frac"] >= TISSUE_MIN_COVER)
        rows.append(r)
    # context scale: tissue facts only (no nucleus annotation outside the FOV); boxes in the FOV frame
    if side0 is not None:
        cboxes = region_boxes((0, 0, CTX_PX, CTX_PX))
        s_ctx = (CTX_MPP / base_mpp) * scale                              # FOV-frame px per context px
        ox = (ctx_l0[0] - meta["left"]) * scale                           # FOV-frame x of the context origin
        oy = (ctx_l0[1] - meta["top"]) * scale
        for rname in REGIONS:
            cb = cboxes[rname]
            r = dict(fov=name, scope="ctx", region=rname, x0=ox + cb[0] * s_ctx, y0=oy + cb[1] * s_ctx,
                     x1=ox + cb[2] * s_ctx, y1=oy + cb[3] * s_ctx,
                     area_um2=(cb[2] - cb[0]) * (cb[3] - cb[1]) * CTX_MPP ** 2, n_total=-1, n_labelled=-1)
            for k in nucls.SUPER:
                r[f"n_{nucls.SUPER_SHORT[k]}"] = -1
                for comp in ("tumor", "stroma"):
                    r[f"n_{nucls.SUPER_SHORT[k]}_in_{comp}"] = -1
            if cmap is None:
                r["valid_frac"] = 0.0
                for c in tissue.TISSUE_CLASSES:
                    r[f"frac_{c}"] = np.nan
                r["frac_comp_tumor"] = np.nan
                r["frac_comp_stroma"] = np.nan
            else:
                sub = cmap[cb[1]:cb[3], cb[0]:cb[2]]
                valid = sub != 0
                nv = int(valid.sum())
                r["valid_frac"] = nv / float(sub.size) if sub.size else 0.0
                for c in tissue.TISSUE_CLASSES:
                    codes = [k for k, v in tissue.CODE_TO_CLASS.items() if v == c]
                    r[f"frac_{c}"] = float(np.isin(sub, codes).sum() / nv) if nv else np.nan
                r["frac_comp_tumor"] = float((sub == 1).sum() / nv) if nv else np.nan
                r["frac_comp_stroma"] = float(np.isin(sub, [2, 6]).sum() / nv) if nv else np.nan
            r["tissue_valid"] = bool(r["valid_frac"] >= TISSUE_MIN_COVER)
            rows.append(r)
    return fov_row, nuc, pd.DataFrame(rows)
