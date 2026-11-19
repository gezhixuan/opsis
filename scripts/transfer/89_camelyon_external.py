#!/usr/bin/env python
"""External validation of the field decoder of Opsis-S on CAMELYON16 (breast carcinoma metastases in lymph nodes, a cohort, scanner and
tissue context never seen in training; lesion masks from the CAMELYON+ references at downsample 32). For sampled 80 um fields
inside lesions and in lesion free tissue, the field decoder answers present / count_bin / cell_frac_bin for tumour cells and
present for lymphocytes through opsis.tools.ask_region (Virchow2 encoded on the fly). Reports AUROC of P(tumour present)
against the mask, the expected fraction against the mask fraction, and per slide numbers.
    OPSIS_TOOL_DEVICE=cuda python scripts/transfer/89_camelyon_external.py [--slides 40] [--per-slide 30] [--seed 0]"""
import argparse, json, os, time
import numpy as np, pandas as pd, openslide
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from opsis import paths, tools as T
ap = argparse.ArgumentParser(); ap.add_argument("--slides", type=int, default=40); ap.add_argument("--per-slide", type=int, default=30); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--field-um", type=float, default=80.0); ap.add_argument("--name", default="camelyon_external")
a = ap.parse_args(); D = paths.CAMELYON; OUT = f"{paths.RUNS}/{a.name}"; paths.ensure(OUT); rng = np.random.default_rng(a.seed)
S = pd.read_csv(f"{D}/splits.csv"); S = S[(S.dataset == "c16") & (S.positive == True) & (S.split == "test")]
cands = [s for s in sorted(S.slide) if os.path.exists(f"{D}/references_camplus/{s}.npz") and os.path.exists(f"{D}/CAMELYON16/images/{s}.tif")]
slides = [cands[i] for i in sorted(rng.choice(len(cands), min(a.slides, len(cands)), replace=False))]; print(f"{len(slides)} CAMELYON16 test slides with lesions", flush=True)
PATH = {s: f"{D}/CAMELYON16/images/{s}.tif" for s in slides}; T._wsi = lambda s: openslide.OpenSlide(PATH[s])      # the tool module otherwise looks slides up in the TCGA store
rows = []; t0 = time.time()
for si, s in enumerate(slides):
    z = np.load(f"{D}/references_camplus/{s}.npz", allow_pickle=True); lab = z["labels"]; ds = int(z["downsample"]); mpp = float(z["mpp"]); tumour = lab > 0
    ms = openslide.OpenSlide(f"{D}/CAMELYON16/masks/{s}_mask.tif"); lev = ms.get_best_level_for_downsample(ds); tis = np.asarray(ms.read_region((0, 0), lev, ms.level_dimensions[lev]).convert("L")) > 0; ms.close()
    H = min(tis.shape[0], tumour.shape[0]); W = min(tis.shape[1], tumour.shape[1]); tis = tis[:H, :W]; tumour = tumour[:H, :W]
    half = int(round(a.field_um / mpp / ds / 2)) or 1                                                 # half field in mask pixels (about 5)
    def frac(m, r, c): return float(m[max(0, r - half): r + half + 1, max(0, c - half): c + half + 1].mean())
    pos = np.argwhere(tumour); neg = np.argwhere(tis & ~tumour)
    if len(pos) == 0 or len(neg) == 0: continue
    pick = []
    for arr, want, test in ((pos, a.per_slide // 2, lambda r, c: frac(tumour, r, c) >= 0.5), (neg, a.per_slide - a.per_slide // 2, lambda r, c: frac(tumour, r, c) == 0.0 and frac(tis, r, c) >= 0.8)):
        tries = 0; got = 0
        while got < want and tries < 50 * want:
            r, c = arr[rng.integers(len(arr))]; tries += 1
            if test(r, c): pick.append((r, c)); got += 1
    for r, c in pick:
        x, y = int((c + 0.5) * ds), int((r + 0.5) * ds)
        try:
            q = T.ask_region(s, x, y, ["present", "count_bin", "cell_frac_bin"], "tumor"); q2 = T.ask_region(s, x, y, ["present"], "sTIL")
        except Exception as e: print("  skip", s, x, y, type(e).__name__, str(e)[:120], flush=True); continue
        A = {r_["operator"]: r_ for r_ in q["answers"]}; FR = {"0": 0.0, "(0,0.1]": 0.05, "(0.1,0.3]": 0.2, "(0.3,0.6]": 0.45, ">0.6": 0.8}
        cf = A["cell_frac_bin"]; exp_frac = float(sum(p * FR[cnd] for p, cnd in zip(cf["probabilities"], cf["candidates"])))
        rows.append(dict(slide=s, x=x, y=y, mask_tumour_frac=frac(tumour, r, c), mask_tissue_frac=frac(tis, r, c), p_tumour_present=A["present"]["probabilities"][1], tumour_count_bin=A["count_bin"]["answer"], tumour_frac_bin=cf["answer"], tumour_expected_frac=exp_frac, p_lymphocytes_present=q2["answers"][0]["probabilities"][1]))
    print(f"  [{si + 1}/{len(slides)}] {s}: {len(pick)} fields, {len(rows)} total, {time.time() - t0:.0f} s", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/fields.csv", index=False); y = (R.mask_tumour_frac >= 0.5).astype(int).to_numpy()
res = dict(n_fields=len(R), n_slides=int(R.slide.nunique()), positives=int(y.sum()), auroc_present=float(roc_auc_score(y, R.p_tumour_present)), ap_present=float(average_precision_score(y, R.p_tumour_present)),
           acc_present_at_0_5=float(((R.p_tumour_present >= 0.5).astype(int) == y).mean()), spearman_expected_frac=float(spearmanr(R.tumour_expected_frac, R.mask_tumour_frac).correlation),
           lymphocytes_present_rate=float((R.p_lymphocytes_present >= 0.5).mean()), per_slide_auroc_median=float(np.median([roc_auc_score((g.mask_tumour_frac >= 0.5).astype(int), g.p_tumour_present) for _, g in R.groupby("slide") if g.mask_tumour_frac.ge(0.5).nunique() == 2])),
           count_bin_on_positive=R[y == 1].tumour_count_bin.value_counts().to_dict(), count_bin_on_negative=R[y == 0].tumour_count_bin.value_counts().to_dict(), field_um=a.field_um, seed=a.seed)
json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=1); paths.ensure(f"{paths.RESULTS}/runs/{a.name}"); os.system(f"cp {OUT}/metrics.json {paths.RESULTS}/runs/{a.name}/"); print("CAMELYON", json.dumps(res)); print("RUN_DONE", OUT)
