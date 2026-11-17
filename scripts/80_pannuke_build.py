#!/usr/bin/env python
"""PanNuke as a second FOV question set (cross organ): 256 px patches at 0.25 mpm, instance masks with 5 nucleus classes,
19 tissue types. Writes $OPSIS_DATA/derived/pannuke/{rgb/<id>.png, fovs.csv, nuclei.csv, regions.csv, questions_seed0.parquet}
with two split columns: split_fold (folds 1+2 train / 10 percent of fold 2 val / fold 3 test) and split_organ (train on the
15 other tissues, test on Breast, Colon, Lung, Prostate). Also questions_nucls_seed0.parquet with the NuCLS class names
(neoplastic -> tumor, inflammatory -> sTIL, connective -> stromal, dead and epithelial -> other) for zero shot transfer.
    python scripts/80_pannuke_build.py"""
import argparse, hashlib, io, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
ap = argparse.ArgumentParser(); ap.add_argument("--abstain", action="store_true", help="organ question with a 'cannot tell' answer: patches of four meta held out organs (Kidney, Stomach, Skin, Thyroid) are labelled cannot tell in training; Breast, Colon, Lung, Prostate stay the unseen test organs"); A = ap.parse_args()
from PIL import Image
from opsis import paths, facts
from opsis.questions import count_bin, cell_frac_bin, area_bin, COUNT_LABELS, CELL_FRAC_LABELS, AREA_LABELS, NOUL, FAMILY, REGION_TEXT

RAW = f"{paths.RAW}/pannuke_hf/data"; OUT = f"{paths.DERIVED}/pannuke" + ("_abstain" if A.abstain else ""); paths.ensure(f"{OUT}/rgb")
if A.abstain and not os.path.islink(f"{OUT}/rgb"):
    os.rmdir(f"{OUT}/rgb"); os.symlink(f"{paths.DERIVED}/pannuke/rgb", f"{OUT}/rgb")
META_ABSTAIN = {"Kidney", "Stomach", "Skin", "Thyroid"}
CLASSES = ["neoplastic", "inflammatory", "connective", "dead", "epithelial"]; CTEXT = {"neoplastic": "neoplastic cells", "inflammatory": "inflammatory cells", "connective": "connective tissue cells", "dead": "dead cells", "epithelial": "non neoplastic epithelial cells"}
TO_NUCLS = {"neoplastic": "tumor", "inflammatory": "sTIL", "connective": "stromal", "dead": "other", "epithelial": "other"}; NUCLS_CLASSES = ["tumor", "stromal", "sTIL", "other"]
NUCLS_TEXT = {"tumor": "tumour cells", "stromal": "stromal cells", "sTIL": "lymphocytes or plasma cells", "other": "other cells"}
TISSUES = ["Adrenal Gland", "Bile Duct", "Bladder", "Breast", "Cervix", "Colon", "Esophagus", "Head & Neck", "Kidney", "Liver", "Lung", "Ovarian", "Pancreatic", "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus"]
HELD_OUT = {"Breast", "Colon", "Lung", "Prostate"}; MPP = 0.25; PAIRS = [("neoplastic", "inflammatory"), ("connective", "inflammatory"), ("neoplastic", "connective")]
fov_rows, nuc_rows, reg_rows = [], [], []
for fold in (1, 2, 3):
    t = pq.read_table(f"{RAW}/fold{fold}-00000-of-00001.parquet")
    for i in range(t.num_rows):
        r = t.slice(i, 1).to_pylist()[0]; fid = f"pannuke_f{fold}_{i:04d}"; p = f"{OUT}/rgb/{fid}.png"
        if not os.path.exists(p): Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(p)
        nuc = []
        for k, (m, c) in enumerate(zip(r["instances"], r["categories"])):
            a = np.asarray(Image.open(io.BytesIO(m["bytes"]))); ys, xs = np.nonzero(a)
            if len(xs) == 0: continue
            nuc.append(dict(fov=fid, nucleus_id=k, cls=CLASSES[c], xmin=int(xs.min()), ymin=int(ys.min()), xmax=int(xs.max()) + 1, ymax=int(ys.max()) + 1, cx=float(xs.mean()), cy=float(ys.mean()), area_um2=float(len(xs)) * MPP * MPP))
        nuc_rows += nuc; N = pd.DataFrame(nuc) if nuc else pd.DataFrame(columns=["cx", "cy", "cls"])
        tissue = TISSUES[r["tissue"]]
        fov_rows.append(dict(fov=fid, slide=fid, patient=fid, fold=fold, tissue=tissue, W=256, H=256, scale=1.0, left=0, top=0, right=256, bottom=256, ctx_left=0, ctx_top=0, ctx_right=256, ctx_bottom=256, base_mpp=MPP, n_nuclei=len(nuc)))
        boxes = facts.region_boxes((0, 0, 256, 256))
        for rname, box in boxes.items():
            sel = N[facts.in_box(N["cx"].to_numpy(), N["cy"].to_numpy(), box)] if len(N) else N
            row = dict(fov=fid, scope="fov", region=rname, x0=box[0], y0=box[1], x1=box[2], y1=box[3], n_total=int(len(sel)), n_labelled=int(len(sel)))
            for c in CLASSES: row[f"n_{c}"] = int((sel["cls"] == c).sum()) if len(sel) else 0
            reg_rows.append(row)
    print(f"fold {fold}: {t.num_rows} patches", flush=True)
F = pd.DataFrame(fov_rows); NU = pd.DataFrame(nuc_rows); RE = pd.DataFrame(reg_rows)
rng = np.random.default_rng(0)
F["split_fold"] = np.where(F["fold"] == 3, "test", np.where((F["fold"] == 2) & (rng.random(len(F)) < 0.1), "val", "train"))
F["split_organ"] = np.where(F["tissue"].isin(HELD_OUT), "test", np.where((F["fold"] == 2) & (rng.random(len(F)) < 0.1), "val", "train"))
F.to_csv(f"{OUT}/fovs.csv", index=False); NU.to_csv(f"{OUT}/nuclei.csv", index=False); RE.to_csv(f"{OUT}/regions.csv", index=False)
tr = NU[NU["fov"].isin(F.loc[F["split_fold"] == "train", "fov"])]; area_q = np.quantile(tr["area_um2"], [0.25, 0.5, 0.75]).tolist(); json.dump({"area_q": area_q}, open(f"{OUT}/bins_seed0.json", "w"))
def build(classes, ctext, cmap):
    rows = []
    for fr in F.itertuples():
        nuc = NU[NU["fov"] == fr.fov]; reg = {r["region"]: r for r in RE[RE["fov"] == fr.fov].to_dict("records")}; rng_ = np.random.default_rng(int(hashlib.sha1(fr.fov.encode()).hexdigest()[:8], 16))
        base = dict(fov=fr.fov, slide=fr.fov, patient=fr.fov, box2="", derivation="pannuke", split_fold=fr.split_fold, split_organ=fr.split_organ)
        def emit(op, args, atype, cands, ans, box, text, box2=None):
            key = f"{fr.fov}|{op}|{json.dumps(args, sort_keys=True)}|{box}|{box2}"
            rows.append(dict(base, qid=hashlib.sha1(key.encode()).hexdigest()[:16], family=FAMILY[op], op=op, args=json.dumps(args, sort_keys=True), atype=atype, candidates=json.dumps(cands), answer=int(ans), box=json.dumps([round(float(v), 1) for v in box]), box2=json.dumps([round(float(v), 1) for v in box2]) if box2 else "", text=text))
        # cell level: up to 12 nuclei stratified by class
        chosen = []
        for c in CLASSES:
            ids = nuc.index[nuc["cls"] == c].to_numpy(); rng_.shuffle(ids); chosen += list(ids[:3])
        rest = [i for i in nuc.index if i not in set(chosen)]; rng_.shuffle(rest); chosen += rest[: max(0, 12 - len(chosen))]
        for i in chosen[:12]:
            n = nuc.loc[i]; box = (n.xmin, n.ymin, n.xmax, n.ymax); c = cmap[n.cls]
            emit("cell_super", {}, "choice", classes, classes.index(c), box, "Which class of cell is the marked nucleus?")
            k = c if rng_.random() < 0.5 else str(rng_.choice([x for x in classes if x != c])); emit("cell_is", {"k": k}, "noul", NOUL, int(k == c), box, f"Is the marked nucleus one of the {ctext[k]}?")
            emit("cell_area_bin", {}, "ordinal", AREA_LABELS, area_bin(n.area_um2, area_q), box, "In which size quartile is the area of the marked nucleus?")
        halves = ["left", "right"] if rng_.random() < 0.5 else ["top", "bottom"]; quad = str(rng_.choice(["q00", "q01", "q10", "q11"]))
        for rname in ["whole"] + halves + [quad]:
            r = reg[rname]; box = (r["x0"], r["y0"], r["x1"], r["y1"]); rt = REGION_TEXT[rname]
            cnt = {c: 0 for c in classes}
            for pc in CLASSES: cnt[cmap[pc]] += int(r[f"n_{pc}"])
            for k in classes:
                emit("count_bin", {"k": k, "region": rname}, "ordinal", COUNT_LABELS, count_bin(cnt[k]), box, f"How many {ctext[k]} are in {rt}?")
                emit("present", {"k": k, "region": rname}, "noul", NOUL, int(cnt[k] >= 1), box, f"Are there any {ctext[k]} in {rt}?")
                if r["n_labelled"] >= 5: emit("cell_frac_bin", {"k": k, "region": rname}, "ordinal", CELL_FRAC_LABELS, cell_frac_bin(cnt[k] / r["n_labelled"]), box, f"What fraction of the cells in {rt} are {ctext[k]}?")
            for k1, k2 in (PAIRS if classes is CLASSES else [("tumor", "sTIL"), ("stromal", "sTIL"), ("tumor", "stromal")]):
                if abs(cnt[k1] - cnt[k2]) >= 2: emit("more_cells", {"k1": k1, "k2": k2, "region": rname}, "choice", [k1, k2], int(cnt[k2] > cnt[k1]), box, f"In {rt}, are there more {ctext[k1]} or more {ctext[k2]}?")
        ra, rb = reg[halves[0]], reg[halves[1]]
        for k in classes:
            na = sum(int(ra[f"n_{pc}"]) for pc in CLASSES if cmap[pc] == k); nb = sum(int(rb[f"n_{pc}"]) for pc in CLASSES if cmap[pc] == k)
            if abs(na - nb) >= 3: emit("compare_regions", {"k": k, "A": halves[0], "B": halves[1]}, "choice", ["A", "B"], int(nb > na), (ra["x0"], ra["y0"], ra["x1"], ra["y1"]), f"Which region has more {ctext[k]}: A ({REGION_TEXT[halves[0]]}) or B ({REGION_TEXT[halves[1]]})?", box2=(rb["x0"], rb["y0"], rb["x1"], rb["y1"]))
        if classes is CLASSES:                                           # organ questions: the true tissue and three negatives
            negs = rng_.choice([t for t in TISSUES if t != fr.tissue], 3, replace=False)
            for t in [fr.tissue] + list(negs):
                if A.abstain:
                    ans = 2 if fr.tissue in META_ABSTAIN else int(t == fr.tissue)
                    emit("tissue_is", {"t": t}, "choice", ["no", "yes", "cannot tell"], ans, (0, 0, 256, 256), f"Is this patch from {t.lower()} tissue? Answer cannot tell if the tissue type is unfamiliar.")
                else:
                    emit("tissue_is", {"t": t}, "noul", NOUL, int(t == fr.tissue), (0, 0, 256, 256), f"Is this patch from {t.lower()} tissue?")
    return pd.DataFrame(rows)
Q = build(CLASSES, CTEXT, {c: c for c in CLASSES}); Q["split"] = Q["split_fold"]; Q.to_parquet(f"{OUT}/questions_seed0.parquet", index=False)
Qn = build(NUCLS_CLASSES, NUCLS_TEXT, TO_NUCLS); Qn["split"] = Qn["split_fold"]; Qn.to_parquet(f"{OUT}/questions_nucls_seed0.parquet", index=False)
print(f"patches {len(F)} (split_fold {F.split_fold.value_counts().to_dict()}, split_organ {F.split_organ.value_counts().to_dict()}); nuclei {len(NU)}; questions {len(Q)} / nucls-vocabulary {len(Qn)}")
print(Q.groupby("op").size().to_string())
