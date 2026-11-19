#!/usr/bin/env python
"""Zero shot slide level metastasis detection on CAMELYON16 with the field decoder of Opsis-S (never trained on CAMELYON): up to
--fields tissue fields of 80 um per slide are encoded (Virchow2, batched) and asked "are there tumour cells"; the slide score
is the maximum (and top 5 mean) field probability. AUROC against the official labels on the official test set (role T1,
129 slides) and on the training pool (270 slides), overall and by lesion size (macro / micro). Writes runs/camelyon_slide/.
    OPSIS_TOOL_DEVICE=cuda python scripts/transfer/90_camelyon_slide.py [--fields 300] [--limit N]"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, openslide
from PIL import Image
from sklearn.metrics import roc_auc_score
from opsis import paths, facts, tools as T, dataset as D
from opsis.data import nucls
ap = argparse.ArgumentParser(); ap.add_argument("--fields", type=int, default=300); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--name", default="camelyon_slide")
a = ap.parse_args(); DD = paths.CAMELYON; OUT = f"{paths.RUNS}/{a.name}"; paths.ensure(OUT); rng = np.random.default_rng(a.seed); dev = T.DEV
S = pd.read_csv(f"{DD}/splits.csv"); S = S[(S.dataset == "c16")].copy(); S["label"] = S.label_official
if a.limit: S = pd.concat([S[S.role == "T1"].head(a.limit // 2), S[S.role == "T1"].tail(a.limit - a.limit // 2)]).drop_duplicates("slide")
enc = T._encoder(); m, vocab, cfg, tmpl = T._fov_model(); MEAN, STD = T._M["mean"].to(dev), T._M["std"].to(dev); SZ = 448
t = tmpl["present"]; args = json.loads(t.args); args["k"] = "tumor"; t2 = tmpl["cell_frac_bin"]; args2 = json.loads(t2.args); args2["k"] = "tumor"
def fields_for(slide):
    ms = openslide.OpenSlide(f"{DD}/CAMELYON16/masks/{slide}_mask.tif"); lev = ms.get_best_level_for_downsample(32); ds = ms.level_downsamples[lev]
    tis = np.asarray(ms.read_region((0, 0), lev, ms.level_dimensions[lev]).convert("L")) > 0; ms.close()
    cand = np.argwhere(tis); half = 5; keep = []
    if len(cand) == 0: return np.zeros((0, 2), int)
    for r, c in cand[rng.permutation(len(cand))[: 40 * a.fields]]:
        if tis[max(0, r - half): r + half + 1, max(0, c - half): c + half + 1].mean() >= 0.8: keep.append((int((c + 0.5) * ds), int((r + 0.5) * ds)))
        if len(keep) >= a.fields: break
    return np.array(keep, int)
def read_state(sl, mpp0, x, y, fov_um=80.0):
    side0 = int(round(fov_um / mpp0)); Wp = int(round(fov_um / nucls.FOV_MPP)); left, top = int(x - side0 / 2), int(y - side0 / 2)
    fov = sl.read_region((left, top), 0, (side0, side0)).convert("RGB").resize((Wp, Wp), Image.BICUBIC); r = nucls.FOV_MPP / 0.25; w = int(round(Wp * r))
    canvas = Image.new("RGB", (SZ, SZ), (255, 255, 255)); ox = (SZ - w) // 2; canvas.paste(fov.resize((w, w), Image.BICUBIC), (ox, ox))
    cside0 = facts.CTX_PX * facts.CTX_MPP / mpp0; cl, ct = int(x - cside0 / 2), int(y - cside0 / 2); lev = sl.get_best_level_for_downsample(facts.CTX_MPP / mpp0); f = sl.level_downsamples[lev]
    ctx = sl.read_region((cl, ct), lev, (int(cside0 / f), int(cside0 / f))).convert("RGB").resize((SZ, SZ), Image.BILINEAR)
    fov_row = dict(scale=Wp / side0, left=left, top=top, ctx_left=cl, ctx_top=ct, ctx_right=cl + cside0, ctx_bottom=ct + cside0); geo_row = dict(fov_scale=r, off_x=ox, off_y=ox)
    return canvas, ctx, fov_row, geo_row, Wp
def to_t(ims): x = torch.from_numpy(np.stack([np.asarray(im, np.float32) / 255.0 for im in ims])).permute(0, 3, 1, 2).to(dev); return (x - MEAN) / STD
rows = []; t0 = time.time(); per_field = []
for si, r in enumerate(S.itertuples()):
    slide = r.slide; path = f"{DD}/CAMELYON16/images/{slide}.tif"
    if not os.path.exists(path): continue
    try: pts = fields_for(slide); sl = openslide.OpenSlide(path); mpp0 = float(sl.properties.get("openslide.mpp-x", 0.243))
    except Exception as e: print("  skip", slide, type(e).__name__, flush=True); continue
    probs = []; fracs = []
    for b0 in range(0, len(pts), a.batch):
        bs = pts[b0:b0 + a.batch]; states = [read_state(sl, mpp0, int(x), int(y)) for x, y in bs]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            tok = enc.forward_features(to_t([s[0] for s in states] + [s[1] for s in states]))[:, enc.n_prefix:].half().cpu()
        n = len(states); items = []
        for i, (canvas, ctx, fov_row, geo_row, Wp) in enumerate(states):
            g = D.FovGeometry(fov_row, geo_row, SZ, facts.CTX_PX); pos = g.token_positions(32)
            q = pd.DataFrame([dict(fov=slide, slide=slide, patient="", family=t.family, op="present", args=json.dumps(args), atype=t.atype, candidates=t.candidates, answer=0, text=t.text, box=json.dumps([0.0, 0.0, float(Wp), float(Wp)]), box2=""),
                              dict(fov=slide, slide=slide, patient="", family=t2.family, op="cell_frac_bin", args=json.dumps(args2), atype=t2.atype, candidates=t2.candidates, answer=0, text=t2.text, box=json.dumps([0.0, 0.0, float(Wp), float(Wp)]), box2="")])
            e = D.encode_questions(q, vocab, "op"); box = np.stack([g.box_features(bb) for bb in e["box"]]); box2 = np.stack([g.box_features(bb) for bb in e["box2"]])
            items.append(dict(tok_fov=tok[i], tok_ctx=tok[n + i], pos=torch.from_numpy(pos), op=torch.from_numpy(e["op"]), key=torch.from_numpy(e["key"]), args=torch.from_numpy(e["args"]), cands=torch.from_numpy(e["cands"]), cand_mask=torch.from_numpy(e["cand_mask"]), answer=torch.from_numpy(e["answer"]), box=torch.from_numpy(box), box2=torch.from_numpy(box2), has_box2=torch.from_numpy(e["has_box2"]), text_id=torch.from_numpy(e["text_id"]), cand_text=torch.from_numpy(e["cand_text"]), atype=torch.from_numpy(e["atype"]), qidx=torch.arange(2), fov_index=i))
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in D.collate(items).items()}
        with torch.no_grad(): p = torch.softmax(m(b, True).float(), -1).cpu().numpy()
        probs += p[:, 0, 1].tolist(); FR = np.array([0.0, 0.05, 0.2, 0.45, 0.8]); fracs += (p[:, 1, :5] @ FR).tolist()
    sl.close(); probs = np.array(probs); fracs = np.array(fracs)
    if len(probs) == 0: continue
    top = np.sort(probs)[::-1]
    rows.append(dict(slide=slide, role=r.role, label=r.label, label_camplus=r.label_camplus, n_fields=len(probs), score_max=float(top[0]), score_top5=float(top[:5].mean()), frac_fields_pos=float((probs >= 0.5).mean()), frac_max=float(fracs.max())))
    per_field.append(pd.DataFrame(dict(slide=slide, x=pts[:len(probs), 0], y=pts[:len(probs), 1], p_tumour=probs, exp_frac=fracs)))
    if si % 20 == 0: print(f"  [{si + 1}/{len(S)}] {slide} {r.label} fields {len(probs)} max {top[0]:.3f} top5 {top[:5].mean():.3f}  {time.time() - t0:.0f} s", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/slides.csv", index=False); pd.concat(per_field).to_csv(f"{OUT}/fields.csv", index=False)
res = {"fields_per_slide": a.fields, "n_slides": len(R), "rows": {}}
for role in ("T1", "pool", "all"):
    g = R if role == "all" else R[R.role == role]; y = (g.label != "negative").astype(int).to_numpy()
    if y.sum() == 0 or (1 - y).sum() == 0: continue
    for sc in ("score_max", "score_top5", "frac_fields_pos"):
        rec = {"n": len(g), "positives": int(y.sum()), "auroc": float(roc_auc_score(y, g[sc]))}
        for size in ("macro", "micro"):
            gg = g[(g.label == size) | (g.label == "negative")]; yy = (gg.label == size).astype(int).to_numpy()
            if yy.sum(): rec[f"auroc_{size}_vs_negative"] = float(roc_auc_score(yy, gg[sc]))
        res["rows"][f"{role}:{sc}"] = rec
json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=1); paths.ensure(f"{paths.RESULTS}/runs/{a.name}"); os.system(f"cp {OUT}/metrics.json {paths.RESULTS}/runs/{a.name}/")
for k, v in res["rows"].items(): print(f"  {k:24s} n={v['n']:4d} pos={v['positives']:3d} AUROC {v['auroc']:.3f} macro {v.get('auroc_macro_vs_negative', float('nan')):.3f} micro {v.get('auroc_micro_vs_negative', float('nan')):.3f}", flush=True)
print("RUN_DONE", OUT)
