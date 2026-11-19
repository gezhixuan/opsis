#!/usr/bin/env python
"""Dense zero shot slide level metastasis detection on CAMELYON16: every 80 um tissue field on a non overlapping grid is
encoded (Virchow2, parallel region reading) and asked "are there tumour cells" by the field decoder of Opsis-S; slide scores are the
maximum, top k means and exceedance counts over all fields. --roles T1 (official test, 129 slides) or pool; --subset N takes a
balanced subset of a role (for choosing the aggregate on the training pool without training anything).
    OPSIS_TOOL_DEVICE=cuda python scripts/transfer/91_camelyon_dense.py --roles T1 --name camelyon_dense_T1
Writes runs/<name>/slides.csv, fields/<slide>.npz (per field p, x, y) and metrics.json."""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, openslide
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from opsis import paths, facts, tools as T, dataset as D
from opsis.data import nucls
ap = argparse.ArgumentParser(); ap.add_argument("--roles", default="T1"); ap.add_argument("--subset", type=int, default=None); ap.add_argument("--stride-um", type=float, default=80.0); ap.add_argument("--field-um", type=float, default=80.0); ap.add_argument("--min-tissue", type=float, default=0.5)
ap.add_argument("--batch", type=int, default=48); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--name", default="camelyon_dense_T1"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--max-fields", type=int, default=60000); ap.add_argument("--shard", default=None, help="i/n: process every n-th slide starting at i (parallel jobs share one run directory)")
a = ap.parse_args(); DD = paths.CAMELYON; OUT = f"{paths.RUNS}/{a.name}"; paths.ensure(OUT); paths.ensure(f"{OUT}/fields"); rng = np.random.default_rng(a.seed); dev = T.DEV; SZ = 448
S = pd.read_csv(f"{DD}/splits.csv"); S = S[(S.dataset == "c16") & S.role.isin(a.roles.split(","))].copy(); S["label"] = S.label_official
if a.subset: S = pd.concat([g.sample(min(len(g), a.subset // 3), random_state=a.seed) for _, g in S.groupby("label")])
if a.limit: S = S.head(a.limit)
if a.shard: i, n = map(int, a.shard.split("/")); S = S.iloc[i::n]
done = {f[:-4] for f in os.listdir(f"{OUT}/fields")}; print(f"{len(S)} slides, {len(done)} already done", flush=True)
enc = T._encoder(); m, vocab, cfg, tmpl = T._fov_model(); MEAN, STD = T._M["mean"].to(dev), T._M["std"].to(dev)
t = tmpl["present"]; args = json.loads(t.args); args["k"] = "tumor"
def grid_for(slide):
    ms = openslide.OpenSlide(f"{DD}/CAMELYON16/masks/{slide}_mask.tif"); lev = ms.get_best_level_for_downsample(32); ds = ms.level_downsamples[lev]
    tis = np.asarray(ms.read_region((0, 0), lev, ms.level_dimensions[lev]).convert("L")) > 0; ms.close(); mpp_mask = 0.243 * ds
    step = max(1, int(round(a.stride_um / mpp_mask))); half = max(1, int(round(a.field_um / mpp_mask / 2)))
    pts = []
    for r in range(half, tis.shape[0] - half, step):
        row = tis[r - half: r + half + 1]
        for c in range(half, tis.shape[1] - half, step):
            if row[:, c - half: c + half + 1].mean() >= a.min_tissue: pts.append((int((c + 0.5) * ds), int((r + 0.5) * ds)))
    pts = np.array(pts, int)
    if len(pts) > a.max_fields: pts = pts[np.sort(rng.choice(len(pts), a.max_fields, replace=False))]
    return pts
class Fields(Dataset):
    def __init__(self, path, pts): self.path, self.pts, self.sl = path, pts, None
    def __len__(self): return len(self.pts)
    def __getitem__(self, i):
        if self.sl is None: self.sl = openslide.OpenSlide(self.path); self.mpp0 = float(self.sl.properties.get("openslide.mpp-x", 0.243))
        x, y = map(int, self.pts[i]); sl, mpp0 = self.sl, self.mpp0; side0 = int(round(a.field_um / mpp0)); Wp = int(round(a.field_um / nucls.FOV_MPP)); left, top = x - side0 // 2, y - side0 // 2
        fov = sl.read_region((left, top), 0, (side0, side0)).convert("RGB").resize((Wp, Wp), Image.BICUBIC); r = nucls.FOV_MPP / 0.25; w = int(round(Wp * r))
        canvas = Image.new("RGB", (SZ, SZ), (255, 255, 255)); ox = (SZ - w) // 2; canvas.paste(fov.resize((w, w), Image.BICUBIC), (ox, ox))
        cside0 = facts.CTX_PX * facts.CTX_MPP / mpp0; cl, ct = int(x - cside0 / 2), int(y - cside0 / 2); lev = sl.get_best_level_for_downsample(facts.CTX_MPP / mpp0); f = sl.level_downsamples[lev]
        ctx = sl.read_region((cl, ct), lev, (int(cside0 / f), int(cside0 / f))).convert("RGB").resize((SZ, SZ), Image.BILINEAR)
        geo = np.array([Wp / side0, left, top, cl, ct, cl + cside0, ct + cside0, r, ox], np.float64)
        return torch.from_numpy(np.asarray(canvas, np.uint8)).permute(2, 0, 1), torch.from_numpy(np.asarray(ctx, np.uint8)).permute(2, 0, 1), torch.from_numpy(geo), i
def make_item(tf, tc, geo, i):
    fov_row = dict(scale=geo[0], left=geo[1], top=geo[2], ctx_left=geo[3], ctx_top=geo[4], ctx_right=geo[5], ctx_bottom=geo[6]); geo_row = dict(fov_scale=geo[7], off_x=geo[8], off_y=geo[8]); Wp = int(round(a.field_um / nucls.FOV_MPP))
    g = D.FovGeometry(fov_row, geo_row, SZ, facts.CTX_PX); pos = g.token_positions(32)
    q = pd.DataFrame([dict(fov="s", slide="s", patient="", family=t.family, op="present", args=json.dumps(args), atype=t.atype, candidates=t.candidates, answer=0, text=t.text, box=json.dumps([0.0, 0.0, float(Wp), float(Wp)]), box2="")])
    e = D.encode_questions(q, vocab, "op"); box = np.stack([g.box_features(bb) for bb in e["box"]]); box2 = np.stack([g.box_features(bb) for bb in e["box2"]])
    return dict(tok_fov=tf, tok_ctx=tc, pos=torch.from_numpy(pos), op=torch.from_numpy(e["op"]), key=torch.from_numpy(e["key"]), args=torch.from_numpy(e["args"]), cands=torch.from_numpy(e["cands"]), cand_mask=torch.from_numpy(e["cand_mask"]), answer=torch.from_numpy(e["answer"]), box=torch.from_numpy(box), box2=torch.from_numpy(box2), has_box2=torch.from_numpy(e["has_box2"]), text_id=torch.from_numpy(e["text_id"]), cand_text=torch.from_numpy(e["cand_text"]), atype=torch.from_numpy(e["atype"]), qidx=torch.arange(1), fov_index=i)
t0 = time.time()
for si, r in enumerate(S.itertuples()):
    slide = r.slide; path = f"{DD}/CAMELYON16/images/{slide}.tif"
    if slide in done or not os.path.exists(path): continue
    pts = grid_for(slide); probs = np.zeros(len(pts), np.float32)
    dl = DataLoader(Fields(path, pts), a.batch, num_workers=a.workers, shuffle=False, persistent_workers=False)
    for canv, ctx, geo, idx in dl:
        x = torch.cat([canv, ctx], 0).to(dev, non_blocking=True).float() / 255.0; x = (x - MEAN) / STD
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"): tok = enc.forward_features(x)[:, enc.n_prefix:].half()
        n = len(idx); items = [make_item(tok[i].cpu(), tok[n + i].cpu(), geo[i].numpy(), int(idx[i])) for i in range(n)]
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in D.collate(items).items()}
        with torch.no_grad(): p = torch.softmax(m(b, True).float(), -1)[:, 0, 1].cpu().numpy()
        probs[idx.numpy()] = p
    np.savez_compressed(f"{OUT}/fields/{slide}.npz", p=probs, x=pts[:, 0], y=pts[:, 1], label=str(r.label), role=str(r.role))
    print(f"  [{si + 1}/{len(S)}] {slide} {r.label} fields {len(pts)} max {probs.max():.3f} n>0.9 {(probs > 0.9).sum()} {time.time() - t0:.0f} s", flush=True)
rows = []
for f in sorted(os.listdir(f"{OUT}/fields")):
    z = np.load(f"{OUT}/fields/{f}"); p = np.sort(z["p"])[::-1]
    rows.append(dict(slide=f[:-4], role=str(z["role"]), label=str(z["label"]), n_fields=len(p), score_max=float(p[0]), top5=float(p[:5].mean()), top20=float(p[:20].mean()), n_gt_0_9=int((p > 0.9).sum()), n_gt_0_99=int((p > 0.99).sum()), frac_gt_0_5=float((p > 0.5).mean())))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/slides.csv", index=False); res = {"stride_um": a.stride_um, "n_slides": len(R), "rows": {}}
for role in sorted(R.role.unique()):
    g = R[R.role == role]; y = (g.label != "negative").astype(int).to_numpy()
    if y.sum() == 0 or (1 - y).sum() == 0: continue
    for sc in ("score_max", "top5", "top20", "n_gt_0_9", "n_gt_0_99", "frac_gt_0_5"):
        rec = {"n": len(g), "positives": int(y.sum()), "auroc": float(roc_auc_score(y, g[sc]))}
        for size in ("macro", "micro"):
            gg = g[(g.label == size) | (g.label == "negative")]; yy = (gg.label == size).astype(int).to_numpy()
            if yy.sum(): rec[f"auroc_{size}_vs_negative"] = float(roc_auc_score(yy, gg[sc]))
        res["rows"][f"{role}:{sc}"] = rec; print(f"  {role}:{sc:12s} n={len(g):4d} pos={int(y.sum()):3d} AUROC {rec['auroc']:.3f} macro {rec.get('auroc_macro_vs_negative', float('nan')):.3f} micro {rec.get('auroc_micro_vs_negative', float('nan')):.3f}", flush=True)
json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=1); paths.ensure(f"{paths.RESULTS}/runs/{a.name}"); os.system(f"cp {OUT}/metrics.json {paths.RESULTS}/runs/{a.name}/"); print("RUN_DONE", OUT)
