#!/usr/bin/env python
"""Per tile scoring with noisy-OR aggregation over the FULL bag for the localised findings (LVI, PNI, necrosis,
precursor / in situ component), the per tile heads of Opsis-S. Every 20x tile of the slide gets a score per finding from a small
MLP on its frozen Virchow2 feature; the slide answer is the noisy-OR of the tile scores (P(present) = 1 - prod_i (1 - p_i)),
so the slide label trains tile evidence and the tile scores are a localisation map for free. Alternatives for the ablation:
--pool lse (log mean exp, a soft max) and --pool topk (mean of the top 1 percent of tile logits).
    python scripts/opsis_s/69_train_tile_mil.py --name tile_noisyor_s0 [--ops pan_lvi+pan_pni+pan_necrosis+pan_precursor] [--pool noisyor] [--dump-slides file]
Writes runs/<name>/{best.pt,history.json,metrics_test.json,predictions_<split>.parquet} in the format of scripts/opsis_s/51_train_slide.py (per_op, per_op_heldout_project)
and, for the slides listed in --dump-slides, runs/<name>/tile_scores/<slide>.npz (per tile logits per op, coords) for the tile level evaluation."""
import argparse, json, os, shutil, time
import numpy as np, pandas as pd, torch, h5py
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from opsis import paths, metrics as M
from opsis.questions import FAMILY

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--split-seed", type=int, default=0)
ap.add_argument("--qset", default="slide_questions_pan"); ap.add_argument("--ops", default="pan_lvi+pan_pni+pan_necrosis+pan_precursor")
ap.add_argument("--pool", default="noisyor", choices=["noisyor", "lse", "topk", "attn"]); ap.add_argument("--topk-frac", type=float, default=0.01)
ap.add_argument("--hidden", type=int, default=512); ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--patience", type=int, default=4); ap.add_argument("--accum", type=int, default=8, help="slides per optimiser step")
ap.add_argument("--max-tiles", type=int, default=0, help="training only: random subset per slide (0 = full bag); evaluation always uses the full bag")
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--eval-splits", default="test+heldout_project"); ap.add_argument("--dump-slides", default=None); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--eval-only", action="store_true", help="skip training: evaluate runs/<name>/best.pt")
ap.add_argument("--resume", action="store_true", help="continue from runs/<name>/ckpt.pt, written after every epoch (model, optimiser, schedule, early stopping state and random generators), so a run can be split over short jobs")
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); OUT = f"{paths.RUNS}/{a.name}"; paths.ensure(OUT); RES = f"{paths.RESULTS}/runs/{a.name}"; paths.ensure(RES)
LIST = lambda v: [x for x in v.replace("+", ",").split(",") if x]
OPS = LIST(a.ops); Q = pd.read_parquet(f"{paths.DERIVED}/{a.qset}/questions_seed{a.split_seed}.parquet"); Q = Q[Q["op"].isin(OPS)].reset_index(drop=True)
for op in OPS: assert Q[Q.op == op]["atype"].iloc[0] == "noul" and len(json.loads(Q[Q.op == op]["candidates"].iloc[0])) == 2, op
if a.smoke: Q = Q[Q["slide"].isin(sorted(Q["slide"].unique())[:6])]; Q = pd.concat([Q.assign(split=s) for s in ("train", "val", "test")], ignore_index=True)
print("ops", OPS, "questions", len(Q), Q.groupby("split")["slide"].nunique().to_dict(), flush=True)
SPLITS = ["train", "val"] + LIST(a.eval_splits)

class Bags(Dataset):
    def __init__(self, q, train):
        self.slides = sorted(q["slide"].unique()); self.train = train; self.rng = np.random.default_rng(a.seed if train else 0)
        self.path = dict(zip(q["slide"], q["h5_path"])); self.y = {s: np.full(len(OPS), -1, np.int64) for s in self.slides}
        for r in q.itertuples(): self.y[r.slide][OPS.index(r.op)] = int(r.answer)
    def __len__(self): return len(self.slides)
    def __getitem__(self, i):
        s = self.slides[i]
        with h5py.File(self.path[s], "r") as h:
            n = h["features"].shape[0]
            if self.train and a.max_tiles and n > a.max_tiles: idx = np.sort(self.rng.choice(n, a.max_tiles, replace=False)); X = h["features"][idx]
            else: X = h["features"][:]
        return dict(slide=s, x=torch.from_numpy(np.ascontiguousarray(X)), y=torch.from_numpy(self.y[s]), idx=i)

ds = {sp: Bags(Q[Q["split"] == sp], sp == "train") for sp in SPLITS if (Q["split"] == sp).any()}
dl = {sp: DataLoader(ds[sp], 1, shuffle=(sp == "train"), num_workers=a.workers, collate_fn=lambda b: b[0], persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None) for sp in ds}
print({sp: len(ds[sp]) for sp in ds}, flush=True)

class TileHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Sequential(torch.nn.LayerNorm(2560), torch.nn.Linear(2560, a.hidden), torch.nn.GELU(), torch.nn.Dropout(a.dropout), torch.nn.Linear(a.hidden, a.hidden), torch.nn.GELU(), torch.nn.Dropout(a.dropout))
        self.head = torch.nn.Linear(a.hidden, len(OPS))
        if a.pool == "noisyor": torch.nn.init.constant_(self.head.bias, -10.0)      # a typical bag has 1.4e4 tiles: start with P(present) = 1 - (1 - sigmoid(-10))^1.4e4 = 0.47
        if a.pool == "attn": self.att_V = torch.nn.Linear(a.hidden, 128); self.att_U = torch.nn.Linear(a.hidden, 128); self.att_w = torch.nn.Linear(128, len(OPS))   # gated attention MIL (ABMIL / CLAM style), one attention per operator
    def attention(self, h): return self.att_w(torch.tanh(self.att_V(h)) * torch.sigmoid(self.att_U(h)))   # [N, n_ops] attention logits: the tile score of the attention model
    def tile_logits(self, x):
        h = self.trunk(x); return self.attention(h) if a.pool == "attn" else self.head(h)           # [N, n_ops]
    def forward(self, x):
        if a.pool == "attn":
            h = self.trunk(x).float(); A = self.attention(h).float(); w = torch.softmax(A, 0)             # attention over tiles, per operator
            z = w.t() @ h                                                                                  # [n_ops, hidden] one pooled embedding per operator
            return (self.head.weight.float() * z).sum(1) + self.head.bias.float(), A
        s = self.tile_logits(x).float(); n = s.shape[0]
        if a.pool == "noisyor":
            log_absent = -torch.nn.functional.softplus(s).sum(0)                     # sum_i log(1 - p_i)
            log_present = torch.log(-torch.expm1(log_absent).clamp(max=-1e-6))        # log(1 - exp(log_absent)), stable for both regimes
            return log_present - log_absent, s                                         # slide logit
        if a.pool == "lse": return torch.logsumexp(s, 0) - np.log(n), s
        k = max(1, int(round(a.topk_frac * n))); return s.topk(k, 0).values.mean(0), s

model = TileHead().to(dev); print(f"trainable parameters {sum(p.numel() for p in model.parameters())/1e6:.2f} M", flush=True)
opt = torch.optim.AdamW([{"params": model.trunk.parameters()}, {"params": model.head.weight}, {"params": [model.head.bias], "lr": 10 * a.lr, "weight_decay": 0.0}], lr=a.lr, weight_decay=a.wd)   # the bias sets the tile prior and has to move by units
steps = a.epochs * len(dl["train"]) // a.accum; warm = max(1, len(dl["train"]) // a.accum)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(steps, 1)))))
bce = torch.nn.functional.binary_cross_entropy_with_logits
def slide_loss(logit, y):
    m = y >= 0
    return bce(logit[m], y[m].float(), reduction="sum"), int(m.sum())

@torch.no_grad()
def run_eval(sp, collect=False):
    model.eval(); tot = n = 0; rows = []
    for b in dl[sp]:
        x = b["x"].to(dev, non_blocking=True).float(); y = b["y"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"): logit, _ = model(x)
        l, m = slide_loss(logit, y); tot += l.item(); n += m
        if collect: rows.append((b["slide"], torch.sigmoid(logit).cpu().numpy()))
    model.train(); return tot / max(n, 1), rows

hist = []; best = float("inf"); bad = 0; t0 = time.time(); ep0 = 0; stopped = False; CK = f"{OUT}/ckpt.pt"
def save_ckpt(ep, stopped):
    c = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), hist=hist, best=best, bad=bad, ep=ep, stopped=stopped, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None)
    torch.save(c, CK + ".tmp"); os.replace(CK + ".tmp", CK)
if a.resume and os.path.exists(CK):
    c = torch.load(CK, map_location=dev, weights_only=False); model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"])
    hist, best, bad, ep0, stopped = c["hist"], c["best"], c["bad"], c["ep"] + 1, c["stopped"]; torch.set_rng_state(c["rng"].cpu()); t0 = time.time() - (hist[-1]["time"] if hist else 0)      # map_location moved the saved states to the GPU, the setters take CPU byte tensors
    if c["cuda_rng"] is not None and dev.type == "cuda": torch.cuda.set_rng_state_all([x.cpu() for x in c["cuda_rng"]])
    print(f"resumed after epoch {c['ep']} (best val {best:.4f}, bad {bad}, stopped {stopped})", flush=True)
if a.eval_only: a.epochs = 0; best = float("nan"); print("eval only from", f"{OUT}/best.pt", flush=True)
for ep in range(ep0, 0 if stopped else a.epochs):
    model.train(); tl = tn = 0; opt.zero_grad(set_to_none=True)
    for i, b in enumerate(dl["train"]):
        x = b["x"].to(dev, non_blocking=True).float(); y = b["y"].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"): logit, _ = model(x)
        l, m = slide_loss(logit, y); (l / a.accum).backward(); tl += l.item(); tn += m
        if (i + 1) % a.accum == 0: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    va, _ = run_eval("val"); hist.append(dict(epoch=ep, train_nll=tl / max(tn, 1), val_nll=va, time=time.time() - t0)); print(json.dumps(hist[-1]), flush=True)
    torch.save(model.state_dict(), f"{OUT}/last.pt")
    if va < best - 1e-4: best = va; bad = 0; torch.save(model.state_dict(), f"{OUT}/best.pt")
    else: bad += 1
    save_ckpt(ep, bad >= a.patience)
    if os.environ.get("OPSIS_STOP_AFTER_EPOCH") == str(ep): raise SystemExit(f"stopped after epoch {ep} (OPSIS_STOP_AFTER_EPOCH, used to test --resume)")
    if bad >= a.patience: print("early stop", flush=True); break
json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1); model.load_state_dict(torch.load(f"{OUT}/best.pt", map_location=dev))

res = {"best_val_nll": best, "pool": a.pool, "ops": OPS}
for sp in [s for s in ds if s != "train"]:
    nll, rows = run_eval(sp, collect=True); P = {s: p for s, p in rows}; q = ds[sp]
    rec = []
    for r in Q[Q["split"] == sp].itertuples():
        p = float(P[r.slide][OPS.index(r.op)]); rec.append(dict(qid=r.qid, fov=r.fov, slide=r.slide, patient=r.patient, project=r.project, family=r.family, op=r.op, atype=r.atype, candidates=r.candidates, answer=int(r.answer), probs=[1 - p, p]))
    Pd = pd.DataFrame(rec); Pd.to_parquet(f"{OUT}/predictions_{sp}.parquet", index=False)
    key = "per_op" if sp == "test" else f"per_op_{sp}"; res[key] = {}; res["test_nll" if sp == "test" else f"test_nll_{sp}"] = nll
    for op, g in Pd.groupby("op"):
        pr = np.stack(g["probs"].to_numpy()); yy = g["answer"].to_numpy(); m = M.summarize(pr, yy, 2, g["atype"].iloc[0], g["patient"].to_numpy())
        m["auroc"] = float(roc_auc_score(yy, pr[:, 1])) if len(set(yy)) > 1 else None; res[key][op] = dict(family=FAMILY[op], **m)
    print(f"EVAL_{sp}", json.dumps({k: (round(v["acc"], 3), None if v["auroc"] is None else round(v["auroc"], 3)) for k, v in res[key].items()}), flush=True)
json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1); json.dump(vars(a), open(f"{OUT}/config.json", "w"), indent=1)
for f in ("config.json", "history.json", "metrics_test.json"): shutil.copy(f"{OUT}/{f}", f"{RES}/{f}")
if a.dump_slides:                                                                  # per tile logits for the tile level evaluation (e.g. BCSS necrosis masks)
    paths.ensure(f"{OUT}/tile_scores"); path = dict(zip(Q["slide"], Q["h5_path"])); split = dict(zip(Q["slide"], Q["split"])); model.eval(); done = 0
    for s in [l.strip() for l in open(a.dump_slides) if l.strip()]:
        if s not in path: continue
        with h5py.File(path[s], "r") as h: X = torch.from_numpy(h["features"][:]).to(dev).float(); C0 = h["coords_level0"][:]; ps = float(h.attrs.get("patch_size_level0", 512))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"): _, sc = model(X)
        np.savez_compressed(f"{OUT}/tile_scores/{s}.npz", logits=sc.cpu().numpy().astype(np.float16), coords_level0=C0, patch_size_level0=ps, ops=np.array(OPS), split=split[s]); done += 1
    print("tile scores written for", done, "slides", flush=True)
print("RUN_DONE", OUT)
