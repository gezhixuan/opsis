#!/usr/bin/env python
"""Tuned state: the field decoder of Opsis-S (scripts/20) with the last blocks of Virchow2 unfrozen. The canvases (FOV png
resampled to 0.25 mpp on a 448 white canvas; 448 um context at 1.0 mpp; PanNuke 256 px pasted at the centre) are encoded
online instead of read from the token cache, so gradients from the question loss (and the density loss) reach the encoder.
Inference cost is unchanged: the encoder still runs once per state. The decoder is initialised from a frozen-state run
(--init-from runs/final_fov_s0, same vocabulary), so epoch 0 reproduces that run and the difference is the tuned encoder.
    python scripts/21_train_tuned_state.py --name final_fov_tuned_s0 --init-from $OPSIS_DATA/runs/final_fov_s0 --qset questions_final --extra-qset pannuke --pool --aux-density 1.0 [--tune-blocks 4] [--freeze-encoder]
Outputs like scripts/20 (metrics_test.json with per_op / per_op_extra) plus encoder_tuned.pt (the trainable encoder parameters)."""
import argparse, json, os, shutil, time
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from opsis import paths, facts, encoder as E, dataset as D, metrics as M
from opsis.data import nucls
from opsis.model import OpsisNet, question_loss, density_loss, load_compat
from opsis.questions import FAMILY

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--init-from", required=True, help="run directory (vocab.json + best.pt) of the frozen state run")
ap.add_argument("--qset", default="questions_final"); ap.add_argument("--extra-qset", default="pannuke"); ap.add_argument("--extra-enc", default="pannuke_virchow2"); ap.add_argument("--extra-split-col", default="split_fold"); ap.add_argument("--extra-states-per-epoch", type=int, default=1800)
ap.add_argument("--protocol", default="all", choices=["all", "comp_holdout"]); ap.add_argument("--query", default="op"); ap.add_argument("--pool", action="store_true"); ap.add_argument("--aux-density", type=float, default=1.0); ap.add_argument("--no-ctx", action="store_true")
ap.add_argument("--tune-blocks", type=int, default=4, help="number of final Virchow2 blocks (of 32) that are trained; 0 with --freeze-encoder reproduces the frozen run"); ap.add_argument("--freeze-encoder", action="store_true")
ap.add_argument("--enc-lr", type=float, default=1e-5); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--patience", type=int, default=3)
ap.add_argument("--q-per-state", type=int, default=32); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--eval-batch", type=int, default=8); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--aug", action="store_true", help="colour jitter on the training canvases")
ap.add_argument("--d", type=int, default=384); ap.add_argument("--enc-layers", type=int, default=2); ap.add_argument("--dec-layers", type=int, default=2); ap.add_argument("--dropout", type=float, default=0.1); ap.add_argument("--size", type=int, default=448)
ap.add_argument("--split-seed", type=int, default=0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda"); OUT = f"{paths.RUNS}/{a.name}"; paths.ensure(OUT); RES = f"{paths.RESULTS}/runs/{a.name}"; paths.ensure(RES); use_ctx = not a.no_ctx
CTX = f"{paths.DERIVED}/context/mpp{facts.CTX_MPP}_{facts.CTX_PX}"

Q, F, G, tokens_path = D.load_tables(a.split_seed, "virchow2", a.size, a.qset); Q = Q[Q["fov"].isin(set(G["fov"]))]; Q = Q[Q["split"].isin(["train", "val", "test"])]
Q2, F2, G2, tokens2 = D.load_tables(a.split_seed, a.extra_enc, a.size, a.extra_qset); Q2 = Q2[Q2["fov"].isin(set(G2["fov"]))]
if a.extra_split_col != "split": Q2 = Q2.assign(split=Q2[a.extra_split_col])
Q2 = Q2[Q2["split"].isin(["train", "val", "test"])]
if a.smoke:
    Q = pd.concat([Q[Q["fov"].isin(sorted(Q["fov"].unique())[:4])].assign(split=s) for s in ("train", "val", "test")], ignore_index=True)
    Q2 = pd.concat([Q2[Q2["fov"].isin(sorted(Q2["fov"].unique())[:4])].assign(split=s) for s in ("train", "val", "test")], ignore_index=True); a.epochs = 1; a.extra_states_per_epoch = 4
vocab = D.Vocab.from_dict(json.load(open(f"{a.init_from}/vocab.json"))); json.dump(vocab.to_dict(), open(f"{OUT}/vocab.json", "w"))
train_q = Q[Q["split"] == "train"]
if a.protocol == "comp_holdout": train_q = train_q[train_q["family"] != "comp"]
def dens_tables(qset, Fx):
    if a.aux_density <= 0: return None, None
    nd = pd.read_csv(f"{paths.DERIVED}/facts/nuclei.csv" if qset == "questions" else f"{paths.DERIVED}/{qset}/nuclei.csv")
    if "super" not in nd.columns: nd["super"] = nd["cls"].map({"neoplastic": "tumor_any", "inflammatory": "sTIL", "connective": "nonTIL_stromal", "dead": "other_nucleus", "epithelial": "other_nucleus"}); nd["in_rect"] = True; nd["labelled"] = True
    rects = {r["fov"]: (r.get("rect_x0", 0), r.get("rect_y0", 0), r.get("rect_x1", r["W"]), r.get("rect_y1", r["H"])) for r in Fx.to_dict("records")}
    return nd, rects
DENS, RECTS = dens_tables(a.qset, F); DENS2, RECTS2 = dens_tables(a.extra_qset, F2)
JIT = transforms.ColorJitter(0.1, 0.1, 0.1, 0.02)

class CanvasDataset(D.StateQuestionDataset):
    """The token cache is replaced by the canvases themselves (uint8), rebuilt from the recorded geometry."""
    def __init__(self, *args, rgb=None, ctx_dir=None, train=False, **kw):
        super().__init__(*args, train=train, **kw); self.rgb = rgb; self.ctx_dir = ctx_dir; self.aug = a.aug and train
    def _tokens(self, i): return np.zeros((1, 1280), np.float16), (np.zeros((1, 1280), np.float16) if self.use_ctx else None)
    def canvas(self, f):
        g = self.geo.loc[f]; im = Image.open(self.rgb(f)).convert("RGB"); w, h = int(g["fov_w"]), int(g["fov_h"])
        if im.size != (w, h): im = im.resize((w, h), Image.BICUBIC)
        c = Image.new("RGB", (self.size, self.size), (255, 255, 255)); c.paste(im, (int(g["off_x"]), int(g["off_y"]))); return c
    def __getitem__(self, k):
        it = super().__getitem__(k); f = self.fov_list[k]; ims = [self.canvas(f)] + ([Image.open(f"{self.ctx_dir}/{f}.png").convert("RGB")] if self.use_ctx else [])
        if self.aug: ims = [JIT(im) for im in ims]
        it["img"] = torch.from_numpy(np.stack([np.asarray(im, dtype=np.uint8) for im in ims])).permute(0, 3, 1, 2)      # [n_canvas, 3, S, S]
        return it
def collate(items):
    out = D.collate(items); out["img"] = torch.stack([it["img"] for it in items]); return out
mk = lambda q, train, seed: CanvasDataset(q, F, G, tokens_path, vocab, a.q_per_state, train=train, use_ctx=use_ctx, seed=seed, query=a.query, density=DENS, rects=RECTS, rgb=nucls.rgb_path, ctx_dir=CTX)
mk2 = lambda q, train, seed: CanvasDataset(q, F2, G2, tokens2, vocab, a.q_per_state, train=train, use_ctx=False, seed=seed, query=a.query, density=DENS2, rects=RECTS2, rgb=lambda f: f"{paths.DERIVED}/{a.extra_qset}/rgb/{f}.png", ctx_dir=None)
ds = {"train": mk(train_q, True, a.seed), "val": mk(Q[Q["split"] == "val"], False, 0), "test": mk(Q[Q["split"] == "test"], False, 0)}
ds2 = {"train": mk2(Q2[Q2["split"] == "train"], True, a.seed), "val": mk2(Q2[Q2["split"] == "val"], False, 0), "test": mk2(Q2[Q2["split"] == "test"], False, 0)}
DL = lambda d, bs, **kw: DataLoader(d, bs, num_workers=a.workers, collate_fn=collate, persistent_workers=a.workers > 0, **kw)
dl = {"train": DL(ds["train"], a.batch, shuffle=True, drop_last=True), "val": DL(ds["val"], a.eval_batch), "test": DL(ds["test"], a.eval_batch)}
sampler2 = RandomSampler(ds2["train"], replacement=False, num_samples=min(a.extra_states_per_epoch, len(ds2["train"])))
dl2 = {"train": DL(ds2["train"], a.batch, sampler=sampler2, drop_last=True), "val": DL(ds2["val"], a.eval_batch), "test": DL(ds2["test"], a.eval_batch)}
print(f"states train {len(ds['train'])} val {len(ds['val'])} test {len(ds['test'])}; extra train {len(ds2['train'])} (sampled {sampler2.num_samples}) val {len(ds2['val'])} test {len(ds2['test'])}", flush=True)

enc = E.load_virchow2(dev, img_size=a.size); enc.set_grad_checkpointing(True)
for p in enc.parameters(): p.requires_grad_(False)
enc_params = []
if not a.freeze_encoder and a.tune_blocks > 0:
    for blk in enc.blocks[-a.tune_blocks:]:
        for p in blk.parameters(): p.requires_grad_(True); enc_params.append(p)
    for p in enc.norm.parameters(): p.requires_grad_(True); enc_params.append(p)
MEAN = torch.tensor(E.IMAGENET_MEAN, device=dev).view(1, 3, 1, 1); STD = torch.tensor(E.IMAGENET_STD, device=dev).view(1, 3, 1, 1)
def encode(img, train):
    """img uint8 [B, n_canvas, 3, S, S] -> tokens [B, n_canvas, N, 1280] (bf16); gradient only when the encoder is trained."""
    B, C = img.shape[:2]; x = (img.flatten(0, 1).to(dev, non_blocking=True).float() / 255.0 - MEAN) / STD
    with torch.set_grad_enabled(train and bool(enc_params)), torch.autocast("cuda", dtype=torch.bfloat16):
        t = enc.forward_features(x)[:, enc.n_prefix:]
    return t.reshape(B, C, t.shape[1], t.shape[2])
model = OpsisNet(len(vocab.ops), len(vocab.args), len(vocab.cands), a.d, a.enc_layers, a.dec_layers, dropout=a.dropout, pool=a.pool, density=a.aux_density > 0).to(dev)
load_compat(model, torch.load(f"{a.init_from}/best.pt", map_location=dev)); print(f"readout initialised from {a.init_from}; encoder trainable parameters {sum(p.numel() for p in enc_params)/1e6:.1f} M in the last {a.tune_blocks} blocks", flush=True)
groups = [{"params": list(model.parameters()), "lr": a.lr}] + ([{"params": enc_params, "lr": a.enc_lr}] if enc_params else [])
opt = torch.optim.AdamW(groups, weight_decay=a.wd); steps = a.epochs * (len(dl["train"]) + len(dl2["train"])); warm = max(1, len(dl["train"]) // 2)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(steps, 1)))))
def to_dev(b): return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
def with_tokens(b, uc, train):
    t = encode(b["img"], train); b = to_dev(b); b["tok_fov"] = t[:, 0]
    if uc: b["tok_ctx"] = t[:, 1]
    return b
@torch.no_grad()
def run_eval(loader, dset, uc, collect=False):
    model.eval(); enc.eval(); tot = n = 0; rows = []
    for b in loader:
        b = with_tokens(b, uc, False)
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(b, uc)
        loss = question_loss(logits.float(), b); m = b["qmask"].sum().item(); tot += loss.item() * m; n += m
        if collect:
            p = torch.softmax(logits.float(), -1).cpu().numpy(); qi = b["qidx"].cpu().numpy(); qm = b["qmask"].cpu().numpy()
            for i in range(p.shape[0]):
                for j in np.flatnonzero(qm[i]): rows.append((int(qi[i, j]), p[i, j].tolist()))
    model.train(); enc.train(bool(enc_params)); return tot / max(n, 1), rows
def save_best():
    torch.save(model.state_dict(), f"{OUT}/best.pt"); torch.save({n: p.detach().cpu() for n, p in enc.named_parameters() if p.requires_grad}, f"{OUT}/encoder_tuned.pt")
hist = []; best = float("inf"); bad = 0; t0 = time.time()
va0, _ = run_eval(dl["val"], ds["val"], use_ctx); va20, _ = run_eval(dl2["val"], ds2["val"], False); print(json.dumps(dict(epoch=-1, val_nll=va0, val_nll_extra=va20, note="initial readout, untuned encoder")), flush=True)
best = 0.5 * (va0 + va20); save_best()
for ep in range(a.epochs):
    model.train(); enc.train(bool(enc_params)); tl = tn = 0; it2 = iter(dl2["train"])
    def batches():
        for b in dl["train"]:
            yield b, use_ctx
            try: yield next(it2), False
            except StopIteration: pass
    for b, uc in batches():
        b = with_tokens(b, uc, True)
        with torch.autocast("cuda", dtype=torch.bfloat16): out = model(b, uc, return_density=a.aux_density > 0)
        logits, dens = out if isinstance(out, tuple) else (out, None)
        loss = question_loss(logits.float(), b)
        if dens is not None: loss = loss + a.aux_density * density_loss(dens, b)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(list(model.parameters()) + enc_params, 1.0); opt.step(); sched.step()
        m = b["qmask"].sum().item(); tl += loss.item() * m; tn += m
    va, _ = run_eval(dl["val"], ds["val"], use_ctx); va2, _ = run_eval(dl2["val"], ds2["val"], False)
    hist.append(dict(epoch=ep, train_nll=tl / max(tn, 1), val_nll=va, val_nll_extra=va2, time=time.time() - t0)); print(json.dumps(hist[-1]), flush=True)
    crit = 0.5 * (va + va2)
    if crit < best - 1e-4: best = crit; bad = 0; save_best()
    else:
        bad += 1
        if bad >= a.patience: print("early stop", flush=True); break
json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1); model.load_state_dict(torch.load(f"{OUT}/best.pt", map_location=dev))
if enc_params:
    sd = torch.load(f"{OUT}/encoder_tuned.pt", map_location=dev); ps = dict(enc.named_parameters())
    with torch.no_grad():
        for n, v in sd.items(): ps[n].copy_(v)
def predictions(loader, dset, uc):
    nll, rows = run_eval(loader, dset, uc, collect=True); P = pd.DataFrame({"qidx": [r[0] for r in rows], "probs": [r[1] for r in rows]})
    return nll, P.merge(dset.q[["qid", "fov", "patient", "family", "op", "atype", "candidates", "answer"]], left_on="qidx", right_index=True)
def per_op(P):
    out = {}
    for op, g in P.groupby("op"):
        K = int(g["candidates"].map(lambda c: len(json.loads(c))).max()); out[op] = dict(family=FAMILY[op], **M.summarize(np.stack(g["probs"].to_numpy()), g["answer"].to_numpy(), K, g["atype"].iloc[0], g["patient"].to_numpy()))
    return out
_, Pv = predictions(dl["val"], ds["val"], use_ctx); Pv.to_parquet(f"{OUT}/predictions_val.parquet", index=False)
te_nll, P = predictions(dl["test"], ds["test"], use_ctx); P.to_parquet(f"{OUT}/predictions_test.parquet", index=False)
res = {"test_nll": te_nll, "best_val_crit": best, "protocol": a.protocol, "use_ctx": use_ctx, "model": "jev_tuned_state", "query": a.query, "pool": a.pool, "aux_density": a.aux_density, "tune_blocks": 0 if a.freeze_encoder else a.tune_blocks, "init_from": a.init_from, "per_op": per_op(P)}
for fam in ["cell", "region", "comp", "organ"]:
    ops = [o for o in res["per_op"] if FAMILY[o] == fam]
    if ops: res[f"mean_acc_{fam}"] = float(np.mean([res["per_op"][o]["acc"] for o in ops])); res[f"mean_nll_{fam}"] = float(np.mean([res["per_op"][o]["nll"] for o in ops]))
te2, P2 = predictions(dl2["test"], ds2["test"], False); P2.to_parquet(f"{OUT}/predictions_test_extra.parquet", index=False); res["test_nll_extra"] = te2; res["per_op_extra"] = per_op(P2); res["mean_acc_extra"] = float(np.mean([v["acc"] for v in res["per_op_extra"].values()]))
print("TEST_EXTRA", a.extra_qset, json.dumps({k: round(v["acc"], 3) for k, v in res["per_op_extra"].items()}), flush=True)
json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1); json.dump(vars(a), open(f"{OUT}/config.json", "w"), indent=1)
print("TEST", json.dumps({k: v for k, v in res.items() if not k.startswith("per_op")}))
for op, r in res["per_op"].items(): print(f"  {op:26s} n={r['n']:6d} acc={r['acc']:.3f} (maj {r['majority_acc']:.3f}) nll={r['nll']:.3f} ece={r['ece']:.3f}")
for f in ("config.json", "vocab.json", "history.json", "metrics_test.json"): shutil.copy(f"{OUT}/{f}", f"{RES}/{f}")
print("RUN_DONE", OUT)
