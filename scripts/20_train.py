#!/usr/bin/env python
"""Train and evaluate the field decoder of Opsis-S (OpsisNet) on cached tokens.
    python scripts/20_train.py --name jev_all_s0 [--protocol all|comp_holdout] [--no-ctx] [--epochs 30] [--seed 0]
Outputs under $OPSIS_DATA/runs/<name>/: config.json, vocab.json, best.pt, history.json, predictions_test.parquet,
metrics_test.json (per operator with patient bootstrap), timing.json. A copy of the json files goes to results/runs/<name>/."""
import argparse, json, os, shutil, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from opsis import paths, dataset as D, metrics as M
from opsis.model import OpsisNet, FixedHeadNet, question_loss, density_loss
from opsis.questions import FAMILY

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--protocol", default="all", choices=["all", "comp_holdout"]); ap.add_argument("--no-ctx", action="store_true"); ap.add_argument("--model", default="jev", choices=["jev", "fixed"]); ap.add_argument("--query", default="op", choices=["op", "primitives", "text", "text+op"]); ap.add_argument("--text-emb", default=None, help="npz from scripts/70_text_embeddings.py (default sapbert_questions)"); ap.add_argument("--text-trainable", action="store_true"); ap.add_argument("--ordinal-tau", type=float, default=0.0)
ap.add_argument("--aux-density", type=float, default=0.0, help="weight of the per token nucleus density loss (joint training)"); ap.add_argument("--obj-token", action="store_true", help="object token from a 2x crop of the target nucleus (scripts/90_object_crops.py)")
ap.add_argument("--qset", default="questions"); ap.add_argument("--extra-qset", default=None, help="second question set trained jointly (e.g. pannuke; no context tokens)"); ap.add_argument("--extra-enc", default="pannuke_virchow2"); ap.add_argument("--extra-split-col", default="split_fold"); ap.add_argument("--extra-states-per-epoch", type=int, default=1800); ap.add_argument("--split-col", default="split"); ap.add_argument("--init-from", default=None, help="run directory whose vocab.json and best.pt initialise the model (same question vocabulary required)"); ap.add_argument("--pool", action="store_true", help="add box pooled state tokens to the query (hybrid query)")
ap.add_argument("--split-seed", type=int, default=0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--q-per-state", type=int, default=32); ap.add_argument("--batch", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--wd", type=float, default=0.05)
ap.add_argument("--d", type=int, default=384); ap.add_argument("--enc-layers", type=int, default=2); ap.add_argument("--dec-layers", type=int, default=2); ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--enc", default="virchow2"); ap.add_argument("--size", type=int, default=448); ap.add_argument("--extra-size", type=int, default=None, help="canvas size of the extra token set (default: --size)"); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--eval-batch", type=int, default=4); ap.add_argument("--limit-train", type=int, default=None); ap.add_argument("--smoke", action="store_true", help="use the FOVs present in the token file for every split (pipeline test only)")
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = f"{paths.RUNS}/{a.name}"; paths.ensure(OUT); RES = f"{paths.RESULTS}/runs/{a.name}"; paths.ensure(RES)
Q, F, G, tokens_path = D.load_tables(a.split_seed, a.enc, a.size, a.qset)
Q = Q[Q["fov"].isin(set(G["fov"]))]
if a.split_col != "split": Q = Q.assign(split=Q[a.split_col])
Q = Q[Q["split"].isin(["train", "val", "test"])]
if a.smoke: Q = pd.concat([Q[Q["fov"].isin(sorted(Q["fov"].unique())[:4])].assign(split=s) for s in ("train", "val", "test")], ignore_index=True)   # 4 states, all splits
Q2 = F2 = G2 = tokens2 = None
if a.extra_qset:                                                                     # joint training with a second FOV set (PanNuke: FOV canvas only, no context)
    Q2, F2, G2, tokens2 = D.load_tables(a.split_seed, a.extra_enc, a.extra_size or a.size, a.extra_qset); Q2 = Q2[Q2["fov"].isin(set(G2["fov"]))]
    if a.extra_split_col != "split": Q2 = Q2.assign(split=Q2[a.extra_split_col])
    Q2 = Q2[Q2["split"].isin(["train", "val", "test"])]
    if a.smoke: Q2 = pd.concat([Q2[Q2["fov"].isin(sorted(Q2["fov"].unique())[:4])].assign(split=s) for s in ("train", "val", "test")], ignore_index=True)
    if a.limit_train: Q2 = Q2[(Q2["split"] != "train") | Q2["fov"].isin(sorted(Q2[Q2["split"] == "train"]["fov"].unique())[: a.limit_train])]
vocab = D.Vocab.from_dict(json.load(open(f"{a.init_from}/vocab.json"))) if a.init_from else D.Vocab(pd.concat([Q, Q2], ignore_index=True) if Q2 is not None else Q); json.dump(vocab.to_dict(), open(f"{OUT}/vocab.json", "w"))
train_q = Q[Q["split"] == "train"]
if a.protocol == "comp_holdout": train_q = train_q[train_q["family"] != "comp"]
if a.limit_train: train_q = train_q[train_q["fov"].isin(sorted(train_q["fov"].unique())[: a.limit_train])]
use_ctx = not a.no_ctx
DENS = OBJ = RECTS = None
if a.aux_density > 0:
    nuc_path = f"{paths.DERIVED}/facts/nuclei.csv" if a.qset == "questions" else f"{paths.DERIVED}/{a.qset}/nuclei.csv"
    DENS = pd.read_csv(nuc_path)
    if "super" not in DENS.columns: DENS["super"] = DENS["cls"].map({"neoplastic": "tumor_any", "inflammatory": "sTIL", "connective": "nonTIL_stromal", "dead": "other_nucleus", "epithelial": "other_nucleus"}); DENS["in_rect"] = True; DENS["labelled"] = True
    RECTS = {r["fov"]: (r.get("rect_x0", 0), r.get("rect_y0", 0), r.get("rect_x1", r["W"]), r.get("rect_y1", r["H"])) for r in F.to_dict("records")}
if a.obj_token:
    z = np.load(f"{paths.FEATURES}/virchow2/object_tokens_{a.qset}.npz"); OBJ = dict(zip(z["key"], z["cls"])); print("object tokens", len(OBJ), flush=True)
mk = lambda q, train, seed: D.StateQuestionDataset(q, F, G, tokens_path, vocab, a.q_per_state, train=train, use_ctx=use_ctx, seed=seed, query=a.query, density=DENS, obj=OBJ, rects=RECTS)
ds_tr = mk(train_q, True, a.seed); ds_va = mk(Q[Q["split"] == "val"], False, 0); ds_te = mk(Q[Q["split"] == "test"], False, 0)
print(f"states train {len(ds_tr)} val {len(ds_va)} test {len(ds_te)}; questions train {len(train_q)} val {len(ds_va.q)} test {len(ds_te.q)}; use_ctx={use_ctx}", flush=True)
dl_tr = DataLoader(ds_tr, a.batch, shuffle=True, num_workers=a.workers, collate_fn=D.collate, drop_last=True, persistent_workers=a.workers > 0)
dl_va = DataLoader(ds_va, a.eval_batch, shuffle=False, num_workers=a.workers, collate_fn=D.collate)
dl_te = DataLoader(ds_te, a.eval_batch, shuffle=False, num_workers=a.workers, collate_fn=D.collate)
dl_tr2 = dl_va2 = dl_te2 = ds_va2 = ds_te2 = None
if Q2 is not None:
    DENS2 = RECTS2 = None
    if a.aux_density > 0:
        DENS2 = pd.read_csv(f"{paths.DERIVED}/{a.extra_qset}/nuclei.csv")
        if "super" not in DENS2.columns: DENS2["super"] = DENS2["cls"].map({"neoplastic": "tumor_any", "inflammatory": "sTIL", "connective": "nonTIL_stromal", "dead": "other_nucleus", "epithelial": "other_nucleus"}); DENS2["in_rect"] = True; DENS2["labelled"] = True
        RECTS2 = {r["fov"]: (r.get("rect_x0", 0), r.get("rect_y0", 0), r.get("rect_x1", r["W"]), r.get("rect_y1", r["H"])) for r in F2.to_dict("records")}
    mk2 = lambda q, train, seed: D.StateQuestionDataset(q, F2, G2, tokens2, vocab, a.q_per_state, train=train, use_ctx=False, seed=seed, query=a.query, density=DENS2, obj=None, rects=RECTS2)
    ds_tr2 = mk2(Q2[Q2["split"] == "train"], True, a.seed); ds_va2 = mk2(Q2[Q2["split"] == "val"], False, 0); ds_te2 = mk2(Q2[Q2["split"] == "test"], False, 0)
    from torch.utils.data import RandomSampler
    sampler2 = RandomSampler(ds_tr2, replacement=False, num_samples=min(a.extra_states_per_epoch, len(ds_tr2)))
    dl_tr2 = DataLoader(ds_tr2, a.batch, sampler=sampler2, num_workers=a.workers, collate_fn=D.collate, drop_last=True, persistent_workers=a.workers > 0)
    dl_va2 = DataLoader(ds_va2, a.eval_batch, shuffle=False, num_workers=a.workers, collate_fn=D.collate); dl_te2 = DataLoader(ds_te2, a.eval_batch, shuffle=False, num_workers=a.workers, collate_fn=D.collate)
    print(f"extra set {a.extra_qset}: states train {len(ds_tr2)} (sampled {sampler2.num_samples} per epoch) val {len(ds_va2)} test {len(ds_te2)}; questions train {int((Q2['split'] == 'train').sum())}", flush=True)
text = None
if a.query in ("text", "text+op"):
    z = np.load(a.text_emb or f"{paths.DERIVED}/text_emb/sapbert_questions.npz"); tix = {t: i for i, t in enumerate(z["texts"])}; cix = {c: i for i, c in enumerate(z["cands"])}
    Et = np.stack([z["E_text"][tix[t]] for t in sorted(vocab.texts, key=vocab.texts.get)]); Ec = np.stack([z["E_cand"][cix[c]] for c in sorted(vocab.cand_texts, key=vocab.cand_texts.get)])
    text = {"E_text": Et, "E_cand": Ec, "mode": a.query}; print("text embeddings", Et.shape, Ec.shape, "mode", a.query, flush=True)
model = (OpsisNet(len(vocab.ops), len(vocab.args), len(vocab.cands), a.d, a.enc_layers, a.dec_layers, dropout=a.dropout, pool=a.pool, text=text, text_trainable=a.text_trainable, density=a.aux_density > 0) if a.model == "jev"
         else FixedHeadNet(len(vocab.keys), a.d, a.enc_layers, dropout=a.dropout)).to(dev)
n_par = sum(p.numel() for p in model.parameters()); print(f"trainable parameters {n_par/1e6:.1f} M", flush=True)
if a.init_from:
    from opsis.model import load_compat; load_compat(model, torch.load(f"{a.init_from}/best.pt", map_location=dev)); print("initialised from", a.init_from, flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
steps = a.epochs * (len(dl_tr) + (len(dl_tr2) if dl_tr2 is not None else 0)); warm = len(dl_tr)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(steps, 1)))))
def to_dev(b): return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
@torch.no_grad()
def run_eval(dl, ds, collect=False, uc=None):
    uc = use_ctx if uc is None else uc
    model.eval(); tot = n = 0; rows = []
    for b in dl:
        b = to_dev(b)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(b, uc)
        loss = question_loss(logits.float(), b); m = b["qmask"].sum().item(); tot += loss.item() * m; n += m
        if collect:
            p = torch.softmax(logits.float(), -1).cpu().numpy(); qi = b["qidx"].cpu().numpy(); qm = b["qmask"].cpu().numpy()
            for i in range(p.shape[0]):
                for j in np.flatnonzero(qm[i]): rows.append((int(qi[i, j]), p[i, j].tolist()))
    model.train(); return tot / max(n, 1), rows
hist = []; best = float("inf"); bad = 0; t0 = time.time()
for ep in range(a.epochs):
    model.train(); tl = tn = 0
    def train_batches():                                                              # NuCLS batches (with context) interleaved with extra set batches (no context)
        it2 = iter(dl_tr2) if dl_tr2 is not None else None
        for b in dl_tr:
            yield b, use_ctx
            if it2 is not None:
                try: yield next(it2), False
                except StopIteration: it2 = None
    for b, uc in train_batches():
        b = to_dev(b)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            out = model(b, uc, return_density=a.aux_density > 0) if a.model == "jev" else model(b, uc)
        logits, dens = out if isinstance(out, tuple) else (out, None)
        loss = question_loss(logits.float(), b, a.ordinal_tau)
        if dens is not None: loss = loss + a.aux_density * density_loss(dens, b)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        m = b["qmask"].sum().item(); tl += loss.item() * m; tn += m
    va, _ = run_eval(dl_va, ds_va); va2 = run_eval(dl_va2, ds_va2, uc=False)[0] if dl_va2 is not None else None
    hist.append(dict(epoch=ep, train_nll=tl / max(tn, 1), val_nll=va, val_nll_extra=va2, time=time.time() - t0)); print(json.dumps(hist[-1]), flush=True)
    if va2 is not None: va = 0.5 * (va + va2)
    if va < best - 1e-4: best = va; bad = 0; torch.save(model.state_dict(), f"{OUT}/best.pt")
    else:
        bad += 1
        if bad >= a.patience: print("early stop", flush=True); break
json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
if os.path.exists(f"{OUT}/best.pt"): model.load_state_dict(torch.load(f"{OUT}/best.pt", map_location=dev))
else: torch.save(model.state_dict(), f"{OUT}/best.pt")
@torch.no_grad()
def density_rules(dl, ds, S=500):
    """Counting family answered from the density head alone: Poisson samples of the summed token rates inside the box."""
    from opsis.questions import count_bin, cell_frac_bin, SUPER_SHORT
    CMAP = {c: c for c in SUPER_SHORT}; CMAP.update({"neoplastic": "tumor", "inflammatory": "sTIL", "connective": "stromal"})   # PanNuke names; dead and epithelial are merged into other and skipped
    def cidx(k): return SUPER_SHORT.index(CMAP[k]) if k in CMAP else None
    model.eval(); rows = []; rng = np.random.default_rng(0)
    for b in dl:
        b = to_dev(b)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            _, dens = model(b, use_ctx, return_density=True)
        rate = torch.exp(dens.float()).cpu().numpy(); N = rate.shape[1]; pos = b["pos"][:, :N, :2].cpu().numpy()
        box = b["box"].cpu().numpy(); box2 = b["box2"].cpu().numpy(); qi = b["qidx"].cpu().numpy(); qm = b["qmask"].cpu().numpy()
        for i in range(rate.shape[0]):
            for j in np.flatnonzero(qm[i]):
                q = ds.q.loc[int(qi[i, j])]; op = q["op"]
                if op not in ("count_bin", "present", "cell_frac_bin", "more_cells", "compare_regions"): continue
                args = json.loads(q["args"]); K = len(json.loads(q["candidates"]))
                if any(cidx(args[kk]) is None for kk in ("k", "k1", "k2") if kk in args): continue
                def lam(bx):
                    m = (pos[i, :, 0] >= bx[0]) & (pos[i, :, 0] < bx[2]) & (pos[i, :, 1] >= bx[1]) & (pos[i, :, 1] < bx[3]); return rate[i][m].sum(0)
                L1 = lam(box[i, j]); n1 = rng.poisson(np.maximum(L1, 1e-6), size=(S, 4))
                if op == "count_bin": h = np.bincount([count_bin(v) for v in n1[:, cidx(args["k"])]], minlength=5)[:5]
                elif op == "present": h = np.bincount((n1[:, cidx(args["k"])] >= 1).astype(int), minlength=2)[:2]
                elif op == "cell_frac_bin": tot = np.maximum(n1.sum(1), 1); h = np.bincount([cell_frac_bin(v) for v in n1[:, cidx(args["k"])] / tot], minlength=5)[:5]
                elif op == "more_cells": a1, a2 = n1[:, cidx(args["k1"])], n1[:, cidx(args["k2"])]; p2 = ((a2 > a1) + 0.5 * (a1 == a2)).mean(); h = np.array([1 - p2, p2]) * S
                else: n2 = rng.poisson(np.maximum(lam(box2[i, j]), 1e-6), size=(S, 4)); k = cidx(args["k"]); pb = ((n2[:, k] > n1[:, k]) + 0.5 * (n1[:, k] == n2[:, k])).mean(); h = np.array([1 - pb, pb]) * S
                pr = (np.asarray(h, float) + 0.5) / (np.asarray(h, float) + 0.5).sum(); rows.append((int(qi[i, j]), pr.tolist()))
    model.train(); return rows
def predictions(dl, ds, uc=None):
    nll, rows = run_eval(dl, ds, collect=True, uc=uc)
    P = pd.DataFrame({"qidx": [r[0] for r in rows], "probs": [r[1] for r in rows]})
    return nll, P.merge(ds.q[["qid", "fov", "patient", "family", "op", "atype", "candidates", "answer"]], left_on="qidx", right_index=True)
_, Pv = predictions(dl_va, ds_va); Pv.to_parquet(f"{OUT}/predictions_val.parquet", index=False)
te_nll, P = predictions(dl_te, ds_te); P.to_parquet(f"{OUT}/predictions_test.parquet", index=False)
res = {"test_nll": te_nll, "best_val_nll": best, "protocol": a.protocol, "use_ctx": use_ctx, "model": a.model, "query": a.query, "pool": a.pool, "ordinal_tau": a.ordinal_tau, "aux_density": a.aux_density, "obj_token": a.obj_token, "per_op": {}}
for op, g in P.groupby("op"):
    K = len(json.loads(g["candidates"].iloc[0])); res["per_op"][op] = dict(family=FAMILY[op], **M.summarize(np.stack(g["probs"].to_numpy()), g["answer"].to_numpy(), K, g["atype"].iloc[0], g["patient"].to_numpy()))
for fam in ["cell", "region", "comp", "organ"]:
    ops = [o for o in res["per_op"] if FAMILY[o] == fam]
    if ops: res[f"mean_acc_{fam}"] = float(np.mean([res["per_op"][o]["acc"] for o in ops])); res[f"mean_nll_{fam}"] = float(np.mean([res["per_op"][o]["nll"] for o in ops]))
if dl_te2 is not None:                                                                 # the extra set's own test split (fold or held out organs)
    te2_nll, P2 = predictions(dl_te2, ds_te2, uc=False); P2.to_parquet(f"{OUT}/predictions_test_extra.parquet", index=False); res["test_nll_extra"] = te2_nll; res["per_op_extra"] = {}
    for op, g in P2.groupby("op"):
        K = len(json.loads(g["candidates"].iloc[0])); res["per_op_extra"][op] = dict(family=FAMILY[op], **M.summarize(np.stack(g["probs"].to_numpy()), g["answer"].to_numpy(), K, g["atype"].iloc[0], g["patient"].to_numpy()))
    res["mean_acc_extra"] = float(np.mean([v["acc"] for v in res["per_op_extra"].values()]))
    print("TEST_EXTRA", a.extra_qset, json.dumps({k: round(v["acc"], 3) for k, v in res["per_op_extra"].items()}), flush=True)
if a.aux_density > 0 and a.model == "jev":
    rows = density_rules(dl_te, ds_te); R = pd.DataFrame({"qidx": [r[0] for r in rows], "probs": [r[1] for r in rows]}).merge(ds_te.q[["qid", "op", "atype", "candidates", "answer", "patient"]], left_on="qidx", right_index=True)
    res["density_rules"] = {}
    for op, g in R.groupby("op"):
        K = len(json.loads(g["candidates"].iloc[0])); pr = np.zeros((len(g), 7))
        for i, p in enumerate(g["probs"]): pr[i, :len(p)] = p
        res["density_rules"][op] = M.summarize(pr, g["answer"].to_numpy(), K, g["atype"].iloc[0], g["patient"].to_numpy(), n_boot=0)
    print("DENSITY RULES", {op: round(v["acc"], 3) for op, v in res["density_rules"].items()})
json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1)
print("TEST", json.dumps({k: v for k, v in res.items() if k not in ("per_op", "density_rules")}))
for op, r in res["per_op"].items(): print(f"  {op:26s} n={r['n']:6d} acc={r['acc']:.3f} (maj {r['majority_acc']:.3f}) f1={r['macro_f1']:.3f} nll={r['nll']:.3f} brier={r['brier']:.3f} ece={r['ece']:.3f}")
# timing: shared state + batched branches vs recomputing the state per question, on one test state
if dev.type == "cuda":
    b = to_dev(D.collate([ds_te[0]])); timing = {}
    for nq in (1, 8, 32):
        if b["op"].shape[1] < nq: continue
        sub = {k: (v[:, :nq] if torch.is_tensor(v) and v.dim() >= 2 and k not in ("tok_fov", "tok_ctx", "pos") else v) for k, v in b.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3): model(sub, use_ctx)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(10): model(sub, use_ctx)
            torch.cuda.synchronize(); shared = (time.time() - t) / 10
            t = time.time()
            for _ in range(10):
                for j in range(nq): model({k: (v[:, j:j + 1] if torch.is_tensor(v) and v.dim() >= 2 and k not in ("tok_fov", "tok_ctx", "pos") else v) for k, v in sub.items()}, use_ctx)
            torch.cuda.synchronize(); serial = (time.time() - t) / 10
        timing[nq] = dict(shared_ms=shared * 1e3, serial_ms=serial * 1e3, speedup=serial / shared)
    json.dump(timing, open(f"{OUT}/timing.json", "w"), indent=1); print("timing (adapter + branches, excludes the frozen encoder):", json.dumps(timing))
json.dump(vars(a), open(f"{OUT}/config.json", "w"), indent=1)
for f in ("config.json", "vocab.json", "history.json", "metrics_test.json", "timing.json"):
    if os.path.exists(f"{OUT}/{f}"): shutil.copy(f"{OUT}/{f}", f"{RES}/{f}")
print("RUN_DONE", OUT)
