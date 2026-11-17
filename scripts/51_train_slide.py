#!/usr/bin/env python
"""The slide decoder of Opsis-S: the question decoder of scripts/20 over a slide state made of K random 20x tile
features (Virchow2, 2,560 d) plus their slide coordinates; questions = report slots and molecular status.
    python scripts/51_train_slide.py --name slide_jev_s0 [--model jev|fixed] [--k-tiles 4096] [--epochs 40] [--qset slide_questions_v2] [--query text+op]
Outputs like scripts/20_train.py (runs/<name>/, results/runs/<name>/metrics_test.json)."""
import argparse, json, os, shutil, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from opsis import paths, dataset as D, metrics as M
from opsis.slide_dataset import SlideQuestionDataset
from opsis.model import OpsisNet, FixedHeadNet, question_loss
from opsis.questions import FAMILY

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--model", default="jev", choices=["jev", "fixed"]); ap.add_argument("--split-seed", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--patience", type=int, default=8); ap.add_argument("--k-tiles", type=int, default=4096); ap.add_argument("--q-per-state", type=int, default=16)
ap.add_argument("--batch", type=int, default=4); ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--d", type=int, default=384); ap.add_argument("--enc-layers", type=int, default=2); ap.add_argument("--dec-layers", type=int, default=2); ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit-train", type=int, default=None); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--h5-dir", default=f"{paths.TCGA}/gigapath_h5/brca_trident_virchow2/h5_files")
ap.add_argument("--qset", default="slide_questions", help="question set directory under derived/ (slide_questions, slide_questions_v2, slidebench_brca); comma separated sets are concatenated")
ap.add_argument("--ops", default=None, help="comma separated operators to keep (default all); --ops-exclude removes operators")
ap.add_argument("--ops-exclude", default=None)
ap.add_argument("--exclude-families", default=None, help="families removed from TRAINING only (e.g. composite): their questions stay in val/test for the held out evaluation")
ap.add_argument("--eval-splits", default="test", help="extra splits evaluated and written as predictions_<split>.parquet (e.g. test+heldout_project)")
ap.add_argument("--blank-state", action="store_true", help="control: slide tokens replaced by zeros, so only the question is available (language prior)")
ap.add_argument("--eval-only", action="store_true", help="skip training: load runs/<name>/best.pt and write the predictions and metrics (recovers a run cut by the time limit)")
ap.add_argument("--init-from", default=None, help="run dir whose best.pt initialises the model (load_compat; vocab must match)")
ap.add_argument("--query", default="op", choices=["op", "primitives", "text", "text+op"]); ap.add_argument("--text-emb", default=None, help="npz from scripts/70_text_embeddings.py (default sapbert_<qset>)"); ap.add_argument("--text-trainable", action="store_true")
a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); OUT = f"{paths.RUNS}/{a.name}"; paths.ensure(OUT); RES = f"{paths.RESULTS}/runs/{a.name}"; paths.ensure(RES)
LIST = lambda v: [x for x in v.replace("+", ",").split(",") if x]            # "+" works where commas are eaten (sbatch --export)
QSETS = LIST(a.qset); Q = pd.concat([pd.read_parquet(f"{paths.DERIVED}/{q}/questions_seed{a.split_seed}.parquet") for q in QSETS], ignore_index=True)
if a.ops: Q = Q[Q["op"].isin(LIST(a.ops))].reset_index(drop=True)
if a.ops_exclude: Q = Q[~Q["op"].isin(LIST(a.ops_exclude))].reset_index(drop=True)
print("question sets", QSETS, "questions", len(Q), "ops", sorted(Q["op"].unique()), flush=True)
if a.smoke:
    keep = sorted(Q["fov"].unique())[:4]; Q = Q[Q["fov"].isin(keep)]; Q = pd.concat([Q.assign(split=s) for s in ("train", "val", "test")], ignore_index=True)
vocab = D.Vocab.from_dict(json.load(open(f"{a.init_from}/vocab.json"))) if a.init_from else D.Vocab(Q); json.dump(vocab.to_dict(), open(f"{OUT}/vocab.json", "w"))
tq = Q[Q["split"] == "train"]
if a.exclude_families: tq = tq[~tq["family"].isin(LIST(a.exclude_families))]
if a.limit_train: tq = tq[tq["fov"].isin(sorted(tq["fov"].unique())[: a.limit_train])]
mk = lambda q, train, seed: SlideQuestionDataset(q, a.h5_dir, vocab, a.k_tiles, a.q_per_state, train=train, seed=seed, query=a.query, blank=a.blank_state)
EVAL = [x for x in LIST(a.eval_splits) if x != "test"]
ds = {"train": mk(tq, True, a.seed), "val": mk(Q[Q["split"] == "val"], False, 0), "test": mk(Q[Q["split"] == "test"], False, 0), **{sp: mk(Q[Q["split"] == sp], False, 0) for sp in EVAL if (Q["split"] == sp).any()}}
print({s: (len(ds[s]), len(ds[s].q)) for s in ds}, flush=True)
dl = {s: DataLoader(ds[s], a.batch, shuffle=(s == "train"), num_workers=a.workers, collate_fn=D.collate, drop_last=(s == "train"), persistent_workers=a.workers > 0) for s in ds}
text = None
if a.query in ("text", "text+op"):
    tix, cix = {}, {}
    for z in [np.load(f) for f in (LIST(a.text_emb) if a.text_emb else [f"{paths.DERIVED}/text_emb/sapbert_{q}.npz" for q in QSETS])]:
        tix.update({t: z["E_text"][i] for i, t in enumerate(z["texts"])}); cix.update({c: z["E_cand"][i] for i, c in enumerate(z["cands"])})
    Et = np.stack([tix[t] for t in sorted(vocab.texts, key=vocab.texts.get)]); Ec = np.stack([cix[c] for c in sorted(vocab.cand_texts, key=vocab.cand_texts.get)])
    text = {"E_text": Et, "E_cand": Ec, "mode": a.query}; print("text embeddings", Et.shape, Ec.shape, "mode", a.query, flush=True)
model = (OpsisNet(len(vocab.ops), len(vocab.args), len(vocab.cands), a.d, a.enc_layers, a.dec_layers, dropout=a.dropout, d_in=2560, text=text, text_trainable=a.text_trainable) if a.model == "jev"
         else FixedHeadNet(len(vocab.keys), a.d, a.enc_layers, dropout=a.dropout, d_in=2560)).to(dev)
if a.init_from:
    from opsis.model import load_compat; load_compat(model, torch.load(f"{a.init_from}/best.pt", map_location=dev)); print("initialised from", a.init_from, flush=True)
print(f"trainable parameters {sum(p.numel() for p in model.parameters())/1e6:.1f} M", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); steps = a.epochs * len(dl["train"]); warm = max(1, len(dl["train"]))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(steps, 1)))))
def to_dev(b): return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
@torch.no_grad()
def run_eval(split, collect=False):
    model.eval(); tot = n = 0; rows = []
    for b in dl[split]:
        b = to_dev(b)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"): logits = model(b, False)
        loss = question_loss(logits.float(), b); m = b["qmask"].sum().item(); tot += loss.item() * m; n += m
        if collect:
            p = torch.softmax(logits.float(), -1).cpu().numpy(); qi = b["qidx"].cpu().numpy(); qm = b["qmask"].cpu().numpy()
            for i in range(p.shape[0]):
                for j in np.flatnonzero(qm[i]): rows.append((int(qi[i, j]), p[i, j].tolist()))
    model.train(); return tot / max(n, 1), rows
hist = []; best = float("inf"); bad = 0; t0 = time.time()
if a.eval_only: a.epochs = 0; hist = json.load(open(f"{OUT}/history.json")) if os.path.exists(f"{OUT}/history.json") else []; best = min([h["val_nll"] for h in hist], default=float("nan")); print("eval only from", f"{OUT}/best.pt", flush=True)
for ep in range(a.epochs):
    model.train(); tl = tn = 0
    for b in dl["train"]:
        b = to_dev(b)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"): logits = model(b, False)
        loss = question_loss(logits.float(), b); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        m = b["qmask"].sum().item(); tl += loss.item() * m; tn += m
    va, _ = run_eval("val"); hist.append(dict(epoch=ep, train_nll=tl / max(tn, 1), val_nll=va, time=time.time() - t0)); print(json.dumps(hist[-1]), flush=True)
    if va < best - 1e-4: best = va; bad = 0; torch.save(model.state_dict(), f"{OUT}/best.pt")
    else:
        bad += 1
        if bad >= a.patience: print("early stop", flush=True); break
json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1); model.load_state_dict(torch.load(f"{OUT}/best.pt", map_location=dev))
def predictions(split):
    nll, rows = run_eval(split, collect=True); P = pd.DataFrame({"qidx": [r[0] for r in rows], "probs": [r[1] for r in rows]})
    return nll, P.merge(ds[split].q[["qid", "fov", "patient", "family", "op", "atype", "candidates", "answer"]], left_on="qidx", right_index=True)
_, Pv = predictions("val"); Pv.to_parquet(f"{OUT}/predictions_val.parquet", index=False)
te_nll, P = predictions("test"); P.to_parquet(f"{OUT}/predictions_test.parquet", index=False)
res = {"test_nll": te_nll, "best_val_nll": best, "model": a.model, "k_tiles": a.k_tiles, "per_op": {}}
for op, g in P.groupby("op"):
    K = int(g["candidates"].map(lambda c: len(json.loads(c))).max()); res["per_op"][op] = dict(family=FAMILY[op], **M.summarize(np.stack(g["probs"].to_numpy()), g["answer"].to_numpy(), K, g["atype"].iloc[0], g["patient"].to_numpy()))
for sp in [x for x in ds if x not in ("train", "val", "test")]:                   # held out projects etc.
    sp_nll, Ps = predictions(sp); Ps.to_parquet(f"{OUT}/predictions_{sp}.parquet", index=False); res[f"test_nll_{sp}"] = sp_nll; res[f"per_op_{sp}"] = {}
    for op, g in Ps.groupby("op"):
        K = int(g["candidates"].map(lambda c: len(json.loads(c))).max()); res[f"per_op_{sp}"][op] = dict(family=FAMILY[op], **M.summarize(np.stack(g["probs"].to_numpy()), g["answer"].to_numpy(), K, g["atype"].iloc[0], g["patient"].to_numpy()))
    print(f"EVAL_{sp}", json.dumps({k: round(v["acc"], 3) for k, v in res[f"per_op_{sp}"].items()}), flush=True)
for fam in ["slide", "molecular", "claim", "external", "composite"]:
    ops = [o for o in res["per_op"] if FAMILY[o] == fam]
    if ops: res[f"mean_acc_{fam}"] = float(np.mean([res["per_op"][o]["acc"] for o in ops])); res[f"mean_nll_{fam}"] = float(np.mean([res["per_op"][o]["nll"] for o in ops]))
json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1); json.dump(vars(a), open(f"{OUT}/config.json", "w"), indent=1)
print("TEST", json.dumps({k: v for k, v in res.items() if k != "per_op"}))
for op, r in res["per_op"].items(): print(f"  {op:18s} n={r['n']:5d} acc={r['acc']:.3f} (maj {r['majority_acc']:.3f}) f1={r['macro_f1']:.3f} nll={r['nll']:.3f} ece={r['ece']:.3f}")
for f in ("config.json", "vocab.json", "history.json", "metrics_test.json"): shutil.copy(f"{OUT}/{f}", f"{RES}/{f}")
print("RUN_DONE", OUT)
