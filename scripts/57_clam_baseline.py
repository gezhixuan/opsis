#!/usr/bin/env python
"""CLAM-SB baseline on the slide level labels (external comparable), and with --model abmil the gated
attention MIL of Ilse et al. 2018 (the same attention network and classifier as CLAM-SB, no instance clustering loss: the
default slide level read out of the foundation model papers; prefix abmil_ / abmil_pan_). The CLAM model code from
the cloned repository (baselines/CLAM beside the repository root) trained with its default recipe (Adam 1e-4, weight decay 1e-5,
dropout 0.25, gated attention, instance clustering loss with B = 8, bag weight 0.7, early stopping patience 20 after
epoch 50, max 200 epochs, one bag per step, all cached tiles of the slide) on our patient split, one head per label.
    python scripts/57_clam_baseline.py --task er_positive [--seeds 0,1,2] [--model clam_sb|clam_mb]
    python scripts/57_clam_baseline.py --qset v2 --task lvi_present            (label from the v2 question set, not the labels csv)
    python scripts/57_clam_baseline.py --qset pan --task pan_necrosis --max-tiles 4096 --prefix clam_pan_   (pan TCGA question set)
--qset reads slide, patient, split, answer (and h5_path for pan) from derived/slide_questions_<qset>/questions_seed0.parquet,
one row per slide and operator; the heldout_project split is not used. The pan runs keep 4,096 random tiles per slide
(the full bags of 9,567 slides do not fit in memory); the BRCA runs keep every tile.
Writes runs/<prefix><task>_s<k>/metrics_test.json in the format of scripts/51 (per_op with one operator) plus AUROC / AP.
--task takes several operators joined by + (the bags of all of them are loaded once, then every task and seed is trained in turn,
with the same seeding as a single task call). --heldout (pan) also scores the heldout_project slides into per_op_heldout_project.
--resume does not retrain a (task, seed) whose best.pt exists: it reloads it and rewrites the metrics (used to add the held out
projects to finished runs)."""
import argparse, json, os, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, torch, h5py
from sklearn.metrics import roc_auc_score, average_precision_score
from opsis import paths, metrics as M
from opsis.questions import FAMILY
sys.path.insert(0, os.path.join(os.path.dirname(paths.ROOT), "baselines", "CLAM"))
from models.model_clam import CLAM_SB, CLAM_MB

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True); ap.add_argument("--seeds", default="0"); ap.add_argument("--model", default="clam_sb", choices=["clam_sb", "clam_mb", "abmil"])
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--reg", type=float, default=1e-5); ap.add_argument("--drop-out", type=float, default=0.25); ap.add_argument("--bag-weight", type=float, default=0.7); ap.add_argument("--B", type=int, default=8)
ap.add_argument("--max-epochs", type=int, default=200); ap.add_argument("--patience", type=int, default=20); ap.add_argument("--stop-epoch", type=int, default=50)
ap.add_argument("--labels", default=None); ap.add_argument("--h5-dir", default=f"{paths.TCGA}/gigapath_h5/brca_trident_virchow2/h5_files")
ap.add_argument("--max-tiles", type=int, default=0, help="0 = all tiles; otherwise a fixed random subset per slide (memory)"); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--heldout", action="store_true", help="pan: also score the heldout_project slides (per_op_heldout_project)"); ap.add_argument("--resume", action="store_true", help="do not retrain a (task, seed) whose best.pt exists, only evaluate it"); ap.add_argument("--load-workers", type=int, default=8)
ap.add_argument("--qset", default=None, choices=[None, "v2", "pan"], help="take labels, splits (and h5_path for pan) from this question set instead of the labels csv"); ap.add_argument("--prefix", default="clam_")
a = ap.parse_args()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TASKS = [t for t in a.task.replace("+", ",").split(",") if t]
Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_{a.qset or 'v2'}/questions_seed0.parquet"); SPLITS = ["train", "val", "test"] + (["heldout_project"] if a.heldout else [])
def table_of(task):
    cands = json.loads(Q[Q.op == task]["candidates"].iloc[0]); atype = Q[Q.op == task]["atype"].iloc[0]
    if a.qset:
        g = Q[(Q.op == task) & Q.split.isin(SPLITS)].drop_duplicates("slide")
        L = pd.DataFrame({"slide": g.slide.values, "patient": g.patient.values, "split": g.split.values, task: g.answer.astype(int).values, "h5": (g.h5_path.values if "h5_path" in g else [f"{a.h5_dir}/{s}.h5" for s in g.slide])})
    else:
        L = pd.read_csv(a.labels or f"{paths.DERIVED}/slide_questions_v2/slide_labels_seed0.csv"); L = L[L[task].notna()].copy(); L[task] = L[task].astype(int); L["h5"] = [f"{a.h5_dir}/{s}.h5" for s in L.slide]
    if a.smoke: L = L.groupby("split").head(6)
    print(f"task {task} K={len(cands)} slides {L.split.value_counts().to_dict()}", flush=True)
    return L, len(cands), atype
TAB = {t: table_of(t) for t in TASKS}
def load_bag(item):
    s, h5 = item
    with h5py.File(h5, "r") as h: X = h["features"][:]
    if a.max_tiles and X.shape[0] > a.max_tiles: X = X[np.random.default_rng(0).choice(X.shape[0], a.max_tiles, replace=False)]      # the same subset for every task and seed
    return s, X.astype(np.float16)
items = sorted({(s, h5) for L, _, _ in TAB.values() for s, h5 in zip(L["slide"], L["h5"])}); feats = {}; t0 = time.time()
with ProcessPoolExecutor(a.load_workers) as ex:
    for i, (s, X) in enumerate(ex.map(load_bag, items, chunksize=4)):
        feats[s] = torch.from_numpy(X)
        if (i + 1) % 1000 == 0: print(f"  {i + 1}/{len(items)} bags, {time.time()-t0:.0f} s", flush=True)
print(f"loaded {len(feats)} bags, {sum(v.shape[0] for v in feats.values())/1e6:.1f} M tiles, {time.time()-t0:.0f} s", flush=True)

def make_model(K):
    kw = dict(gate=True, size_arg="small", dropout=a.drop_out, k_sample=a.B, n_classes=K, instance_loss_fn=torch.nn.CrossEntropyLoss(), subtyping=K > 2, embed_dim=2560)
    return (CLAM_MB(**kw) if a.model == "clam_mb" else CLAM_SB(**kw)).to(dev)      # abmil: CLAM_SB without the instance loss

def run_epoch(model, slides, y, opt=None):
    train = opt is not None; model.train(train); tot = 0.0; probs = {}
    for s in slides:
        X = feats[s].to(dev, non_blocking=True).float(); label = torch.tensor([y[s]], device=dev)
        with torch.set_grad_enabled(train):
            logits, Y_prob, Y_hat, _, inst = model(X, label=label, instance_eval=a.model != "abmil")
            loss = torch.nn.functional.cross_entropy(logits, label)
            total = loss if a.model == "abmil" else a.bag_weight * loss + (1 - a.bag_weight) * inst["instance_loss"]
        if train: opt.zero_grad(set_to_none=True); total.backward(); opt.step()
        tot += loss.item(); probs[s] = Y_prob.detach().float().cpu().numpy()[0]
    return tot / max(len(slides), 1), probs

def score(P, slides, y, pat, K, atype):
    probs = np.stack([P[s] for s in slides]); labels = np.array([y[s] for s in slides]); patients = np.array([pat[s] for s in slides])
    m = M.summarize(probs, labels, K, atype, patients)
    try:
        if K == 2: m["auroc"] = float(roc_auc_score(labels, probs[:, 1])); m["ap"] = float(average_precision_score(labels, probs[:, 1]))
        else: m["auroc"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro", labels=list(range(K)))); m["ap"] = float(np.mean([average_precision_score(labels == k, probs[:, k]) for k in range(K) if (labels == k).any()]))
    except ValueError as e: m["auroc"] = None; m["ap"] = None; print("auroc skipped:", e)
    return m, probs, labels, patients

res_all = {}
for task in TASKS:
    L, K, atype = TAB[task]; D = L["slide"].tolist(); y = dict(zip(L["slide"], L[task])); split = dict(zip(L["slide"], L["split"])); pat = dict(zip(L["slide"], L["patient"]))
    for seed in [int(x) for x in a.seeds.replace("+", ",").split(",") if x]:                      # "+" survives sbatch --export, commas do not
        torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
        name = f"{a.prefix}{task}_s{seed}" + ("_smoke" if a.smoke else ""); OUT = f"{paths.RUNS}/{name}"; paths.ensure(OUT); RES = f"{paths.RESULTS}/runs/{name}"; paths.ensure(RES)
        tr = [s for s in D if split[s] == "train"]; va = [s for s in D if split[s] == "val"]; te = [s for s in D if split[s] == "test"]; ho = [s for s in D if split[s] == "heldout_project"]
        model = make_model(K); t0 = time.time(); old = None
        if a.resume and os.path.exists(f"{OUT}/best.pt") and os.path.exists(f"{OUT}/metrics_test.json"):
            old = json.load(open(f"{OUT}/metrics_test.json")); hist = json.load(open(f"{OUT}/history.json")); best = old["best_val_loss"]; print(f"resume {name}: best.pt reused, not retrained", flush=True)
        else:
            opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=a.lr, weight_decay=a.reg); best = float("inf"); bad = 0; hist = []
            for ep in range(2 if a.smoke else a.max_epochs):
                rng.shuffle(tr); tl, _ = run_epoch(model, tr, y, opt); vl, _ = run_epoch(model, va, y)
                hist.append(dict(epoch=ep, train_loss=tl, val_loss=vl, time=time.time() - t0)); print(json.dumps(hist[-1]), flush=True)
                if vl < best: best = vl; bad = 0; torch.save(model.state_dict(), f"{OUT}/best.pt")
                else:
                    bad += 1
                    if ep >= a.stop_epoch and bad >= a.patience: print("early stop", flush=True); break
            json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
        model.load_state_dict(torch.load(f"{OUT}/best.pt", map_location=dev))
        _, P = run_epoch(model, te, y); m, probs, labels, patients = score(P, te, y, pat, K, atype)
        if old is not None and abs(old["per_op"][task]["acc"] - m["acc"]) > 1e-9: print(f"WARNING {name}: test accuracy on reload {m['acc']:.4f} differs from the stored {old['per_op'][task]['acc']:.4f}", flush=True)
        res = {"test_nll": m["nll"], "best_val_loss": best, "model": a.model, "task": task, "seed": seed, "n_tiles_total": int(sum(feats[s].shape[0] for s in D if split[s] != "heldout_project")), "per_op": {task: dict(family=FAMILY.get(task, "slide"), **m)}}
        pd.DataFrame({"slide": te, "patient": patients, "answer": labels, "probs": [p.tolist() for p in probs]}).to_parquet(f"{OUT}/predictions_test.parquet", index=False)
        if ho:
            _, PH = run_epoch(model, ho, y); mh, ph, lh, pth = score(PH, ho, y, pat, K, atype)
            res["test_nll_heldout_project"] = mh["nll"]; res["per_op_heldout_project"] = {task: dict(family=FAMILY.get(task, "slide"), **mh)}
            pd.DataFrame({"slide": ho, "patient": pth, "answer": lh, "probs": [p.tolist() for p in ph]}).to_parquet(f"{OUT}/predictions_heldout_project.parquet", index=False)
            print(f"HELDOUT {name} n={mh['n']} acc={mh['acc']:.3f} (maj {mh['majority_acc']:.3f})", flush=True)
        json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1)
        if old is None: json.dump(dict(vars(a), task=task), open(f"{OUT}/config.json", "w"), indent=1)
        for f in ("config.json", "history.json", "metrics_test.json"): shutil.copy(f"{OUT}/{f}", f"{RES}/{f}")
        print(f"TEST {name} n={m['n']} acc={m['acc']:.3f} (maj {m['majority_acc']:.3f}) f1={m['macro_f1']:.3f} auroc={m['auroc']} ap={m['ap']} nll={m['nll']:.3f} ece={m['ece']:.3f} epochs={len(hist)} {time.time()-t0:.0f}s", flush=True)
        res_all[name] = res
print("RUN_DONE", list(res_all))
