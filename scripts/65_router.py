#!/usr/bin/env python
"""Routing between Opsis-S and Opsis-V (one question interface, two instantiations): evaluated offline on the predictions both
made on the same questions. Rules: per operator (chosen on val), confidence escalation (Opsis-S first, Opsis-V when the
Opsis-S max probability is below a val quantile), their combination, a learned router (logistic regression on
operator one hot + Opsis-S confidence + entropy, trained on val), and the oracle upper bound.
    python scripts/65_router.py [--structured jevpool_dens_v2_s0] [--vlm vlm_all_v2_s0] [--out results/collected/RESULTS_router.md]
    python scripts/65_router.py --structured final_fov_s0+final_fov_s1+final_fov_s2 --vlm final_vlm_s0+final_vlm_s1+final_vlm_s2
        (run lists pair up by position: every rule is evaluated per seed pair, the table reports mean and sd over pairs)
Cost model (defaults of --cost-structured and --cost-vlm, ms per question, H200): Opsis-S = adapter + branches shared over 32 questions
(2.8 ms per state) plus the frozen encoder amortised over the questions of a state (measured 41.7 ms per state over 139 questions),
Opsis-V = shared prefix at 32 questions per state (8.1 ms per question, scripts/41)."""
import argparse, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from opsis import paths
ap = argparse.ArgumentParser(); ap.add_argument("--structured", default="jevpool_dens_v2_s0"); ap.add_argument("--vlm", default="vlm_all_v2_s0"); ap.add_argument("--out", default=None)
ap.add_argument("--cost-structured", type=float, default=2.8 / 32 + 41.7 / 139); ap.add_argument("--cost-vlm", type=float, default=8.1); a = ap.parse_args()
LIST = lambda v: [x for x in v.replace("+", ",").split(",") if x]
PAIRS = list(zip(LIST(a.structured), LIST(a.vlm))); assert PAIRS and len(LIST(a.structured)) == len(LIST(a.vlm)), "the two run lists must pair up by position"

def load(run, split):
    P = pd.read_parquet(f"{paths.RUNS}/{run}/predictions_{split}.parquet"); P["probs"] = P["probs"].map(lambda p: np.asarray(p, dtype=float)); return P[["qid", "op", "family", "answer", "probs"]]
def merge(structured, vlm, split):
    S = load(structured, split).rename(columns={"probs": "ps"}); V = load(vlm, split)[["qid", "probs"]].rename(columns={"probs": "pv"}); M = S.merge(V, on="qid")
    for c in ("ps", "pv"): M[c] = M[c].map(lambda p: p / p.sum())
    M["cs"] = M.apply(lambda r: int(np.argmax(r.ps)) == r.answer, axis=1); M["cv"] = M.apply(lambda r: int(np.argmax(r.pv)) == r.answer, axis=1)
    M["conf_s"] = M["ps"].map(np.max); M["ent_s"] = M["ps"].map(lambda p: -(p * np.log(p + 1e-9)).sum())
    M["nll_s"] = M.apply(lambda r: -np.log(max(r.ps[r.answer], 1e-6)), axis=1); M["nll_v"] = M.apply(lambda r: -np.log(max(r.pv[r.answer], 1e-6)), axis=1); return M

def evaluate_pair(structured, vlm):
    val, test = merge(structured, vlm, "val"), merge(structured, vlm, "test"); rows = []
    def report(name, use_v, M=test):
        use_v = np.asarray(use_v, bool); acc = np.where(use_v, M.cv, M.cs).mean(); nll = np.where(use_v, M.nll_v, M.nll_s).mean(); f = use_v.mean(); cost = (1 - f) * a.cost_structured + f * a.cost_vlm
        rows.append(dict(rule=name, acc=acc, nll=nll, vlm_share=f, ms_per_question=cost)); print(f"{name:52s} acc {acc:.3f} NLL {nll:.3f} VLM {f:5.1%} {cost:5.2f} ms/q")
    n = len(test); report("structured only", np.zeros(n, bool)); report("VLM only", np.ones(n, bool)); report("oracle", (~test.cs & test.cv).to_numpy())
    better_v = {op: g.cv.mean() > g.cs.mean() for op, g in val.groupby("op")}; route_op = test.op.map(better_v).fillna(False).to_numpy(); report("per operator (chosen on val)", route_op); print("   operators routed to the VLM:", sorted(o for o, b in better_v.items() if b))
    for b in (0.1, 0.25, 0.5):
        tau = np.quantile(val.conf_s, b); report(f"confidence escalation, {b:.0%} of val", (test.conf_s < tau).to_numpy())
    for b in (0.1, 0.25):
        rest = val[~val.op.map(better_v).fillna(False)]; tau = np.quantile(rest.conf_s, b); report(f"per operator + confidence ({b:.0%} of the rest)", route_op | (test.conf_s < tau).to_numpy())
    ops = sorted(val.op.unique())
    def feats(M): X = pd.get_dummies(M.op).reindex(columns=ops, fill_value=0).astype(float); X["conf"] = M.conf_s.values; X["ent"] = M.ent_s.values; return X.values
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(feats(val), (~val.cs & val.cv).astype(int).values); pv = clf.predict_proba(feats(val))[:, 1]; pt = clf.predict_proba(feats(test))[:, 1]
    for b in (0.1, 0.25): report(f"learned router, {b:.0%} escalated", pt >= np.quantile(pv, 1 - b))
    # both instantiations on every question: probability fusion (no routing, costs both)
    def fused(M, w):
        ps = np.stack([np.pad(p, (0, 32 - len(p))) for p in M.ps]); pv = np.stack([np.pad(p, (0, 32 - len(p))) for p in M.pv]); return w * ps + (1 - w) * pv
    def geo(M):
        ps = np.stack([np.pad(p, (0, 32 - len(p))) for p in M.ps]); pv = np.stack([np.pad(p, (0, 32 - len(p))) for p in M.pv]); g = np.sqrt(ps * pv); return g / g.sum(1, keepdims=True)
    y = test.answer.to_numpy(); yv = val.answer.to_numpy()
    for name, P in (("fusion: arithmetic mean of both tiers", fused(test, 0.5)), ("fusion: geometric mean of both tiers", geo(test))):
        acc = (P.argmax(1) == y).mean(); nll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1)).mean()
        rows.append(dict(rule=name, acc=acc, nll=nll, vlm_share=1.0, ms_per_question=a.cost_structured + a.cost_vlm)); print(f"{name:52s} acc {acc:.3f} NLL {nll:.3f} VLM 100.0% {a.cost_structured + a.cost_vlm:5.2f} ms/q")
    return pd.DataFrame(rows).assign(pair=f"{structured} vs {vlm}")

D = pd.concat([(print(f"\n=== {s} vs {v}") or evaluate_pair(s, v)) for s, v in PAIRS], ignore_index=True); G = D.groupby("rule", sort=False)
df = pd.DataFrame({"rule": list(G.groups), "acc": G.acc.mean().values, "acc_sd": G.acc.std(ddof=1).fillna(0).values, "nll": G.nll.mean().values, "nll_sd": G.nll.std(ddof=1).fillna(0).values, "vlm_share": G.vlm_share.mean().values, "ms_per_question": G.ms_per_question.mean().values})
if len(PAIRS) > 1: print(f"\n=== mean over {len(PAIRS)} seed pairs"); print(df.round(3).to_string(index=False))
if a.out:
    sd = len(PAIRS) > 1
    lines = [f"| {r.rule} | {r.acc:.3f}" + (f" ± {r.acc_sd:.3f}" if sd else "") + f" | {r.nll:.3f}" + (f" ± {r.nll_sd:.3f}" if sd else "") + f" | {r.vlm_share:.1%} | {r.ms_per_question:.2f} |" for r in df.itertuples()]
    open(a.out, "w").write("# Routing between Opsis-S and Opsis-V (generated by scripts/65_router.py)\n\nPairs: " + ", ".join(f"{s} vs {v}" for s, v in PAIRS) + (" (mean ± sd over seed pairs)" if sd else "") + "\n\n| rule | acc | NLL | VLM share | ms per question |\n|---|---|---|---|---|\n" + "\n".join(lines) + "\n")
