"""Metrics per operator: accuracy, macro F1, Brier, NLL, ECE, selective accuracy, ordinal MAE, patient-clustered bootstrap."""
import numpy as np


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def selective_accuracy(conf, correct, coverage):
    order = np.argsort(-conf)
    k = max(1, int(round(coverage * len(conf))))
    return float(correct[order[:k]].mean())


def macro_f1(y, yhat, K):
    f = []
    for k in range(K):
        tp = ((yhat == k) & (y == k)).sum()
        fp = ((yhat == k) & (y != k)).sum()
        fn = ((yhat != k) & (y == k)).sum()
        if tp + fp + fn == 0:
            continue
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f)) if f else float("nan")


def summarize(probs, y, K, atype, patients=None, n_boot=1000, seed=0):
    """probs [n, K] (padded columns zero), y [n]. Returns a dict of metrics with bootstrap CIs over patients."""
    probs = np.asarray(probs, dtype=np.float64)[:, :K]
    probs = probs / probs.sum(1, keepdims=True).clip(1e-12)
    y = np.asarray(y)
    yhat = probs.argmax(1)
    correct = (yhat == y).astype(float)
    conf = probs.max(1)
    onehot = np.eye(K)[y]
    nll = -np.log(probs[np.arange(len(y)), y].clip(1e-12))
    brier = ((probs - onehot) ** 2).sum(1)
    out = dict(n=int(len(y)), acc=float(correct.mean()), macro_f1=macro_f1(y, yhat, K), nll=float(nll.mean()), brier=float(brier.mean()),
               ece=ece(conf, correct), sel_acc_50=selective_accuracy(conf, correct, 0.5), sel_acc_80=selective_accuracy(conf, correct, 0.8),
               majority_acc=float(np.bincount(y, minlength=K).max() / len(y)))
    if atype == "ordinal":
        out["mae_bins"] = float(np.abs(yhat - y).mean())
        out["expected_mae_bins"] = float(np.abs((probs * np.arange(K)).sum(1) - y).mean())
    if patients is not None and n_boot:
        pts = np.asarray(patients)
        uniq, inv = np.unique(pts, return_inverse=True)
        cnt = np.bincount(inv, minlength=len(uniq)).astype(float)
        sc = np.bincount(inv, weights=correct, minlength=len(uniq))
        sn = np.bincount(inv, weights=nll, minlength=len(uniq))
        rng = np.random.default_rng(seed)
        accs, nlls = [], []
        for _ in range(n_boot):
            w = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)))
            tot = (w * cnt).sum()
            accs.append((w * sc).sum() / tot)
            nlls.append((w * sn).sum() / tot)
        out["acc_ci"] = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]
        out["nll_ci"] = [float(np.percentile(nlls, 2.5)), float(np.percentile(nlls, 97.5))]
    return out
