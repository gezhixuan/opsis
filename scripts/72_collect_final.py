#!/usr/bin/env python
"""Collect the final runs into results/collected/RESULTS_final.md: mean and sd over seeds per operator for the field
decoder of Opsis-S (final_fov_s*, joint NuCLS + PanNuke), its PanNuke fold / held out organ evaluation, Opsis-V (final_vlm_s*),
the pan TCGA slide decoder (final_slide_pan_s*, test and held out projects) and the full bag tile heads (final_tile_*).
Runs that have not finished are skipped, so the script can be rerun as results land.
    python scripts/72_collect_final.py"""
import glob, json, os, re
import numpy as np, pandas as pd
from opsis import paths
R = paths.RUNS
def load(pattern):
    out = {}
    for d in sorted(glob.glob(f"{R}/{pattern}")):
        f = f"{d}/metrics_test.json"
        if os.path.exists(f): out[os.path.basename(d)] = json.load(open(f))
    return out
def ms(v, nd=3):
    v = [x for x in v if x is not None]
    return "" if not v else (f"{np.mean(v):.{nd}f}" if len(v) == 1 else f"{np.mean(v):.{nd}f} ± {np.std(v, ddof=1):.{nd}f}")
def table(runs, key, cols=("acc", "nll"), extra=None, ref=None):
    ops = sorted({op for m in runs.values() for op in m.get(key, {})}); rows = []
    for op in ops:
        r = {"operator": op, "n": next((m[key][op]["n"] for m in runs.values() if op in m.get(key, {})), ""), "maj": f"{next((m[key][op]['majority_acc'] for m in runs.values() if op in m.get(key, {})), 0):.2f}"}
        for c in cols: r[c] = ms([m[key][op].get(c) for m in runs.values() if op in m.get(key, {})])
        if extra:
            for name, rr in extra.items(): r[name] = ms([x[key][op].get(cols[0]) for x in rr.values() if op in x.get(key, {})])
        if ref:
            for name, rm in ref.items(): r[name] = f"{rm[key][op][cols[0]]:.3f}" if op in rm.get(key, {}) else ""
        rows.append(r)
    T = pd.DataFrame(rows)
    return ["| " + " | ".join(T.columns) + " |", "|" + "---|" * len(T.columns), *("| " + " | ".join(str(x) for x in r) + " |" for r in T.itertuples(index=False))] if len(T) else ["(no finished run)"]
md = ["# Final results, collected by scripts/72_collect_final.py", ""]
fov = load("final_fov_s[0-9]"); org = load("final_fov_organ_s[0-9]"); v2 = load("jevpool_dens_v2_s0"); vlm2 = load("vlm_all_v2_s0")
md += [f"## Opsis-S on fields, joint NuCLS + PanNuke training ({len(fov)} seeds: {', '.join(fov)})", "",
       f"NuCLS v2 test, mean ± sd over seeds. Overall test NLL {ms([m['test_nll'] for m in fov.values()])} (earlier single seed jevpool_dens_v2_s0: {ms([m['test_nll'] for m in v2.values()])}, Opsis-V vlm_all_v2_s0: {ms([m['test_nll'] for m in vlm2.values()])}).", ""]
md += table(fov, "per_op", ("acc", "nll", "ece"), ref={"r5 structured acc": next(iter(v2.values()), {}), "r5 VLM acc": next(iter(vlm2.values()), {})} if v2 and vlm2 else None)
md += ["", f"PanNuke test (fold split, {len(fov)} seeds) and held out organs (final_fov_organ_s*, {len(org)} seed): accuracy.", ""]
md += table(fov, "per_op_extra", ("acc", "nll"), extra={"held out organs acc": org} if org else None)
tuned = load("final_fov_tuned_s[0-9]"); ctl = load("final_fov_tunedctl_s[0-9]")
if tuned:
    md += ["", f"## Tuned state: Virchow2 last 4 blocks unfrozen, readout initialised from final_fov_s0 ({len(tuned)} seed); control = same continued training with the encoder frozen ({len(ctl)} seed)", "",
           f"NuCLS test NLL: tuned {ms([m['test_nll'] for m in tuned.values()])}, control {ms([m['test_nll'] for m in ctl.values()])}, frozen reference final_fov_s0 {ms([m['test_nll'] for m in load('final_fov_s0').values()])}. Validation NLL did not improve over the initial readout in either run (train NLL kept falling), so the encoder is not the lever at this data scale.", ""]
    md += table(tuned, "per_op", ("acc", "nll"), extra={"control acc": ctl, "frozen final_fov_s0 acc": load("final_fov_s0")})
    md += ["", "PanNuke test:", ""] + table(tuned, "per_op_extra", ("acc", "nll"), extra={"control acc": ctl, "frozen final_fov_s0 acc": load("final_fov_s0")})
vlm = load("final_vlm_s[0-9]")
md += ["", f"## Opsis-V, Qwen3-VL-8B LoRA on questions_final ({len(vlm)} seeds)", "", f"Test NLL {ms([m['test_nll'] for m in vlm.values()])}.", ""] + table(vlm, "per_op", ("acc", "nll", "ece"))
vs = load("vlmstate_8b_s[0-9]"); vb = load("vlmstate_8b_both_s[0-9]")
if vs or vb:
    md += ["", f"## VLM tier reading the shared state: Qwen3-VL-8B LoRA r=64 with the 8x8 pooled Virchow2 tokens injected through a projector, joint PanNuke ({len(vs)} seed tokens only, {len(vb)} seed tokens + own image tokens)", "",
           f"NuCLS test NLL: tokens only {ms([m['test_nll'] for m in vs.values()])}, tokens + pixels {ms([m['test_nll'] for m in vb.values()])}, pixels only (final_vlm, 3 seeds) {ms([m['test_nll'] for m in vlm.values()])}, structured tier {ms([m['test_nll'] for m in fov.values()])}. The tokens-only arm is the diagnostic: the language model reading nothing but the structured tier's state reaches the pixel path, so the accuracy of the VLM tier comes from the pathology state, not from the base model's own vision. Router complementarity does not depend on different visual evidence either: on the same seed pair the oracle over (structured, VLM reading the same state) is 0.856 against 0.859 for (structured, VLM reading pixels), and the learned router reaches 0.800 in both (results/collected/RESULTS_router_state.md). The one operator the tokens-only arm loses is compare_regions (0.59 against 0.80): two boxes on an 8x8 pooled grid are too coarse, which is where the base model's fine image tokens still matter.", ""]
    md += table(vs, "per_op", ("acc", "nll"), extra={"tokens + pixels acc": vb, "pixels only acc (3 seeds)": vlm, "structured acc (3 seeds)": fov})
    md += ["", "PanNuke test:", ""] + table(vs, "per_op_extra", ("acc", "nll"), extra={"tokens + pixels acc": vb, "structured acc (3 seeds)": fov})
sl = load("final_slide_pan_s[0-9]")
md += ["", f"## Opsis-S on pan cancer slides, K = 4,096 tiles ({len(sl)} seeds)", "", f"Test NLL {ms([m['test_nll'] for m in sl.values()])}; held out projects (KIRP, STAD, CESC, PAAD) NLL {ms([m.get('test_nll_heldout_project') for m in sl.values()])}. Composite operators were excluded from training and are evaluated held out.", "", "Test split:", ""]
md += table(sl, "per_op", ("acc", "nll", "ece")) + ["", "Held out projects (KIRP, STAD, CESC, PAAD were never seen in training, so cancer_type is 0 and molecular_subtype near chance by construction; the pan operators are the informative rows):", ""] + table(sl, "per_op_heldout_project", ("acc", "nll", "ece"))
for pool in ("noisyor", "lse", "topk"):
    ti = load(f"final_tile_{pool}_s[0-9]")
    if not ti: continue
    md += ["", f"## Full bag tile heads, {pool} pooling ({len(ti)} seeds): localised findings", "", "Test split (acc, AUROC, NLL):", ""] + table(ti, "per_op", ("acc", "auroc", "nll")) + ["", "Held out projects:", ""] + table(ti, "per_op_heldout_project", ("acc", "auroc", "nll"))
    for name in ti:
        f = f"{R}/{name}/tile_eval_bcss.json"
        if os.path.exists(f):
            e = json.load(open(f)); md += ["", f"Tile level necrosis against the BCSS masks ({name}, necrotic = necrosis_or_debris >= {e['frac']:.0%} of the tile):", "", "| split | slides | tiles | positive rate | AUROC necrosis head | AP necrosis head | AUROC LVI head | AUROC PNI head | AUROC precursor head |", "|---|---|---|---|---|---|---|---|---|"]
            for r in e["table"]: md.append(f"| {r['split']} | {r['n_slides']} | {r['n_tiles']} | {r['pos_rate']:.3f} | {r.get('auroc_pan_necrosis', float('nan')):.3f} | {r.get('ap_pan_necrosis', float('nan')):.3f} | {r.get('auroc_pan_lvi', float('nan')):.3f} | {r.get('auroc_pan_pni', float('nan')):.3f} | {r.get('auroc_pan_precursor', float('nan')):.3f} |")
ADD = f"{paths.ROOT}/results/collected/RESULTS_final_addenda.md"; TXT = "\n".join(md) + "\n" + ("\n" + open(ADD).read() if os.path.exists(ADD) else "")
open(f"{paths.ROOT}/results/collected/RESULTS_final.md", "w").write(TXT); print(TXT)
