#!/usr/bin/env python
"""Pan cancer slide level question set (final version): every TCGA diagnostic slide with cached Virchow2 features,
questions from (a) the shared report schema extraction (derived/report_pan/extracted_pan_all.jsonl, gpt-6-luna v1.1),
(b) clinical and molecular tables (cancer type, TCGASubtype selected subtype, ABSOLUTE purity bins, leukocyte fraction
bins), (c) composite held out operators whose answers are functions of trained operators. Split: patient level 70 / 15 /
15 inside every project (seed), plus four held out projects (KIRP, STAD, CESC, PAAD) that never enter train or val.
    python scripts/questions/66_build_slide_questions_pan.py [--seed 0]
Writes $OPSIS_DATA/derived/slide_questions_pan/{questions_seed<k>.parquet, split_seed<k>.json, summary.json}."""
import argparse, glob, hashlib, json, os, re
import numpy as np, pandas as pd
from opsis import paths
from opsis.questions import FAMILY

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--heldout", default="KIRP+STAD+CESC+PAAD"); a = ap.parse_args()
PP = f"{paths.TCGA}"; OUT = f"{paths.DERIVED}/slide_questions_pan"; paths.ensure(OUT)
HELD = set(a.heldout.split("+"))
# slides and features
lay = []
for f in glob.glob(f"{PP}/gigapath_h5/*_trident_virchow2/*_trident.csv"):
    d = pd.read_csv(f); proj = os.path.basename(os.path.dirname(f)).split("_")[0].upper(); d["project"] = proj
    d["h5_path"] = [f"{os.path.dirname(f)}/h5_files/{s}.h5" for s in d["slide_id"]]; lay.append(d)
lay = pd.concat(lay, ignore_index=True); lay = lay[lay["h5_path"].map(os.path.exists)].drop_duplicates("slide_id")
# report extraction
ext = {}
for l in open(f"{paths.DERIVED}/report_pan/extracted_pan_all.jsonl"):
    j = json.loads(l); ext[j["participant"]] = j["fields"]
val = lambda c: (c.get("value") if isinstance(c, dict) else c)
NR = lambda v: v is None or str(v).strip().lower() in ("not reported", "", "none reported")
# tables
pur = pd.read_csv(f"{PP}/clinical/batch2/TCGA_mastercalls.abs_tables_JSedit.fixed.txt", sep="\t"); pur["participant"] = pur["array"].str[:12]; pur = pur.dropna(subset=["purity"]).drop_duplicates("participant").set_index("participant")["purity"]
leu = pd.read_csv(f"{PP}/immune_landscape/TCGA_all_leuk_estimate.masked.20170107.tsv", sep="\t", header=None, names=["project", "sample", "leuk"]); leu["participant"] = leu["sample"].str[:12]; leu = leu[leu["sample"].str[13:15] == "01"].drop_duplicates("participant").set_index("participant")["leuk"]
sub = pd.read_csv(f"{PP}/clinical/batch2/TCGASubtype.20170308.tsv.gz", sep="\t"); sub["participant"] = sub["sampleID"].str[:12]; sub = sub.dropna(subset=["Subtype_Selected"]).drop_duplicates("participant").set_index("participant")["Subtype_Selected"]
# v2 BRCA extraction for the composites that need Nottingham components / DCIS comedo
v2 = {}
f2 = f"{paths.DERIVED}/slide_questions_v2/extraction_merged.jsonl"
if os.path.exists(f2):
    for l in open(f2): j = json.loads(l); v2[j["participant"]] = {k: v.get("value") for k, v in j["fields"].items()}
# split: patient level inside each project; held out projects separate
rng = np.random.default_rng(a.seed); fold = {}
for proj, g in lay.groupby("project"):
    pats = sorted(g["participant"].unique()); rng.shuffle(pats); n = len(pats)
    if proj in HELD:
        for p in pats: fold[p] = "heldout_project"
    else:
        for i, p in enumerate(pats): fold[p] = "train" if i < 0.7 * n else "val" if i < 0.85 * n else "test"
json.dump({k: sorted(p for p, v in fold.items() if v == k) for k in ("train", "val", "test", "heldout_project")}, open(f"{OUT}/split_seed{a.seed}.json", "w"))
PROJECTS = sorted(lay["project"].unique()); sub_cands = {}
for proj in PROJECTS:
    ps = lay[lay.project == proj]["participant"].unique(); vals = sub.reindex(ps).dropna().value_counts(); sub_cands[proj] = (lambda c: c if len(c) >= 2 else [])(sorted(vals[vals >= 20].index.tolist())[:12])
PUR_BINS = [0.5, 0.65, 0.8]; LEU_BINS = [0.09, 0.17, 0.30]
def obin(x, edges): return int(np.searchsorted(edges, x, side="right"))
PREC_CATS = ["in_situ_carcinoma", "adenoma", "dysplasia_or_intraepithelial", "other"]
def prec_cat(t):
    t = str(t).lower()
    if re.search(r"in situ|dcis|lcis|\bcis\b|intraductal carcinoma", t): return 0
    if "adenoma" in t: return 1
    if re.search(r"dysplasia|intraepithelial|panin|\bpin\b|\bais\b|\bcin\b|\bvin\b|metaplasia with dysplasia", t): return 2
    return 3
rows = []
for r in lay.itertuples():
    p = r.participant; e = ext.get(p); base = dict(fov=r.slide_id, slide=r.slide_id, patient=p, project=r.project, h5_path=r.h5_path, box=json.dumps([0.0, 0.0, 1.0, 1.0]), box2="", split=fold.get(p, ""))
    def emit(op, atype, cands, ans, text, derivation, args=None):
        key = f"{r.slide_id}|{op}|{json.dumps(args or {}, sort_keys=True)}"
        rows.append(dict(base, derivation=derivation, qid=hashlib.sha1(key.encode()).hexdigest()[:16], family=FAMILY[op], op=op, args=json.dumps(args or {"scope": "slide"}, sort_keys=True), atype=atype, candidates=json.dumps(cands), answer=int(ans), text=text))
    emit("cancer_type", "choice", PROJECTS, PROJECTS.index(r.project), "Which TCGA cancer type is this slide from?", "table")
    if p in pur.index: emit("tumor_purity_bin", "ordinal", ["<0.5", "0.5-0.65", "0.65-0.8", ">=0.8"], obin(pur[p], PUR_BINS), "What fraction of the cells in this tumour are tumour cells (ABSOLUTE purity)?", "table")
    if p in leu.index: emit("leukocyte_fraction_bin", "ordinal", ["<0.09", "0.09-0.17", "0.17-0.30", ">=0.30"], obin(leu[p], LEU_BINS), "What fraction of the cells in this tumour sample are leukocytes (methylation estimate)?", "table")
    if p in sub.index and sub[p] in sub_cands.get(r.project, []): emit("molecular_subtype", "choice", sub_cands[r.project], sub_cands[r.project].index(sub[p]), f"Which {r.project} molecular subtype is this tumour?", "table", args={"scope": "slide", "project": r.project})
    if e is not None:
        g = {k: val(v) for k, v in e.items()}; S = lambda k: None if NR(g.get(k)) else str(g[k]).strip().lower()
        if S("histologic_grade") in ("1", "2", "3", "4"): emit("pan_grade", "choice", ["1", "2", "3", "4"], int(S("histologic_grade")) - 1, "What is the histologic grade of the primary tumour on the report's grading system?", "report_pan")
        for op, k, text in (("pan_lvi", "lymphovascular_invasion", "Is lymphovascular invasion present?"), ("pan_pni", "perineural_invasion", "Is perineural invasion present?"), ("pan_necrosis", "necrosis", "Is tumour necrosis present?"), ("pan_precursor", "precursor_or_in_situ_component", "Is a precursor or in situ component of the tumour present?"), ("pan_treatment_effect", "treatment_effect", "")):
            if op == "pan_treatment_effect": continue
            if S(k) in ("present", "absent"): emit(op, "noul", ["no", "yes"], int(S(k) == "present"), text, "report_pan")
        if S("multifocal") in ("yes", "no"): emit("pan_multifocal", "noul", ["no", "yes"], int(S("multifocal") == "yes"), "Is the primary tumour multifocal?", "report_pan")
        if S("mitotic_activity") in ("low", "moderate", "high"): emit("pan_mitotic", "ordinal", ["low", "moderate", "high"], ["low", "moderate", "high"].index(S("mitotic_activity")), "How high is the mitotic activity of the tumour?", "report_pan")
        if S("lymphocytic_infiltrate") in ("none", "mild", "moderate", "marked"): emit("pan_infiltrate", "ordinal", ["none", "mild", "moderate", "marked"], ["none", "mild", "moderate", "marked"].index(S("lymphocytic_infiltrate")), "How dense is the lymphocytic infiltrate in the tumour?", "report_pan")
        # composites (held out from training)
        if S("histologic_grade") in ("1", "2", "3", "4") and S("lymphovascular_invasion") in ("present", "absent"):
            emit("high_grade_and_lvi", "noul", ["no", "yes"], int(int(S("histologic_grade")) >= 3 and S("lymphovascular_invasion") == "present"), "Is this a high grade tumour (grade 3 or 4) with lymphovascular invasion?", "composite")
        if S("necrosis") in ("present", "absent") and S("mitotic_activity") in ("low", "moderate", "high"):
            emit("necrosis_and_high_mitotic", "noul", ["no", "yes"], int(S("necrosis") == "present" and S("mitotic_activity") == "high"), "Does the tumour show both necrosis and high mitotic activity?", "composite")
        if S("precursor_or_in_situ_component") == "present" and not NR(g.get("precursor_type")):
            emit("precursor_type", "choice", PREC_CATS, prec_cat(g["precursor_type"]), "Given that a precursor lesion is present, which kind is it?", "composite")
    v = v2.get(p)
    if v:
        comp = [v.get(k) for k in ("tubule_score", "pleomorphism_score", "mitotic_score")]
        if all(str(c) in ("1", "2", "3") for c in comp):
            tot = sum(int(c) for c in comp); emit("grade_from_components", "choice", ["1", "2", "3"], 0 if tot <= 5 else 1 if tot <= 7 else 2, "What is the Nottingham grade implied by the tubule, pleomorphism and mitotic scores?", "composite")
        if str(v.get("dcis_present")).lower() in ("yes", "no") and str(v.get("dcis_comedo_necrosis")).lower() in ("yes", "no", "present", "absent"):
            emit("dcis_with_comedo", "noul", ["no", "yes"], int(str(v["dcis_present"]).lower() == "yes" and str(v["dcis_comedo_necrosis"]).lower() in ("yes", "present")), "Is there ductal carcinoma in situ with comedo necrosis?", "composite")
Q = pd.DataFrame(rows); Q = Q[Q["split"] != ""]
Q.to_parquet(f"{OUT}/questions_seed{a.seed}.parquet", index=False)
summ = {"slides": int(Q["slide"].nunique()), "questions": int(len(Q)), "per_split": Q["split"].value_counts().to_dict(), "per_op": Q.groupby("op").size().to_dict(),
        "majority": {op: round(float(g["answer"].value_counts(normalize=True).max()), 3) for op, g in Q.groupby("op")}, "heldout_projects": sorted(HELD), "purity_bins": PUR_BINS, "leukocyte_bins": LEU_BINS}
json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
print(f"questions {len(Q)} on {Q['slide'].nunique()} slides; per split {summ['per_split']}")
for op, g in Q.groupby("op"): print(f"  {op:26s} n={len(g):6d} majority={summ['majority'][op]:.2f} classes={len(json.loads(g['candidates'].iloc[0]))}")
