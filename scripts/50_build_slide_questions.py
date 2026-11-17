#!/usr/bin/env python
"""Slide level question set for TCGA-BRCA: report slots (gpt-5.6 extraction of the TCGA reports)
and molecular status (clinical batch 2) as typed questions on the whole slide.
    python scripts/50_build_slide_questions.py [--seed 0]
Writes $OPSIS_DATA/derived/slide_questions/{slides.csv, split_seed<k>.json, questions_seed<k>.parquet}. The participant split
keeps the NuCLS patients in the fold of the FOV split (derived/questions/split_seed<k>.json) so that later experiments can
combine FOV and slide states without leakage."""
import argparse, hashlib, json, os, re
import numpy as np, pandas as pd
from opsis import paths, splits
from opsis.questions import FAMILY

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
PP = f"{paths.TCGA}"; OUT = f"{paths.DERIVED}/slide_questions"; paths.ensure(OUT)
lay = pd.read_csv(f"{PP}/gigapath_h5/brca_trident_virchow2/brca_trident.csv"); lay["b15"] = lay["rna_barcode"].str[:15]
rep = {j["participant"]: j for j in (json.loads(l) for l in open(f"{PP}/reports/report_matrix.jsonl")) if j["cohort"] == "brca"}
cat = pd.read_csv(f"{PP}/maps/report_cat.csv", index_col=0)
b2 = pd.read_csv(f"{PP}/clinical/batch2/tcga_probe_labels_batch2.tsv", sep="\t"); b2 = b2[b2["project"] == "TCGA-BRCA"].set_index("participant")
def htype(t):
    t = str(t).lower(); lob = bool(re.search(r"\blobular\b", t)) and "lobular carcinoma in situ" not in t.replace("invasive lobular", "").replace("infiltrating lobular", "")
    duc = bool(re.search(r"duct|no special type|\bnst\b|\bnos\b", t))
    return "mixed" if lob and duc else "lobular" if lob else "ductal" if duc else "other"
def cat_val(b15, k):
    if b15 not in cat.columns or f"{k}_reported" not in cat.index: return None
    if float(cat.loc[f"{k}_reported", b15]) <= 0: return None
    return float(cat.loc[k, b15])
rows = []; srows = []
for r in lay.itertuples():
    p = r.participant; j = rep.get(p); b = b2.loc[p] if p in b2.index else None
    base = dict(fov=r.slide_id, slide=r.slide_id, patient=p, atype="", box=json.dumps([0.0, 0.0, 1.0, 1.0]), box2="", derivation="report/batch2", split="")
    def emit(op, atype, cands, ans, text):
        key = f"{r.slide_id}|{op}"; rows.append(dict(base, qid=hashlib.sha1(key.encode()).hexdigest()[:16], family=FAMILY[op], op=op, args=json.dumps({"scope": "slide"}), atype=atype, candidates=json.dumps(cands), answer=int(ans), text=text))
    if j is not None:
        ht = htype(j.get("histologic_type", "")); emit("histologic_type", "choice", ["ductal", "lobular", "mixed", "other"], ["ductal", "lobular", "mixed", "other"].index(ht), "What is the histologic type of the invasive carcinoma in this slide?")
        g = cat_val(r.b15, "grade")
        if g in (1.0, 2.0, 3.0): emit("histologic_grade", "ordinal", ["1", "2", "3"], int(g) - 1, "What is the histologic (Nottingham) grade of the carcinoma in this slide?")
        g = cat_val(r.b15, "nuclear_grade")
        if g in (1.0, 2.0, 3.0): emit("nuclear_grade", "ordinal", ["1", "2", "3"], int(g) - 1, "What is the nuclear grade of the carcinoma in this slide?")
        for k, op, text in (("in_situ", "in_situ_present", "Is an in situ carcinoma component present in this slide?"), ("lvi", "lvi_present", "Is lymphovascular invasion present in this slide?"),
                            ("necrosis", "necrosis_present", "Is tumour necrosis present in this slide?"), ("multifocal", "multifocal", "Is the carcinoma multifocal?"), ("pni", "pni_present", "Is perineural invasion present in this slide?")):
            v = cat_val(r.b15, k)
            if v in (0.0, 1.0): emit(op, "noul", ["no", "yes"], int(v), text)
    if b is not None:
        for k, op, text in (("er_status", "er_positive", "Is the carcinoma oestrogen receptor positive?"), ("pr_status", "pr_positive", "Is the carcinoma progesterone receptor positive?"), ("her2_status", "her2_positive", "Is the carcinoma HER2 positive?")):
            v = str(b.get(k, ""))
            if v in ("Positive", "Negative"): emit(op, "noul", ["no", "yes"], int(v == "Positive"), text)
        pam = str(b.get("pam50", "")); P5 = ["LumA", "LumB", "Basal", "Her2", "Normal"]
        if pam in P5: emit("pam50", "choice", P5, P5.index(pam), "Which PAM50 intrinsic subtype does this carcinoma belong to?")
    srows.append(dict(slide=r.slide_id, participant=p, sample_key=r.sample_key, n_tiles=r.n_tiles, has_report=j is not None, has_batch2=b is not None))
S = pd.DataFrame(srows); Q = pd.DataFrame(rows)
# participant split: NuCLS patients keep their FOV fold, the rest 70/15/15
fov_split = splits.read_split(f"{paths.DERIVED}/questions/split_seed{a.seed}.json"); fixed = splits.fold_of(fov_split)
rest = sorted(set(S["participant"]) - set(fixed)); sp = splits.split_patients(rest, seed=a.seed)
for k in sp: sp[k] = sorted(sp[k] + [p for p, f in fixed.items() if f == k and p in set(S["participant"])])
splits.write_split(sp, f"{OUT}/split_seed{a.seed}.json"); fold = splits.fold_of(sp); S["split"] = S["participant"].map(fold); Q["split"] = Q["patient"].map(fold)
S.to_csv(f"{OUT}/slides.csv", index=False); Q.to_parquet(f"{OUT}/questions_seed{a.seed}.parquet", index=False)
print(f"slides {len(S)} (participants {S.participant.nunique()}, with report {int(S.has_report.sum())}, with batch2 {int(S.has_batch2.sum())}); questions {len(Q)}; split slides", S["split"].value_counts().to_dict())
for op, g in Q.groupby("op"): vc = g["answer"].value_counts(normalize=True); print(f"  {op:18s} n={len(g):5d} majority={vc.iloc[0]:.2f} classes={len(vc)}")
