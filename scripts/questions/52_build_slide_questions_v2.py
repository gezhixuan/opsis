#!/usr/bin/env python
"""Slide level question set v2 from the report extraction v2 (Sonnet subagents, schema configs/reports/report_v2_schema.json)
merged with the molecular labels: the v1 operators re-derived from the new extraction, the new Nottingham / DCIS /
findings operators, and claim level Noul questions ("Is this statement supported by the slide?").
    python scripts/questions/52_build_slide_questions_v2.py [--extraction <jsonl>] [--seed 0]
Writes $OPSIS_DATA/derived/slide_questions_v2/{extraction_merged.jsonl, questions_seed<k>.parquet, split_seed<k>.json (copied from v1)}."""
import argparse, glob, hashlib, json, os, shutil
import numpy as np, pandas as pd
from opsis import paths, splits
from opsis.questions import FAMILY

ap = argparse.ArgumentParser(); ap.add_argument("--extraction", default=None, help="merged jsonl; default: pilot + batches under derived/report_v2"); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
R2 = f"{paths.DERIVED}/report_v2"; OUT = f"{paths.DERIVED}/slide_questions_v2"; paths.ensure(OUT); PP = f"{paths.TCGA}"
files = [a.extraction] if a.extraction else [f"{R2}/pilot/extracted_sonnet.jsonl"] + sorted(glob.glob(f"{R2}/batches/extracted_b*.jsonl"))
ext = {}
for f in files:
    if not os.path.exists(f): continue
    for l in open(f):
        try: j = json.loads(l); ext[j["participant"]] = j
        except Exception: pass
with open(f"{OUT}/extraction_merged.jsonl", "w") as f:
    for j in ext.values(): f.write(json.dumps(j) + "\n")
print(f"extraction records: {len(ext)} from {len(files)} files")
lay = pd.read_csv(f"{PP}/gigapath_h5/brca_trident_virchow2/brca_trident.csv")
b2 = pd.read_csv(f"{PP}/clinical/batch2/tcga_probe_labels_batch2.tsv", sep="\t"); b2 = b2[b2["project"] == "TCGA-BRCA"].set_index("participant")
sp = splits.read_split(f"{paths.DERIVED}/slide_questions/split_seed{a.seed}.json"); fold = splits.fold_of(sp); shutil.copy(f"{paths.DERIVED}/slide_questions/split_seed{a.seed}.json", f"{OUT}/split_seed{a.seed}.json")
NR = lambda v: v is None or str(v).strip().lower().startswith("not reported") or str(v).strip() == ""
TYPES = ["ductal_nst", "lobular", "mixed_ductal_lobular", "mucinous", "metaplastic", "medullary", "papillary", "tubular", "apocrine", "micropapillary", "other"]
DCIS_TYPES = ["comedo", "cribriform", "solid", "micropapillary", "papillary", "mixed", "other"]; DCIS_GRADES = ["low", "intermediate", "high"]; TILS = ["none", "sparse", "moderate", "dense"]
import re
NEG_PATTERNS = [(r'(?i)^(?:no|without|absence of|absent|negative for)\s+(?:evidence of\s+|definite\s+|definitive\s+)?(.+?)(?:\s+(?:is|are|was|were))?\s*(?:identified|seen|present|noted|found|observed|detected|demonstrated|evident)?\.?$', r'\1'),
                (r'(?i)^(.+?)\s+(?:is|are|was|were)?\s*(?:not|non)\s+(?:identified|seen|present|noted|found|observed|detected|demonstrated|evident|involved|appreciated)\.?$', r'\1'),
                (r'(?i)^(.+?)[,:]?\s+(?:is|are|was|were)?\s*(?:absent|negative|not identified|not seen|not present|none identified|none seen|none|lacking)\.?$', r'\1'),
                (r'(?i)^(.+?)\s+absent\s+in\s+(.+)$', r'\1 in \2'),
                (r'(?i)^(?:lack|lacks|lacking) of\s+(.+)$', r'\1')]
NEG_WORDS = re.compile(r'(?i)\b(no|not|absent|negative|without|none|free of|lacks?|lacking)\b')
def claim_finding(claim, polarity):
    """Claims are stored as (finding, polarity). Some absent findings were written as negated sentences ("No LVI identified"):
    strip the negation so the question always asks about the positive finding; returns (finding, ok)."""
    t = claim.strip().rstrip('.')
    if polarity == "present": return t, True
    for p, r in NEG_PATTERNS:
        if re.match(p, t): t = re.sub(p, r, t).strip(' ,.;:'); break
    return t, not NEG_WORDS.search(t)
TOPICS = [("lvi", r"lymph|vascular|angio"), ("pni", r"perineural"), ("necrosis", r"necro"), ("dcis", r"in situ|dcis|intraductal|comedo|cribriform"), ("lcis", r"lcis|lobular neoplasia|lobular carcinoma in situ"),
          ("calcification", r"calcif"), ("grade", r"grade|tubule|pleomorph|mitot|mitos|differentiat|nuclear"), ("tils", r"lymphocyt|inflamm|plasma"), ("type", r"ductal|lobular|mucinous|metaplastic|medullary|papillary|tubular|apocrine|micropapillary|carcinoma"),
          ("benign", r"fibrocystic|hyperplasia|adenosis|papilloma|fibroadenoma|metaplasia|cyst|columnar|biopsy site")]
def claim_topic(finding):
    for name, pat in TOPICS:
        if re.search(pat, finding, flags=re.I): return name
    return "other"
rows = []
for r in lay.itertuples():
    p = r.participant; e = ext.get(p); b = b2.loc[p] if p in b2.index else None
    base = dict(fov=r.slide_id, slide=r.slide_id, patient=p, box=json.dumps([0.0, 0.0, 1.0, 1.0]), box2="", derivation="report_v2", split=fold.get(p, ""))
    def emit(op, atype, cands, ans, text, args=None, topic=""):
        key = f"{r.slide_id}|{op}|{json.dumps(args or {}, sort_keys=True)}"; rows.append(dict(base, qid=hashlib.sha1(key.encode()).hexdigest()[:16], family=FAMILY[op], op=op, args=json.dumps(args or {"scope": "slide"}, sort_keys=True), atype=atype, candidates=json.dumps(cands), answer=int(ans), text=text, topic=topic))
    if e is not None:
        F = {k: v["value"] for k, v in e["fields"].items()}
        g = lambda k: None if NR(F.get(k)) else str(F[k]).strip()
        comp = [g(k) for k in ("tubule_score", "pleomorphism_score", "mitotic_score")]
        if g("nottingham_total") is None and all(c in ("1", "2", "3") for c in comp):       # the grade is a deterministic function of the three scores
            F["nottingham_total"] = {3: "1", 4: "1", 5: "1", 6: "2", 7: "2", 8: "3", 9: "3"}[sum(int(c) for c in comp)]
        t = g("histologic_type")
        if t in TYPES: emit("histologic_type_fine", "choice", TYPES, TYPES.index(t), "What is the histologic type of the invasive carcinoma in this slide?")
        if t: coarse = "mixed" if t == "mixed_ductal_lobular" else "lobular" if t == "lobular" else "ductal" if t == "ductal_nst" else "other"; emit("histologic_type", "choice", ["ductal", "lobular", "mixed", "other"], ["ductal", "lobular", "mixed", "other"].index(coarse), "What is the histologic type of the invasive carcinoma in this slide?")
        for k, op, text in (("nottingham_total", "histologic_grade", "What is the histologic (Nottingham) grade of the carcinoma in this slide?"), ("nuclear_grade", "nuclear_grade", "What is the nuclear grade of the carcinoma in this slide?"),
                            ("tubule_score", "tubule_score", "What is the Nottingham tubule formation score of the carcinoma in this slide?"), ("pleomorphism_score", "pleomorphism_score", "What is the Nottingham nuclear pleomorphism score of the carcinoma in this slide?"), ("mitotic_score", "mitotic_score", "What is the Nottingham mitotic count score of the carcinoma in this slide?")):
            v = g(k)
            if v in ("1", "2", "3"): emit(op, "ordinal", ["1", "2", "3"], int(v) - 1, text)
        for k, op, text, yes in (("dcis_present", "in_situ_present", "Is an in situ carcinoma component present in this slide?", "yes"), ("lcis_present", "lcis_present", "Is lobular carcinoma in situ present in this slide?", "yes"),
                                 ("lvi", "lvi_present", "Is lymphovascular invasion present in this slide?", "present"), ("pni", "pni_present", "Is perineural invasion present in this slide?", "present"), ("necrosis", "necrosis_present", "Is tumour necrosis present in this slide?", "present"),
                                 ("calcifications", "calcifications", "Are calcifications present in this slide?", "present"), ("dcis_comedo_necrosis", "dcis_comedo_necrosis", "Does the in situ component show comedo necrosis?", "yes")):
            v = g(k)
            if v in (yes, {"yes": "no", "present": "absent"}[yes]): emit(op, "noul", ["no", "yes"], int(v == yes), text)
        v = g("focality")
        if v in ("unifocal", "multifocal"): emit("multifocal", "noul", ["no", "yes"], int(v == "multifocal"), "Is the carcinoma multifocal?")
        v = g("dcis_type")
        if v in DCIS_TYPES: emit("dcis_type", "choice", DCIS_TYPES, DCIS_TYPES.index(v), "What is the architectural type of the ductal carcinoma in situ in this slide?")
        v = g("dcis_grade")
        if v in DCIS_GRADES: emit("dcis_grade", "ordinal", DCIS_GRADES, DCIS_GRADES.index(v), "What is the nuclear grade of the ductal carcinoma in situ in this slide?")
        v = g("tils_description")
        if v in TILS: emit("tils_description", "ordinal", TILS, TILS.index(v), "How dense are the stromal tumour infiltrating lymphocytes in this slide?")
        for i, c in enumerate(e.get("h_and_e_visible_claims", [])):
            if c.get("polarity") in ("present", "absent") and c.get("claim"):
                finding, ok = claim_finding(c["claim"], c["polarity"])
                if not ok: continue                                                  # absent claim still phrased as a negation: ambiguous, dropped
                emit("claim_supported", "noul", ["no", "yes"], int(c["polarity"] == "present"), f"Does this slide show the following finding: {finding}?", args={"scope": "slide", "claim": i}, topic=claim_topic(finding))
    if b is not None:
        for k, op, text in (("er_status", "er_positive", "Is the carcinoma oestrogen receptor positive?"), ("pr_status", "pr_positive", "Is the carcinoma progesterone receptor positive?"), ("her2_status", "her2_positive", "Is the carcinoma HER2 positive?")):
            v = str(b.get(k, ""))
            if v in ("Positive", "Negative"): emit(op, "noul", ["no", "yes"], int(v == "Positive"), text)
        pam = str(b.get("pam50", "")); P5 = ["LumA", "LumB", "Basal", "Her2", "Normal"]
        if pam in P5: emit("pam50", "choice", P5, P5.index(pam), "Which PAM50 intrinsic subtype does this carcinoma belong to?")
Q = pd.DataFrame(rows); Q = Q[Q["split"] != ""]; Q.to_parquet(f"{OUT}/questions_seed{a.seed}.parquet", index=False)
print(f"questions {len(Q)} on {Q.fov.nunique()} slides; per split", Q["split"].value_counts().to_dict())
for op, g in Q.groupby("op"): vc = g["answer"].value_counts(normalize=True); print(f"  {op:22s} n={len(g):5d} majority={vc.iloc[0]:.2f} classes={len(vc)}")
