#!/usr/bin/env python
"""Install the question release (Hugging Face dataset gezx1004/opsis-questions, downloaded to <folder>) into $OPSIS_DATA/derived
with the file names the code reads, and add to the pan cancer slide questions the path of each slide's TRIDENT tile features under
$OPSIS_TCGA (gigapath_h5/<project>_trident_virchow2/h5_files/<slide>.h5). The organiser episodes go to results/organiser.
    python release/install_questions.py <folder>"""
import argparse, os, shutil, sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from opsis import paths

ap = argparse.ArgumentParser(); ap.add_argument("src"); a = ap.parse_args()
NAMES = {"questions.parquet": "questions_seed0.parquet", "split.json": "split_seed0.json", "bins.json": "bins_seed0.json",
         "questions_nucls_classes.parquet": "questions_nucls_seed0.parquet", "slide_labels.csv": "slide_labels_seed0.csv",
         "report_fields.jsonl": "extraction_merged.jsonl", "pan_cancer_fields.jsonl": "extracted_pan_all.jsonl",
         "pan_cancer_fields.csv": "extracted_pan_all_values.csv"}
DIRS = {"fields/nucls": "questions_final", "fields/pannuke": "pannuke", "slides/pan_cancer": "slide_questions_pan",
        "slides/breast": "slide_questions_v2", "reports": "report_pan"}
for src, dst in DIRS.items():
    for root, _, files in os.walk(f"{a.src}/{src}"):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), f"{a.src}/{src}"); out = f"{paths.DERIVED}/{dst}/{os.path.dirname(rel)}/{NAMES.get(f, f)}"
            os.makedirs(os.path.dirname(out), exist_ok=True); shutil.copy2(os.path.join(root, f), out)
if os.path.isdir(f"{a.src}/organiser"):
    shutil.copytree(f"{a.src}/organiser", f"{paths.RESULTS}/organiser", dirs_exist_ok=True)
q = f"{paths.DERIVED}/slide_questions_pan/questions_seed0.parquet"; Q = pd.read_parquet(q)
Q["h5_path"] = paths.TCGA + "/gigapath_h5/" + Q["project"].str.lower() + "_trident_virchow2/h5_files/" + Q["slide"] + ".h5"
Q.to_parquet(q, index=False)
missing = (~Q.drop_duplicates("slide")["h5_path"].map(os.path.exists)).sum()
print(f"installed into {paths.DERIVED}; {missing} of {Q.slide.nunique()} slides have no tile features under {paths.TCGA} yet")
