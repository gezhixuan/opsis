#!/usr/bin/env python
"""Assemble the two Hugging Face releases from a working data root:
  <out>/opsis-questions   the question sets, the audited report fields and the organiser episodes (--organiser)
  <out>/opsis-v           the Opsis-V LoRA adapter (one seed, chosen by validation NLL) with its model card
Report text is not redistributed: the TCGA-Reports corpus is cited and downloaded by the user. Columns that hold absolute paths
are dropped (the tile feature path of a slide is derived from $OPSIS_TCGA, see src/opsis/paths.py).
    python release/build_release.py --data $OPSIS_DATA --organiser <results/organiser> --out <folder>"""
import argparse, glob, json, os, shutil

import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--organiser", required=True, help="the organiser episodes and summaries, paths and connector listings already removed")
ap.add_argument("--vlm-runs", default="final_vlm_s0,final_vlm_s1,final_vlm_s2"); a = ap.parse_args()
D = f"{a.data}/derived"; R = f"{a.data}/runs"; HERE = os.path.dirname(os.path.abspath(__file__))
Q = f"{a.out}/opsis-questions"; V = f"{a.out}/opsis-v"
for d in (Q, V):
    if os.path.exists(d): shutil.rmtree(d)


def put(src, dst, drop=()):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if src.endswith(".parquet"):
        df = pd.read_parquet(src); df.drop(columns=[c for c in drop if c in df.columns]).to_parquet(dst, index=False)
    elif src.endswith(".jsonl") and drop:
        with open(dst, "w") as f:
            for line in open(src):
                r = json.loads(line); f.write(json.dumps({k: v for k, v in r.items() if k not in drop}) + "\n")
    else:
        shutil.copy2(src, dst)


# fields: NuCLS (with the BCSS tissue context) and PanNuke
for f, g in [("questions_seed0.parquet", "questions.parquet"), ("fovs.csv", "fovs.csv"), ("nuclei.csv", "nuclei.csv"),
             ("bins_seed0.json", "bins.json"), ("split_seed0.json", "split.json")]:
    put(f"{D}/questions_final/{f}", f"{Q}/fields/nucls/{g}")
for f, g in [("questions_seed0.parquet", "questions.parquet"), ("questions_nucls_seed0.parquet", "questions_nucls_classes.parquet"),
             ("fovs.csv", "fovs.csv"), ("nuclei.csv", "nuclei.csv"), ("regions.csv", "regions.csv"), ("bins_seed0.json", "bins.json")]:
    put(f"{D}/pannuke/{f}", f"{Q}/fields/pannuke/{g}")
# slides: pan cancer (32 TCGA projects, composites held out of training) and the breast synoptic set
for f, g in [("questions_seed0.parquet", "questions.parquet"), ("split_seed0.json", "split.json"), ("summary.json", "summary.json"),
             ("bcss_slides.txt", "bcss_slides.txt")]:
    put(f"{D}/slide_questions_pan/{f}", f"{Q}/slides/pan_cancer/{g}", drop=("h5_path",))
for f, g in [("questions_seed0.parquet", "questions.parquet"), ("split_seed0.json", "split.json"), ("slide_labels_seed0.csv", "slide_labels.csv"),
             ("extraction_merged.jsonl", "report_fields.jsonl")]:
    put(f"{D}/slide_questions_v2/{f}", f"{Q}/slides/breast/{g}")
# report fields of the pan cancer set (gpt-6-luna with verbatim quotes) and the independent audit extraction (Claude Sonnet 5.5)
put(f"{D}/report_pan/extracted_pan_all.jsonl", f"{Q}/reports/pan_cancer_fields.jsonl", drop=("_usage",))
put(f"{D}/report_pan/extracted_pan_all_values.csv", f"{Q}/reports/pan_cancer_fields.csv")
for f in sorted(glob.glob(f"{D}/report_pan/audit/extracted_sonnet_a*.jsonl")):
    put(f, f"{Q}/reports/audit/{os.path.basename(f).replace('extracted_sonnet_', 'audit_extraction_')}", drop=("_usage",))
put(f"{D}/report_pan/audit/disagreements.csv", f"{Q}/reports/audit/disagreements.csv")
# organiser episodes: one jsonl per condition (question, tool calls with results, answer, tokens, time) and its summary
shutil.copytree(a.organiser, f"{Q}/organiser")
shutil.copy2(f"{HERE}/dataset_card.md", f"{Q}/README.md")

# Opsis-V: the seed with the lowest validation NLL
runs = {r: json.load(open(f"{R}/{r}/metrics_test.json")) for r in a.vlm_runs.split(",")}
best = min(runs, key=lambda r: runs[r]["best_val_nll"])
os.makedirs(V); shutil.copy2(f"{R}/{best}/adapter/adapter_model.safetensors", f"{V}/adapter_model.safetensors")
cfg = json.load(open(f"{R}/{best}/adapter/adapter_config.json")); cfg["base_model_name_or_path"] = "Qwen/Qwen3-VL-8B-Instruct"
json.dump(cfg, open(f"{V}/adapter_config.json", "w"), indent=2)
train = json.load(open(f"{R}/{best}/config.json"))
json.dump({k: v for k, v in train.items() if k not in ("model",)} | {"base_model": "Qwen/Qwen3-VL-8B-Instruct"}, open(f"{V}/training_config.json", "w"), indent=1)
json.dump({r: dict(val_nll=m["best_val_nll"], test_nll=m["test_nll"]) for r, m in runs.items()} | {"released": best}, open(f"{V}/seeds.json", "w"), indent=1)
shutil.copy2(f"{HERE}/model_card.md", f"{V}/README.md")
print("released seed", best, {r: round(m["best_val_nll"], 4) for r, m in runs.items()})
for d in (Q, V):
    n = sum(len(fs) for _, _, fs in os.walk(d)); size = sum(os.path.getsize(os.path.join(p, f)) for p, _, fs in os.walk(d) for f in fs)
    print(f"{d}: {n} files, {size / 1e6:.1f} MB")
