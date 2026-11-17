#!/usr/bin/env python
"""Frozen biomedical text embeddings (SapBERT, mean pooled, 768 d) for every unique question text and candidate string of a
question set: $OPSIS_DATA/derived/text_emb/sapbert_<set>.npz with texts, cands, E_text, E_cand.
    python scripts/70_text_embeddings.py [--set questions|slide_questions] [--seed 0]"""
import argparse, json, os
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModel
from opsis import paths
ap = argparse.ArgumentParser(); ap.add_argument("--set", default="questions"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"); a = ap.parse_args()
Q = pd.read_parquet(f"{paths.DERIVED}/{a.set}/questions_seed{a.seed}.parquet")
texts = sorted(Q["text"].unique()); cands = sorted({c for cs in Q["candidates"].unique() for c in json.loads(cs)})
CAND_TEXT = {"tumor": "tumour cells", "stromal": "stromal cells", "sTIL": "lymphocytes or plasma cells", "other": "other cells", "no": "no", "yes": "yes",
             "stroma": "stroma", "inflamed_stroma": "lymphocytic infiltrate", "in_situ": "in situ carcinoma", "necrosis": "necrosis", "A": "region A", "B": "region B",
             "0": "zero", "1-3": "one to three", "4-10": "four to ten", "11-30": "eleven to thirty", ">30": "more than thirty", "q1": "smallest quartile", "q2": "second quartile", "q3": "third quartile", "q4": "largest quartile",
             "(0,0.1]": "up to ten percent", "(0.1,0.3]": "ten to thirty percent", "(0.3,0.6]": "thirty to sixty percent", ">0.6": "more than sixty percent",
             "<0.05": "under five percent", "[0.05,0.25)": "five to twenty five percent", "[0.25,0.5)": "twenty five to fifty percent", "[0.5,0.75)": "fifty to seventy five percent", ">=0.75": "at least seventy five percent",
             "tumor_nonMitotic": "non mitotic tumour cell", "tumor_mitotic": "mitotic tumour cell", "nonTILnonMQ_stromal": "stromal cell that is not an immune cell", "macrophage": "macrophage", "lymphocyte": "lymphocyte", "plasma_cell": "plasma cell", "other_nucleus": "other nucleus"}
tok = AutoTokenizer.from_pretrained(a.model); model = AutoModel.from_pretrained(a.model).eval()
@torch.no_grad()
def embed(strings):
    out = []
    for i in range(0, len(strings), 64):
        enc = tok(strings[i:i + 64], padding=True, truncation=True, max_length=64, return_tensors="pt"); h = model(**enc).last_hidden_state; m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).numpy())
    return np.concatenate(out).astype(np.float32)
E_text = embed(texts); E_cand = embed([CAND_TEXT.get(c, c) for c in cands])
out = f"{paths.DERIVED}/text_emb"; paths.ensure(out); np.savez(f"{out}/sapbert_{a.set}.npz", texts=np.array(texts), cands=np.array(cands), E_text=E_text, E_cand=E_cand)
print("TEXT_EMB_DONE", len(texts), "texts", len(cands), "candidates", E_text.shape, "->", f"{out}/sapbert_{a.set}.npz")
