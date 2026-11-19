#!/usr/bin/env python
"""Report model comparable: PRISM2 writes a free text report for every test and held out project slide of the pan TCGA set
("Write a report", the model card's recommended prompt). The reports go through the same extraction (scripts/questions/54_extract_openai.py, schema v1.1)
as the TCGA reports, so the report model is scored on exactly the synoptic fields our questions use.
    python scripts/baselines/85_prism2_reports.py [--max-tiles 8192] [--limit N]     -> derived/report_models/prism2_reports.jsonl"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch, h5py
from opsis import paths
ap = argparse.ArgumentParser(); ap.add_argument("--model", default=f"{paths.MODELS}/PRISM2"); ap.add_argument("--max-tiles", type=int, default=8192); ap.add_argument("--batch", type=int, default=4); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--max-new-tokens", type=int, default=300)
a = ap.parse_args(); dev = torch.device("cuda"); OUT = f"{paths.DERIVED}/report_models"; paths.ensure(OUT); F = f"{OUT}/prism2_reports.jsonl"
Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_pan/questions_seed0.parquet"); S = Q[Q["split"].isin(["test", "heldout_project"])].drop_duplicates("slide").set_index("slide"); slides = sorted(S.index)
if a.limit: slides = slides[: a.limit]
done = {json.loads(l)["slide"] for l in open(F)} if os.path.exists(F) else set(); todo = [s for s in slides if s not in done]; print(f"{len(slides)} slides, {len(done)} done, {len(todo)} to write", flush=True)
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained(a.model, trust_remote_code=True, torch_dtype="auto").to(dev).eval(); processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True); rng = np.random.default_rng(0)
def tiles(slide):
    with h5py.File(S.loc[slide, "h5_path"], "r") as h:
        n = h["features"].shape[0]; X = h["features"][np.sort(rng.choice(n, a.max_tiles, replace=False)), :1280] if n > a.max_tiles else h["features"][:, :1280]
    return torch.from_numpy(X.astype(np.float32))
t0 = time.time()
with open(F, "a") as f:
    for b0 in range(0, len(todo), a.batch):
        bs = todo[b0:b0 + a.batch]; batch = processor(tile_embeddings=[tiles(s) for s in bs]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16): ans = model.get_response(**batch, prompt="Write a report", max_new_tokens=a.max_new_tokens)
        for s, txt in zip(bs, ans): f.write(json.dumps(dict(participant=s, slide=s, project=str(S.loc[s, "project"]), split=str(S.loc[s, "split"]), text=str(txt).strip())) + "\n")
        f.flush()
        if (b0 // a.batch) % 25 == 0: print(f"  {b0 + len(bs)}/{len(todo)} slides, {time.time() - t0:.0f} s; sample: {str(ans[0])[:160]!r}", flush=True)
print("RUN_DONE", F)
