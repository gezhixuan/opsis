#!/usr/bin/env python
"""Measured cost of the frozen encoder per state (Virchow2 ViT-H/14, bf16, one H200): 448 px canvas (32 x 32 tokens) and 896 px
(64 x 64), batch 1 and 16, ms per canvas; a FOV state is two canvases (field and context). Writes results/runs/encoder_timing.json."""
import json, time, torch
from opsis import paths, encoder as E
dev = torch.device("cuda"); out = {}
for S in (448, 896):
    m = E.load_virchow2(dev, img_size=S)
    for B in (1, 16):
        x = torch.randn(B, 3, S, S, device=dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3): m.forward_features(x)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(10): m.forward_features(x)
            torch.cuda.synchronize(); ms = (time.time() - t) / 10 / B * 1e3
        out[f"{S}px_batch{B}"] = dict(ms_per_canvas=round(ms, 2), ms_per_fov_state=round(2 * ms, 2)); print(S, B, f"{ms:.1f} ms per canvas", flush=True)
    del m; torch.cuda.empty_cache()
paths.ensure(f"{paths.RESULTS}/runs"); json.dump(out, open(f"{paths.RESULTS}/runs/encoder_timing.json", "w"), indent=1); print("RUN_DONE", out)
