#!/usr/bin/env python
"""Virchow2 tokens for the PanNuke patches: 256 px at 0.25 mpp pasted at the centre of a white 448 canvas (same layout as
the NuCLS FOV canvas). No context image. Writes features/pannuke_virchow2/{tokens_448.h5, geometry_448.csv}."""
import argparse, os, time
import numpy as np, pandas as pd, torch, h5py
from PIL import Image
from opsis import paths, encoder
ap = argparse.ArgumentParser(); ap.add_argument("--batch", type=int, default=32); a = ap.parse_args()
Q = f"{paths.DERIVED}/pannuke"; OUT = f"{paths.FEATURES}/pannuke_virchow2"; paths.ensure(OUT); S = 448; dev = torch.device("cuda")
F = pd.read_csv(f"{Q}/fovs.csv"); names = F["fov"].tolist(); N = len(names)
model = encoder.load_virchow2(dev, img_size=S); mean = torch.tensor(encoder.IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(encoder.IMAGENET_STD).view(1, 3, 1, 1)
def prep(n):
    im = Image.open(f"{Q}/rgb/{n}.png").convert("RGB"); canvas = Image.new("RGB", (S, S), (255, 255, 255)); canvas.paste(im, (96, 96)); return canvas
geo = [dict(fov=n, W=256, H=256, fov_scale=1.0, off_x=96, off_y=96, fov_w=256, fov_h=256) for n in names]; n_tok = (S // 14) ** 2; t0 = time.time()
with h5py.File(f"{OUT}/tokens_{S}.h5.tmp", "w") as h:
    h.create_dataset("fov", data=np.array(names, dtype="S")); cf = h.create_dataset("cls_fov", (N, 1280), dtype="f2"); tf = h.create_dataset("tok_fov", (N, n_tok, 1280), dtype="f2", chunks=(1, n_tok, 1280))
    for i in range(0, N, a.batch):
        ims = [prep(n) for n in names[i:i + a.batch]]; x = ((torch.from_numpy(np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in ims])).permute(0, 3, 1, 2) - mean) / std).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): c, t = encoder.encode_batch(model, x)
        cf[i:i + len(ims)] = c.float().cpu().numpy(); tf[i:i + len(ims)] = t.float().cpu().numpy()
        if (i // a.batch) % 20 == 0: print(f"  {i + len(ims)}/{N} {time.time() - t0:.0f}s", flush=True)
    h.attrs.update(dict(encoder="virchow2", size=S, patch=14, fov_mpp=0.25, ctx_mpp=1.0, ctx_px=448, n_prefix=5))
os.replace(f"{OUT}/tokens_{S}.h5.tmp", f"{OUT}/tokens_{S}.h5"); pd.DataFrame(geo).to_csv(f"{OUT}/geometry_{S}.csv", index=False); print("TOKENS_DONE", N, f"{time.time() - t0:.0f}s")
