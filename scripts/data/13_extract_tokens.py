#!/usr/bin/env python
"""Frozen Virchow2 tokens for every FOV and its context: features/<enc>/<set>.h5 with
    cls_fov [N, 1280], tok_fov [N, 32*32, 1280], cls_ctx, tok_ctx (fp16), and the fov list.
The FOV png (0.2 mpp) is resampled to 0.25 mpp, centre-padded with white to 448 x 448 (so 32 x 32 tokens; the FOV
occupies a known sub-window recorded in geometry.csv). The context png (448 px at 1.0 mpp) is used as is.
    python scripts/data/13_extract_tokens.py [--enc virchow2] [--batch 16] [--device cuda]"""
import argparse, os, time
import numpy as np, pandas as pd, torch, h5py
from PIL import Image
from opsis import paths, facts, encoder
from opsis.data import nucls

ap = argparse.ArgumentParser(); ap.add_argument("--enc", default="virchow2"); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--device", default="cuda"); ap.add_argument("--size", type=int, default=448); ap.add_argument("--fov-mpp", type=float, default=0.25); ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
OUT = f"{paths.FEATURES}/{a.enc}"; paths.ensure(OUT)
CTX = f"{paths.DERIVED}/context/mpp{facts.CTX_MPP}_{facts.CTX_PX}"
F = pd.read_csv(f"{paths.DERIVED}/facts/fovs.csv")
has_ctx = F["fov"].map(lambda n: os.path.exists(f"{CTX}/{n}.png")); print(f"{int((~has_ctx).sum())} FOVs without a context crop are skipped", flush=True); F = F[has_ctx]
if a.limit: F = F.head(a.limit)
names = F["fov"].tolist(); S = a.size
mean = torch.tensor(encoder.IMAGENET_MEAN).view(1, 3, 1, 1); std = torch.tensor(encoder.IMAGENET_STD).view(1, 3, 1, 1)
def prep_fov(name):
    im = Image.open(nucls.rgb_path(name)).convert("RGB"); W, H = im.size
    r = nucls.FOV_MPP / a.fov_mpp                            # 0.8: shrink to 0.25 mpp
    w, h = max(1, int(round(W * r))), max(1, int(round(H * r)))
    if max(w, h) > S:                                         # rare large FOVs: fit inside the canvas
        f = S / float(max(w, h)); w, h = int(w * f), int(h * f); r = r * f
    im = im.resize((w, h), Image.BICUBIC); canvas = Image.new("RGB", (S, S), (255, 255, 255)); ox, oy = (S - w) // 2, (S - h) // 2
    canvas.paste(im, (ox, oy))
    return canvas, dict(fov=name, W=W, H=H, fov_scale=r, off_x=ox, off_y=oy, fov_w=w, fov_h=h)
def to_tensor(ims):
    x = torch.from_numpy(np.stack([np.asarray(im, dtype=np.float32) / 255.0 for im in ims])).permute(0, 3, 1, 2)
    return ((x - mean) / std)
model = encoder.load_virchow2(a.device, img_size=S); N = len(names); n_tok = (S // model.patch) ** 2
geo = []
with h5py.File(f"{OUT}/tokens_{S}.h5.tmp", "w") as h:
    dn = h.create_dataset("fov", data=np.array(names, dtype="S"))
    cf = h.create_dataset("cls_fov", (N, model.dim), dtype="f2"); tf = h.create_dataset("tok_fov", (N, n_tok, model.dim), dtype="f2", chunks=(1, n_tok, model.dim))
    cc = h.create_dataset("cls_ctx", (N, model.dim), dtype="f2"); tc = h.create_dataset("tok_ctx", (N, n_tok, model.dim), dtype="f2", chunks=(1, n_tok, model.dim))
    t0 = time.time()
    for i in range(0, N, a.batch):
        bn = names[i:i + a.batch]; fovs, metas = zip(*[prep_fov(n) for n in bn]); geo += metas
        ctxs = [Image.open(f"{CTX}/{n}.png").convert("RGB") for n in bn]
        if S != facts.CTX_PX: ctxs = [im.resize((S, S), Image.BICUBIC) for im in ctxs]        # one token grid for both canvases (the context carries no new detail, only finer tokens)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.device == "cuda")):
            c1, t1 = encoder.encode_batch(model, to_tensor(fovs).to(a.device)); c2, t2 = encoder.encode_batch(model, to_tensor(ctxs).to(a.device))
        cf[i:i + len(bn)] = c1.float().cpu().numpy(); tf[i:i + len(bn)] = t1.float().cpu().numpy(); cc[i:i + len(bn)] = c2.float().cpu().numpy(); tc[i:i + len(bn)] = t2.float().cpu().numpy()
        if (i // a.batch) % 10 == 0: print(f"  {i + len(bn)}/{N}  {time.time() - t0:.0f}s", flush=True)
    h.attrs.update(dict(encoder=a.enc, size=S, patch=model.patch, fov_mpp=a.fov_mpp, ctx_mpp=facts.CTX_MPP, ctx_px=facts.CTX_PX, n_prefix=model.n_prefix))
os.replace(f"{OUT}/tokens_{S}.h5.tmp", f"{OUT}/tokens_{S}.h5"); pd.DataFrame(geo).to_csv(f"{OUT}/geometry_{S}.csv", index=False)
print("TOKENS_DONE", f"{OUT}/tokens_{S}.h5", "N", N, "tokens per image", n_tok, f"{time.time() - t0:.0f}s")
