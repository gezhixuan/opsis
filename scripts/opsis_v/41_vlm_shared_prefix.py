#!/usr/bin/env python
"""Shared prefix inference for Opsis-V (the execution pattern of Visual Jev, arXiv 2609.25845): the image prefix is prefilled once,
its KV cache is expanded over the N question suffixes of the state, and the candidate letter logits are read at the end of
each suffix. Checks the probabilities against the plain full sequence computation and times both for N in --n-questions.
    python scripts/opsis_v/41_vlm_shared_prefix.py --run vlm_all_v2_s0 [--states 20] [--n-questions 1 8 32]
Writes runs/<run>/shared_prefix_timing.json (and a copy under results/runs/<run>/)."""
import argparse, json, os, shutil, time
import numpy as np, pandas as pd, torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from opsis import paths, facts
from opsis.data import nucls

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--model", default=f"{paths.MODELS}/Qwen3-VL-8B-Instruct"); ap.add_argument("--states", type=int, default=20); ap.add_argument("--n-questions", type=int, nargs="+", default=[1, 8, 32]); ap.add_argument("--attn", default="sdpa"); ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"]); ap.add_argument("--tag", default="")
a = ap.parse_args(); dev = torch.device("cuda"); OUT = f"{paths.RUNS}/{a.run}"; RES = f"{paths.RESULTS}/runs/{a.run}"; paths.ensure(RES)
CTX = f"{paths.DERIVED}/context/mpp{facts.CTX_MPP}_{facts.CTX_PX}"; cfg = json.load(open(f"{OUT}/config.json"))
Q = pd.read_parquet(f"{paths.DERIVED}/questions/questions_seed{cfg['split_seed']}.parquet"); F = pd.read_csv(f"{paths.DERIVED}/facts/fovs.csv").set_index("fov")
Q = Q[(Q["split"] == "test") & Q["fov"].map(lambda f: os.path.exists(f"{CTX}/{f}.png"))]
proc = AutoProcessor.from_pretrained(a.model); tok = proc.tokenizer; tok.padding_side = "right"
LETTERS = "ABCDEFG"; LID = torch.tensor([tok.encode(c, add_special_tokens=False)[0] for c in LETTERS], device=dev)
SYSTEM = "You are a pathology image analysis assistant. Answer each multiple choice question about the images with the option letter only."
DT = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=DT, attn_implementation=a.attn).to(dev)
model = PeftModel.from_pretrained(model, f"{OUT}/adapter").eval(); model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
def prefix_text(fov):
    W, H = int(F.loc[fov, "W"]), int(F.loc[fov, "H"])
    user = [{"type": "image"}, {"type": "text", "text": f"Image 1: a breast cancer H&E field of view at 0.2 micrometres per pixel ({W} x {H} px)."},
            {"type": "image"}, {"type": "text", "text": f"Image 2: the {facts.CTX_PX * facts.CTX_MPP:.0f} micrometre context window around the centre of image 1 at {facts.CTX_MPP} micrometres per pixel."}, {"type": "text", "text": "QSPLIT"}]
    t = proc.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
    return t.split("QSPLIT")[0], t.split("QSPLIT")[1]                 # prefix (ends right before the question), tail (im_end + assistant prompt)
def suffix_text(fov, q, tail):
    W, H = int(F.loc[fov, "W"]), int(F.loc[fov, "H"]); cands = json.loads(q["candidates"]); text = q["text"]
    if q["family"] == "cell":
        x0, y0, x1, y1 = json.loads(q["box"]); nb = [int(round(1000 * x0 / W)), int(round(1000 * y0 / H)), int(round(1000 * x1 / W)), int(round(1000 * y1 / H))]
        text += f" The marked nucleus is the one inside bbox_2d {nb} of image 1 (coordinates normalised to 0-1000)."
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(cands))
    return f"{text}\nOptions:\n{opts}\nAnswer with the option letter." + tail, len(cands)
@torch.no_grad()
def full_probs(fov, qs):
    pre, tail = prefix_text(fov); texts = [pre + suffix_text(fov, q, tail)[0] for q in qs]; ims = []
    for _ in texts: ims += [Image.open(nucls.rgb_path(fov)).convert("RGB"), Image.open(f"{CTX}/{fov}.png").convert("RGB")]
    tok.padding_side = "left"; enc = proc(text=texts, images=ims, return_tensors="pt", padding=True).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DT == torch.bfloat16): lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
    return torch.softmax(lg[:, LID], -1)
@torch.no_grad()
def shared_probs(fov, qs):
    pre, tail = prefix_text(fov); ims = [Image.open(nucls.rgb_path(fov)).convert("RGB"), Image.open(f"{CTX}/{fov}.png").convert("RGB")]
    enc = proc(text=[pre], images=ims, return_tensors="pt").to(dev); P = enc["input_ids"].shape[1]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DT == torch.bfloat16): out = model(**enc, use_cache=True, logits_to_keep=1)
    cache = out.past_key_values; rope_deltas = model.model.rope_deltas if hasattr(model, "model") else None
    sufs = [tok(suffix_text(fov, q, tail)[0], add_special_tokens=False)["input_ids"] for q in qs]; L = max(len(s) for s in sufs); N = len(sufs)
    ids = torch.full((N, L), tok.pad_token_id, dtype=torch.long, device=dev); am = torch.zeros((N, P + L), dtype=torch.long, device=dev); am[:, :P] = 1; last = []
    for i, s in enumerate(sufs): ids[i, :len(s)] = torch.tensor(s, device=dev); am[i, P:P + len(s)] = 1; last.append(len(s) - 1)
    cache.batch_repeat_interleave(N)
    cache_position = torch.arange(P, P + L, device=dev)
    inner = model.model if hasattr(model, "model") else model.base_model.model.model
    delta = inner.rope_deltas.reshape(-1)[0] if getattr(inner, "rope_deltas", None) is not None else torch.tensor(0, device=dev)
    pos = (torch.arange(L, device=dev) + P + delta).view(1, 1, L).expand(3, N, L).contiguous()      # text continuation of the mrope positions
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DT == torch.bfloat16): lg = model(input_ids=ids, attention_mask=am, past_key_values=cache, cache_position=cache_position, position_ids=pos, use_cache=True).logits.float()
    except Exception as e:
        print("explicit position ids failed:", type(e).__name__, str(e)[:120], "-> model computed positions", flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DT == torch.bfloat16): lg = model(input_ids=ids, attention_mask=am, past_key_values=cache, cache_position=cache_position, use_cache=True).logits.float()
    lg = lg[torch.arange(N), torch.tensor(last, device=dev)]
    return torch.softmax(lg[:, LID], -1)
groups = [(f, g) for f, g in Q.groupby("fov")][: a.states]; res = {}
for n in a.n_questions:
    diffs = []; agree = []; t_full = t_shared = 0.0; k = 0
    for fov, g in groups:
        qs = g.head(n).to_dict("records")
        if len(qs) < n: continue
        torch.cuda.synchronize(); t0 = time.time(); pf = full_probs(fov, qs); torch.cuda.synchronize(); t_full += time.time() - t0
        torch.cuda.synchronize(); t0 = time.time(); ps = shared_probs(fov, qs); torch.cuda.synchronize(); t_shared += time.time() - t0
        K = [len(json.loads(q["candidates"])) for q in qs]; d = max(float((pf[i, :K[i]] / pf[i, :K[i]].sum() - ps[i, :K[i]] / ps[i, :K[i]].sum()).abs().max()) for i in range(len(qs))); diffs.append(d); k += 1
        agree.append(float(np.mean([int(pf[i, :K[i]].argmax() == ps[i, :K[i]].argmax()) for i in range(len(qs))])))
    res[n] = dict(dtype=a.dtype, states=k, ms_full_per_state=1e3 * t_full / max(k, 1), ms_shared_per_state=1e3 * t_shared / max(k, 1), ms_shared_per_question=1e3 * t_shared / max(k * n, 1), speedup=t_full / max(t_shared, 1e-9), max_abs_prob_diff=float(np.max(diffs)) if diffs else None, mean_abs_prob_diff=float(np.mean(diffs)) if diffs else None, argmax_agreement=float(np.mean(agree)) if agree else None)
    print(f"N={n}: " + json.dumps(res[n]), flush=True)
fn = f"shared_prefix_timing{a.tag}.json"; json.dump(res, open(f"{OUT}/{fn}", "w"), indent=1); shutil.copy(f"{OUT}/{fn}", f"{RES}/{fn}"); print("RUN_DONE")
