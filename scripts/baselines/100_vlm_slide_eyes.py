#!/usr/bin/env python
"""Vision language models answering slide level questions by looking at the slide (the looking condition of scripts/organiser/81_organiser_eval.py: a 1024 px
thumbnail and six random 256 um tissue tiles at 0.5 um per pixel), without any label of the task, on the question sets of the
main table: --qset pan (the pan TCGA test split, scripts/questions/66_build_slide_questions_pan.py questions), brca (the TCGA-BRCA v2 bench, scripts/questions/52_build_slide_questions_v2.py), camelyon
(CAMELYON16 official test, 129 slides, "is metastatic carcinoma present", AUROC), bcss (the 8,830 20x tiles inside BCSS
regions of scripts/transfer/71_eval_tile_necrosis.py, one tile image, "is necrosis present", AUROC against the mask) and nucls (the NuCLS field questions of
the main table, FOV plus context image as scripts/opsis_v/40_train_vlm.py, API backend only; the local models are scored by scripts/opsis_v/40_train_vlm.py --eval-only).
Local backend: letter logits of the next token (as scripts/opsis_v/40_train_vlm.py zero shot), Qwen3-VL-8B or Patho-R1-7B, one GPU. API backend:
the OpenAI Responses API with the JSON answer of scripts/organiser/81_organiser_eval.py (confidence gives the score for AUROC).
    python scripts/baselines/100_vlm_slide_eyes.py --name eyes_qwen3vl8b_pan --qset pan --shard 0/3
    python scripts/baselines/100_vlm_slide_eyes.py --name eyes_sol_brca --backend api --model gpt-6-sol --qset brca
    python scripts/baselines/100_vlm_slide_eyes.py --name eyes_opus55_pan --backend anthropic --model claude-opus-5-5 --qset pan --shard 0/4
Anthropic backend: the Messages API through the anthropic SDK, same prompt, images and JSON reply as the OpenAI backend, --reasoning
is the output_config effort (low by default, thinking is always on for Claude Opus 5.5), key read from $ANTHROPIC_API_KEY or --key-file.
Writes results/organiser/<name>.jsonl (one row per question, resumable); scripts/collect/collect.py eyes scores them."""
import argparse, base64, io, json, os, re, sys, time
import numpy as np, pandas as pd
from PIL import Image, PngImagePlugin
PngImagePlugin.MAX_TEXT_CHUNK = 256 * 1024 * 1024  # some TCGA slides (e.g. TCGA-02) carry an ICC profile larger than the 1 MB default, which is saved into the PNG
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from opsis import paths, tools as T
from opsis.keys import api_key
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--model", default=f"{paths.MODELS}/Qwen3-VL-8B-Instruct"); ap.add_argument("--backend", default="local", choices=["local", "api", "anthropic"])
ap.add_argument("--qset", required=True, choices=["pan", "brca", "camelyon", "bcss", "nucls"]); ap.add_argument("--ops", default="main", help="main = the rows of the main table, all, or a comma list")
ap.add_argument("--n-tiles", type=int, default=6); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--shard", default=None); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--resume", action="store_true")
ap.add_argument("--attn", default="flash_attention_2"); ap.add_argument("--key-file", default=None, help="file holding the API key (default: $OPENAI_API_KEY, or $ANTHROPIC_API_KEY for the anthropic backend)"); ap.add_argument("--base", default="https://api.openai.com/v1"); ap.add_argument("--reasoning", default="low")
ap.add_argument("--blank-image", action="store_true", help="control: every image replaced by a uniform grey image of the same size, captions unchanged (language prior, as scripts/opsis_v/40_train_vlm.py --blank-image)"); ap.add_argument("--bcss-run", default="final_tile_noisyor_s0"); ap.add_argument("--frac", type=float, default=0.5, help="bcss: a tile is necrotic when at least this fraction of its mask is necrosis or debris")
a = ap.parse_args(); OUT = f"{paths.ROOT}/results/organiser"; os.makedirs(OUT, exist_ok=True); LOG = f"{OUT}/{a.name}.jsonl"; rng = np.random.default_rng(a.seed)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def LAB(i): return LETTERS[i] if i < 26 else LETTERS[i // 26 - 1] + LETTERS[i % 26]
MAIN = {"pan": ["cancer_type", "pan_necrosis", "pan_lvi", "pan_pni", "pan_mitotic", "tumor_purity_bin", "high_grade_and_lvi", "necrosis_and_high_mitotic", "grade_from_components", "precursor_type", "dcis_with_comedo"],
        "brca": ["er_positive", "histologic_grade", "lvi_present", "pam50"], "camelyon": ["metastasis"], "bcss": ["necrosis"],
        "nucls": ["cell_super", "cell_main", "cell_area_bin", "compare_regions", "dominant_tissue", "stil_score_bin"]}
OPS = MAIN[a.qset] if a.ops == "main" else None if a.ops == "all" else a.ops.split(",")
CAM = paths.CAMELYON
# ---------- tasks: (unit, question dict, image spec) ----------
def tissue_tiles_h5(slide, n):
    import h5py
    _, S = T._registry()
    with h5py.File(S.loc[slide, "h5_path"], "r") as h: C0 = h["coords_level0"][:]; ps = float(h.attrs.get("patch_size_level0", 512))
    idx = rng.choice(len(C0), min(n, len(C0)), replace=False); return [(int(C0[i, 0] + ps / 2), int(C0[i, 1] + ps / 2)) for i in idx]
def q_of(r): return dict(qid=r.qid, operator=r.op, question=r.text, candidates=json.loads(r.candidates), answer=int(r.answer))
tasks = []
if a.qset == "pan":
    for s in T.list_slides("test", None, 5000, a.seed):
        for q in T.ground_truth(s, OPS): tasks.append((s, q, "slide"))
elif a.qset == "brca":
    Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_v2/questions_seed0.parquet"); Q = Q[(Q.split == "test") & Q.family.isin(["slide", "molecular"])]
    if OPS: Q = Q[Q.op.isin(OPS)]
    for r in Q.sort_values(["slide", "qid"]).itertuples(): tasks.append((r.slide, q_of(r), "slide"))
elif a.qset == "camelyon":
    import openslide
    S = pd.read_csv(f"{CAM}/splits.csv"); S = S[(S.dataset == "c16") & (S.role == "T1")].sort_values("slide")
    PATH = {s: f"{CAM}/CAMELYON16/images/{s}.tif" for s in S.slide}; T._wsi = lambda s: openslide.OpenSlide(PATH[s])
    def cam_tiles(slide, n):
        ms = openslide.OpenSlide(f"{CAM}/CAMELYON16/masks/{slide}_mask.tif"); lev = ms.get_best_level_for_downsample(32); ds = ms.level_downsamples[lev]
        tis = np.asarray(ms.read_region((0, 0), lev, ms.level_dimensions[lev]).convert("L")) > 0; ms.close(); half = max(1, int(round(128 / (0.243 * ds))))
        pts = [(c, r) for r in range(half, tis.shape[0] - half, half) for c in range(half, tis.shape[1] - half, half) if tis[r - half: r + half, c - half: c + half].mean() >= 0.5]
        pick = rng.choice(len(pts), min(n, len(pts)), replace=False); return [(int((pts[i][0] + 0.5) * ds), int((pts[i][1] + 0.5) * ds)) for i in pick]
    for r in S.itertuples():
        tasks.append((r.slide, dict(qid=f"cam_{r.slide}", operator="metastasis", question="Is metastatic carcinoma present in this sentinel lymph node slide?", candidates=["no", "yes"], answer=int(r.label_official in ("micro", "macro"))), "slide"))
elif a.qset == "bcss":
    import glob
    E = pd.read_parquet(f"{paths.RUNS}/{a.bcss_run}/tile_eval_bcss.parquet"); TS = f"{paths.RUNS}/{a.bcss_run}/tile_scores"
    files = {re.sub(r"-01Z-00-", "-", os.path.basename(f).split(".")[0]): f for f in glob.glob(f"{TS}/*.npz")}
    for stem, g in E.groupby("stem"):
        z = np.load(files[stem]); C = z["coords_level0"]; ps = float(z["patch_size_level0"]); slide = os.path.basename(files[stem]).replace(".npz", "")
        for r in g.itertuples():
            cx, cy = int(C[r.tile, 0] + ps / 2), int(C[r.tile, 1] + ps / 2)
            tasks.append((slide, dict(qid=f"bcss_{stem}_{r.tile}", operator="necrosis", question="Is tumour necrosis present in this 128 um tile of a breast cancer slide?", candidates=["no", "yes"], answer=int(r.frac_necrosis_or_debris >= a.frac), frac=float(r.frac_necrosis_or_debris)), ("tile", cx, cy, ps)))
elif a.qset == "nucls":
    from opsis import facts; from opsis.data import nucls
    Q = pd.read_parquet(f"{paths.DERIVED}/questions_final/questions_seed0.parquet"); Q = Q[Q.split == "test"]
    if OPS: Q = Q[Q.op.isin(OPS)]
    F = pd.read_parquet(f"{paths.DERIVED}/facts/fovs.parquet").set_index("fov") if os.path.exists(f"{paths.DERIVED}/facts/fovs.parquet") else None
    CTX = f"{paths.DERIVED}/context/mpp{facts.CTX_MPP}_{facts.CTX_PX}"
    for r in Q.sort_values(["fov", "qid"]).itertuples():
        q = q_of(r); q["family"] = r.family; q["box"] = r.box; tasks.append((r.fov, q, "fov"))
tasks.sort(key=lambda t: (t[0], t[1]["qid"]))
if a.shard: i_, n_ = map(int, a.shard.split("/")); tasks = tasks[i_::n_]
if a.limit: tasks = tasks[: a.limit]
done = set()
if a.resume and os.path.exists(LOG): done = {json.loads(l)["qid"] for l in open(LOG)}
print(f"{a.qset}: {len(tasks)} questions in this shard, {len(done)} done, backend {a.backend} model {a.model}", flush=True)
SYSTEM = "You are a pathologist answering one multiple choice question about an H&E image. Look at every image given, then answer."
# ---------- images for a task: list of (caption, PIL image) ----------
def b2i(b64): return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
def grey(im): return Image.new("RGB", im.size, (128, 128, 128))
def images_for(unit, spec):
    if a.blank_image and spec == "slide":   # the thumbnail is read for its size and caption, the tiles are 512 px by construction
        th = T.thumbnail(unit, 1024); pts = cam_tiles(unit, a.n_tiles) if a.qset == "camelyon" else tissue_tiles_h5(unit, a.n_tiles)
        return [(f"Thumbnail of the whole slide ({th['mpp']} um per pixel):", grey(b2i(th["image_png_base64"])))] + [(f"Tile at level 0 ({x}, {y}), 256 um at 0.5 um per pixel:", Image.new("RGB", (512, 512), (128, 128, 128))) for (x, y) in pts]
    if a.blank_image: a_blank = images_real(unit, spec); return [(cap, grey(im)) for cap, im in a_blank]
    return images_real(unit, spec)
def images_real(unit, spec):
    if spec == "slide":
        th = T.thumbnail(unit, 1024); ims = [(f"Thumbnail of the whole slide ({th['mpp']} um per pixel):", b2i(th["image_png_base64"]))]
        pts = cam_tiles(unit, a.n_tiles) if a.qset == "camelyon" else tissue_tiles_h5(unit, a.n_tiles)
        for (x, y) in pts: ims.append((f"Tile at level 0 ({x}, {y}), 256 um at 0.5 um per pixel:", b2i(T.view_region(unit, x, y, 256, 0.5, 512)["image_png_base64"])))
        return ims
    if spec[0] == "tile":
        _, cx, cy, ps = spec; return [("The tile, 128 um at 0.25 um per pixel:", b2i(T.view_region(unit, cx, cy, 128, 0.25, 512)["image_png_base64"]))]
    if spec == "fov":
        im = Image.open(nucls.rgb_path(unit)).convert("RGB"); W, H = im.size
        return [(f"Image 1: a breast cancer H&E field of view at 0.2 micrometres per pixel ({W} x {H} px).", im), (f"Image 2: the {facts.CTX_PX * facts.CTX_MPP:.0f} micrometre context window around the centre of image 1 at {facts.CTX_MPP} micrometres per pixel.", Image.open(f"{CTX}/{unit}.png").convert("RGB"))]
def question_text(unit, q, spec):
    text = q["question"]
    if spec == "fov" and q.get("family") == "cell" and q.get("box"):
        W, H = Image.open(nucls.rgb_path(unit)).size; x0, y0, x1, y1 = json.loads(q["box"]); nb = [int(round(1000 * x0 / W)), int(round(1000 * y0 / H)), int(round(1000 * x1 / W)), int(round(1000 * y1 / H))]
        text += f" The marked nucleus is the one inside bbox_2d {nb} of image 1 (coordinates normalised to 0-1000)."
    return text
# ---------- backends ----------
if a.backend == "local":
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    dev = torch.device("cuda"); proc = AutoProcessor.from_pretrained(a.model); tok = proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation=a.attn).to(dev).eval()
    LID = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]; assert all(len(tok.encode(c, add_special_tokens=False)) == 1 for c in LETTERS)
    def answer(unit, q, spec):
        ims = images_for(unit, spec); K = len(q["candidates"]); opts = "\n".join(f"{LAB(i)}. {c}" for i, c in enumerate(q["candidates"]))
        user = []
        for cap, im in ims: user += [{"type": "text", "text": cap}, {"type": "image"}]
        user.append({"type": "text", "text": f"{question_text(unit, q, spec)}\nOptions:\n{opts}\nAnswer with the option letter only."})
        text = proc.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        enc = proc(text=[text], images=[im for _, im in ims], return_tensors="pt"); enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): lg = model(**enc, logits_to_keep=1).logits[0, -1].float()
        if K <= 26: cl = lg[LID[:K]]
        else: cl = torch.stack([lg[LID[i]] if i < 26 else lg[LID[i // 26 - 1]] for i in range(K)])   # two letter labels: scored by their first letter (cancer type only)
        p = torch.softmax(cl, -1).cpu().numpy(); pred = int(p.argmax())
        return dict(pred=pred, confidence=float(p[pred]), probs=[round(float(x), 5) for x in p], usage=dict(prompt=int(enc["input_ids"].shape[1]), completion=0), text="")
def parse_reply(text, labs, K):
    """the JSON reply of an API model -> (pred, confidence, probs); falls back to the first option letter in the text"""
    ans = None; conf = None
    try: j = json.loads(text[text.index("{"): text.rindex("}") + 1]); ans = str(j.get("answer", "")).strip().upper().rstrip("."); conf = float(j.get("confidence", 0.5))
    except Exception: ans = next((c for c in re.findall(r"[A-Z]+", text.upper()) if c in labs), None)
    pred = labs.index(ans) if ans in labs else None
    probs = [round((conf if i == pred else (1 - conf) / max(K - 1, 1)), 5) for i in range(K)] if pred is not None and conf is not None else None
    return pred, conf, probs
REPLY = "Reply with a JSON object only: {\"answer\": \"<option letter>\", \"confidence\": <0 to 1>, \"reason\": \"<one sentence>\"}"
def png64(im): b = io.BytesIO(); im.save(b, format="PNG"); return base64.b64encode(b.getvalue()).decode()
if a.backend == "anthropic":
    import anthropic
    client = anthropic.Anthropic(api_key=api_key(a.key_file, env="ANTHROPIC_API_KEY"), max_retries=6, timeout=600)
    def answer(unit, q, spec):
        ims = images_for(unit, spec); K = len(q["candidates"]); labs = [LAB(i) for i in range(K)]; opts = "\n".join(f"{l}. {c}" for l, c in zip(labs, q["candidates"]))
        content = [{"type": "text", "text": f"Question: {question_text(unit, q, spec)}\nOptions:\n{opts}"}]
        for cap, im in ims: content += [{"type": "text", "text": cap}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png64(im)}}]
        content.append({"type": "text", "text": REPLY})
        kw = dict(model=a.model, max_tokens=4000, system=SYSTEM, messages=[{"role": "user", "content": content}])
        if a.reasoning: kw["output_config"] = {"effort": a.reasoning}
        for attempt in range(4):
            try: d = client.messages.create(**kw); break
            except anthropic.APIStatusError as e:
                print("api error", type(e).__name__, str(e)[:200], flush=True); time.sleep(10 * (attempt + 1))
        else: raise RuntimeError("anthropic request failed four times")
        text = "".join(b.text for b in d.content if b.type == "text"); u = d.usage
        pred, conf, probs = parse_reply(text, labs, K)
        return dict(pred=pred, confidence=conf, probs=probs, usage=dict(prompt=int(u.input_tokens), completion=int(u.output_tokens), cache_read=int(getattr(u, "cache_read_input_tokens", 0) or 0)), text=text[:400], stop=d.stop_reason)
elif a.backend == "api":
    import ssl, certifi, urllib.request
    KEY = api_key(a.key_file); SSL = ssl.create_default_context(cafile=certifi.where())
    def api(body):
        req = urllib.request.Request(a.base + "/responses", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=600, context=SSL) as r: return json.load(r)
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                if hasattr(e, "read"):
                    try: err += " | " + e.read().decode()[:400]
                    except Exception: pass
                print("api error", err, flush=True); time.sleep(5 * (attempt + 1))
        raise RuntimeError(err)
    def answer(unit, q, spec):
        ims = images_for(unit, spec); K = len(q["candidates"]); labs = [LAB(i) for i in range(K)]; opts = "\n".join(f"{l}. {c}" for l, c in zip(labs, q["candidates"]))
        content = [{"type": "input_text", "text": f"Question: {question_text(unit, q, spec)}\nOptions:\n{opts}"}]
        for cap, im in ims: content += [{"type": "input_text", "text": cap}, {"type": "input_image", "image_url": "data:image/png;base64," + png64(im), "detail": "high"}]
        content.append({"type": "input_text", "text": REPLY})
        body = {"model": a.model, "instructions": SYSTEM, "input": [{"role": "user", "content": content}], "max_output_tokens": 600, "store": False}
        if a.reasoning and a.model.startswith(("gpt-5", "gpt-6", "o")): body["reasoning"] = {"effort": a.reasoning}
        d = api(body); u = d.get("usage", {}); out = d.get("output", [])
        text = d.get("output_text") or "".join(c.get("text", "") for o in out if o.get("type") == "message" for c in o.get("content", []) if c.get("type") == "output_text")
        pred, conf, probs = parse_reply(text, labs, K)
        return dict(pred=pred, confidence=conf, probs=probs, usage=dict(prompt=u.get("input_tokens", 0), completion=u.get("output_tokens", 0)), text=text[:400])
# ---------- loop ----------
t00 = time.time(); n_done = 0
with open(LOG, "a") as f:
    for i, (unit, q, spec) in enumerate(tasks):
        if q["qid"] in done: continue
        t0 = time.time()
        try: r = answer(unit, q, spec)
        except Exception as e:
            print(f"[{i + 1}/{len(tasks)}] {unit[:20]} {q['operator']} ERROR {type(e).__name__}: {str(e)[:200]}", flush=True); continue
        row = dict(unit=unit, qid=q["qid"], operator=q["operator"], answer=q["answer"], correct=(r["pred"] == q["answer"]), seconds=round(time.time() - t0, 1), **{k: v for k, v in q.items() if k in ("frac",)}, **r)
        f.write(json.dumps(row) + "\n"); f.flush(); n_done += 1
        if n_done <= 5 or n_done % 50 == 0: print(f"[{i + 1}/{len(tasks)}] {unit[:20]} {q['operator']:26s} truth {q['candidates'][q['answer']]!s:12s} pred {q['candidates'][r['pred']] if r['pred'] is not None else None!s:12s} {'ok' if row['correct'] else 'x '} conf {r['confidence'] if r['confidence'] is None else round(r['confidence'], 2)} tok {r['usage']['prompt']} {time.time() - t0:.1f}s ({(time.time() - t00) / n_done:.1f} s/q)", flush=True)
print("DONE", a.name, n_done, "answered", flush=True)
