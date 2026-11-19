#!/usr/bin/env python
"""Opsis-V on Qwen3-VL: shared image prefix (FOV + context + scale text), the natural language
questions of the field question set with lettered candidates, answer probability = softmax over the candidate letters'
LM head logits at the answer position. Vision tower frozen, LoRA on the language tower, answer token cross entropy over
the full vocabulary (the Visual Jev recipe, arXiv 2609.25845).
    python scripts/opsis_v/40_train_vlm.py --name vlm_all_s0 [--protocol all|comp_holdout] [--no-ctx] [--target coords|none]
Outputs under $OPSIS_DATA/runs/<name>/: adapter/, history.json, predictions_{val,test}.parquet, metrics_test.json, config.json;
json files copied to results/runs/<name>/ (same format as the Opsis-S runs, so the collectors read them)."""
import argparse, json, os, shutil, time
import numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, PeftModel
from opsis import paths, facts, metrics as M
from opsis.data import nucls
from opsis.questions import FAMILY

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--protocol", default="all", choices=["all", "comp_holdout"]); ap.add_argument("--qset", default="questions", help="question set directory under derived/ (NuCLS FOV sets only: questions, questions_final)")
ap.add_argument("--model", default=f"{paths.MODELS}/Qwen3-VL-8B-Instruct"); ap.add_argument("--split-seed", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--q-per-state", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--val-states", type=int, default=64); ap.add_argument("--limit-states", type=int, default=None); ap.add_argument("--test-states", type=int, default=None)
ap.add_argument("--eval-batch", type=int, default=16); ap.add_argument("--no-ctx", action="store_true"); ap.add_argument("--target", default="coords", choices=["coords", "none"])
ap.add_argument("--attn", default="flash_attention_2"); ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--blank-image", action="store_true", help="control: every image replaced by a uniform grey canvas (language prior)")
ap.add_argument("--v2tok", default="none", choices=["none", "pool8"], help="inject 8x8 pooled Virchow2 tokens per canvas through a trained projector")
ap.add_argument("--vision", default="qwen", choices=["qwen", "none"], help="qwen: native image tokens in the prefix; none: text and Virchow2 tokens only")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev = torch.device("cuda"); OUT = f"{paths.RUNS}/{a.name}"; paths.ensure(OUT); RES = f"{paths.RESULTS}/runs/{a.name}"; paths.ensure(RES)
CTX = f"{paths.DERIVED}/context/mpp{facts.CTX_MPP}_{facts.CTX_PX}"
Q = pd.read_parquet(f"{paths.DERIVED}/{a.qset}/questions_seed{a.split_seed}.parquet"); F = pd.read_csv(f"{paths.DERIVED}/facts/fovs.csv").set_index("fov")
Q = Q[Q["fov"].map(lambda f: os.path.exists(f"{CTX}/{f}.png"))].reset_index(drop=True)
states = {s: sorted(Q.loc[Q["split"] == s, "fov"].unique()) for s in ("train", "val", "test")}
if a.limit_states: states["train"] = states["train"][: a.limit_states]
states["val"] = list(np.random.default_rng(0).permutation(states["val"])[: a.val_states])
if a.test_states: states["test"] = states["test"][: a.test_states]
train_q = Q[Q["split"] == "train"]
if a.protocol == "comp_holdout": train_q = train_q[train_q["family"] != "comp"]
groups = {s: {f: g for f, g in Q[Q["fov"].isin(states[s])].groupby("fov")} for s in ("val", "test")}
groups["train"] = {f: g for f, g in train_q[train_q["fov"].isin(states["train"])].groupby("fov")}
print({s: (len(states[s]), sum(len(g) for g in groups[s].values())) for s in groups}, flush=True)

proc = AutoProcessor.from_pretrained(a.model); tok = proc.tokenizer; tok.padding_side = "left"
assert not (a.vision == "none" and a.v2tok == "none"), "vision none needs --v2tok"
PH = "<|fim_pad|>"; PH_ID = tok.convert_tokens_to_ids(PH); NV2 = 64; assert tok.encode(PH, add_special_tokens=False) == [PH_ID]
V2 = None
if a.v2tok != "none":
    z = np.load(f"{paths.FEATURES}/virchow2/tokens_448_pool8.npz"); V2 = {"idx": {n: i for i, n in enumerate(z["fov"])}, "fov": z["tok_fov"], "ctx": z["tok_ctx"]}
    print("Virchow2 pooled tokens:", V2["fov"].shape, flush=True)
LETTERS = "ABCDEFG"; LETTER_IDS = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]; assert all(len(tok.encode(c, add_special_tokens=False)) == 1 for c in LETTERS)
SYSTEM = "You are a pathology image analysis assistant. Answer each multiple choice question about the images with the option letter only."
def prompt(fov, q):
    W, H = int(F.loc[fov, "W"]), int(F.loc[fov, "H"]); cands = json.loads(q["candidates"]); text = q["text"]
    if q["family"] == "cell" and a.target == "coords":
        x0, y0, x1, y1 = json.loads(q["box"]); nb = [int(round(1000 * x0 / W)), int(round(1000 * y0 / H)), int(round(1000 * x1 / W)), int(round(1000 * y1 / H))]
        text += f" The marked nucleus is the one inside bbox_2d {nb} of image 1 (coordinates normalised to 0-1000)."
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(cands))
    img = [{"type": "image"}] if a.vision == "qwen" else []
    v2 = (lambda k: [{"type": "text", "text": f" Pathology encoder features of image {k}: " + PH * NV2 + "."}]) if a.v2tok != "none" else (lambda k: [])
    user = img + [{"type": "text", "text": f"Image 1: a breast cancer H&E field of view at 0.2 micrometres per pixel ({W} x {H} px)."}] + v2(1)
    if not a.no_ctx: user += img + [{"type": "text", "text": f"Image 2: the {facts.CTX_PX * facts.CTX_MPP:.0f} micrometre context window around the centre of image 1 at {facts.CTX_MPP} micrometres per pixel."}] + v2(2)
    user += [{"type": "text", "text": f"{text}\nOptions:\n{opts}\nAnswer with the option letter."}]
    return proc.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True), len(cands)
_imcache = {}
def images(fov):
    if fov not in _imcache:
        ims = [Image.open(nucls.rgb_path(fov)).convert("RGB")] + ([] if a.no_ctx else [Image.open(f"{CTX}/{fov}.png").convert("RGB")])
        if a.blank_image: ims = [Image.new("RGB", im.size, (128, 128, 128)) for im in ims]
        if len(_imcache) > 64: _imcache.clear()
        _imcache[fov] = ims
    return _imcache[fov]
CUR = {}
def build(fov, qs):
    texts, ks = zip(*[prompt(fov, q) for q in qs.to_dict("records")]); ims = []
    if a.vision == "qwen":
        for _ in texts: ims += images(fov)
    enc = proc(text=list(texts), images=ims if ims else None, return_tensors="pt", padding=True)
    enc = {k: v.to(dev) for k, v in enc.items()}
    if V2 is not None:                                              # Virchow2 tokens for the hook: [B, n_canvas*64, 1280] and the placeholder mask
        i = V2["idx"][fov]; t = [V2["fov"][i]] + ([] if a.no_ctx else [V2["ctx"][i]])
        CUR["tok"] = torch.from_numpy(np.concatenate(t, 0)).to(dev).unsqueeze(0).expand(len(texts), -1, -1)
        CUR["mask"] = enc["input_ids"] == PH_ID
        assert int(CUR["mask"].sum()) == CUR["tok"].shape[0] * CUR["tok"].shape[1], "placeholder count mismatch"
    return enc, list(ks)
def sample_questions(g):
    if len(g) <= a.q_per_state: return g
    ops = g["op"].to_numpy(); uniq = np.unique(ops); per = {o: np.flatnonzero(ops == o) for o in uniq}; chosen = set()
    while len(chosen) < a.q_per_state:
        for o in rng.permutation(uniq):
            chosen.add(int(rng.choice(per[o])))
            if len(chosen) >= a.q_per_state: break
    return g.iloc[sorted(chosen)]

model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation=a.attn).to(dev)   # Qwen3-VL, Qwen2.5-VL (Patho-R1), Gemma 3 (MedGemma)
for p in model.parameters(): p.requires_grad_(False)
if a.eval_only and os.path.isdir(f"{OUT}/adapter"):
    model = PeftModel.from_pretrained(model, f"{OUT}/adapter")
else:
    lcfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none", target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, lcfg)
model.print_trainable_parameters()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
projector = None
if V2 is not None:
    import torch.nn as nn
    D = model.config.text_config.hidden_size
    class Projector(nn.Module):
        def __init__(self):
            super().__init__(); self.net = nn.Sequential(nn.LayerNorm(1280), nn.Linear(1280, D), nn.GELU(), nn.Linear(D, D))
            self.src = nn.Parameter(torch.zeros(2, D)); self.pos = nn.Parameter(torch.zeros(NV2, D)); nn.init.normal_(self.src, std=0.02); nn.init.normal_(self.pos, std=0.02)
        def forward(self, x):                                       # x [B, n_canvas*64, 1280] -> [B, n_canvas*64, D]
            B, T, _ = x.shape; n = T // NV2; y = self.net(x.float())
            y = y + self.src[:n].repeat_interleave(NV2, 0)[None] + self.pos.repeat(n, 1)[None]
            return y
    projector = Projector().to(dev)
    if a.eval_only and os.path.exists(f"{OUT}/projector.pt"): projector.load_state_dict(torch.load(f"{OUT}/projector.pt", map_location=dev))
    lm = model.base_model.model.model.language_model if hasattr(model, "base_model") else model.model.language_model
    def _inject(module, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        if emb is None or "tok" not in CUR: return None
        emb = emb.clone(); emb[CUR["mask"]] = projector(CUR["tok"]).reshape(-1, emb.shape[-1]).to(emb.dtype); kwargs["inputs_embeds"] = emb
        return args, kwargs
    lm.register_forward_pre_hook(_inject, with_kwargs=True); print("Virchow2 injection hook on", type(lm).__name__, flush=True)
def trainable_named():
    for n, p in model.named_parameters():
        if p.requires_grad: yield n, p
    if projector is not None:
        for n, p in projector.named_parameters(): yield "projector." + n, p
LID = torch.tensor(LETTER_IDS, device=dev)
_LTK = [True]
def answer_logits(enc, train):
    """Full vocabulary logits at the answer position (the token after the generation prompt): [B, V]."""
    if _LTK[0]:
        try:
            return model(**enc, logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            _LTK[0] = False; print("logits_to_keep unsupported, using full logits", flush=True)
    return model(**enc).logits[:, -1, :]
def loss_and_probs(enc, ks, answers):
    lg = answer_logits(enc, True).float()                           # [B, V]
    loss = Fn.cross_entropy(lg, torch.tensor([LETTER_IDS[x] for x in answers], device=dev))
    cl = lg[:, LID]; mask = torch.arange(7, device=dev)[None] < torch.tensor(ks, device=dev)[:, None]
    probs = torch.softmax(cl.masked_fill(~mask, -1e4), -1)
    return loss, probs
@torch.no_grad()
def evaluate(split, collect=False):
    model.eval(); tot = n = 0; rows = []
    for fov, g in groups[split].items():
        for i in range(0, len(g), a.eval_batch):
            qs = g.iloc[i:i + a.eval_batch]; enc, ks = build(fov, qs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = answer_logits(enc, False).float()
            ans = torch.tensor([LETTER_IDS[x] for x in qs["answer"]], device=dev)
            cl = lg[:, LID]; mask = torch.arange(7, device=dev)[None] < torch.tensor(ks, device=dev)[:, None]; probs = torch.softmax(cl.masked_fill(~mask, -1e4), -1)
            nll = -torch.log(probs[torch.arange(len(qs)), qs["answer"].to_numpy()].clamp(min=1e-12)); tot += nll.sum().item(); n += len(qs)
            if collect:
                for q, p in zip(qs.itertuples(), probs.cpu().numpy()):
                    rows.append(dict(qid=q.qid, fov=q.fov, patient=q.patient, family=q.family, op=q.op, atype=q.atype, candidates=q.candidates, answer=int(q.answer), probs=p[: json.loads(q.candidates).__len__()].tolist()))
    model.train(); return tot / max(n, 1), pd.DataFrame(rows)
hist = []; best = float("inf"); t0 = time.time()
if not a.eval_only:
    params = [p for _, p in trainable_named()]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * len(groups["train"]); warm = max(1, len(groups["train"]) // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(steps, 1)))))
    model.train(); step = 0
    for ep in range(a.epochs):
        order = rng.permutation(list(groups["train"])); tl = []
        for fov in order:
            qs = sample_questions(groups["train"][fov]); enc, ks = build(fov, qs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = loss_and_probs(enc, ks, qs["answer"].to_numpy())
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); tl.append(loss.item()); step += 1
            if step % 100 == 0: print(f"  ep {ep} step {step} train nll {np.mean(tl[-100:]):.3f} {time.time() - t0:.0f}s", flush=True)
        va, _ = evaluate("val"); hist.append(dict(epoch=ep, train_nll=float(np.mean(tl)), val_nll=va, time=time.time() - t0)); print(json.dumps(hist[-1]), flush=True)
        if va < best:
            best = va; model.save_pretrained(f"{OUT}/adapter"); best_sd = {n: p.detach().clone() for n, p in trainable_named()}
            if projector is not None: torch.save(projector.state_dict(), f"{OUT}/projector.pt")
    json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    with torch.no_grad():                                           # restore the best epoch's LoRA and projector weights
        for n, p in trainable_named(): p.copy_(best_sd[n])
va, Pv = evaluate("val", collect=True); Pv.to_parquet(f"{OUT}/predictions_val.parquet", index=False)
te, P = evaluate("test", collect=True); P.to_parquet(f"{OUT}/predictions_test.parquet", index=False)
res = {"test_nll": te, "best_val_nll": best if hist else va, "protocol": a.protocol, "use_ctx": not a.no_ctx, "model": f"qwen3vl8b_lora_vision-{a.vision}_v2tok-{a.v2tok}", "per_op": {}}
for op, g in P.groupby("op"):
    K = len(json.loads(g["candidates"].iloc[0])); pr = np.zeros((len(g), 7))
    for i, p in enumerate(g["probs"]): pr[i, :len(p)] = p
    res["per_op"][op] = dict(family=FAMILY[op], **M.summarize(pr, g["answer"].to_numpy(), K, g["atype"].iloc[0], g["patient"].to_numpy()))
for fam in ["cell", "region", "comp"]:
    ops = [o for o in res["per_op"] if FAMILY[o] == fam]
    if ops: res[f"mean_acc_{fam}"] = float(np.mean([res["per_op"][o]["acc"] for o in ops])); res[f"mean_nll_{fam}"] = float(np.mean([res["per_op"][o]["nll"] for o in ops]))
json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1); json.dump(vars(a), open(f"{OUT}/config.json", "w"), indent=1)
print("TEST", json.dumps({k: v for k, v in res.items() if k != "per_op"}))
for op, r in res["per_op"].items(): print(f"  {op:26s} n={r['n']:6d} acc={r['acc']:.3f} (maj {r['majority_acc']:.3f}) nll={r['nll']:.3f} ece={r['ece']:.3f}")
for f in ("config.json", "history.json", "metrics_test.json"):
    if os.path.exists(f"{OUT}/{f}"): shutil.copy(f"{OUT}/{f}", f"{RES}/{f}")
print("RUN_DONE", OUT)
