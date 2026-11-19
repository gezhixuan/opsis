#!/usr/bin/env python
"""Opsis-V, second recipe: any image-text base (Qwen3-VL, Qwen3.5 / 3.6 dense or MoE), the shared visual state of
Opsis-S (8x8 pooled Virchow2 tokens per canvas, injected through a trained projector) with or without the base's own
image tokens, joint training on the NuCLS question set and an extra FOV set (PanNuke, canvas only), lettered candidates,
answer probability = softmax over the candidate letters at the answer position, LoRA on the language tower, early stopping.
    python scripts/opsis_v/41_train_vlm_state.py --name vlmstate_8b_s0 --qset questions_final \
        --extra-qset pannuke --v2tok pool8 --vision none [--lora-r 64] [--epochs 10 --patience 3]
Outputs under $OPSIS_DATA/runs/<name>/: adapter/, projector.pt, history.json, predictions_{val,test[,test_extra]}.parquet, metrics_test.json
(per_op, per_op_extra; same format as scripts/opsis_s/20_train.py and 40), config.json; json files copied to results/runs/<name>/."""
import argparse, json, os, shutil, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, PeftModel
from opsis import paths, facts, metrics as M
from opsis.data import nucls
from opsis.questions import FAMILY

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--model", default=f"{paths.MODELS}/Qwen3-VL-8B-Instruct"); ap.add_argument("--model-tag", default=None)
ap.add_argument("--qset", default="questions_final"); ap.add_argument("--extra-qset", default=None, help="second FOV set trained jointly (pannuke: canvas only, no context)")
ap.add_argument("--extra-split-col", default="split_fold"); ap.add_argument("--extra-states-per-epoch", type=int, default=1800); ap.add_argument("--extra-val-states", type=int, default=64); ap.add_argument("--extra-test-states", type=int, default=600)
ap.add_argument("--split-seed", type=int, default=0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--protocol", default="all", choices=["all", "comp_holdout"])
ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--patience", type=int, default=3); ap.add_argument("--q-per-state", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--lora-r", type=int, default=64)
ap.add_argument("--lora-targets", default="auto", help="auto (by architecture) or a regex over module names")
ap.add_argument("--val-states", type=int, default=64); ap.add_argument("--limit-states", type=int, default=None); ap.add_argument("--test-states", type=int, default=None); ap.add_argument("--eval-batch", type=int, default=16)
ap.add_argument("--no-ctx", action="store_true"); ap.add_argument("--target", default="coords", choices=["coords", "none"]); ap.add_argument("--attn", default="flash_attention_2"); ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--v2tok", default="pool8", choices=["none", "pool8"], help="inject the 8x8 pooled Virchow2 tokens of each canvas through a trained projector")
ap.add_argument("--vision", default="none", choices=["qwen", "none"], help="qwen: the base's own image tokens are also in the prefix; none: text and Virchow2 tokens only")
ap.add_argument("--blank-image", action="store_true"); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
assert not (a.vision == "none" and a.v2tok == "none"), "vision none needs --v2tok"
dev = torch.device("cuda"); OUT = f"{paths.RUNS}/{a.name}"; paths.ensure(OUT); RES = f"{paths.RESULTS}/runs/{a.name}"; paths.ensure(RES)
CTX = f"{paths.DERIVED}/context/mpp{facts.CTX_MPP}_{facts.CTX_PX}"
if a.smoke: a.limit_states, a.val_states, a.test_states, a.extra_states_per_epoch, a.extra_val_states, a.extra_test_states, a.epochs = 4, 4, 4, 4, 4, 4, 1

# ---- state sets: NuCLS FOVs (canvas + context) and the extra set (canvas only) ----------------------------------------------
class StateSet:
    def __init__(self, name, Q, geom, rgb, ctx, tokens, desc, with_ctx):
        self.name, self.Q, self.geom, self.rgb, self.ctx, self.tok, self.desc, self.with_ctx = name, Q, geom, rgb, ctx, tokens, desc, with_ctx
def load_main():
    Q = pd.read_parquet(f"{paths.DERIVED}/{a.qset}/questions_seed{a.split_seed}.parquet"); F = pd.read_csv(f"{paths.DERIVED}/facts/fovs.csv").set_index("fov")
    Q = Q[Q["fov"].map(lambda f: os.path.exists(f"{CTX}/{f}.png"))].reset_index(drop=True)
    tok = None
    if a.v2tok != "none":
        z = np.load(f"{paths.FEATURES}/virchow2/tokens_448_pool8.npz"); tok = {"idx": {n: i for i, n in enumerate(z["fov"])}, "fov": z["tok_fov"], "ctx": z["tok_ctx"]}
    desc = lambda fov: f"Image 1: a breast cancer H&E field of view at 0.2 micrometres per pixel ({int(F.loc[fov, 'W'])} x {int(F.loc[fov, 'H'])} px)."
    return StateSet("main", Q, F, nucls.rgb_path, lambda f: f"{CTX}/{f}.png", tok, desc, not a.no_ctx)
def load_extra():
    D = f"{paths.DERIVED}/{a.extra_qset}"; Q = pd.read_parquet(f"{D}/questions_seed{a.split_seed}.parquet"); Q = Q.assign(split=Q[a.extra_split_col]) if a.extra_split_col != "split" else Q
    F = pd.read_csv(f"{D}/fovs.csv").set_index("fov"); tok = None
    if a.v2tok != "none":
        z = np.load(f"{paths.FEATURES}/{a.extra_qset}_virchow2/tokens_448_pool8.npz"); tok = {"idx": {n: i for i, n in enumerate(z["fov"])}, "fov": z["tok_fov"], "ctx": None}
        Q = Q[Q["fov"].isin(tok["idx"])].reset_index(drop=True)
    desc = lambda fov: f"Image 1: an H&E field of view at 0.25 micrometres per pixel ({int(F.loc[fov, 'W'])} x {int(F.loc[fov, 'H'])} px)."
    return StateSet("extra", Q, F, lambda f: f"{D}/rgb/{f}.png", None, tok, desc, False)
SETS = {"main": load_main()}
if a.extra_qset: SETS["extra"] = load_extra()
def pick_states(S, limit, val_n, test_n):
    st = {s: sorted(S.Q.loc[S.Q["split"] == s, "fov"].unique()) for s in ("train", "val", "test")}
    if limit: st["train"] = list(np.random.default_rng(0).permutation(st["train"])[:limit])
    st["val"] = list(np.random.default_rng(0).permutation(st["val"])[:val_n])
    if test_n: st["test"] = list(np.random.default_rng(1).permutation(st["test"])[:test_n])
    return st
STATES = {"main": pick_states(SETS["main"], a.limit_states, a.val_states, a.test_states)}
if "extra" in SETS: STATES["extra"] = pick_states(SETS["extra"], None, a.extra_val_states, a.extra_test_states)
GROUPS = {}
for k, S in SETS.items():
    tq = S.Q[S.Q["split"] == "train"]
    if a.protocol == "comp_holdout" and k == "main": tq = tq[tq["family"] != "comp"]
    GROUPS[k] = {"train": {f: g for f, g in tq[tq["fov"].isin(STATES[k]["train"])].groupby("fov")}, **{s: {f: g for f, g in S.Q[S.Q["fov"].isin(STATES[k][s])].groupby("fov")} for s in ("val", "test")}}
    print(k, {s: (len(GROUPS[k][s]), sum(len(g) for g in GROUPS[k][s].values())) for s in GROUPS[k]}, flush=True)

proc = AutoProcessor.from_pretrained(a.model); tok = proc.tokenizer; tok.padding_side = "left"
PH = "<|fim_pad|>"; PH_ID = tok.convert_tokens_to_ids(PH); NV2 = 64; assert tok.encode(PH, add_special_tokens=False) == [PH_ID]
LETTERS = "ABCDEFG"; LETTER_IDS = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]; assert all(len(tok.encode(c, add_special_tokens=False)) == 1 for c in LETTERS)
SYSTEM = "You are a pathology image analysis assistant. Answer each multiple choice question about the images with the option letter only."
def prompt(S, fov, q):
    W, H = int(S.geom.loc[fov, "W"]), int(S.geom.loc[fov, "H"]); cands = json.loads(q["candidates"]); text = q["text"]
    if q["family"] == "cell" and a.target == "coords" and q.get("box"):
        x0, y0, x1, y1 = json.loads(q["box"]) if isinstance(q["box"], str) else q["box"]; nb = [int(round(1000 * x0 / W)), int(round(1000 * y0 / H)), int(round(1000 * x1 / W)), int(round(1000 * y1 / H))]
        text += f" The marked nucleus is the one inside bbox_2d {nb} of image 1 (coordinates normalised to 0-1000)."
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(cands))
    img = [{"type": "image"}] if a.vision == "qwen" else []
    v2 = (lambda k: [{"type": "text", "text": f" Pathology encoder features of image {k}: " + PH * NV2 + "."}]) if a.v2tok != "none" else (lambda k: [])
    user = img + [{"type": "text", "text": S.desc(fov)}] + v2(1)
    if S.with_ctx: user += img + [{"type": "text", "text": f"Image 2: the {facts.CTX_PX * facts.CTX_MPP:.0f} micrometre context window around the centre of image 1 at {facts.CTX_MPP} micrometres per pixel."}] + v2(2)
    user += [{"type": "text", "text": f"{text}\nOptions:\n{opts}\nAnswer with the option letter."}]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False), len(cands)
_imcache = {}
def images(S, fov):
    key = (S.name, fov)
    if key not in _imcache:
        ims = [Image.open(S.rgb(fov)).convert("RGB")] + ([Image.open(S.ctx(fov)).convert("RGB")] if S.with_ctx else [])
        if a.blank_image: ims = [Image.new("RGB", im.size, (128, 128, 128)) for im in ims]
        if len(_imcache) > 64: _imcache.clear()
        _imcache[key] = ims
    return _imcache[key]
CUR = {}
def build(S, fov, qs):
    texts, ks = zip(*[prompt(S, fov, q) for q in qs.to_dict("records")]); ims = []
    if a.vision == "qwen":
        for _ in texts: ims += images(S, fov)
    enc = proc(text=list(texts), images=ims if ims else None, return_tensors="pt", padding=True); enc = {k: v.to(dev) for k, v in enc.items()}
    CUR.clear()
    if S.tok is not None:
        i = S.tok["idx"][fov]; t = [S.tok["fov"][i]] + ([S.tok["ctx"][i]] if S.with_ctx else [])
        CUR["tok"] = torch.from_numpy(np.concatenate(t, 0)).to(dev).unsqueeze(0).expand(len(texts), -1, -1); CUR["mask"] = enc["input_ids"] == PH_ID
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

model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation=a.attn).to(dev)
for p in model.parameters(): p.requires_grad_(False)
TARGETS = {"auto": None, "dense": r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
           "hybrid": r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)"}   # hybrid: Qwen3.5 linear attention layers too; experts are 3D tensors (not targeted), shared expert is
tgt = a.lora_targets if a.lora_targets not in TARGETS else TARGETS[a.lora_targets]
if tgt is None: tgt = TARGETS["hybrid"] if "linear_attention" in str(getattr(model.config.text_config, "layer_types", "")) else TARGETS["dense"]
if a.eval_only and os.path.isdir(f"{OUT}/adapter"): model = PeftModel.from_pretrained(model, f"{OUT}/adapter")
else: model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, bias="none", target_modules=tgt))
model.print_trainable_parameters(); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
projector = None
if a.v2tok != "none":
    D = model.config.text_config.hidden_size
    class Projector(nn.Module):
        def __init__(self):
            super().__init__(); self.net = nn.Sequential(nn.LayerNorm(1280), nn.Linear(1280, D), nn.GELU(), nn.Linear(D, D))
            self.src = nn.Parameter(torch.zeros(2, D)); self.pos = nn.Parameter(torch.zeros(NV2, D)); nn.init.normal_(self.src, std=0.02); nn.init.normal_(self.pos, std=0.02)
        def forward(self, x):
            B, T, _ = x.shape; n = T // NV2; y = self.net(x.float()); return y + self.src[:n].repeat_interleave(NV2, 0)[None] + self.pos.repeat(n, 1)[None]
    projector = Projector().to(dev)
    if a.eval_only and os.path.exists(f"{OUT}/projector.pt"): projector.load_state_dict(torch.load(f"{OUT}/projector.pt", map_location=dev))
    lm = next(m for n, m in model.named_modules() if n.endswith("language_model"))
    def _inject(module, args, kwargs):
        emb = kwargs.get("inputs_embeds")
        if emb is None or "tok" not in CUR: return None
        emb = emb.clone(); emb[CUR["mask"]] = projector(CUR["tok"]).reshape(-1, emb.shape[-1]).to(emb.dtype); kwargs["inputs_embeds"] = emb; return args, kwargs
    lm.register_forward_pre_hook(_inject, with_kwargs=True); print("Virchow2 injection hook on", type(lm).__name__, flush=True)
def trainable_named():
    for n, p in model.named_parameters():
        if p.requires_grad: yield n, p
    if projector is not None:
        for n, p in projector.named_parameters(): yield "projector." + n, p
LID = torch.tensor(LETTER_IDS, device=dev)
def answer_logits(enc): return model(**enc, logits_to_keep=1).logits[:, -1, :]
def probs_from(lg, ks):
    cl = lg[:, LID]; mask = torch.arange(7, device=dev)[None] < torch.tensor(ks, device=dev)[:, None]; return torch.softmax(cl.masked_fill(~mask, -1e4), -1)
@torch.no_grad()
def evaluate(k, split, collect=False):
    model.eval(); S = SETS[k]; tot = n = 0; rows = []
    for fov, g in GROUPS[k][split].items():
        for i in range(0, len(g), a.eval_batch):
            qs = g.iloc[i:i + a.eval_batch]; enc, ks = build(S, fov, qs)
            with torch.autocast("cuda", dtype=torch.bfloat16): lg = answer_logits(enc).float()
            probs = probs_from(lg, ks); nll = -torch.log(probs[torch.arange(len(qs)), qs["answer"].to_numpy()].clamp(min=1e-12)); tot += nll.sum().item(); n += len(qs)
            if collect:
                for q, p in zip(qs.itertuples(), probs.cpu().numpy()):
                    rows.append(dict(qid=q.qid, fov=q.fov, patient=q.patient, family=q.family, op=q.op, atype=q.atype, candidates=q.candidates, answer=int(q.answer), probs=p[: len(json.loads(q.candidates))].tolist()))
    model.train(); return tot / max(n, 1), pd.DataFrame(rows)
hist = []; best = float("inf"); bad = 0; t0 = time.time()
if not a.eval_only:
    params = [p for _, p in trainable_named()]; opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    n_extra = min(a.extra_states_per_epoch, len(GROUPS["extra"]["train"])) if "extra" in GROUPS else 0
    steps = a.epochs * (len(GROUPS["main"]["train"]) + n_extra); warm = max(1, steps // (10 * a.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / max(steps, 1)))))
    model.train(); step = 0
    for ep in range(a.epochs):
        order = [("main", f) for f in GROUPS["main"]["train"]] + ([("extra", f) for f in rng.choice(list(GROUPS["extra"]["train"]), n_extra, replace=False)] if n_extra else [])
        order = [order[i] for i in rng.permutation(len(order))]; tl = []
        for k, fov in order:
            qs = sample_questions(GROUPS[k]["train"][fov]); enc, ks = build(SETS[k], fov, qs)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = answer_logits(enc).float(); loss = Fn.cross_entropy(lg, torch.tensor([LETTER_IDS[x] for x in qs["answer"].to_numpy()], device=dev))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); tl.append(loss.item()); step += 1
            if step % 100 == 0: print(f"  ep {ep} step {step} train nll {np.mean(tl[-100:]):.3f} {time.time() - t0:.0f}s", flush=True)
        va, _ = evaluate("main", "val"); va2 = evaluate("extra", "val")[0] if "extra" in GROUPS else None
        hist.append(dict(epoch=ep, train_nll=float(np.mean(tl)), val_nll=va, val_nll_extra=va2, time=time.time() - t0)); print(json.dumps(hist[-1]), flush=True)
        crit = va if va2 is None else 0.5 * (va + va2)
        if crit < best - 1e-4:
            best = crit; bad = 0; model.save_pretrained(f"{OUT}/adapter"); best_sd = {n: p.detach().clone() for n, p in trainable_named()}
            if projector is not None: torch.save(projector.state_dict(), f"{OUT}/projector.pt")
        else:
            bad += 1
            if bad >= a.patience: print("early stop", flush=True); break
    json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    with torch.no_grad():
        for n, p in trainable_named(): p.copy_(best_sd[n])
def per_op(P):
    out = {}
    for op, g in P.groupby("op"):
        K = len(json.loads(g["candidates"].iloc[0])); pr = np.zeros((len(g), 7))
        for i, p in enumerate(g["probs"]): pr[i, :len(p)] = p
        out[op] = dict(family=FAMILY[op], **M.summarize(pr, g["answer"].to_numpy(), K, g["atype"].iloc[0], g["patient"].to_numpy()))
    return out
va, Pv = evaluate("main", "val", collect=True); Pv.to_parquet(f"{OUT}/predictions_val.parquet", index=False)
te, P = evaluate("main", "test", collect=True); P.to_parquet(f"{OUT}/predictions_test.parquet", index=False)
res = {"test_nll": te, "best_val_crit": best if hist else va, "protocol": a.protocol, "use_ctx": not a.no_ctx, "model": a.model_tag or f"{os.path.basename(a.model.rstrip('/'))}_lora{a.lora_r}_vision-{a.vision}_v2tok-{a.v2tok}" + (f"_joint-{a.extra_qset}" if a.extra_qset else ""), "per_op": per_op(P)}
for fam in ["cell", "region", "comp"]:
    ops = [o for o in res["per_op"] if FAMILY[o] == fam]
    if ops: res[f"mean_acc_{fam}"] = float(np.mean([res["per_op"][o]["acc"] for o in ops])); res[f"mean_nll_{fam}"] = float(np.mean([res["per_op"][o]["nll"] for o in ops]))
if "extra" in GROUPS:
    te2, P2 = evaluate("extra", "test", collect=True); P2.to_parquet(f"{OUT}/predictions_test_extra.parquet", index=False); res["test_nll_extra"] = te2; res["per_op_extra"] = per_op(P2)
    print("TEST_EXTRA", a.extra_qset, json.dumps({k: round(v["acc"], 3) for k, v in res["per_op_extra"].items()}), flush=True)
json.dump(res, open(f"{OUT}/metrics_test.json", "w"), indent=1); json.dump(vars(a), open(f"{OUT}/config.json", "w"), indent=1)
print("TEST", json.dumps({k: v for k, v in res.items() if not k.startswith("per_op")}))
for op, r in res["per_op"].items(): print(f"  {op:26s} n={r['n']:6d} acc={r['acc']:.3f} (maj {r['majority_acc']:.3f}) nll={r['nll']:.3f} ece={r['ece']:.3f}")
for f in ("config.json", "history.json", "metrics_test.json"):
    if os.path.exists(f"{OUT}/{f}"): shutil.copy(f"{OUT}/{f}", f"{RES}/{f}")
print("RUN_DONE", OUT)
