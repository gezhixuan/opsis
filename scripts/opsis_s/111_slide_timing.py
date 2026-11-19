#!/usr/bin/env python
"""Measured cost of a slide question for Opsis-S (final_slide_pan_s0, K = 4,096 cached tiles), with the protocol of the field
timing in scripts/opsis_s/20_train.py: bf16, 3 warm up passes, then timed passes of the adapter and decoder alone, 1, 12 and 32 questions
about one slide in one pass and one by one (the slide question set has 12 operators, 32 cycles through them). Per slide it also
times the read of the 4,096 sampled tile vectors from the cached h5 and the whole ask_slide call of the organiser's tool (read,
encode the questions, decode, softmax). The field timing of final_fov_s0 is repeated in the same job, so the two numbers come from
one card. Writes results/runs/slide_timing.json.
    python scripts/opsis_s/111_slide_timing.py [--slides 20]"""
import argparse, json, os, platform, time
import numpy as np, pandas as pd, torch
os.environ.setdefault("OPSIS_TOOL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
from opsis import paths, tools as T, dataset as D

ap = argparse.ArgumentParser(); ap.add_argument("--slides", type=int, default=20); ap.add_argument("--reps", type=int, default=10)
ap.add_argument("--skip-field", action="store_true"); ap.add_argument("--out", default=None); a = ap.parse_args()
dev = T.DEV; cuda = dev.type == "cuda"; KEEP = ("tok_fov", "tok_ctx", "pos")
def sync():
    if cuda: torch.cuda.synchronize()
def to_dev(b): return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
def cut(b, lo, hi): return {k: (v[:, lo:hi] if torch.is_tensor(v) and v.dim() >= 2 and k not in KEEP else v) for k, v in b.items()}
def timed(fn, reps, warm=3):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
        for _ in range(warm): fn()
        sync(); t = time.perf_counter()
        for _ in range(reps): fn()
        sync(); return (time.perf_counter() - t) / reps * 1e3
def shared_and_serial(model, b, nq, flag, reps):
    sub = cut(b, 0, nq); one = [cut(sub, j, j + 1) for j in range(nq)]
    return dict(shared_ms=timed(lambda: model(sub, flag), reps), serial_ms=timed(lambda: [model(o, flag) for o in one], reps))

out = dict(device=(torch.cuda.get_device_name(0) if cuda else "cpu"), host=platform.node(), torch=torch.__version__, reps=a.reps, slide_run=T.RUNS["slide"], slides=[])
print(out, flush=True)

# ---------- slide questions ----------
m, vocab, cfg = T._slide_model(); OPS = list(T.SLIDE_OPS); _, S = T._registry()
slides = [s for s in T.list_slides("test", n=a.slides * 2, seed=0) if os.path.exists(S.loc[s, "h5_path"])][: a.slides]
def slide_batch(slide, X, pos, nq):
    q = T._slide_question_rows(slide, OPS); q = pd.concat([q] * (nq // len(q) + 1), ignore_index=True).iloc[:nq]; e = D.encode_questions(q, vocab, "op"); n = len(q)
    box = np.tile(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.float32), (n, 1))      # the item of tools.ask_slide
    item = dict(tok_fov=torch.from_numpy(X), pos=torch.from_numpy(pos), op=torch.from_numpy(e["op"]), key=torch.from_numpy(e["key"]), args=torch.from_numpy(e["args"]), cands=torch.from_numpy(e["cands"]), cand_mask=torch.from_numpy(e["cand_mask"]), answer=torch.from_numpy(e["answer"]), box=torch.from_numpy(box), box2=torch.from_numpy(np.zeros_like(box)), has_box2=torch.zeros(n), text_id=torch.from_numpy(e["text_id"]), cand_text=torch.from_numpy(e["cand_text"]), atype=torch.from_numpy(e["atype"]), qidx=torch.arange(n), fov_index=0)
    return to_dev(D.collate([item]))
for i, sl in enumerate(slides):
    t = time.perf_counter(); X, pos = T._slide_tokens(sl, cfg["k_tiles"]); read_ms = (time.perf_counter() - t) * 1e3      # first read of this slide in the job (the file system may have it cached)
    t = time.perf_counter(); T._slide_tokens(sl, cfg["k_tiles"]); read2_ms = (time.perf_counter() - t) * 1e3
    b = slide_batch(sl, X, pos, 32); r = dict(slide=sl, n_tiles=int(T.slide_info(sl)["n_tiles_20x"]), read_ms=read_ms, read_again_ms=read2_ms)
    for nq in (1, 12, 32): r[f"q{nq}"] = shared_and_serial(m, b, nq, False, a.reps)
    sync(); t = time.perf_counter(); ans = T.ask_slide(sl, OPS); sync(); r["ask_slide_12_ms"] = (time.perf_counter() - t) * 1e3; r["answered"] = len(ans.get("answers", []))
    out["slides"].append(r); print(i, json.dumps(r), flush=True)
med = lambda f: float(np.median([f(r) for r in out["slides"]])); rng = lambda f: [float(np.min([f(r) for r in out["slides"]])), float(np.max([f(r) for r in out["slides"]]))]
out["slide_summary"] = dict(n_slides=len(out["slides"]), k_tiles=cfg["k_tiles"], median_tiles_in_bag=med(lambda r: r["n_tiles"]),
    **{f"q{nq}_shared_ms_median": med(lambda r, nq=nq: r[f"q{nq}"]["shared_ms"]) for nq in (1, 12, 32)}, **{f"q{nq}_shared_ms_range": rng(lambda r, nq=nq: r[f"q{nq}"]["shared_ms"]) for nq in (1, 12, 32)},
    **{f"q{nq}_serial_ms_median": med(lambda r, nq=nq: r[f"q{nq}"]["serial_ms"]) for nq in (1, 12, 32)},
    read_ms_median=med(lambda r: r["read_ms"]), read_again_ms_median=med(lambda r: r["read_again_ms"]), ask_slide_12_ms_median=med(lambda r: r["ask_slide_12_ms"]))
print("SLIDE", json.dumps(out["slide_summary"]), flush=True)

# ---------- control: the field timing of scripts/opsis_s/20_train.py on this card ----------
if not a.skip_field:
    from opsis.model import OpsisNet, load_compat
    run = f"{paths.RUNS}/final_fov_s0"; c = json.load(open(f"{run}/config.json")); fv = D.Vocab.from_dict(json.load(open(f"{run}/vocab.json")))
    Q, F, G, tokens_path = D.load_tables(c["split_seed"], c["enc"], c["size"], c["qset"]); Q = Q[Q["fov"].isin(set(G["fov"]))]; Q = Q[Q["split"] == "test"]
    DENS = RECTS = None
    if c["aux_density"] > 0:
        DENS = pd.read_csv(f"{paths.DERIVED}/facts/nuclei.csv" if c["qset"] == "questions" else f"{paths.DERIVED}/{c['qset']}/nuclei.csv")
        RECTS = {r["fov"]: (r.get("rect_x0", 0), r.get("rect_y0", 0), r.get("rect_x1", r["W"]), r.get("rect_y1", r["H"])) for r in F.to_dict("records")}
    ds = D.StateQuestionDataset(Q, F, G, tokens_path, fv, c["q_per_state"], train=False, use_ctx=not c["no_ctx"], seed=0, query=c["query"], density=DENS, obj=None, rects=RECTS)
    fm = OpsisNet(len(fv.ops), len(fv.args), len(fv.cands), c["d"], c["enc_layers"], c["dec_layers"], dropout=c["dropout"], pool=c["pool"], text=None, text_trainable=False, density=c["aux_density"] > 0).to(dev)
    load_compat(fm, torch.load(f"{run}/best.pt", map_location=dev)); fm.eval(); b = to_dev(D.collate([ds[0]])); out["field_control"] = {}
    for nq in (1, 8, 32):
        if b["op"].shape[1] < nq: continue
        out["field_control"][f"q{nq}"] = shared_and_serial(fm, b, nq, not c["no_ctx"], a.reps)
    out["field_control"]["tokens"] = {k: list(b[k].shape) for k in KEEP if k in b and torch.is_tensor(b[k])}
    try: out["field_control"]["recorded_h200"] = json.load(open(f"{paths.RESULTS}/runs/final_fov_s0/timing.json"))
    except Exception: pass
    print("FIELD", json.dumps({k: v for k, v in out["field_control"].items() if k != "recorded_h200"}), flush=True)
dst = a.out or f"{paths.RESULTS}/runs/slide_timing.json"; paths.ensure(os.path.dirname(dst)); json.dump(out, open(dst, "w"), indent=1); print("RUN_DONE", dst)
