#!/usr/bin/env python
"""The reading condition of the organiser study (a written description): a slide is described once without any question in view, and
Claude then answers each question of the slide from that text alone (no image, no tool). Three sources of the description:
  prism2   the report PRISM2 writes for the slide (derived/report_models/prism2_reports.jsonl)
  opus     Claude describes the 1,024 px thumbnail and the six tiles of the looking condition of scripts/81 (--n-tiles 6), or these
           plus more random tissue tiles (--n-tiles 48), eight images per call, one paragraph per image
Both stages run through the Claude Code CLI as in scripts/81 (--restricted, no local settings). A description is
cached per slide under results/organiser/descriptions/<desc-name>/, so a rerun or a second answering run reuses it.
Writes results/organiser/<name>.jsonl in the format of scripts/81 plus desc_words and the tokens spent on describing.
    python scripts/106_describe_then_answer.py --name opus55_describe6_brca30 --source opus --n-tiles 6 --project BRCA --resume
    python scripts/106_describe_then_answer.py --name prism2report_brca30 --source prism2 --project BRCA --resume
--answer-backend api: the answering stage goes to the OpenAI Responses API instead (the request of scripts/81, reasoning low), so another
frontier model reads the same cached text. No description is written in this mode: the cache of an earlier run is read (--desc-name).
    python scripts/106_describe_then_answer.py --name sol_prism2report_brca30 --source prism2 --project BRCA --answer-backend api --model gpt-6-sol --key-file <file> --resume
    python scripts/106_describe_then_answer.py --name sol_describe6_brca30 --source opus --n-tiles 6 --project BRCA --answer-backend api --model gpt-6-sol --key-file <file> --resume
Stop it with its pid."""
import argparse, base64, json, os, re, shutil, ssl, subprocess, tempfile, time, urllib.request
import numpy as np, pandas as pd
from opsis import paths, tools as T
from opsis.keys import api_key
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--source", required=True, choices=["prism2", "opus"]); ap.add_argument("--n-tiles", type=int, default=6); ap.add_argument("--desc-name", default=None)
ap.add_argument("--slides", type=int, default=30); ap.add_argument("--project", default=None); ap.add_argument("--split", default="test"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--model", default="claude-opus-5-5"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--shard", default=None, help="i/n over slides"); ap.add_argument("--limit-slides", type=int, default=None); ap.add_argument("--describe-only", action="store_true")
ap.add_argument("--claude-bin", default="claude")
ap.add_argument("--answer-backend", default="claude", choices=["claude", "api"]); ap.add_argument("--key-file", default=None, help="file holding the API key (default: $OPENAI_API_KEY)"); ap.add_argument("--base", default="https://api.openai.com/v1"); ap.add_argument("--max-tokens", type=int, default=1500); ap.add_argument("--reasoning", default="low")
a = ap.parse_args(); OUT = f"{paths.ROOT}/results/organiser"; LOG = f"{OUT}/{a.name}.jsonl"
DESC = f"{OUT}/descriptions/{a.desc_name or ('prism2' if a.source == 'prism2' else f'opus55_{a.n_tiles}tiles')}"; os.makedirs(DESC, exist_ok=True)
OPS = ["high_grade_and_lvi", "necrosis_and_high_mitotic", "grade_from_components", "dcis_with_comedo", "precursor_type", "pan_grade", "pan_lvi", "pan_pni", "pan_necrosis", "pan_precursor", "pan_mitotic", "pan_multifocal"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ENV = {"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin", "TERM": "dumb"}
DESCRIBE_SYSTEM = ("You are a pathologist describing a whole slide image for a colleague who cannot see it and who will later answer questions about the case from your text alone. "
                   "You do not know the questions. Write plain text, one paragraph per image, each headed by the name of the image.")
DESCRIBE_USER = ("Look at every image below with the Read tool, then write the description. For a thumbnail give the overall layout of the tissue. For each tile describe the tissue types present, "
                 "the tumour architecture, gland or tubule formation, nuclear size and pleomorphism, mitotic figures, necrosis, any in situ or precursor component, vessels and whether tumour lies inside them, "
                 "nerves and whether tumour surrounds them, inflammatory cells, and anything else of diagnostic value. Say what is absent as well as what is present.\n")
ANSWER_SYSTEM = ("You are a pathologist's assistant answering one multiple choice question about a whole slide image. You cannot see the slide. A written description of it is given, and it was written before the question was known. Answer from the description. "
                 "When you are done, answer with a JSON object only: {\"answer\": \"<option letter>\", \"confidence\": <0 to 1>, \"reason\": \"<one sentence>\"}.")
def claude(user, system, tools, cwd, max_turns):
    """One CLI call. Returns (text, usage, failed)."""
    cmd = [a.claude_bin, "-p", user, "--restricted", "--model", a.model, "--output-format", "stream-json", "--verbose", "--no-session-persistence", "--max-turns", str(max_turns), "--system-prompt", system, "--tools", tools, "--permission-mode", "dontAsk"]
    if tools: cmd += ["--allowedTools", tools]
    try: r = subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900, env=ENV)
    except subprocess.TimeoutExpired: print("claude timeout after 900 s", flush=True); return "", dict(prompt=0, completion=0), True
    text = ""; usage = dict(prompt=0, completion=0); err = False
    for line in r.stdout.splitlines():
        try: e = json.loads(line)
        except Exception: continue
        if e.get("type") == "result":
            text = e.get("result") or ""; u = e.get("usage", {}); usage = dict(prompt=u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0), completion=u.get("output_tokens", 0)); err = bool(e.get("is_error"))
    if r.returncode or not text: print("claude exit", r.returncode, (r.stderr or r.stdout)[-300:].replace("\n", " | "), flush=True)
    return text, usage, bool(r.returncode or err or not text)
def openai(user, system):
    """One Responses API call with text only, as scripts/81 sends it. Returns (text, usage, failed)."""
    import certifi
    body = {"model": a.model, "instructions": system, "input": [{"role": "user", "content": [{"type": "input_text", "text": user}]}], "max_output_tokens": a.max_tokens, "store": False}
    if a.reasoning and a.model.startswith(("gpt-5", "gpt-6", "o")): body["reasoning"] = {"effort": a.reasoning}
    key = api_key(a.key_file)
    req = urllib.request.Request(a.base + "/responses", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600, context=ssl.create_default_context(cafile=certifi.where())) as r: d = json.load(r)
            u = d.get("usage", {}); text = d.get("output_text") or "".join(c.get("text", "") for o in d.get("output", []) if o.get("type") == "message" for c in o.get("content", []) if c.get("type") == "output_text")
            return text, dict(prompt=u.get("input_tokens", 0), completion=u.get("output_tokens", 0)), not text
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            if hasattr(e, "read"):
                try: err += " | " + e.read().decode()[:300]
                except Exception: pass
            print("api error", err, flush=True); time.sleep(5 * (attempt + 1))
    return "", dict(prompt=0, completion=0), True
def tiles_of(slide, n):
    """The six tiles of the looking condition of scripts/81 first (same generator, same seed), then further tissue tiles from a second generator."""
    import h5py
    _, S = T._registry()
    with h5py.File(S.loc[slide, "h5_path"], "r") as h: C0 = h["coords_level0"][:]; ps = float(h.attrs.get("patch_size_level0", 512))
    first = list(np.random.default_rng(a.seed).choice(len(C0), min(6, len(C0)), replace=False)); F = set(int(i) for i in first); rest = [int(i) for i in np.random.default_rng(a.seed + 1).permutation(len(C0)) if int(i) not in F]
    return [(int(C0[i, 0] + ps / 2), int(C0[i, 1] + ps / 2)) for i in (first + rest)[:n]]
def describe(slide):
    """The description of one slide, cached. None when a call failed."""
    f = f"{DESC}/{slide}.json"
    if os.path.exists(f): return json.load(open(f))
    if a.answer_backend == "api" and a.source != "prism2": raise SystemExit(f"no cached description for {slide} in {DESC}: the api backend only reads descriptions an earlier run wrote")
    if a.source == "prism2":
        R = getattr(describe, "R", None) or {j["slide"]: j["text"] for j in map(json.loads, open(f"{paths.DERIVED}/report_models/prism2_reports.jsonl"))}; describe.R = R
        d = dict(slide=slide, source="prism2", text=R.get(slide, ""), usage=dict(prompt=0, completion=0), calls=0, seconds=0.0)
    else:
        tmp = tempfile.mkdtemp(prefix="describe_", dir=OUT); t0 = time.time(); parts = []; usage = dict(prompt=0, completion=0); XY = tiles_of(slide, a.n_tiles); calls = 0
        try:
            for b in range(0, len(XY), 8 if a.n_tiles > 6 else 6):
                chunk = XY[b: b + (8 if a.n_tiles > 6 else 6)]; lines = []
                if b == 0:
                    th = T.thumbnail(slide, 1024); open(f"{tmp}/thumbnail.png", "wb").write(base64.b64decode(th["image_png_base64"])); lines.append(f"Thumbnail of the slide ({th['mpp']} um per pixel): {tmp}/thumbnail.png")
                for k, (x, y) in enumerate(chunk):
                    v = T.view_region(slide, x, y, 256, 0.5, 512); open(f"{tmp}/tile{b + k + 1}.png", "wb").write(base64.b64decode(v["image_png_base64"])); lines.append(f"Tile {b + k + 1} at level 0 ({x}, {y}), 256 um at 0.5 um per pixel: {tmp}/tile{b + k + 1}.png")
                text, u, failed = claude(DESCRIBE_USER + "\n".join(lines), DESCRIBE_SYSTEM, "Read", tmp, 24); calls += 1
                if failed: return None
                parts.append(text.replace(tmp + "/", "")); usage = {k: usage[k] + u[k] for k in usage}
        finally: shutil.rmtree(tmp, ignore_errors=True)
        d = dict(slide=slide, source=f"opus {a.n_tiles} tiles", text="\n\n".join(parts), usage=usage, calls=calls, seconds=round(time.time() - t0, 1))
    json.dump(d, open(f, "w"), indent=1); return d
def answer(slide, q, d):
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["candidates"])); tmp = tempfile.mkdtemp(prefix="answer_", dir=OUT); t0 = time.time()
    user = f"Description of the slide:\n{d['text']}\n\nQuestion: {q['question']}\nOptions:\n{opts}"
    try: text, u, failed = openai(user, ANSWER_SYSTEM) if a.answer_backend == "api" else claude(user, ANSWER_SYSTEM, "", tmp, 3)
    finally: shutil.rmtree(tmp, ignore_errors=True)
    labs = list(LETTERS[: len(q["candidates"])]); ans = None; conf = None
    try: j = json.loads(text[text.index("{"): text.rindex("}") + 1]); ans = str(j.get("answer", "")).strip().upper().rstrip("."); conf = float(j.get("confidence", 0.5))
    except Exception: ans = next((c for c in re.findall(r"[A-Z]+", text.upper()) if c in labs), None)
    pred = labs.index(ans) if ans in labs else None
    return dict(slide=slide, qid=q["qid"], operator=q["operator"], answer=q["answer"], pred=pred, correct=(pred == q["answer"]), confidence=conf, steps=0, tools=[], usage=u, seconds=round(time.time() - t0, 1), text=text[:400],
                desc_words=len(d["text"].split()), desc_usage=d["usage"], desc_calls=d["calls"], failed=failed)
done = {json.loads(l)["qid"] for l in open(LOG)} if (a.resume and os.path.exists(LOG)) else set()
slides = T.list_slides(a.split, a.project, a.slides, a.seed)
if a.shard: i_, n_ = map(int, a.shard.split("/")); slides = slides[i_::n_]
if a.limit_slides: slides = slides[: a.limit_slides]
print(f"{a.name}: source {a.source} tiles {a.n_tiles if a.source == 'opus' else '-'} slides {len(slides)} descriptions in {DESC}", flush=True); nfail = 0
with open(LOG, "a") as f:
    for si, s in enumerate(slides):
        d = describe(s)
        if d is None:
            nfail += 1; print(f"[slide {si + 1}/{len(slides)}] {s[:16]} description FAILED ({nfail} in a row)", flush=True)
            if nfail >= 3: raise SystemExit("three failed calls in a row (rate limit or login?), stopping; rerun with --resume")
            continue
        print(f"[slide {si + 1}/{len(slides)}] {s[:16]} description {len(d['text'].split())} words, {d['calls']} calls, {d['usage']['prompt']}+{d['usage']['completion']} tokens, {d['seconds']}s", flush=True)
        if a.describe_only: continue
        for q in T.ground_truth(s, OPS):
            if q["qid"] in done: continue
            r = answer(s, q, d)
            if r.pop("failed"):
                nfail += 1; print(f"   {q['operator']:26s} FAILED ({nfail} in a row), not logged", flush=True)
                if nfail >= 3: raise SystemExit("three failed calls in a row (rate limit or login?), stopping; rerun with --resume")
                continue
            nfail = 0; f.write(json.dumps(r) + "\n"); f.flush()
            print(f"   {q['operator']:26s} truth {q['candidates'][q['answer']]:<12s} pred {(q['candidates'][r['pred']] if r['pred'] is not None else '-'):<12s} {'ok' if r['correct'] else 'no'}  tok {r['usage']['prompt']}+{r['usage']['completion']} {r['seconds']}s", flush=True)
if os.path.exists(LOG) and os.path.getsize(LOG):
    R = pd.DataFrame([json.loads(l) for l in open(LOG)]).drop_duplicates("qid", keep="last")
    summ = dict(name=a.name, source=a.source, n_tiles=a.n_tiles if a.source == "opus" else None, model=a.model, n=len(R), acc=float(R.correct.mean()), per_op={op: dict(n=len(g), acc=float(g.correct.mean())) for op, g in R.groupby("operator")},
                answer_prompt_tokens_per_question=float(R.usage.map(lambda u: u["prompt"]).mean()), desc_words_per_slide=float(R.drop_duplicates("slide").desc_words.mean()), describe_tokens_per_slide=float(R.drop_duplicates("slide").desc_usage.map(lambda u: u["prompt"] + u["completion"]).mean()))
    json.dump(summ, open(f"{OUT}/{a.name}_summary.json", "w"), indent=1); print("SUMMARY", json.dumps({k: v for k, v in summ.items() if k != "per_op"}))
