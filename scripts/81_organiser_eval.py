#!/usr/bin/env python
"""Organiser evaluation: a language model answers slide level questions of the pan TCGA test split, either through the typed
Opsis tools (asking, --arm tools), through its own vision on a thumbnail and tiles (looking, --arm vision), or both (--arm both).
The rules condition (--arm structured) is the slide decoder alone, with rule composition for the composite operators. Backend: OpenAI compatible
chat completions with function calling (key read from a file, never printed). Writes results/organiser/<name>.jsonl and a
summary json with accuracy per operator, tool calls and token usage.
    python scripts/81_organiser_eval.py --name sol_tools --arm tools --model gpt-6-sol --slides 30 [--ops composites|synoptic|all] [--project BRCA]"""
import argparse, base64, json, os, ssl, sys, time, urllib.request
import numpy as np, pandas as pd, certifi
from opsis import paths, tools as T
from opsis.keys import api_key
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--arm", default="tools", choices=["tools", "vision", "both", "free", "look", "structured"]); ap.add_argument("--model", default="gpt-6-sol")
ap.add_argument("--slides", type=int, default=30); ap.add_argument("--project", default=None); ap.add_argument("--split", default="test"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ops", default="all", help="composites | synoptic | all | comma list"); ap.add_argument("--max-steps", type=int, default=8); ap.add_argument("--n-tiles", type=int, default=6, help="looking conditions: tiles shown besides the thumbnail")
ap.add_argument("--key-file", default=None, help="file holding the API key (default: $OPENAI_API_KEY)"); ap.add_argument("--base", default="https://api.openai.com/v1"); ap.add_argument("--max-tokens", type=int, default=1500); ap.add_argument("--reasoning", default="low")
ap.add_argument("--alias", action="store_true", help="show the organiser an opaque slide name (slide_<hash>) instead of the TCGA barcode, resolved inside the MCP server, and hide project and patient in slide_info: the barcode's tissue source site code identifies the cancer type"); ap.add_argument("--blank-image", action="store_true", help="control for the looking condition of the API backend: the thumbnail and every tile replaced by a uniform grey image of the same size (language prior)"); ap.add_argument("--limit-q", type=int, default=None); ap.add_argument("--resume", action="store_true"); ap.add_argument("--shard", default=None, help="i/n: this process takes every n-th question starting at i (parallel shards write separate logs)")
ap.add_argument("--backend", default="api", choices=["api", "codex", "claude"], help="codex: the Codex CLI runs the episode with the MCP server attached, images reach the model through MCP tool results. claude: the Claude Code CLI in print mode, --bare with our own system prompt, the MCP server attached, no built in tools (--arm vision gets Read for the prescribed images)")
ap.add_argument("--codex-bin", default="codex")
ap.add_argument("--claude-bin", default="claude")
a = ap.parse_args(); OUT = f"{paths.ROOT}/results/organiser"; os.makedirs(OUT, exist_ok=True); LOG = f"{OUT}/{a.name}.jsonl"
COMPOSITES = ["high_grade_and_lvi", "necrosis_and_high_mitotic", "grade_from_components", "dcis_with_comedo", "precursor_type"]
SYNOPTIC = ["pan_grade", "pan_lvi", "pan_pni", "pan_necrosis", "pan_precursor", "pan_mitotic", "pan_multifocal"]
OPS = COMPOSITES if a.ops == "composites" else SYNOPTIC if a.ops == "synoptic" else COMPOSITES + SYNOPTIC if a.ops == "all" else a.ops.split(",")
KEY = api_key(a.key_file); CTX = ssl.create_default_context(cafile=certifi.where())
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def LAB(i): return LETTERS[i] if i < 26 else LETTERS[i // 26 - 1] + LETTERS[i % 26]  # option labels A to Z, then AA, AB, ... (cancer type has 32 candidates)

def api(body):
    req = urllib.request.Request(a.base + "/responses", data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600, context=CTX) as r: return json.load(r)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
            if hasattr(e, "read"):
                try: err += " | " + e.read().decode()[:400]
                except Exception: pass
            print("api error", err, flush=True); time.sleep(5 * (attempt + 1))
    raise RuntimeError(err)

TOOLS = [
    {"type": "function", "name": "slide_info", "description": "Project, size, microns per pixel, tile count and the operators available for this slide.", "parameters": {"type": "object", "properties": {"slide": {"type": "string"}}, "required": ["slide"]}},
    {"type": "function", "name": "ask_slide", "description": "Slide level typed questions answered by a trained structured readout over 4,096 20x tiles; returns calibrated probabilities over the candidates. Operators: cancer_type, pan_grade (1/2/3), pan_lvi, pan_pni, pan_necrosis, pan_precursor, pan_mitotic (low/moderate/high), pan_multifocal, pan_infiltrate, tumor_purity_bin, leukocyte_fraction_bin, molecular_subtype.", "parameters": {"type": "object", "properties": {"slide": {"type": "string"}, "operators": {"type": "array", "items": {"type": "string"}}}, "required": ["slide", "operators"]}},
    {"type": "function", "name": "tile_evidence", "description": "Where a localised finding is strongest according to a full bag tile head trained on report labels: slide probability and the top tiles (level 0 x, y, score). finding: necrosis, lvi, pni, precursor.", "parameters": {"type": "object", "properties": {"slide": {"type": "string"}, "finding": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["slide", "finding"]}},
    {"type": "function", "name": "ask_region", "description": "Cell and tissue questions about the 80 um field centred at level 0 (x, y), answered by a trained field readout: present, count_bin, cell_frac_bin, more_cells (cell_class tumor | stromal | sTIL | other), tissue_present, tissue_frac_bin, dominant_tissue (448 um context), stil_score_bin, mitotic_present. Slow (about 20 s): use sparingly.", "parameters": {"type": "object", "properties": {"slide": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}, "operators": {"type": "array", "items": {"type": "string"}}, "cell_class": {"type": "string"}}, "required": ["slide", "x", "y", "operators"]}},
]
VISION_TOOLS = [
    {"type": "function", "name": "view_region", "description": "Look at an H&E crop centred at level 0 (x, y): size_um microns wide at mpp microns per pixel.", "parameters": {"type": "object", "properties": {"slide": {"type": "string"}, "x": {"type": "integer"}, "y": {"type": "integer"}, "size_um": {"type": "integer"}, "mpp": {"type": "number"}}, "required": ["slide", "x", "y"]}},
]
def run_tool(name, args, alias=None):
    """alias = (shown name, slide id): the model's name for the slide is resolved here and put back in the result, project and patient are dropped (as scripts/80 does with OPSIS_SLIDE_ALIAS and OPSIS_HIDE_PROJECT)"""
    if alias and args.get("slide") == alias[0]: args = {**args, "slide": alias[1]}
    f = getattr(T, name); r = f(**args); img = r.pop("image_png_base64", None) if name == "view_region" else None
    if alias and isinstance(r, dict): r = {k: v for k, v in r.items() if k not in ("project", "patient")}
    txt = json.dumps(r); return (txt.replace(alias[1], alias[0]) if alias else txt), img
def grey64(size):
    import io; from PIL import Image
    b = io.BytesIO(); Image.new("RGB", size, (128, 128, 128)).save(b, format="PNG"); return base64.b64encode(b.getvalue()).decode()
def size64(b64):
    import io; from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).size

SYSTEM = ("You are a pathologist's assistant answering one multiple choice question about a whole slide image. "
          + ("You have tools that read the slide with trained models; they return calibrated probabilities. Compose composite questions from the primitive questions (for example 'high grade AND lymphovascular invasion' from pan_grade and pan_lvi), and prefer the tool probabilities over guessing. " if a.arm in ("tools", "both") else "")
          + ("You can look at the slide yourself: a thumbnail and tiles are given, and view_region shows any location. " if a.arm in ("vision", "both") else "")
          + "When you are done, answer with a JSON object only: {\"answer\": \"<option letter>\", \"confidence\": <0 to 1>, \"reason\": \"<one sentence>\"}.")
def tissue_tiles(slide, n):
    import h5py
    _, S = T._registry(); rng = np.random.default_rng(a.seed)
    with h5py.File(S.loc[slide, "h5_path"], "r") as h: C0 = h["coords_level0"][:]; ps = float(h.attrs.get("patch_size_level0", 512))
    idx = rng.choice(len(C0), min(n, len(C0)), replace=False); return [(int(C0[i, 0] + ps / 2), int(C0[i, 1] + ps / 2)) for i in idx]
def episode(slide, q):
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["candidates"]))
    import hashlib; shown = f"slide_{hashlib.md5(slide.encode()).hexdigest()[:8]}" if a.alias else slide; alias = (shown, slide) if a.alias else None
    content = [{"type": "input_text", "text": f"Slide: {shown}\nQuestion: {q['question']}\nOptions:\n{opts}"}]
    if a.arm in ("vision", "both"):
        th = T.thumbnail(slide, 1024)
        if a.blank_image: th["image_png_base64"] = grey64(size64(th["image_png_base64"]))
        content.append({"type": "input_text", "text": f"Thumbnail ({th['mpp']} um per pixel):"}); content.append({"type": "input_image", "image_url": "data:image/png;base64," + th["image_png_base64"], "detail": "high"})
        for (x, y) in tissue_tiles(slide, a.n_tiles):
            v = {"image_png_base64": grey64((512, 512))} if a.blank_image else T.view_region(slide, x, y, 256, 0.5, 512); content.append({"type": "input_text", "text": f"Tile at level 0 ({x}, {y}), 256 um at 0.5 um per pixel:"}); content.append({"type": "input_image", "image_url": "data:image/png;base64," + v["image_png_base64"], "detail": "high"})
    inp = [{"role": "user", "content": content}]
    tools = (TOOLS if a.arm in ("tools", "both") else []) + (VISION_TOOLS if a.arm == "both" else [])
    usage = dict(prompt=0, completion=0); calls = []; t0 = time.time(); text = None
    for step in range(a.max_steps + 1):
        body = {"model": a.model, "instructions": SYSTEM, "input": inp, "max_output_tokens": a.max_tokens, "store": False}
        if tools and step < a.max_steps: body["tools"] = tools
        if a.reasoning and a.model.startswith(("gpt-5", "gpt-6", "o")): body["reasoning"] = {"effort": a.reasoning}
        d = api(body); u = d.get("usage", {}); usage["prompt"] += u.get("input_tokens", 0); usage["completion"] += u.get("output_tokens", 0)
        out = d.get("output", []); fcs = [o for o in out if o.get("type") == "function_call"]
        for o in out:
            if o.get("type") in ("function_call", "message", "reasoning"): inp.append({k: v for k, v in o.items() if k in ("type", "id", "call_id", "name", "arguments", "role", "content", "summary", "encrypted_content")})
        if fcs:
            for tc in fcs:
                name = tc["name"]; args = json.loads(tc.get("arguments") or "{}"); t1 = time.time()
                try: txt, img = run_tool(name, args, alias)
                except Exception as e: txt, img = json.dumps({"error": f"{type(e).__name__}: {str(e)[:200]}"}), None
                calls.append(dict(tool=name, args=args, seconds=round(time.time() - t1, 1), result=txt[:300]))
                inp.append({"type": "function_call_output", "call_id": tc["call_id"], "output": txt})
                if img: inp.append({"role": "user", "content": [{"type": "input_text", "text": f"Image from {name}({args.get('x')}, {args.get('y')}):"}, {"type": "input_image", "image_url": "data:image/png;base64," + img, "detail": "high"}]})
            continue
        text = d.get("output_text") or "".join(c.get("text", "") for o in out if o.get("type") == "message" for c in o.get("content", []) if c.get("type") == "output_text"); break
    ans = None; conf = None
    if text:
        try: j = json.loads(text[text.index("{"): text.rindex("}") + 1]); ans = str(j.get("answer", "")).strip().upper()[:1]; conf = float(j.get("confidence", 0.5))
        except Exception: ans = next((c for c in text.upper() if c in LETTERS[: len(q["candidates"])]), None)
    pred = LETTERS.index(ans) if (ans and ans in LETTERS[: len(q["candidates"])]) else None
    return dict(slide=slide, qid=q["qid"], operator=q["operator"], answer=q["answer"], pred=pred, correct=(pred == q["answer"]), confidence=conf, steps=len(calls), tools=calls, usage=usage, seconds=round(time.time() - t0, 1), text=(text or "")[:400])
def structured_answer(slide, q):
    """The rules condition: the slide decoder alone, composites composed by rules from the primitive probabilities."""
    op = q["operator"]
    def P(o): r = T.ask_slide(slide, [o])["answers"][0]; return dict(zip(r["candidates"], r["probabilities"]))
    if op in SYNOPTIC + ["cancer_type", "molecular_subtype", "tumor_purity_bin", "leukocyte_fraction_bin", "pan_infiltrate"]:
        p = P(op); c = q["candidates"]; pr = [p.get(x, 0.0) for x in c]
    elif op == "high_grade_and_lvi": g, l = P("pan_grade"), P("pan_lvi"); y = g.get("3", 0) * l.get("yes", 0); pr = [1 - y, y]
    elif op == "necrosis_and_high_mitotic": n, m = P("pan_necrosis"), P("pan_mitotic"); y = n.get("yes", 0) * m.get("high", 0); pr = [1 - y, y]
    # two rules added after the first runs of the study (scripts/81d fills them into the saved runs): the pan cancer
    # operators have no tubule, pleomorphism or DCIS questions, so the nearest primitives stand in for the components
    elif op == "grade_from_components": g = P("pan_grade"); pr = [g.get(x, 0.0) for x in q["candidates"]]
    elif op == "dcis_with_comedo": d, n = P("pan_precursor"), P("pan_necrosis"); y = d.get("yes", 0) * n.get("yes", 0); pr = [1 - y, y]
    else: pr = None
    if pr is None: return dict(slide=slide, qid=q["qid"], operator=op, answer=q["answer"], pred=None, correct=False, confidence=None, steps=0, tools=[], usage=dict(prompt=0, completion=0), seconds=0, text="no rule")
    pred = int(np.argmax(pr)); return dict(slide=slide, qid=q["qid"], operator=op, answer=q["answer"], pred=pred, correct=(pred == q["answer"]), confidence=float(max(pr)), steps=0, tools=[], usage=dict(prompt=0, completion=0), seconds=0, text=json.dumps([round(x, 3) for x in pr]))
def episode_codex(slide, q):
    """One question through the Codex CLI: the prompt names the MCP tools, --arm tools forbids image tools, --arm both allows them,
    --arm vision asks the model to look at the thumbnail and six tiles through view_region before answering."""
    import subprocess, tempfile
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(q["candidates"]))
    allowed = {"tools": "slide_info, ask_slide, tile_evidence, ask_region (do not use view_region or thumbnail)", "both": "slide_info, ask_slide, tile_evidence, ask_region, view_region, thumbnail", "vision": "thumbnail and view_region only (look at the thumbnail and at least six 256 um tiles yourself; do not call ask_slide, tile_evidence or ask_region)", "free": "slide_info, ask_slide, tile_evidence, ask_region, view_region, thumbnail"}[a.arm]
    if a.arm == "free":   # no guidance at all: the model plans what to look at and what to ask
        prompt = (f"You are a pathologist answering one multiple choice question about a whole slide image. The MCP server 'opsis' offers tools: {allowed}; use them however you judge best, in any order and as many times as you need. Do not run shell commands."
                  f"\nSlide: {slide}\nQuestion: {q['question']}\nOptions:\n{opts}\nWhen you are done, reply with a JSON object only: {{\"answer\": \"<option letter>\", \"confidence\": <0 to 1>, \"reason\": \"<one sentence>\"}}")
    else:
        prompt = (f"You are a pathologist's assistant answering one multiple choice question about a whole slide image. Use the MCP server 'opsis' with these tools: {allowed}. "
              + ("The readout tools return calibrated probabilities from trained models; prefer them over guessing and compose composite questions from primitive ones. " if a.arm != "vision" else "")
              + f"Do not run shell commands.\nSlide: {slide}\nQuestion: {q['question']}\nOptions:\n{opts}\nWhen you are done, reply with a JSON object only: {{\"answer\": \"<option letter>\", \"confidence\": <0 to 1>, \"reason\": \"<one sentence>\"}}")
    tmp = tempfile.mkdtemp(prefix="codex_", dir=OUT); last = f"{tmp}/last.txt"; t0 = time.time()
    cmd = [a.codex_bin, "exec", "--json", "--skip-git-repo-check", "-s", "read-only", "-C", tmp, "-m", a.model, "-o", last,
           "-c", f'mcp_servers.opsis.command="{sys.executable}"', "-c", f'mcp_servers.opsis.args=["{paths.ROOT}/scripts/80_mcp_server.py"]',
           "-c", f'mcp_servers.opsis.env={{OPSIS_DATA="{os.environ.get("OPSIS_DATA", "")}",PYTHONPATH="{paths.ROOT}/src"}}', prompt]
    env = dict(os.environ); env["PATH"] = os.path.dirname(a.codex_bin) + ":" + env.get("PATH", "")     # the codex launcher needs node on PATH
    r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1200, env=env)
    if r.returncode or not os.path.exists(f"{tmp}/last.txt"): print("codex exit", r.returncode, r.stderr[-400:].replace("\n", " | "), flush=True)
    calls = []; usage = dict(prompt=0, completion=0, cached=0)
    for line in r.stdout.splitlines():
        try: e = json.loads(line)
        except Exception: continue
        it = e.get("item", {})
        if e.get("type") == "item.completed" and it.get("type") == "mcp_tool_call": calls.append(dict(tool=it.get("tool"), args=it.get("arguments"), status=it.get("status"), result=str(it.get("result"))[:300]))
        if e.get("type") == "turn.completed": u = e.get("usage", {}); usage["prompt"] += u.get("input_tokens", 0); usage["completion"] += u.get("output_tokens", 0); usage["cached"] += u.get("cached_input_tokens", 0)
    text = open(last).read() if os.path.exists(last) else ""
    ans = None; conf = None
    try: j = json.loads(text[text.index("{"): text.rindex("}") + 1]); ans = str(j.get("answer", "")).strip().upper()[:1]; conf = float(j.get("confidence", 0.5))
    except Exception: ans = next((c for c in text.upper() if c in LETTERS[: len(q["candidates"])]), None)
    pred = LETTERS.index(ans) if (ans and ans in LETTERS[: len(q["candidates"])]) else None
    return dict(slide=slide, qid=q["qid"], operator=q["operator"], answer=q["answer"], pred=pred, correct=(pred == q["answer"]), confidence=conf, steps=len(calls), tools=calls, usage=usage, seconds=round(time.time() - t0, 1), text=text[:400], stderr=r.stderr[-300:] if r.returncode else "")
def episode_claude(slide, q):
    """One question through the Claude Code CLI in print mode: --restricted (no local settings, hooks or code running tools) with our own system prompt (SYSTEM of the API backend),
    the opsis MCP server attached with --strict-mcp-config, permission mode dontAsk with only the condition's tools allowed, no built in
    tools except Read in --arm vision, which reads the same prescribed thumbnail and tiles as the API backend from PNG files.
    Token counts come from the result event (prompt = input + cache read + cache creation)."""
    import subprocess, tempfile
    opts = "\n".join(f"{LAB(i)}. {c}" for i, c in enumerate(q["candidates"]))
    tmp = tempfile.mkdtemp(prefix="claude_", dir=OUT); t0 = time.time()
    mcp_tools = {"tools": ["slide_info", "ask_slide", "tile_evidence", "ask_region"], "both": ["slide_info", "ask_slide", "tile_evidence", "ask_region", "view_region", "thumbnail"], "free": ["slide_info", "ask_slide", "tile_evidence", "ask_region", "view_region", "thumbnail"], "vision": [], "look": ["slide_info", "thumbnail", "view_region"]}[a.arm]
    import hashlib; shown = f"slide_{hashlib.md5(slide.encode()).hexdigest()[:8]}" if a.alias else slide
    user = f"Slide: {shown}\nQuestion: {q['question']}\nOptions:\n{opts}"; builtin = ""
    if a.arm == "vision":
        th = T.thumbnail(slide, 1024); open(f"{tmp}/thumbnail.png", "wb").write(base64.b64decode(th["image_png_base64"])); lines = [f"Thumbnail of the slide ({th['mpp']} um per pixel): {tmp}/thumbnail.png"]
        for i, (x, y) in enumerate(tissue_tiles(slide, a.n_tiles)):
            v = T.view_region(slide, x, y, 256, 0.5, 512); open(f"{tmp}/tile{i + 1}.png", "wb").write(base64.b64decode(v["image_png_base64"])); lines.append(f"Tile {i + 1} at level 0 ({x}, {y}), 256 um at 0.5 um per pixel: {tmp}/tile{i + 1}.png")
        user = "Look at every image below with the Read tool, then answer.\n" + "\n".join(lines) + "\n" + user; builtin = "Read"
    if a.arm == "free":
        system = ("You are a pathologist answering one multiple choice question about a whole slide image. The MCP server opsis offers tools that read the slide with trained models (calibrated probabilities) and tools that show you the slide; use them however you judge best, in any order and as many times as you need. "
                  "When you are done, answer with a JSON object only: {\"answer\": \"<option letter>\", \"confidence\": <0 to 1>, \"reason\": \"<one sentence>\"}.")
    elif a.arm == "look":   # looking with free viewing: no trained model behind any tool, the model chooses what to look at and how often
        system = ("You are a pathologist's assistant answering one multiple choice question about a whole slide image. You can look at the slide yourself with the MCP tools of the server opsis: slide_info gives its size and microns per pixel, thumbnail shows the whole slide, and view_region shows any location at the field width and resolution you choose (level 0 coordinates). Start from the thumbnail, then look at as many regions as you need, at low and at high power, before you answer. "
                  "When you are done, answer with a JSON object only: {\"answer\": \"<option letter>\", \"confidence\": <0 to 1>, \"reason\": \"<one sentence>\"}.")
    else:
        system = SYSTEM.replace("You have tools that read the slide", "You have MCP tools (server opsis) that read the slide").replace("view_region shows any location", "the MCP tools thumbnail and view_region show any location")
    cfg = {"mcpServers": {"opsis": {"command": sys.executable, "args": [f"{paths.ROOT}/scripts/80_mcp_server.py"], "env": {"OPSIS_DATA": os.environ.get("OPSIS_DATA", ""), "PYTHONPATH": f"{paths.ROOT}/src", "OPSIS_MCP_TOOLS": ",".join(mcp_tools), **({"OPSIS_SLIDE_ALIAS": f"{shown}={slide}", "OPSIS_HIDE_PROJECT": "1"} if a.alias else {})}}}}
    allowed = [f"mcp__opsis__{t}" for t in mcp_tools] + (["Read"] if builtin else [])
    cmd = [a.claude_bin, "-p", user, "--restricted", "--model", a.model, "--output-format", "stream-json", "--verbose", "--no-session-persistence", "--max-turns", str(a.max_steps + 2),
           "--system-prompt", system, "--tools", builtin, "--permission-mode", "dontAsk", "--allowedTools", ",".join(allowed)]
    if mcp_tools: cmd += ["--mcp-config", json.dumps(cfg), "--strict-mcp-config"]
    env = {"HOME": os.environ["HOME"], "PATH": "/usr/bin:/bin", "TERM": "dumb"}
    try: r = subprocess.run(cmd, cwd=tmp, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=900, env=env)
    except subprocess.TimeoutExpired:   # a hung CLI call is a failed episode (not logged, retried on resume), not a crash of the shard
        import shutil; shutil.rmtree(tmp, ignore_errors=True); print("claude timeout after 900 s", flush=True)
        return dict(slide=slide, qid=q["qid"], operator=q["operator"], answer=q["answer"], pred=None, correct=False, confidence=None, steps=0, tools=[], usage=dict(prompt=0, completion=0, cached=0), seconds=round(time.time() - t0, 1), text="", cost_usd=None, stderr="timeout", failed=True)
    calls = []; usage = dict(prompt=0, completion=0, cached=0); text = ""; cost = None; denied = []; err = False
    for line in r.stdout.splitlines():
        try: e = json.loads(line)
        except Exception: continue
        if e.get("type") == "assistant":
            for c in e.get("message", {}).get("content", []):
                if c.get("type") == "tool_use": calls.append(dict(tool=c.get("name"), args=c.get("input")))
        if e.get("type") == "result":
            text = e.get("result") or ""; u = e.get("usage", {}); usage["prompt"] = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0); usage["cached"] = u.get("cache_read_input_tokens", 0); usage["completion"] = u.get("output_tokens", 0); cost = e.get("total_cost_usd"); denied = e.get("permission_denials", []); err = bool(e.get("is_error"))
    failed = bool(r.returncode or err)  # a CLI or API failure (rate limit, login) is not an answer: the episode is not logged and the run stops after three in a row
    if r.returncode or not text: print("claude exit", r.returncode, (r.stderr or r.stdout)[-500:].replace("\n", " | "), flush=True)
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
    if denied: print("  permission denials:", [d.get("tool_name") for d in denied][:5], flush=True)
    import re; labs = [LAB(i) for i in range(len(q["candidates"]))]; ans = None; conf = None
    try: j = json.loads(text[text.index("{"): text.rindex("}") + 1]); ans = str(j.get("answer", "")).strip().upper().rstrip("."); conf = float(j.get("confidence", 0.5))
    except Exception: ans = next((c for c in re.findall(r"[A-Z]+", text.upper()) if c in labs), None)
    pred = labs.index(ans) if ans in labs else None
    return dict(slide=slide, qid=q["qid"], operator=q["operator"], answer=q["answer"], pred=pred, correct=(pred == q["answer"]), confidence=conf, steps=len(calls), tools=calls, usage=usage, seconds=round(time.time() - t0, 1), text=text[:400], cost_usd=cost, stderr=r.stderr[-300:] if r.returncode else "", failed=failed)
done = set()
if a.resume and os.path.exists(LOG): done = {json.loads(l)["qid"] for l in open(LOG)}
slides = T.list_slides(a.split, a.project, a.slides, a.seed); tasks = [(s, q) for s in slides for q in T.ground_truth(s, OPS)]
if a.limit_q: tasks = tasks[: a.limit_q]
if a.shard: i_, n_ = map(int, a.shard.split("/")); tasks = tasks[i_::n_]
print(f"{a.name}: arm {a.arm} model {a.model} slides {len(slides)} questions {len(tasks)} ops {sorted({q['operator'] for _, q in tasks})}", flush=True)
nfail = 0
with open(LOG, "a") as f:
    for i, (s, q) in enumerate(tasks):
        if q["qid"] in done: continue
        r = structured_answer(s, q) if a.arm == "structured" else episode_codex(s, q) if a.backend == "codex" else episode_claude(s, q) if a.backend == "claude" else episode(s, q)
        if r.pop("failed", False):
            nfail = nfail + 1; print(f"[{i + 1}/{len(tasks)}] {s[:16]} {q['operator']:26s} FAILED ({nfail} in a row), not logged", flush=True)
            if nfail >= 3: raise SystemExit("three failed episodes in a row (rate limit or login?), stopping; rerun with --resume")
            continue
        nfail = 0; f.write(json.dumps(r) + "\n"); f.flush()
        print(f"[{i + 1}/{len(tasks)}] {s[:16]} {q['operator']:26s} truth {q['candidates'][q['answer']]!s:12s} pred {q['candidates'][r['pred']] if r['pred'] is not None else None!s:12s} {'ok ' if r['correct'] else 'no '} steps {r['steps']} tok {r['usage']['prompt']}+{r['usage']['completion']} {r['seconds']}s", flush=True)
R = pd.DataFrame([json.loads(l) for l in open(LOG)]); summ = {"name": a.name, "arm": a.arm, "model": a.model, "n": len(R), "acc": float(R.correct.mean()), "per_op": {op: dict(n=len(g), acc=float(g.correct.mean())) for op, g in R.groupby("operator")},
        "prompt_tokens": int(sum(u["prompt"] for u in R.usage)), "completion_tokens": int(sum(u["completion"] for u in R.usage)), "tool_calls_per_question": float(R.steps.mean()), "seconds_per_question": float(R.seconds.mean())}
json.dump(summ, open(f"{OUT}/{a.name}_summary.json", "w"), indent=1); print("SUMMARY", json.dumps({k: v for k, v in summ.items() if k != "per_op"})); print({op: (v["n"], round(v["acc"], 3)) for op, v in summ["per_op"].items()})
