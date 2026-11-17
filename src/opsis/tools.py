"""Typed Opsis tools for an organiser (a frontier model or a local LLM): the trained decoders of Opsis-S answer isolated typed questions
about a slide or a region, return calibrated probabilities, and expose tile level evidence. Everything runs on CPU from the
cached Virchow2 features (slides) or an on the fly Virchow2 encoding (regions), so the server can live on a login node.
Functions return JSON serialisable dicts; images are base64 PNG strings. Models load lazily on first use."""
import base64, glob, io, json, os
import numpy as np, pandas as pd, torch, h5py
from PIL import Image
from opsis import paths, facts, dataset as D
from opsis.data import nucls

torch.set_num_threads(int(os.environ.get("OPSIS_TOOL_THREADS", "8")))
DEV = torch.device(os.environ.get("OPSIS_TOOL_DEVICE", "cpu"))          # cpu on a login node; cuda inside a GPU job
RUNS = dict(fov=os.environ.get("OPSIS_FOV_RUN", "final_fov_s0"), slide=os.environ.get("OPSIS_SLIDE_RUN", "final_slide_pan_s0"), tile=os.environ.get("OPSIS_TILE_RUN", "final_tile_lse_s0"))
WSI_ROOT = paths.WSI
FINDING_OP = {"necrosis": "pan_necrosis", "lvi": "pan_lvi", "pni": "pan_pni", "precursor": "pan_precursor"}
SLIDE_OPS = {  # operator -> (question, what the candidates mean)
    "cancer_type": "Which TCGA cancer type is this slide from?", "pan_grade": "What is the histologic grade of the tumour (1 = well, 2 = moderately, 3 = poorly differentiated or high grade)?",
    "pan_lvi": "Is lymphovascular invasion present?", "pan_pni": "Is perineural invasion present?", "pan_necrosis": "Is tumour necrosis present?",
    "pan_precursor": "Is a precursor or in situ component of the tumour present?", "pan_mitotic": "How high is the mitotic activity of the tumour?", "pan_multifocal": "Is the tumour multifocal?",
    "pan_infiltrate": "How dense is the lymphocytic infiltrate?", "tumor_purity_bin": "What fraction of the cells in this tumour are tumour cells?", "leukocyte_fraction_bin": "What is the leukocyte fraction of the sample?", "molecular_subtype": "Which molecular subtype is this tumour?"}
REGION_OPS = ["present", "count_bin", "cell_frac_bin", "more_cells", "tissue_present", "tissue_frac_bin", "dominant_tissue", "stil_score_bin", "mitotic_present"]
_M = {}

# ---------- registry ----------
def _registry():
    if "Q" not in _M:
        Q = pd.read_parquet(f"{paths.DERIVED}/slide_questions_pan/questions_seed0.parquet"); _M["Q"] = Q
        _M["slides"] = Q.drop_duplicates("slide").set_index("slide")[["project", "patient", "split", "h5_path"]]
    return _M["Q"], _M["slides"]
def list_slides(split="test", project=None, n=20, seed=0):
    """Slide ids of a split (test, val, heldout_project), optionally one TCGA project, a seeded sample of n."""
    _, S = _registry(); s = S[S.split == split]
    if project: s = s[s.project == project]
    ids = sorted(s.index); rng = np.random.default_rng(seed); return [ids[i] for i in sorted(rng.choice(len(ids), min(n, len(ids)), replace=False))]
def _wsi(slide):
    _, S = _registry(); proj = S.loc[slide, "project"]; hits = glob.glob(f"{WSI_ROOT}/TCGA-{proj}/*/{slide}.svs")
    if not hits: raise FileNotFoundError(f"no WSI for {slide}")
    import openslide; return openslide.OpenSlide(hits[0])
def slide_info(slide):
    """Project, dimensions, microns per pixel, number of cached 20x tiles and the operators the slide decoder answers."""
    _, S = _registry(); r = S.loc[slide]
    with h5py.File(r.h5_path, "r") as h: n = int(h["features"].shape[0]); ps = float(h.attrs.get("patch_size_level0", 512))
    try: sl = _wsi(slide); W, H = sl.dimensions; mpp = float(sl.properties.get("openslide.mpp-x", 0.25)); sl.close()
    except Exception: W = H = None; mpp = None
    return dict(slide=slide, project=str(r.project), patient=str(r.patient), width_px=W, height_px=H, mpp=mpp, n_tiles_20x=n, tile_size_level0_px=ps, slide_operators=list(SLIDE_OPS), findings_with_tile_maps=list(FINDING_OP), region_operators=REGION_OPS)

# ---------- slide decoder ----------
def _slide_model():
    if "slide" not in _M:
        from opsis.model import OpsisNet, load_compat
        run = f"{paths.RUNS}/{RUNS['slide']}"; vocab = D.Vocab.from_dict(json.load(open(f"{run}/vocab.json"))); cfg = json.load(open(f"{run}/config.json"))
        m = OpsisNet(len(vocab.ops), len(vocab.args), len(vocab.cands), cfg["d"], cfg["enc_layers"], cfg["dec_layers"], dropout=0.0, d_in=2560).to(DEV); load_compat(m, torch.load(f"{run}/best.pt", map_location=DEV)); m.eval()
        _M["slide"] = (m, vocab, cfg)
    return _M["slide"]
def _slide_tokens(slide, k):
    _, S = _registry()
    with h5py.File(S.loc[slide, "h5_path"], "r") as h:
        n = h["features"].shape[0]; C = h["coords"][:]; lo, hi = C.min(0), C.max(0); rng = np.random.default_rng(0)
        if n >= k: idx = np.sort(rng.choice(n, k, replace=False)); X = h["features"][idx].astype(np.float16); Cs = C[idx]
        else: idx = rng.integers(0, n, k); X = h["features"][:].astype(np.float16)[idx]; Cs = C[idx]
    uv = ((Cs - lo) / np.maximum(hi - lo, 1.0)).astype(np.float32); return X, np.concatenate([uv, uv], 1)
def _slide_question_rows(slide, ops):
    Q, S = _registry(); proj = S.loc[slide, "project"]; rows = []
    for op in ops:
        g = Q[(Q.op == op) & (Q.project == proj)]
        if len(g) == 0: g = Q[Q.op == op]
        if len(g) == 0: continue
        r = g.iloc[0]; rows.append(dict(fov=slide, slide=slide, patient="", family=r.family, op=op, args=r.args, atype=r.atype, candidates=r.candidates, answer=0, text=r.text, box="[0.0, 0.0, 1.0, 1.0]", box2=""))
    return pd.DataFrame(rows)
def ask_slide(slide, operators):
    """Answer typed slide level questions with the slide decoder of Opsis-S (4,096 cached 20x tiles). operators: list of
    operator names (see slide_info). Returns for each the question, its candidates and calibrated probabilities."""
    if isinstance(operators, str): operators = [operators]
    m, vocab, cfg = _slide_model(); q = _slide_question_rows(slide, operators)
    if len(q) == 0: return {"error": f"no known operator among {operators}"}
    X, pos = _slide_tokens(slide, cfg["k_tiles"]); e = D.encode_questions(q, vocab, "op"); n = len(q)
    box = np.tile(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.float32), (n, 1))
    item = dict(tok_fov=torch.from_numpy(X), pos=torch.from_numpy(pos), op=torch.from_numpy(e["op"]), key=torch.from_numpy(e["key"]), args=torch.from_numpy(e["args"]), cands=torch.from_numpy(e["cands"]), cand_mask=torch.from_numpy(e["cand_mask"]), answer=torch.from_numpy(e["answer"]), box=torch.from_numpy(box), box2=torch.from_numpy(np.zeros_like(box)), has_box2=torch.zeros(n), text_id=torch.from_numpy(e["text_id"]), cand_text=torch.from_numpy(e["cand_text"]), atype=torch.from_numpy(e["atype"]), qidx=torch.arange(n), fov_index=0)
    b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in D.collate([item]).items()}
    with torch.no_grad(): p = torch.softmax(m(b, False).float(), -1)[0].cpu().numpy()
    out = []
    for i, r in enumerate(q.itertuples()):
        c = json.loads(r.candidates); out.append(dict(operator=r.op, question=r.text, candidates=c, probabilities=[round(float(x), 4) for x in p[i, :len(c)]], answer=c[int(np.argmax(p[i, :len(c)]))]))
    return {"slide": slide, "answers": out}

# ---------- tile heads ----------
class _TileHead(torch.nn.Module):
    def __init__(self, n_ops, hidden=512):
        super().__init__(); self.trunk = torch.nn.Sequential(torch.nn.LayerNorm(2560), torch.nn.Linear(2560, hidden), torch.nn.GELU(), torch.nn.Dropout(0.0), torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(0.0)); self.head = torch.nn.Linear(hidden, n_ops)
    def forward(self, x): return self.head(self.trunk(x))
def _tile_model():
    if "tile" not in _M:
        run = f"{paths.RUNS}/{RUNS['tile']}"; cfg = json.load(open(f"{run}/config.json")); ops = [x for x in cfg["ops"].replace("+", ",").split(",") if x]
        m = _TileHead(len(ops), cfg["hidden"]).to(DEV); m.load_state_dict(torch.load(f"{run}/best.pt", map_location=DEV)); m.eval(); _M["tile"] = (m, ops, cfg["pool"])
    return _M["tile"]
def tile_evidence(slide, finding, top_k=6):
    """Per tile evidence for a localised finding (necrosis, lvi, pni, precursor) from the full bag tile head: the slide
    probability and the top_k tiles (level 0 x, y, score) where the evidence is strongest, for viewing with view_region."""
    m, ops, pool = _tile_model(); op = FINDING_OP.get(finding, finding)
    if op not in ops: return {"error": f"finding must be one of {list(FINDING_OP)}"}
    _, S = _registry()
    with h5py.File(S.loc[slide, "h5_path"], "r") as h: X = torch.from_numpy(h["features"][:].astype(np.float32)); C0 = h["coords_level0"][:]; ps = float(h.attrs.get("patch_size_level0", 512))
    with torch.no_grad(): s = m(X)[:, ops.index(op)].numpy()
    n = len(s); slide_logit = float(np.logaddexp.reduce(s) - np.log(n)) if pool == "lse" else float(np.log(-np.expm1(-np.logaddexp(0, s).sum())) + np.logaddexp(0, s).sum())
    p = float(1 / (1 + np.exp(-slide_logit))); order = np.argsort(-s)[:top_k]
    return {"slide": slide, "finding": finding, "slide_probability": round(p, 4), "n_tiles": n, "tile_size_level0_px": ps,
            "top_tiles": [dict(x=int(C0[i, 0]), y=int(C0[i, 1]), score=round(float(s[i]), 3), percentile=round(float((s < s[i]).mean()), 3)) for i in order],
            "note": "scores are per tile logits; the slide probability aggregates all tiles"}

# ---------- images ----------
def _png64(im): buf = io.BytesIO(); im.save(buf, format="PNG"); return base64.b64encode(buf.getvalue()).decode()
def thumbnail(slide, max_px=1024):
    """Low power thumbnail of the whole slide (base64 PNG) with its scale in microns per pixel."""
    sl = _wsi(slide); W, H = sl.dimensions; mpp0 = float(sl.properties.get("openslide.mpp-x", 0.25)); im = sl.get_thumbnail((max_px, max_px)); sl.close()
    return {"slide": slide, "image_png_base64": _png64(im), "mpp": round(mpp0 * W / im.size[0], 2), "width_px": im.size[0], "height_px": im.size[1]}
def view_region(slide, x, y, size_um=256, mpp=0.5, out_px=512):
    """An H&E crop centred at level 0 (x, y): size_um microns wide, rendered at mpp microns per pixel, resized to out_px (base64 PNG)."""
    sl = _wsi(slide); mpp0 = float(sl.properties.get("openslide.mpp-x", 0.25)); side0 = int(size_um / mpp0); lev = sl.get_best_level_for_downsample(mpp / mpp0); f = sl.level_downsamples[lev]
    im = sl.read_region((int(x - side0 / 2), int(y - side0 / 2)), lev, (int(side0 / f), int(side0 / f))).convert("RGB").resize((out_px, out_px), Image.BILINEAR); sl.close()
    return {"slide": slide, "x": x, "y": y, "size_um": size_um, "image_png_base64": _png64(im)}

# ---------- field decoder (on the fly Virchow2 on CPU) ----------
def _encoder():
    if "enc" not in _M:
        from opsis import encoder as E
        _M["enc"] = E.load_virchow2(DEV, img_size=448); _M["mean"] = torch.tensor(E.IMAGENET_MEAN).view(1, 3, 1, 1); _M["std"] = torch.tensor(E.IMAGENET_STD).view(1, 3, 1, 1)
    return _M["enc"]
def _fov_model():
    if "fov" not in _M:
        from opsis.model import OpsisNet, load_compat
        run = f"{paths.RUNS}/{RUNS['fov']}"; vocab = D.Vocab.from_dict(json.load(open(f"{run}/vocab.json"))); cfg = json.load(open(f"{run}/config.json"))
        m = OpsisNet(len(vocab.ops), len(vocab.args), len(vocab.cands), cfg["d"], cfg["enc_layers"], cfg["dec_layers"], dropout=0.0, pool=cfg["pool"], density=cfg["aux_density"] > 0).to(DEV); load_compat(m, torch.load(f"{run}/best.pt", map_location=DEV)); m.eval()
        Qf = pd.read_parquet(f"{paths.DERIVED}/questions_final/questions_seed0.parquet"); tmpl = {op: Qf[Qf.op == op].iloc[0] for op in REGION_OPS if (Qf.op == op).any()}
        _M["fov"] = (m, vocab, cfg, tmpl)
    return _M["fov"]
def _region_state(slide, x, y, fov_um=80.0):
    """Builds the NuCLS style state at (x, y): FOV png at 0.2 mpp (fov_um wide) on the 448 canvas at 0.25 mpp, context 448 um at 1.0 mpp. Returns tokens, geometry rows."""
    sl = _wsi(slide); mpp0 = float(sl.properties.get("openslide.mpp-x", 0.25)); S = 448
    side0 = int(round(fov_um / mpp0)); left, top = int(x - side0 / 2), int(y - side0 / 2); Wp = Hp = int(round(fov_um / nucls.FOV_MPP))
    fov_im = sl.read_region((left, top), 0, (side0, side0)).convert("RGB").resize((Wp, Hp), Image.BICUBIC)
    r = nucls.FOV_MPP / 0.25; w = h = int(round(Wp * r)); canvas = Image.new("RGB", (S, S), (255, 255, 255)); ox = oy = (S - w) // 2; canvas.paste(fov_im.resize((w, h), Image.BICUBIC), (ox, oy))
    cside0 = facts.CTX_PX * facts.CTX_MPP / mpp0; cl, ct = int(x - cside0 / 2), int(y - cside0 / 2); lev = sl.get_best_level_for_downsample(facts.CTX_MPP / mpp0); f = sl.level_downsamples[lev]
    ctx = sl.read_region((cl, ct), lev, (int(cside0 / f), int(cside0 / f))).convert("RGB").resize((S, S), Image.BILINEAR); sl.close()
    enc = _encoder(); xs = torch.from_numpy(np.stack([np.asarray(im, np.float32) / 255.0 for im in (canvas, ctx)])).permute(0, 3, 1, 2); xs = ((xs - _M["mean"]) / _M["std"]).to(DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV.type == "cuda"): t = enc.forward_features(xs)[:, enc.n_prefix:].half().cpu()
    fov_row = dict(scale=Wp / side0, left=left, top=top, ctx_left=cl, ctx_top=ct, ctx_right=cl + cside0, ctx_bottom=ct + cside0); geo_row = dict(fov_scale=r, off_x=ox, off_y=oy)
    return t[0].numpy(), t[1].numpy(), fov_row, geo_row, (Wp, Hp), canvas
def ask_region(slide, x, y, operators, cell_class="tumor"):
    """Answer region level questions about an 80 um field centred at level 0 (x, y) with the field decoder of Opsis-S (Virchow2 encoded on
    the fly). operators: subset of present, count_bin, cell_frac_bin, more_cells (cell_class = tumor | stromal | sTIL | other),
    tissue_present, tissue_frac_bin, dominant_tissue (on the 448 um context), stil_score_bin, mitotic_present."""
    if isinstance(operators, str): operators = [operators]
    m, vocab, cfg, tmpl = _fov_model(); tf, tc, fov_row, geo_row, (Wp, Hp), _ = _region_state(slide, x, y)
    g = D.FovGeometry(fov_row, geo_row, 448, facts.CTX_PX); pos = g.token_positions(32); rows = []
    ctx_box = [(fov_row["ctx_left"] - fov_row["left"]) * fov_row["scale"], (fov_row["ctx_top"] - fov_row["top"]) * fov_row["scale"], (fov_row["ctx_right"] - fov_row["left"]) * fov_row["scale"], (fov_row["ctx_bottom"] - fov_row["top"]) * fov_row["scale"]]
    for op in operators:
        if op not in tmpl: continue
        t = tmpl[op]; args = json.loads(t.args)
        for k in ("k", "c"):
            if k in args: args[k] = cell_class
        if "k1" in args: args["k1"], args["k2"] = (cell_class, "sTIL") if cell_class != "sTIL" else ("tumor", "sTIL")
        box = ctx_box if args.get("scope") == "ctx" else [0.0, 0.0, float(Wp), float(Hp)]
        rows.append(dict(fov=slide, slide=slide, patient="", family=t.family, op=op, args=json.dumps(args), atype=t.atype, candidates=t.candidates, answer=0, text=t.text, box=json.dumps(box), box2=""))
    if not rows: return {"error": f"operators must be among {REGION_OPS}"}
    q = pd.DataFrame(rows)
    try: e = D.encode_questions(q, vocab, "op")
    except KeyError as err: return {"error": f"argument not in the trained vocabulary: {err}"}
    n = len(q); box = np.stack([g.box_features(b) for b in e["box"]]); box2 = np.stack([g.box_features(b) for b in e["box2"]])
    item = dict(tok_fov=torch.from_numpy(tf.astype(np.float16)), tok_ctx=torch.from_numpy(tc.astype(np.float16)), pos=torch.from_numpy(pos), op=torch.from_numpy(e["op"]), key=torch.from_numpy(e["key"]), args=torch.from_numpy(e["args"]), cands=torch.from_numpy(e["cands"]), cand_mask=torch.from_numpy(e["cand_mask"]), answer=torch.from_numpy(e["answer"]), box=torch.from_numpy(box), box2=torch.from_numpy(box2), has_box2=torch.from_numpy(e["has_box2"]), text_id=torch.from_numpy(e["text_id"]), cand_text=torch.from_numpy(e["cand_text"]), atype=torch.from_numpy(e["atype"]), qidx=torch.arange(n), fov_index=0)
    b = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in D.collate([item]).items()}
    with torch.no_grad(): p = torch.softmax(m(b, True).float(), -1)[0].cpu().numpy()
    out = []
    for i, r in enumerate(q.itertuples()):
        c = json.loads(r.candidates); out.append(dict(operator=r.op, question=r.text, candidates=c, probabilities=[round(float(v), 4) for v in p[i, :len(c)]], answer=c[int(np.argmax(p[i, :len(c)]))]))
    return {"slide": slide, "x": x, "y": y, "field_um": 80, "answers": out}

# ---------- ground truth (evaluator only, never exposed to the organiser) ----------
def ground_truth(slide, operators=None):
    Q, _ = _registry(); g = Q[Q.slide == slide]
    if operators: g = g[g.op.isin(operators)]
    return [dict(qid=r.qid, operator=r.op, family=r.family, question=r.text, candidates=json.loads(r.candidates), answer=int(r.answer), atype=r.atype) for r in g.itertuples()]
