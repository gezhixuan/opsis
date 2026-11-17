"""States (cached encoder tokens) and structured questions for training and evaluation.

Geometry: the FOV png (0.2 mpp) was resampled by fov_scale and pasted at (off_x, off_y) on the 448 canvas; the context
canvas covers the level-0 box (ctx_left, ctx_top, ctx_right, ctx_bottom). A box in FOV-frame px maps to both canvases,
and every state token carries its (u, v) in both frames, so query boxes and token positions share one coordinate system.
"""
import json

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from opsis import paths
from opsis.questions import OPS

MAX_ARGS = 8
MAX_CANDS = 32                                                        # cancer_type has 32 candidates (pan TCGA); histologic_type_fine 11
# Operator primitives: a compositional description of every operator in shared vocabulary, so that a held-out operator
# is encoded by tokens that trained operators also use (query="primitives").
PRIMITIVES = {
    "cell_super": ["target=cell", "ask=class", "level=super"],
    "cell_main": ["target=cell", "ask=class", "level=main"],
    "cell_is": ["target=cell", "ask=is_class", "out=noul"],
    "cell_tissue": ["target=cell", "ask=tissue_class"],
    "cell_area_bin": ["target=cell", "ask=area", "out=bin"],
    "count_bin": ["target=region", "ask=count", "out=bin"],
    "present": ["target=region", "ask=count", "out=present"],
    "cell_frac_bin": ["target=region", "ask=cell_fraction", "out=bin"],
    "tissue_present": ["target=region", "ask=tissue_fraction", "out=present"],
    "tissue_frac_bin": ["target=region", "ask=tissue_fraction", "out=bin"],
    "dominant_tissue": ["target=region", "ask=tissue_dominant"],
    "more_cells": ["target=region", "ask=count", "out=compare_classes"],
    "compare_regions": ["target=region", "target2=region", "ask=count", "out=compare_targets"],
    "count_in_compartment_bin": ["target=region", "ask=count", "out=bin", "filter=compartment"],
    "tils_location": ["target=region", "ask=count", "out=compare_compartments", "filter=compartment"],
    "density_compare": ["target=region", "ask=density", "out=compare_compartments", "filter=compartment"],
    "mitotic_present": ["target=region", "ask=mitosis", "out=present"],
    "mitotic_count_bin": ["target=region", "ask=mitosis", "out=bin"],
    "stil_score_bin": ["target=region", "ask=til_fraction", "out=bin", "filter=compartment"],
    "tissue_is": ["target=region", "ask=tissue_class", "out=noul"],
    "cell_is": ["target=cell", "ask=is_class", "out=noul"],
}


ATYPE = {"choice": 0, "noul": 1, "ordinal": 2}


class Vocab:
    def __init__(self, questions):
        args = set()
        cands = set()
        self.texts = {t: i for i, t in enumerate(sorted(questions["text"].unique()))}
        self.cand_texts = {c: i for i, c in enumerate(sorted({c for cs in questions["candidates"].unique() for c in json.loads(cs)}))}
        for prims in PRIMITIVES.values():
            args.update(prims)
        for a, c in zip(questions["args"], questions["candidates"]):
            for k, v in json.loads(a).items():
                args.add(f"{k}={v}")
            for i, _ in enumerate(json.loads(c)):
                cands.add(f"{c}#{i}")
        self.args = {s: i + 1 for i, s in enumerate(sorted(args))}            # 0 = pad
        self.cands = {s: i + 1 for i, s in enumerate(sorted(cands))}          # 0 = pad
        self.ops = {o: i for i, o in enumerate(OPS)}
        keys = sorted({f"{o}|{a}" for o, a in zip(questions["op"], questions["args"]) for a in [self._key_args(a)]})
        self.keys = {k: i for i, k in enumerate(keys)}                        # operator + arguments without the region name

    @staticmethod
    def _key_args(a):
        d = {k: v for k, v in json.loads(a).items() if k not in ("region", "A", "B")}
        return json.dumps(d, sort_keys=True)

    def key_of(self, op, args):
        return self.keys.get(f"{op}|{self._key_args(args)}", 0)

    def to_dict(self):
        return {"args": self.args, "cands": self.cands, "ops": self.ops, "keys": self.keys, "texts": self.texts, "cand_texts": self.cand_texts}

    @classmethod
    def from_dict(cls, d):
        v = cls.__new__(cls)
        v.args, v.cands, v.ops, v.keys = d["args"], d["cands"], d["ops"], d.get("keys", {})
        v.texts, v.cand_texts = d.get("texts", {}), d.get("cand_texts", {})
        return v


def encode_questions(q, vocab, query="op"):
    """q: DataFrame rows of one FOV. Returns dict of numpy arrays (op, args, cands, cand_mask, answer, box, box2, has_box2).
    query="op": one embedding per operator plus its arguments. query="primitives": the operator is replaced by its
    primitive tokens (PRIMITIVES) in the argument slots and the op slot holds a shared generic id."""
    n = len(q)
    op = np.array([vocab.ops[o] if query in ("op", "text+op") else 0 for o in q["op"]], dtype=np.int64)
    key = np.array([vocab.key_of(o, a) for o, a in zip(q["op"], q["args"])], dtype=np.int64)
    text_id = np.array([vocab.texts.get(t, 0) for t in q["text"]], dtype=np.int64)
    atype = np.array([ATYPE.get(t, 0) for t in q["atype"]], dtype=np.int64)
    cand_text = np.zeros((n, MAX_CANDS), dtype=np.int64)
    args = np.zeros((n, MAX_ARGS), dtype=np.int64)
    cands = np.zeros((n, MAX_CANDS), dtype=np.int64)
    box = np.zeros((n, 4), dtype=np.float32)
    box2 = np.zeros((n, 4), dtype=np.float32)
    has2 = np.zeros(n, dtype=np.float32)
    for i, (o, a, c, b, b2) in enumerate(zip(q["op"], q["args"], q["candidates"], q["box"], q["box2"])):
        toks = [vocab.args[f"{k}={v}"] for k, v in json.loads(a).items()]
        if query == "primitives":
            toks = [vocab.args[t] for t in PRIMITIVES[o]] + toks
        args[i, :len(toks)] = toks[:MAX_ARGS]
        cl = json.loads(c)
        cands[i, :len(cl)] = [vocab.cands[f"{c}#{j}"] for j in range(len(cl))]
        cand_text[i, :len(cl)] = [vocab.cand_texts.get(x, 0) for x in cl]
        box[i] = json.loads(b)
        if isinstance(b2, str) and b2:
            box2[i] = json.loads(b2)
            has2[i] = 1.0
    if query in ("text", "text+op"):
        pass                                                     # the model reads text_id / cand_text; op and args stay for text+op
    if query == "text":
        args[:] = 0
    return dict(op=op, key=key, args=args, cands=cands, cand_mask=(cands > 0), answer=q["answer"].to_numpy().astype(np.int64),
                box=box, box2=box2, has_box2=has2, text_id=text_id, cand_text=cand_text, atype=atype)


class FovGeometry:
    """Maps FOV-frame px to normalised coordinates on the FOV canvas and on the context canvas."""

    def __init__(self, fov_row, geo_row, size, ctx_px):
        self.size = size
        self.ctx_px = ctx_px
        self.s = float(geo_row["fov_scale"])
        self.ox, self.oy = float(geo_row["off_x"]), float(geo_row["off_y"])
        self.scale = float(fov_row["scale"])                               # png px per level-0 px
        self.left, self.top = float(fov_row["left"]), float(fov_row["top"])
        self.ctx_left, self.ctx_top = float(fov_row["ctx_left"]), float(fov_row["ctx_top"])
        self.ctx_side0 = float(fov_row["ctx_right"]) - self.ctx_left         # level-0 px covered by the context canvas

    def fov_to_fovcanvas(self, x, y):
        return (self.ox + x * self.s) / self.size, (self.oy + y * self.s) / self.size

    def fov_to_ctxcanvas(self, x, y):
        X = self.left + x / self.scale
        Y = self.top + y / self.scale
        return (X - self.ctx_left) / self.ctx_side0, (Y - self.ctx_top) / self.ctx_side0

    def box_features(self, box):
        """8 numbers: box corners on the FOV canvas and on the context canvas (unit square frames)."""
        x0, y0, x1, y1 = box
        a = self.fov_to_fovcanvas(x0, y0) + self.fov_to_fovcanvas(x1, y1)
        b = self.fov_to_ctxcanvas(x0, y0) + self.fov_to_ctxcanvas(x1, y1)
        return np.array(a + b, dtype=np.float32)

    def token_positions(self, grid):
        """[2 * grid * grid, 4]: (u_fov, v_fov, u_ctx, v_ctx) for FOV-canvas tokens then context-canvas tokens."""
        c = (np.arange(grid) + 0.5) / grid
        vv, uu = np.meshgrid(c, c, indexing="ij")
        uu, vv = uu.ravel(), vv.ravel()
        # FOV canvas token (uu, vv) -> FOV-frame px -> ctx canvas
        xf = (uu * self.size - self.ox) / self.s
        yf = (vv * self.size - self.oy) / self.s
        uc, vc = self.fov_to_ctxcanvas(xf, yf)
        pf = np.stack([uu, vv, uc, vc], 1)
        # context canvas token (uu, vv) -> level-0 -> FOV-frame px -> FOV canvas
        X = self.ctx_left + uu * self.ctx_side0
        Y = self.ctx_top + vv * self.ctx_side0
        xf2 = (X - self.left) * self.scale
        yf2 = (Y - self.top) * self.scale
        uf, vf = self.fov_to_fovcanvas(xf2, yf2)
        pc = np.stack([uf, vf, uu, vv], 1)
        return np.concatenate([pf, pc], 0).astype(np.float32)


class StateQuestionDataset(Dataset):
    """One item = one FOV state with a set of questions (sampled for training, all for evaluation).
    density: DataFrame of nuclei (fov, cx, cy, super, in_rect, labelled) -> per token count targets [grid*grid, 4] on the
    FOV canvas plus a token mask (inside the annotated rectangle); obj: dict key -> 1,280 d object token per cell question."""

    def __init__(self, questions, fovs, geometry, tokens_path, vocab, q_per_state=32, train=True, use_ctx=True, seed=0, query="op",
                 density=None, obj=None, rects=None):
        self.q = questions.reset_index(drop=True)
        self.groups = {f: g.index.to_numpy() for f, g in self.q.groupby("fov")}
        self.fov_list = sorted(self.groups)
        self.fovs = fovs.set_index("fov")
        self.geo = geometry.set_index("fov")
        self.tokens_path = tokens_path
        self.h5 = None
        with h5py.File(tokens_path, "r") as h:
            names = [n.decode() for n in h["fov"][:]]
            self.size = int(h.attrs["size"])
            self.grid = self.size // int(h.attrs["patch"])
            self.ctx_px = int(h.attrs["ctx_px"])
        self.index_of = {n: i for i, n in enumerate(names)}
        self.vocab = vocab
        self.q_per_state = q_per_state
        self.train = train
        self.use_ctx = use_ctx
        self.rng = np.random.default_rng(seed)
        self.enc = {f: encode_questions(self.q.loc[idx], vocab, query) for f, idx in self.groups.items()}
        self.ops = {f: self.q.loc[idx, "op"].to_numpy() for f, idx in self.groups.items()}
        self.density = {f: g for f, g in density.groupby("fov")} if density is not None else None
        self.rects = rects
        self.obj = obj
        if obj is not None:
            self.obj_keys = {f: [f"{f}|{b}" for b in self.q.loc[idx, "box"]] for f, idx in self.groups.items()}
            self.obj_cell = {f: (self.q.loc[idx, "family"] == "cell").to_numpy() for f, idx in self.groups.items()}

    def __len__(self):
        return len(self.fov_list)

    def _tokens(self, i):
        if self.h5 is None:
            self.h5 = h5py.File(self.tokens_path, "r")
        tf = self.h5["tok_fov"][i]
        tc = self.h5["tok_ctx"][i] if self.use_ctx else None
        return tf, tc

    def sample_questions(self, f):
        idx = self.groups[f]
        if not self.train or len(idx) <= self.q_per_state:
            return np.arange(len(idx))
        ops = self.ops[f]
        uniq = np.unique(ops)
        per = {o: np.flatnonzero(ops == o) for o in uniq}
        chosen = []
        while len(chosen) < self.q_per_state:                 # round robin over operators, random inside
            for o in self.rng.permutation(uniq):
                p = per[o]
                if len(p):
                    chosen.append(int(self.rng.choice(p)))
                if len(chosen) >= self.q_per_state:
                    break
        return np.array(sorted(set(chosen)))

    def __getitem__(self, k):
        f = self.fov_list[k]
        i = self.index_of[f]
        tf, tc = self._tokens(i)
        g = FovGeometry(self.fovs.loc[f], self.geo.loc[f], self.size, self.ctx_px)
        pos = g.token_positions(self.grid)
        sel = self.sample_questions(f)
        e = self.enc[f]
        box = np.stack([g.box_features(b) for b in e["box"][sel]])
        box2 = np.stack([g.box_features(b) for b in e["box2"][sel]])
        if self.density is not None:
            G_ = self.grid; cnt = np.zeros((G_ * G_, 4), np.float32); nd = self.density.get(f)
            if nd is not None:
                nd = nd[nd["in_rect"] & nd["labelled"]]
                u, v = g.fov_to_fovcanvas(nd["cx"].to_numpy(), nd["cy"].to_numpy()); ti = np.clip((v * G_).astype(int), 0, G_ - 1); tj = np.clip((u * G_).astype(int), 0, G_ - 1)
                from opsis.data.nucls import SUPER
                for c, sup in enumerate(SUPER):
                    m = (nd["super"].to_numpy() == sup); np.add.at(cnt[:, c], ti[m] * G_ + tj[m], 1.0)
            x0, y0, x1, y1 = self.rects[f]; u0, v0 = g.fov_to_fovcanvas(x0, y0); u1, v1 = g.fov_to_fovcanvas(x1, y1)
            cc = (np.arange(G_) + 0.5) / G_; vv, uu = np.meshgrid(cc, cc, indexing="ij"); dmask = ((uu >= u0) & (uu < u1) & (vv >= v0) & (vv < v1)).ravel()
        item = dict(tok_fov=torch.from_numpy(tf.astype(np.float16)), pos=torch.from_numpy(pos),
                    op=torch.from_numpy(e["op"][sel]), key=torch.from_numpy(e["key"][sel]), args=torch.from_numpy(e["args"][sel]), cands=torch.from_numpy(e["cands"][sel]),
                    cand_mask=torch.from_numpy(e["cand_mask"][sel]), answer=torch.from_numpy(e["answer"][sel]),
                    box=torch.from_numpy(box), box2=torch.from_numpy(box2), has_box2=torch.from_numpy(e["has_box2"][sel]),
                    text_id=torch.from_numpy(e["text_id"][sel]), cand_text=torch.from_numpy(e["cand_text"][sel]), atype=torch.from_numpy(e["atype"][sel]),
                    qidx=torch.from_numpy(self.groups[f][sel]), fov_index=k)
        if self.use_ctx:
            item["tok_ctx"] = torch.from_numpy(tc.astype(np.float16))
        if self.density is not None:
            item["dens_target"] = torch.from_numpy(cnt); item["dens_mask"] = torch.from_numpy(dmask)
        if self.obj is not None:
            keys = [self.obj_keys[f][j] for j in sel]; cell = self.obj_cell[f][sel]
            ov = np.zeros((len(sel), 1280), np.float16)
            for j, (kk, isc) in enumerate(zip(keys, cell)):
                if isc and kk in self.obj: ov[j] = self.obj[kk]
            item["obj"] = torch.from_numpy(ov); item["has_obj"] = torch.from_numpy(cell.astype(np.float32))
        return item


def collate(items):
    """Pads the question dimension; returns tensors [B, ...] and a question mask [B, Q]."""
    B = len(items)
    Q = max(len(it["op"]) for it in items)
    Ns = [it["tok_fov"].shape[0] for it in items]
    if len(set(Ns)) == 1:
        out = {"tok_fov": torch.stack([it["tok_fov"] for it in items]), "pos": torch.stack([it["pos"] for it in items])}
    else:                                                                 # full bag slides: pad the token dimension and mask the padding
        N = max(Ns); out = {"tok_fov": torch.zeros(B, N, items[0]["tok_fov"].shape[1], dtype=items[0]["tok_fov"].dtype), "pos": torch.zeros(B, N, items[0]["pos"].shape[1]), "mem_mask": torch.ones(B, N, dtype=torch.bool)}
        for b, it in enumerate(items):
            n = it["tok_fov"].shape[0]; out["tok_fov"][b, :n] = it["tok_fov"]; out["pos"][b, :n] = it["pos"][:n]; out["mem_mask"][b, :n] = False
    if "tok_ctx" in items[0]:
        out["tok_ctx"] = torch.stack([it["tok_ctx"] for it in items])
    qmask = torch.zeros(B, Q, dtype=torch.bool)
    for key, shape, dt in (("op", (), torch.long), ("key", (), torch.long), ("args", (MAX_ARGS,), torch.long), ("cands", (MAX_CANDS,), torch.long),
                           ("cand_mask", (MAX_CANDS,), torch.bool), ("answer", (), torch.long), ("box", (8,), torch.float32),
                           ("box2", (8,), torch.float32), ("has_box2", (), torch.float32), ("qidx", (), torch.long),
                           ("text_id", (), torch.long), ("cand_text", (MAX_CANDS,), torch.long), ("atype", (), torch.long)):
        t = torch.zeros((B, Q) + shape, dtype=dt)
        for b, it in enumerate(items):
            n = len(it[key])
            t[b, :n] = it[key]
            qmask[b, :n] = True
        out[key] = t
    out["qmask"] = qmask
    out["fov_index"] = torch.tensor([it["fov_index"] for it in items])
    if "dens_target" in items[0]:
        out["dens_target"] = torch.stack([it["dens_target"] for it in items]); out["dens_mask"] = torch.stack([it["dens_mask"] for it in items])
    if "obj" in items[0]:
        out["obj"] = torch.zeros(B, Q, 1280, dtype=torch.float16); out["has_obj"] = torch.zeros(B, Q)
        for b, it in enumerate(items):
            n = len(it["op"]); out["obj"][b, :n] = it["obj"]; out["has_obj"][b, :n] = it["has_obj"]
    return out


def load_tables(split_seed=0, enc="virchow2", size=448, qset="questions"):
    """qset="questions": the NuCLS set (derived/questions + derived/facts/fovs.csv); any other name: derived/<qset>/{questions_seed<k>.parquet, fovs.csv}."""
    Q = pd.read_parquet(f"{paths.DERIVED}/{qset}/questions_seed{split_seed}.parquet")
    F = pd.read_csv(f"{paths.DERIVED}/facts/fovs.csv" if qset == "questions" else f"{paths.DERIVED}/{qset}/fovs.csv")
    G = pd.read_csv(f"{paths.FEATURES}/{enc}/geometry_{size}.csv")
    return Q, F, G, f"{paths.FEATURES}/{enc}/tokens_{size}.h5"
