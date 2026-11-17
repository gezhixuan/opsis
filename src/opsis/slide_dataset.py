"""Slide states for the slide level questions: a random subset of K tile features (Virchow2 20x, 2,560 d) with their
normalised slide coordinates as position features, in the tensor layout that `dataset.collate` expects."""
import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from opsis.dataset import encode_questions, MAX_ARGS, MAX_CANDS


class SlideQuestionDataset(Dataset):
    def __init__(self, questions, h5_dir, vocab, k_tiles=4096, q_per_state=16, train=True, seed=0, query="op", blank=False):
        self.q = questions.reset_index(drop=True)
        self.blank = blank                                                     # control: no slide information (zero tokens)
        self.groups = {f: g.index.to_numpy() for f, g in self.q.groupby("fov")}
        self.slides = sorted(self.groups)
        self.h5_dir = h5_dir
        self.k = k_tiles
        self.q_per_state = q_per_state
        self.train = train
        self.rng = np.random.default_rng(seed)
        self.enc = {f: encode_questions(self.q.loc[idx], vocab, query) for f, idx in self.groups.items()}
        self.h5_path = dict(zip(self.q["slide"], self.q["h5_path"])) if "h5_path" in self.q.columns else None

    def __len__(self):
        return len(self.slides)

    def _tiles(self, slide, k_idx):
        if self.blank:
            return np.zeros((16, 2560), dtype=np.float16), np.zeros((16, 2), dtype=np.float32)
        path = self.h5_path.get(slide) if getattr(self, "h5_path", None) else None     # pan cancer sets carry a per slide h5 path
        with h5py.File(path or f"{self.h5_dir}/{slide}.h5", "r") as h:
            n = h["features"].shape[0]
            rng = self.rng if self.train else np.random.default_rng(k_idx)        # fixed subset at evaluation
            Call = h["coords"][:]
            lo, hi = Call.min(0), Call.max(0)
            if self.k <= 0:                                                    # full bag: every tile (padded and masked in collate)
                X = h["features"][:].astype(np.float16); C = Call.astype(np.float32)
            elif n >= self.k:                                                  # h5 fancy indexing needs sorted unique indices
                idx = np.sort(rng.choice(n, self.k, replace=False))
                X = h["features"][idx].astype(np.float16)
                C = Call[idx].astype(np.float32)
            else:                                                              # small slide: read everything, resample with replacement
                idx = rng.integers(0, n, self.k)
                X = h["features"][:].astype(np.float16)[idx]
                C = Call[idx].astype(np.float32)
        pos = (C - lo) / np.maximum(hi - lo, 1.0)
        return X, pos

    def __getitem__(self, i):
        f = self.slides[i]
        X, uv = self._tiles(f, i)
        pos = np.concatenate([uv, uv], 1).astype(np.float32)                        # (u, v, u, v): one frame only
        idx = self.groups[f]
        sel = np.arange(len(idx)) if (not self.train or len(idx) <= self.q_per_state) else np.sort(self.rng.choice(len(idx), self.q_per_state, replace=False))
        e = self.enc[f]
        box = np.tile(np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.float32), (len(sel), 1))
        return dict(tok_fov=torch.from_numpy(X), pos=torch.from_numpy(pos), op=torch.from_numpy(e["op"][sel]), key=torch.from_numpy(e["key"][sel]),
                    args=torch.from_numpy(e["args"][sel]), cands=torch.from_numpy(e["cands"][sel]), cand_mask=torch.from_numpy(e["cand_mask"][sel]),
                    answer=torch.from_numpy(e["answer"][sel]), box=torch.from_numpy(box), box2=torch.from_numpy(np.zeros_like(box)),
                    has_box2=torch.zeros(len(sel)), text_id=torch.from_numpy(e["text_id"][sel]), cand_text=torch.from_numpy(e["cand_text"][sel]), atype=torch.from_numpy(e["atype"][sel]),
                    qidx=torch.from_numpy(idx[sel]), fov_index=i)


class SlideViewDataset(SlideQuestionDataset):
    """Adds n high power views per slide from the candidate pool (features/fov_pool/<slide>.h5). n_views: an int, or
    (lo, hi) to draw a random count per slide in training. with_ctx adds the 448 um context tokens of each view.
    chooser(slide, n, rng, pool) -> indices into the pool, default uniform without replacement."""

    def __init__(self, questions, h5_dir, pool_dir, vocab, k_tiles=4096, q_per_state=16, train=True, seed=0, query="op",
                 n_views=4, with_ctx=False, chooser=None):
        super().__init__(questions, h5_dir, vocab, k_tiles, q_per_state, train, seed, query)
        self.pool_dir = pool_dir
        self.n_views = n_views
        self.with_ctx = with_ctx
        self.chooser = chooser

    def _views(self, slide, i):
        p = f"{self.pool_dir}/{slide}.h5"
        rng = self.rng if self.train else np.random.default_rng(1000 + i)
        n = self.n_views if isinstance(self.n_views, int) else int(rng.integers(self.n_views[0], self.n_views[1] + 1))
        if n == 0 or not os.path.exists(p):
            return np.zeros((0, 1280), np.float16), np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(0, np.int64)
        with h5py.File(p, "r") as h:
            m = int(h.attrs["n"]); sel = self.chooser(slide, min(n, m), rng, h) if self.chooser else np.sort(rng.choice(m, min(n, m), replace=False))
            tf = h["tok_fov"][sel]                                    # [n, 64, 1280]
            tc = h["tok_ctx"][sel] if self.with_ctx else None
            ti = h["tile_index"][sel]
        with h5py.File(f"{self.h5_dir}/{slide}.h5", "r") as h:
            C = h["coords"][:]; lo, hi = C.min(0), C.max(0); uv = (C[ti] - lo) / np.maximum(hi - lo, 1.0)
        toks, pos, src, slot = [tf], [np.repeat(uv, 64, 0)], [np.zeros(len(sel) * 64, np.int64)], [np.tile(np.arange(64), len(sel))]
        if tc is not None:
            toks.append(tc); pos.append(np.repeat(uv, 64, 0)); src.append(np.ones(len(sel) * 64, np.int64)); slot.append(np.tile(np.arange(64), len(sel)))
        T = np.concatenate([t.reshape(-1, 1280) for t in toks]).astype(np.float16); P = np.concatenate(pos).astype(np.float32)
        return T, np.concatenate([P, P], 1), np.concatenate(src), np.concatenate(slot)

    def __getitem__(self, i):
        it = super().__getitem__(i)
        T, P, S, L = self._views(self.slides[i], i)
        it.update(tok_view=torch.from_numpy(T), pos_view=torch.from_numpy(P), src_view=torch.from_numpy(S), slot_view=torch.from_numpy(L))
        return it


def collate_views(items):
    from opsis.dataset import collate
    out = collate(items)
    E = max(it["tok_view"].shape[0] for it in items); B = len(items)
    out["tok_view"] = torch.zeros(B, E, 1280, dtype=torch.float16); out["pos_view"] = torch.zeros(B, E, 4); out["src_view"] = torch.zeros(B, E, dtype=torch.long)
    out["slot_view"] = torch.zeros(B, E, dtype=torch.long); out["view_mask"] = torch.ones(B, E, dtype=torch.bool)
    for b, it in enumerate(items):
        n = it["tok_view"].shape[0]
        if n:
            out["tok_view"][b, :n] = it["tok_view"]; out["pos_view"][b, :n] = it["pos_view"]; out["src_view"][b, :n] = it["src_view"]; out["slot_view"][b, :n] = it["slot_view"]; out["view_mask"][b, :n] = False
    return out
