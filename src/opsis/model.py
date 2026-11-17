"""The question decoder of Opsis-S (OpsisNet): typed questions answered over one shared state.

State adapter: frozen encoder tokens of the FOV canvas and the context canvas -> linear -> + source and position
embeddings -> transformer encoder -> memory. Question branch: a short query sequence (operator, arguments, target box,
candidates) that self-attends only within its own question (block diagonal mask) and cross-attends to the memory of
its state. Every candidate token yields one logit; the answer distribution is the softmax over the candidates.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from opsis.dataset import MAX_ARGS, MAX_CANDS


class Fourier(nn.Module):
    def __init__(self, n_in, d, n_freq=8):
        super().__init__()
        self.register_buffer("freq", (2.0 ** torch.arange(n_freq)) * math.pi)
        self.mlp = nn.Sequential(nn.Linear(n_in * (2 * n_freq + 1), d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        xf = x.unsqueeze(-1) * self.freq
        feats = torch.cat([x, torch.sin(xf).flatten(-2), torch.cos(xf).flatten(-2)], -1)
        return self.mlp(feats)


class StateAdapter(nn.Module):
    def __init__(self, d_in=1280, d=384, layers=2, heads=8, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.src = nn.Embedding(2, d)
        self.pos = Fourier(4, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers) if layers > 0 else None      # 0 layers: projection and positions only (full bag slides)
        self.norm = nn.LayerNorm(d)

    def forward(self, tok_fov, tok_ctx, pos, extra=None, extra_mask=None, pad_mask=None):
        """extra: [B, E, d] already projected tokens appended to the state (views); extra_mask: [B, E] True = pad;
        pad_mask: [B, N] True where tok_fov is padding (full bag slides of different sizes)."""
        B, N, _ = tok_fov.shape
        x = self.proj(tok_fov.float()) + self.src.weight[0]
        if tok_ctx is not None:
            y = self.proj(tok_ctx.float()) + self.src.weight[1]
            x = torch.cat([x, y], 1)
        else:
            pos = pos[:, :N]
        x = x + self.pos(pos)
        mask = None
        if pad_mask is not None:
            mask = pad_mask if tok_ctx is None else torch.cat([pad_mask, torch.zeros(B, x.shape[1] - pad_mask.shape[1], dtype=torch.bool, device=x.device)], 1)
        if extra is not None:
            x = torch.cat([x, extra], 1)
            base = mask if mask is not None else torch.zeros(B, x.shape[1] - extra.shape[1], dtype=torch.bool, device=x.device)
            mask = torch.cat([base, extra_mask], 1)
        return self.norm(self.enc(x, src_key_padding_mask=mask) if self.enc is not None else x), mask


def box_pool(mem, pos, box):
    """Mean of the state tokens whose centre lies inside the box: mem [B, M, d], pos [B, M, 4] (u_fov, v_fov first),
    box [B, Q, 8] (FOV canvas corners first) -> [B, Q, d]; the global mean when no token falls inside."""
    u = pos[:, None, :, 0]
    v = pos[:, None, :, 1]
    x0, y0, x1, y1 = (box[:, :, i, None] for i in range(4))
    inside = ((u >= x0) & (u < x1) & (v >= y0) & (v < y1)).to(mem.dtype)
    n = inside.sum(-1, keepdim=True)
    pooled = torch.einsum("bqm,bmd->bqd", inside, mem) / n.clamp(min=1)
    glob = mem.mean(1, keepdim=True).expand(-1, box.shape[1], -1)
    return torch.where(n > 0, pooled, glob)


class QuestionDecoder(nn.Module):
    T = 6 + MAX_ARGS + MAX_CANDS      # op, box, box2, pooled box, pooled box2, object, args..., cands...

    def __init__(self, n_ops, n_args, n_cands, d=384, layers=2, heads=8, dropout=0.1, pool=False, text=None, text_trainable=False):
        """text: None, or dict(E_text [T, 768], E_cand [C, 768], mode in {"text", "text+op"}): frozen (or trainable) SapBERT
        vectors of the question text and of each candidate string replace (text) or accompany (text+op) the symbolic tokens."""
        super().__init__()
        self.pool = pool
        self.text_mode = text["mode"] if text else None
        if text:
            Et, Ec = torch.as_tensor(text["E_text"], dtype=torch.float32), torch.as_tensor(text["E_cand"], dtype=torch.float32)
            if text_trainable:
                self.E_text, self.E_cand = nn.Parameter(Et), nn.Parameter(Ec)
            else:
                self.register_buffer("E_text", Et); self.register_buffer("E_cand", Ec)
            self.text_proj = nn.Sequential(nn.LayerNorm(Et.shape[1]), nn.Linear(Et.shape[1], d))
            self.cand_text_proj = nn.Sequential(nn.LayerNorm(Ec.shape[1]), nn.Linear(Ec.shape[1], d))
        self.op = nn.Embedding(n_ops, d)
        self.arg = nn.Embedding(n_args + 1, d, padding_idx=0)
        self.cand = nn.Embedding(n_cands + 1, d, padding_idx=0)
        self.box = Fourier(8, d)
        self.box2 = Fourier(8, d)
        self.slot = nn.Embedding(self.T, d)
        self.obj_proj = nn.Sequential(nn.LayerNorm(1280), nn.Linear(1280, d))
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def queries(self, b, memory=None, pos=None):
        """[B, Q, T, d] query tokens and [B, Q, T] key padding mask (True = pad)."""
        op = self.op(b["op"]).unsqueeze(2)
        if self.text_mode:
            t = self.text_proj(self.E_text[b["text_id"]]).unsqueeze(2)
            op = t if self.text_mode == "text" else op + t
        box = self.box(b["box"]).unsqueeze(2)
        box2 = (self.box2(b["box2"]) * b["has_box2"].unsqueeze(-1)).unsqueeze(2)
        if self.pool:
            p1 = box_pool(memory, pos, b["box"]).unsqueeze(2)
            p2 = (box_pool(memory, pos, b["box2"]) * b["has_box2"].unsqueeze(-1)).unsqueeze(2)
        else:
            p1 = torch.zeros_like(box)
            p2 = torch.zeros_like(box)
        args = self.arg(b["args"])
        cands = self.cand(b["cands"])
        if self.text_mode:
            ct = self.cand_text_proj(self.E_cand[b["cand_text"]])
            cands = ct if self.text_mode == "text" else cands + ct
        if "obj" in b:
            ob = (self.obj_proj(b["obj"].float()) * b["has_obj"].unsqueeze(-1)).unsqueeze(2); obj_pad = (b["has_obj"] == 0).unsqueeze(2)
        else:
            ob = torch.zeros_like(box); obj_pad = torch.ones_like(b["op"], dtype=torch.bool).unsqueeze(2)
        x = torch.cat([op, box, box2, p1, p2, ob, args, cands], 2) + self.slot.weight
        fixed = torch.zeros_like(b["op"], dtype=torch.bool).unsqueeze(2)
        pad = torch.cat([fixed.expand(-1, -1, 3), (~fixed if not self.pool else fixed).expand(-1, -1, 2), obj_pad,
                         b["args"] == 0, ~b["cand_mask"]], 2)
        return x, pad

    def forward(self, memory, b, pos=None, memory_mask=None):
        x, pad = self.queries(b, memory, pos)
        B, Q, T, d = x.shape
        x = x.reshape(B, Q * T, d)
        pad = pad.reshape(B, Q * T)
        qblock = torch.arange(Q, device=x.device).repeat_interleave(T)
        attn_mask = qblock.unsqueeze(0) != qblock.unsqueeze(1)             # True = blocked: questions never see each other
        # padded question slots keep their op/box tokens as keys inside their own block (no all-masked rows), and are
        # dropped from the loss by qmask; they can never influence a valid question.
        h = self.dec(x, memory, tgt_mask=attn_mask, tgt_key_padding_mask=pad, memory_key_padding_mask=memory_mask)
        h = self.norm(h).reshape(B, Q, T, d)
        logits = self.head(h[:, :, 6 + MAX_ARGS:]).squeeze(-1)            # [B, Q, MAX_CANDS]
        return logits.masked_fill(~b["cand_mask"], -1e4)


class OpsisNet(nn.Module):
    def __init__(self, n_ops, n_args, n_cands, d=384, enc_layers=2, dec_layers=2, heads=8, dropout=0.1, d_in=1280, pool=False, text=None, text_trainable=False, density=False):
        super().__init__()
        self.adapter = StateAdapter(d_in, d, enc_layers, heads, dropout)
        self.decoder = QuestionDecoder(n_ops, n_args, n_cands, d, dec_layers, heads, dropout, pool=pool, text=text, text_trainable=text_trainable)
        self.density_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4)) if density else None   # per token nucleus counts per super class (log rate)

    def encode(self, b, use_ctx=True):
        return self.adapter(b["tok_fov"], b.get("tok_ctx") if use_ctx else None, b["pos"], pad_mask=b.get("mem_mask"))[0]

    def forward(self, b, use_ctx=True, return_density=False):
        mem = self.encode(b, use_ctx)
        pos = b["pos"] if use_ctx else b["pos"][:, : mem.shape[1]]
        logits = self.decoder(mem, b, pos, memory_mask=b.get("mem_mask"))
        if return_density and self.density_head is not None:
            N = b["tok_fov"].shape[1]
            return logits, self.density_head(mem[:, :N])            # [B, N, 4] log rates on the FOV canvas tokens
        return logits


def density_loss(log_rate, b):
    """Poisson negative log likelihood of the per token counts inside the annotated rectangle."""
    rate = torch.exp(log_rate.float()).clamp(max=50.0); y = b["dens_target"].float(); m = b["dens_mask"].float().unsqueeze(-1)
    nll = (rate - y * log_rate.float()) * m
    return nll.sum() / m.sum().clamp(min=1) / 4


def question_loss(logits, b, ordinal_tau=0.0):
    """Mean cross entropy over valid questions. ordinal_tau > 0: ordinal questions get a soft target
    softmax(-|k - y| / tau) over their candidates instead of a one hot target."""
    m = b["qmask"]
    if m.sum() == 0:
        return logits.sum() * 0.0
    lg, y = logits[m], b["answer"][m]
    if ordinal_tau <= 0 or "atype" not in b:
        return F.cross_entropy(lg, y)
    K = lg.shape[1]
    dist = (torch.arange(K, device=lg.device)[None] - y[:, None]).abs().float()
    soft = torch.softmax(-dist / ordinal_tau, 1) * b["cand_mask"][m].float()
    soft = soft / soft.sum(1, keepdim=True).clamp(min=1e-9)
    hard = F.one_hot(y, K).float()
    is_ord = (b["atype"][m] == 2).float()[:, None]
    target = is_ord * soft + (1 - is_ord) * hard
    return -(target * F.log_softmax(lg, 1)).sum(1).mean()


class FixedHeadNet(nn.Module):
    """Control: the same state adapter, but no question conditioned attention. Each (operator, arguments) pair owns a
    linear head over [mean of the adapted tokens inside the target box, global mean, mean inside the second box]."""

    def __init__(self, n_keys, d=384, enc_layers=2, heads=8, dropout=0.1, d_in=1280):
        super().__init__()
        self.adapter = StateAdapter(d_in, d, enc_layers, heads, dropout)
        self.W = nn.Parameter(torch.randn(n_keys, 3 * d, MAX_CANDS) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_keys, MAX_CANDS))

    @staticmethod
    def _pool(mem, pos, box):
        return box_pool(mem, pos, box)

    def encode(self, b, use_ctx=True):
        return self.adapter(b["tok_fov"], b.get("tok_ctx") if use_ctx else None, b["pos"])[0]

    def forward(self, b, use_ctx=True):
        mem = self.encode(b, use_ctx)
        pos = b["pos"] if use_ctx else b["pos"][:, : mem.shape[1]]
        f1 = self._pool(mem, pos, b["box"])
        f2 = self._pool(mem, pos, b["box2"]) * b["has_box2"].unsqueeze(-1)
        feat = torch.cat([f1, mem.mean(1, keepdim=True).expand(-1, f1.shape[1], -1), f2], -1)      # [B, Q, 3d]
        W = self.W[b["key"]]                                                                         # [B, Q, 3d, K]
        logits = torch.einsum("bqd,bqdk->bqk", feat, W) + self.b[b["key"]]
        return logits.masked_fill(~b["cand_mask"], -1e4)


class SlideViewNet(nn.Module):
    """Slide state (K tile features, 2,560 d) plus a variable number of high power views, each a set of pooled Virchow2
    tokens (1,280 d) with the view's slide position and a source id (0 = 112 um view, 1 = its 448 um context). The views
    are projected separately, appended to the state and masked where padded; the question decoder is unchanged."""

    def __init__(self, n_ops, n_args, n_cands, d=384, enc_layers=2, dec_layers=2, heads=8, dropout=0.1, d_in=2560, d_view=1280):
        super().__init__()
        self.adapter = StateAdapter(d_in, d, enc_layers, heads, dropout)
        self.view_proj = nn.Linear(d_view, d)
        self.view_src = nn.Embedding(2, d)
        self.view_slot = nn.Embedding(64, d)                 # position of a pooled token inside its 8 x 8 view grid
        self.decoder = QuestionDecoder(n_ops, n_args, n_cands, d, dec_layers, heads, dropout, pool=False)

    def forward(self, b, use_ctx=False):
        extra = extra_mask = None
        if "tok_view" in b and b["tok_view"].shape[1] > 0:
            tv = b["tok_view"]                               # [B, E, 1280]
            extra = self.view_proj(tv.float()) + self.view_src(b["src_view"]) + self.adapter.pos(b["pos_view"]) + self.view_slot(b["slot_view"])
            extra_mask = b["view_mask"]                      # True = pad
        mem, mask = self.adapter(b["tok_fov"], None, b["pos"], extra, extra_mask)
        return self.decoder(mem, b, None, memory_mask=mask)


def load_compat(model, state_dict):
    """Load an older checkpoint: inserts the object query slot (zero row at index 5) and pads candidate slots / fixed head outputs when MAX_CANDS grew."""
    key = "decoder.slot.weight"
    if key in state_dict and key in model.state_dict():
        w = state_dict[key]; want = model.state_dict()[key].shape[0]
        if w.shape[0] == want - 1 or w.shape[0] == want - 1 - (MAX_CANDS - 7):                         # no object slot yet
            w = torch.cat([w[:5], torch.zeros(1, w.shape[1], dtype=w.dtype), w[5:]], 0)
        if w.shape[0] < want:                                                                          # fewer candidate slots (MAX_CANDS was 7)
            w = torch.cat([w, torch.zeros(want - w.shape[0], w.shape[1], dtype=w.dtype)], 0)
        state_dict[key] = w
    for key in ("W", "b"):                                                                             # FixedHeadNet: candidate axis last
        if key in state_dict and key in model.state_dict() and state_dict[key].shape[-1] < model.state_dict()[key].shape[-1]:
            w = state_dict[key]; pad = model.state_dict()[key].shape[-1] - w.shape[-1]
            state_dict[key] = torch.cat([w, torch.zeros(*w.shape[:-1], pad, dtype=w.dtype)], -1)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [m for m in missing if not m.startswith("decoder.obj_proj")]
    assert not missing and not unexpected, (missing, unexpected)
    return model
