"""Frozen tile encoders from the local checkpoint store ($OPSIS_ENCODERS, no network). Virchow2."""
import json
import os

import timm
import torch

from opsis import paths

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_virchow2(device="cuda", img_size=224):
    """Virchow2 (ViT-H/14, 4 register tokens, SwiGLU). Returns the timm model; forward gives [B, 1 + 4 + N, 1280]."""
    d = f"{paths.ENCODERS}/Virchow2"
    cfg = json.load(open(f"{d}/config.json"))
    kw = dict(cfg["model_args"])                           # img_size stays 224: dynamic_img_size interpolates pos_embed for 448 inputs
    kw.update(mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, dynamic_img_size=True)
    model = timm.create_model(cfg["architecture"], pretrained=False, **kw)
    sd = torch.load(f"{d}/pytorch_model.bin", map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    model.n_prefix = 1 + kw.get("reg_tokens", 0)
    model.patch = 14
    model.dim = 1280
    return model


@torch.no_grad()
def encode_batch(model, x):
    """x: float tensor [B, 3, H, W] already normalised. Returns (cls [B, D], patch tokens [B, N, D]) in fp16."""
    out = model.forward_features(x)
    return out[:, 0].half(), out[:, model.n_prefix:].half()
