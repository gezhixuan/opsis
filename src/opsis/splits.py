"""Patient level splits. NuCLS has one slide per patient, so slide and patient coincide."""
import json
import os

import numpy as np

from opsis import paths


def split_patients(patients, seed=0, frac=(0.7, 0.15, 0.15)):
    pts = sorted(set(patients))
    rng = np.random.default_rng(seed)
    rng.shuffle(pts)
    n = len(pts)
    n_tr = int(round(frac[0] * n))
    n_va = int(round(frac[1] * n))
    return {"train": sorted(pts[:n_tr]), "val": sorted(pts[n_tr:n_tr + n_va]), "test": sorted(pts[n_tr + n_va:])}


def write_split(split, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=1)


def read_split(path):
    with open(path) as f:
        s = json.load(f)
    assert not (set(s["train"]) & set(s["test"])) and not (set(s["train"]) & set(s["val"])) and not (set(s["val"]) & set(s["test"]))
    return s


def fold_of(split):
    out = {}
    for k, v in split.items():
        for p in v:
            out[p] = k
    return out
