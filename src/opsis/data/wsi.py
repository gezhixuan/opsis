"""Context crops from the TCGA diagnostic slides in $OPSIS_WSI (read only)."""
import glob
import os
import re

import numpy as np
from PIL import Image

from opsis import paths

SLIDE_RE = re.compile(r"^(TCGA-\w\w-\w{4})-01Z-00-(DX\d)\.")


def slide_index(project="TCGA-BRCA", root=paths.WSI):
    """slide short name (TCGA-XX-XXXX-DXn) -> svs path."""
    out = {}
    for p in sorted(glob.glob(f"{root}/{project}/*/*.svs")):
        m = SLIDE_RE.match(os.path.basename(p))
        if m:
            out.setdefault(f"{m.group(1)}-{m.group(2)}", p)
    return out


def resolve_mpp(sl):
    """openslide.mpp-x, else aperio.MPP, else 0.25 for a 40x scan (AppMag), else nan."""
    pr = sl.properties
    for k in ("openslide.mpp-x", "aperio.MPP"):
        try:
            v = float(pr.get(k, "nan"))
            if np.isfinite(v) and v > 0:
                return v
        except ValueError:
            pass
    mag = pr.get("aperio.AppMag") or pr.get("openslide.objective-power")
    if mag:
        try:
            return 10.0 / float(mag)            # 40x -> 0.25, 20x -> 0.5
        except ValueError:
            pass
    return float("nan")


class SlideReader:
    def __init__(self, path):
        import openslide
        self.sl = openslide.OpenSlide(path)
        self.mpp = resolve_mpp(self.sl)
        self.dims = self.sl.dimensions

    def read_context(self, cx, cy, mpp_target, size_px):
        """A size_px x size_px RGB crop at mpp_target centred on level-0 point (cx, cy); outside the slide is white."""
        scale = mpp_target / self.mpp                    # level-0 px per output px
        side0 = int(round(size_px * scale))
        x0 = int(round(cx - side0 / 2.0))
        y0 = int(round(cy - side0 / 2.0))
        lvl = self.sl.get_best_level_for_downsample(scale)
        ds = self.sl.level_downsamples[lvl]
        side_l = int(np.ceil(side0 / ds))
        reg = self.sl.read_region((x0, y0), lvl, (side_l, side_l)).convert("RGB")
        return reg.resize((size_px, size_px), Image.BICUBIC)

    def read_level0(self, x0, y0, w, h):
        return self.sl.read_region((x0, y0), 0, (w, h)).convert("RGB")
