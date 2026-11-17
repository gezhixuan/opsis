"""Question operators. Every question is derived from the fact tables by code; validity rules are explicit.

Record fields: qid, fov, slide, patient, family, op, args, atype, candidates, answer, box, box2, text, derivation.
Boxes are in FOV png pixels (0.2 mpp). atype is choice, noul or ordinal; the answer is an index into candidates
(noul: candidates = [no, yes]).
"""
import hashlib
import json

import numpy as np

from opsis.data import nucls, tissue

SUPER_SHORT = [nucls.SUPER_SHORT[k] for k in nucls.SUPER]            # tumor, stromal, sTIL, other
COUNT_LABELS = ["0", "1-3", "4-10", "11-30", ">30"]
CELL_FRAC_LABELS = ["0", "(0,0.1]", "(0.1,0.3]", "(0.3,0.6]", ">0.6"]
TISSUE_FRAC_LABELS = ["<0.05", "[0.05,0.25)", "[0.25,0.5)", "[0.5,0.75)", ">=0.75"]
AREA_LABELS = ["q1", "q2", "q3", "q4"]
NOUL = ["no", "yes"]
PAIRS = [("tumor", "sTIL"), ("stromal", "sTIL"), ("tumor", "stromal")]
COMPARTMENTS = ["tumor", "stroma"]
CTX_MIN_COVER = 0.5      # context windows often cross the BCSS ROI border; fractions are over the valid area
FAMILY = {"cell_super": "cell", "cell_main": "cell", "cell_is": "cell", "cell_tissue": "cell", "cell_area_bin": "cell",
          "count_bin": "region", "present": "region", "cell_frac_bin": "region", "tissue_present": "region",
          "tissue_frac_bin": "region", "dominant_tissue": "region", "more_cells": "region",
          "compare_regions": "comp", "count_in_compartment_bin": "comp", "tils_location": "comp", "density_compare": "comp",
          # slide level (TCGA-BRCA, scripts/50): report slots and molecular status of the patient, asked on the whole slide
          "histologic_type": "slide", "histologic_grade": "slide", "nuclear_grade": "slide", "in_situ_present": "slide",
          "lvi_present": "slide", "necrosis_present": "slide", "multifocal": "slide", "pni_present": "slide",
          "er_positive": "molecular", "pr_positive": "molecular", "her2_positive": "molecular", "pam50": "molecular",
          "tissue_is": "organ",
          # slide level, report extraction v2 (scripts/52): Nottingham components, DCIS details, other findings, claims
          "histologic_type_fine": "slide", "tubule_score": "slide", "pleomorphism_score": "slide", "mitotic_score": "slide",
          "dcis_type": "slide", "dcis_grade": "slide", "dcis_comedo_necrosis": "slide", "lcis_present": "slide", "calcifications": "slide",
          "tils_description": "slide", "claim_supported": "claim",
          # external slide level multiple choice (SlideBench-VQA TCGA, BRCA rows): free text question and options
          "slidebench_mcq": "external",
          # pan cancer slide level (final version, scripts/66): shared report schema, tables, composite held out operators
          "pan_grade": "slide", "pan_lvi": "slide", "pan_pni": "slide", "pan_necrosis": "slide", "pan_precursor": "slide", "pan_mitotic": "slide",
          "pan_multifocal": "slide", "pan_infiltrate": "slide", "cancer_type": "slide", "molecular_subtype": "molecular",
          "tumor_purity_bin": "molecular", "leukocyte_fraction_bin": "molecular",
          "grade_from_components": "composite", "high_grade_and_lvi": "composite", "necrosis_and_high_mitotic": "composite", "dcis_with_comedo": "composite", "precursor_type": "composite",
          # final FOV extras (scripts/67)
          "stil_score_bin": "region", "mitotic_present": "region", "mitotic_count_bin": "region"}
OPS = list(FAMILY)


def count_bin(n):
    return 0 if n == 0 else 1 if n <= 3 else 2 if n <= 10 else 3 if n <= 30 else 4


def cell_frac_bin(f):
    return 0 if f == 0 else 1 if f <= 0.1 else 2 if f <= 0.3 else 3 if f <= 0.6 else 4


def tissue_frac_bin(f):
    return 0 if f < 0.05 else 1 if f < 0.25 else 2 if f < 0.5 else 3 if f < 0.75 else 4


def area_bin(a, q):
    return int(np.searchsorted(q, a, side="right"))     # q = [q1, q2, q3]


def _rng(name, salt=""):
    h = hashlib.sha1(f"{name}|{salt}".encode()).hexdigest()
    return np.random.default_rng(int(h[:8], 16))


REGION_TEXT = {"whole": "the field of view", "left": "the left half of the field", "right": "the right half of the field",
               "top": "the top half of the field", "bottom": "the bottom half of the field", "q00": "the top left quadrant",
               "q01": "the top right quadrant", "q10": "the bottom left quadrant", "q11": "the bottom right quadrant"}
CLASS_TEXT = {"tumor": "tumour cells", "stromal": "stromal cells", "sTIL": "lymphocytes or plasma cells", "other": "other cells"}
TISSUE_TEXT = {"tumor": "invasive tumour", "stroma": "stroma", "inflamed_stroma": "lymphocytic infiltrate", "in_situ": "in situ carcinoma",
               "necrosis": "necrosis", "other": "other tissue"}


class QuestionBuilder:
    def __init__(self, fov_row, nuc, regions, area_q, n_cells=12, seed_salt="r1"):
        self.f = fov_row
        self.nuc = nuc
        self.reg = {r["region"]: r for r in regions[regions["scope"] == "fov"].to_dict("records")}
        self.ctx = {r["region"]: r for r in regions[regions["scope"] == "ctx"].to_dict("records")}
        self.area_q = area_q
        self.n_cells = n_cells
        self.rng = _rng(fov_row["fov"], seed_salt)
        self.out = []

    # -- helpers ----------------------------------------------------------------------------------------------------
    def emit(self, op, args, atype, candidates, answer, box, text, derivation, box2=None):
        key = f"{self.f['fov']}|{op}|{json.dumps(args, sort_keys=True)}|{box}|{box2}"
        self.out.append(dict(qid=hashlib.sha1(key.encode()).hexdigest()[:16], fov=self.f["fov"], slide=self.f["slide"],
                             patient=self.f["patient"], family=FAMILY[op], op=op, args=json.dumps(args, sort_keys=True),
                             atype=atype, candidates=json.dumps(candidates), answer=int(answer),
                             box=json.dumps([round(float(v), 1) for v in box]), box2=json.dumps([round(float(v), 1) for v in box2]) if box2 else "",
                             text=text, derivation=derivation))

    def _rbox(self, r, scope="fov"):
        d = self.reg if scope == "fov" else self.ctx
        return (d[r]["x0"], d[r]["y0"], d[r]["x1"], d[r]["y1"])

    # -- cell level -------------------------------------------------------------------------------------------------
    def cells(self):
        cand = self.nuc[self.nuc["in_rect"] & self.nuc["labelled"]]
        if len(cand) == 0:
            return
        chosen = []
        for k in nucls.SUPER:                               # stratified: up to 3 per super class, then fill
            idx = cand.index[cand["super"] == k].to_numpy()
            self.rng.shuffle(idx)
            chosen += list(idx[:3])
        rest = [i for i in cand.index if i not in set(chosen)]
        self.rng.shuffle(rest)
        chosen += rest[:max(0, self.n_cells - len(chosen))]
        for i in chosen[:self.n_cells]:
            n = cand.loc[i]
            box = (n["xmin"], n["ymin"], n["xmax"], n["ymax"])
            sup = nucls.SUPER_SHORT[n["super"]]
            self.emit("cell_super", {}, "choice", SUPER_SHORT, SUPER_SHORT.index(sup), box,
                      "Which class of cell is the marked nucleus?", f"super={n['super']}")
            if n["main"] in nucls.MAIN:
                self.emit("cell_main", {}, "choice", nucls.MAIN, nucls.MAIN.index(n["main"]), box,
                          "Which fine class is the marked nucleus?", f"main={n['main']}")
            k = sup if self.rng.random() < 0.5 else str(self.rng.choice([c for c in SUPER_SHORT if c != sup]))
            self.emit("cell_is", {"k": k}, "noul", NOUL, int(k == sup), box,
                      f"Is the marked nucleus one of the {CLASS_TEXT[k]}?", f"super={n['super']} k={k}")
            if n["tissue_code"] != 0:
                tc = n["tissue_class"]
                self.emit("cell_tissue", {}, "choice", tissue.TISSUE_CLASSES, tissue.TISSUE_CLASSES.index(tc), box,
                          "In which tissue compartment does the marked nucleus lie?", f"tissue_code={n['tissue_code']}")
            if n["polygon"] and np.isfinite(n["area_um2"]):
                self.emit("cell_area_bin", {}, "ordinal", AREA_LABELS, area_bin(n["area_um2"], self.area_q), box,
                          "In which size quartile is the area of the marked nucleus?", f"area_um2={n['area_um2']:.1f}")

    # -- region level -----------------------------------------------------------------------------------------------
    def regions(self):
        halves = ["left", "right"] if self.rng.random() < 0.5 else ["top", "bottom"]
        quad = str(self.rng.choice(["q00", "q01", "q10", "q11"]))
        for rname in ["whole"] + halves + [quad]:
            r = self.reg[rname]
            box = self._rbox(rname)
            rt = REGION_TEXT[rname]
            for k in SUPER_SHORT:
                n = int(r[f"n_{k}"])
                self.emit("count_bin", {"k": k, "region": rname}, "ordinal", COUNT_LABELS, count_bin(n), box,
                          f"How many {CLASS_TEXT[k]} are in {rt}?", f"n={n}")
                self.emit("present", {"k": k, "region": rname}, "noul", NOUL, int(n >= 1), box,
                          f"Are there any {CLASS_TEXT[k]} in {rt}?", f"n={n}")
                if r["n_labelled"] >= 5:
                    fr = n / float(r["n_labelled"])
                    self.emit("cell_frac_bin", {"k": k, "region": rname}, "ordinal", CELL_FRAC_LABELS, cell_frac_bin(fr), box,
                              f"What fraction of the cells in {rt} are {CLASS_TEXT[k]}?", f"n={n}/{r['n_labelled']}")
            for k1, k2 in PAIRS:
                n1, n2 = int(r[f"n_{k1}"]), int(r[f"n_{k2}"])
                if abs(n1 - n2) >= 2:
                    self.emit("more_cells", {"k1": k1, "k2": k2, "region": rname}, "choice", [k1, k2], int(n2 > n1), box,
                              f"In {rt}, are there more {CLASS_TEXT[k1]} or more {CLASS_TEXT[k2]}?", f"n1={n1} n2={n2}")
            if r["tissue_valid"] and rname == "whole":            # FOV scale: only the dominant class (BCSS is a low power annotation)
                fr = {c: float(r[f"frac_{c}"]) for c in tissue.TISSUE_CLASSES}
                dom = max(fr, key=fr.get)
                self.emit("dominant_tissue", {"region": rname, "scope": "fov"}, "choice", tissue.TISSUE_CLASSES, tissue.TISSUE_CLASSES.index(dom), box,
                          f"Which tissue compartment covers most of {rt}?", f"frac={fr[dom]:.3f}")
        # context scale (448 um window): tissue composition questions at the scale of the BCSS annotation
        for rname in ["whole"] + halves + [quad]:
            r = self.ctx.get(rname)
            if r is None or not (float(r["valid_frac"]) >= CTX_MIN_COVER):
                continue
            box = self._rbox(rname, "ctx")
            rt = REGION_TEXT[rname].replace("the field of view", "the 448 um context window").replace("of the field", "of the context window") \
                .replace("quadrant", "quadrant of the context window")
            fr = {c: float(r[f"frac_{c}"]) for c in tissue.TISSUE_CLASSES}
            for c in tissue.TISSUE_CLASSES:
                self.emit("tissue_present", {"c": c, "region": rname, "scope": "ctx"}, "noul", NOUL, int(fr[c] >= 0.05), box,
                          f"Does {rt} contain {TISSUE_TEXT[c]} (at least 5 percent of its area)?", f"frac={fr[c]:.3f}")
                self.emit("tissue_frac_bin", {"c": c, "region": rname, "scope": "ctx"}, "ordinal", TISSUE_FRAC_LABELS, tissue_frac_bin(fr[c]), box,
                          f"What fraction of the area of {rt} is {TISSUE_TEXT[c]}?", f"frac={fr[c]:.3f}")
            dom = max(fr, key=fr.get)
            self.emit("dominant_tissue", {"region": rname, "scope": "ctx"}, "choice", tissue.TISSUE_CLASSES, tissue.TISSUE_CLASSES.index(dom), box,
                      f"Which tissue compartment covers most of {rt}?", f"frac={fr[dom]:.3f}")

    # -- compositional ----------------------------------------------------------------------------------------------
    def compositional(self):
        halves = ["left", "right"] if self.rng.random() < 0.5 else ["top", "bottom"]
        ra, rb = self.reg[halves[0]], self.reg[halves[1]]
        for k in SUPER_SHORT:
            na, nb = int(ra[f"n_{k}"]), int(rb[f"n_{k}"])
            if abs(na - nb) >= 3:
                self.emit("compare_regions", {"k": k, "A": halves[0], "B": halves[1]}, "choice", ["A", "B"], int(nb > na),
                          self._rbox(halves[0]), f"Which region has more {CLASS_TEXT[k]}: A ({REGION_TEXT[halves[0]]}) or B ({REGION_TEXT[halves[1]]})?",
                          f"nA={na} nB={nb}", box2=self._rbox(halves[1]))
        w = self.reg["whole"]
        if not w["tissue_valid"]:
            return
        box = self._rbox("whole")
        fc = {"tumor": float(w["frac_comp_tumor"]), "stroma": float(w["frac_comp_stroma"])}
        for comp in COMPARTMENTS:
            if fc[comp] >= 0.1:
                n = int(w[f"n_sTIL_in_{comp}"])
                self.emit("count_in_compartment_bin", {"k": "sTIL", "comp": comp}, "ordinal", COUNT_LABELS, count_bin(n), box,
                          f"How many lymphocytes or plasma cells lie within the {TISSUE_TEXT[comp]} of the field of view?", f"n={n}")
        nt, ns = int(w["n_sTIL_in_tumor"]), int(w["n_sTIL_in_stroma"])
        if fc["tumor"] >= 0.1 and fc["stroma"] >= 0.1 and nt + ns >= 5 and max(nt, ns) / float(nt + ns) >= 0.6:
            self.emit("tils_location", {}, "choice", COMPARTMENTS, int(ns > nt), box,
                      "Are the lymphocytes and plasma cells of the field of view mostly in the invasive tumour or in the stroma?", f"nt={nt} ns={ns}")
        if fc["tumor"] >= 0.1 and fc["stroma"] >= 0.1:
            for k in ["sTIL", "tumor"]:
                dt = int(w[f"n_{k}_in_tumor"]) / (fc["tumor"] * w["area_um2"])
                ds = int(w[f"n_{k}_in_stroma"]) / (fc["stroma"] * w["area_um2"])
                if max(dt, ds) > 0 and max(dt, ds) >= 1.5 * min(dt, ds):
                    self.emit("density_compare", {"k": k}, "choice", COMPARTMENTS, int(ds > dt), box,
                              f"Is the density of {CLASS_TEXT[k]} higher in the invasive tumour or in the stroma of the field of view?",
                              f"dens_t={dt * 1e4:.2f} dens_s={ds * 1e4:.2f} per 1e4 um2")

    def build(self):
        self.cells()
        self.regions()
        self.compositional()
        return self.out
