"""Project paths, set through environment variables (see env/opsis_env.example.sh).

OPSIS_DATA      data root: raw/ (NuCLS, BCSS, PanNuke, CAMELYON16), derived/ (facts, question sets), features/ (cached Virchow2
                tokens and tile vectors), runs/ (checkpoints and predictions). Default: <repo>/data.
OPSIS_WSI       TCGA diagnostic slides (.svs) as downloaded from the GDC. Default: $OPSIS_DATA/tcga_wsi.
OPSIS_TCGA      TCGA resources: TRIDENT tile features (gigapath_h5/<project>_trident_virchow2/h5_files/<slide>.h5), the
                TCGA-Reports corpus (reports/TCGA_Reports.csv) and the clinical tables. Default: $OPSIS_DATA/tcga.
OPSIS_CAMELYON  CAMELYON16 slides and reference annotations. Default: $OPSIS_DATA/raw/camelyon16.
OPSIS_MODELS    local weights of the comparables (Qwen3-VL-8B-Instruct, Patho-R1-7B, TITAN, PRISM2, CellViT-plus-plus).
                Default: <repo>/checkpoints.
OPSIS_THIRD_PARTY  cloned code of the comparables (CLAM, CellViT-plus-plus). Default: <repo>/third_party.
OPSIS_ENCODERS  local encoder weights, one folder per encoder (Virchow2/config.json and pytorch_model.bin from the gated
                Hugging Face repository paige-ai/Virchow2). Default: <repo>/checkpoints/encoders.
"""
import os

ROOT = os.environ.get("OPSIS_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
DATA = os.environ.get("OPSIS_DATA", f"{ROOT}/data")
WSI = os.environ.get("OPSIS_WSI", f"{DATA}/tcga_wsi")
ENCODERS = os.environ.get("OPSIS_ENCODERS", f"{ROOT}/checkpoints/encoders")
TCGA = os.environ.get("OPSIS_TCGA", f"{DATA}/tcga")
CAMELYON = os.environ.get("OPSIS_CAMELYON", f"{DATA}/raw/camelyon16")
MODELS = os.environ.get("OPSIS_MODELS", f"{ROOT}/checkpoints")
THIRD_PARTY = os.environ.get("OPSIS_THIRD_PARTY", f"{ROOT}/third_party")

RAW = f"{DATA}/raw"
DERIVED = f"{DATA}/derived"
FEATURES = f"{DATA}/features"
RUNS = f"{DATA}/runs"
NUCLS = f"{RAW}/nucls/NuCLS_dataset"
TIGER_ROI = f"{RAW}/tiger/wsirois/roi-level-annotations"
BCSS = f"{RAW}/bcss"
RESULTS = f"{ROOT}/results"  # runs/, collected/, organiser/: written by the scripts, not tracked
for _d in ("runs", "collected", "organiser"):
    os.makedirs(f"{RESULTS}/{_d}", exist_ok=True)


def ensure(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
