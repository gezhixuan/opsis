# Opsis

Code for **One state, many questions: Opsis gives a frontier model typed eyes for pathology**
(Zhixuan Ge and Wei Qiu, Rice University). arXiv: to be added.

Opsis asks pathology images typed questions: an operator, its arguments, a box where the question is about a region, and a fixed
list of candidate answers. The answer is a probability over the candidates. A field or a slide is encoded once into a state, and
any number of questions are answered from it in one pass.

- **Opsis-S** is a question decoder of 10 M parameters on frozen Virchow2 tokens of a field with its context canvas, or on 4,096
  sampled 20x tile vectors of a slide, with per tile heads for findings that live in a few tiles.
- **Opsis-V** is the same interface inside Qwen3-VL-8B (LoRA), which reads the field and its context as pixels with the canvases
  as a shared prefix.
- **The organiser** is a frontier language model that answers slide questions, including composite questions Opsis was never
  trained on, by calling Opsis tools (`slide_info`, `ask_slide`, `tile_evidence`, `ask_region`) through an MCP server.

## What is released

| | where | licence |
|---|---|---|
| code | this repository | MIT |
| question sets, audited report fields, organiser episodes | Hugging Face dataset `gezx1004/opsis-questions` | per source, see its card |
| Opsis-V adapter (one seed, chosen by validation NLL) | Hugging Face model `gezx1004/opsis-v` | Apache 2.0 |

Opsis-S weights and the cached Virchow2 features are not released: the Virchow2 licence (CC-BY-NC-ND 4.0) counts models trained
on its outputs as derivatives. With access to Virchow2 they are retrained with the scripts below (a field decoder trains in about
12 minutes on one H200, a slide decoder in 3.6 hours).

## Setup

Three Python environments were used (`env/requirements*.txt`): the main one, one for Opsis-V and the local vision language
models (transformers 5), and one for TITAN and PRISM2 (the PRISM2 remote code runs with transformers 4.48). Copy
`env/opsis_env.example.sh` to `env/opsis_env.sh`, set the paths and `source env/opsis_env.sh` before running anything.

| variable | content |
|---|---|
| `OPSIS_DATA` | `raw/` (NuCLS, BCSS, PanNuke), `derived/` (facts and question sets), `features/` (Virchow2 tokens), `runs/` (checkpoints, predictions) |
| `OPSIS_WSI` | TCGA diagnostic slides, downloaded by `scripts/data/104_tcga_wsi_manifest.py` and `scripts/data/105_tcga_wsi_download.py` from the GDC |
| `OPSIS_TCGA` | TRIDENT tile features (`gigapath_h5/<project>_trident_virchow2/h5_files/<slide>.h5`), `reports/TCGA_Reports.csv`, the clinical tables |
| `OPSIS_ENCODERS` | `Virchow2/` (gated, `paige-ai/Virchow2` on Hugging Face) |
| `OPSIS_MODELS` | Qwen3-VL-8B-Instruct, Patho-R1-7B, TITAN, PRISM2, CellViT-plus-plus checkpoints |
| `OPSIS_THIRD_PARTY` | clones of CLAM (commit 53e2409) and CellViT-plus-plus (commit 463c5c4) |
| `OPSIS_CAMELYON` | CAMELYON16 slides and reference annotations |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | only for the report extraction, the organiser and the frontier model comparables |

Data sources: NuCLS and BCSS (CC0), PanNuke (CC BY-NC-SA 4.0), TCGA diagnostic slides and clinical tables (GDC open access),
TCGA-Reports (Kefeli et al., CC BY 4.0), CAMELYON16. The slide tile features are TRIDENT 0.3.2 outputs (HEST tissue
segmentation, 20x, 256 px tiles, Virchow2, class token concatenated with the mean patch token, 2,560 dimensions). A new
extraction gives slightly different vectors than ours.

The question sets can be rebuilt from these sources with the scripts below, or taken from the dataset release:

```bash
huggingface-cli download gezx1004/opsis-questions --repo-type dataset --local-dir opsis-questions
python release/install_questions.py opsis-questions      # into $OPSIS_DATA/derived with the names the code reads
```

The install also copies the organiser episodes of the paper into `results/organiser/`, so the organiser tables (`collect.py organiser`,
`collect.py ways_to_read`) are recomputed from them without running a frontier model.

`release/build_release.py` is the script that assembled both Hugging Face releases from our data root.

## Pipeline

`scripts/` has one folder per stage. Within a folder the number gives the order in which the scripts ran. A long job has its SLURM
wrapper next to it, which takes the arguments through `ARGS` and is submitted from the repository root
(`sbatch --export=ALL,ARGS="--name final_fov_s0 ..." scripts/opsis_s/20_train.sbatch`).

| folder | content |
|---|---|
| `data` | BCSS masks (`01`), facts of the NuCLS and BCSS fields (`10`), context canvases (`12`), Virchow2 tokens of fields and canvases (`13`, `81` for PanNuke), operator text embeddings for one ablation (`70`), TCGA slides from the GDC (`104`, `105`) |
| `questions` | NuCLS field questions and patient split (`11`, `67`), PanNuke (`80`), breast slide questions (`50`, `52`, `53`), report field extraction with verbatim quotes (`54`, `55`), pan cancer slide questions and split (`66`), audit agreement (`68`) |
| `opsis_s` | field decoder (`20`), unfrozen encoder ablation (`21`), slide decoder (`51`), per tile heads (`69`), encoder and slide question timing (`92`, `111`) |
| `opsis_v` | LoRA training and evaluation (`40`, also the untrained Qwen3-VL-8B and Patho-R1-7B with `--eval-only`), Opsis-V reading the Virchow2 state (`41_train_vlm_state`), shared prefix timing (`41_vlm_shared_prefix`) |
| `organiser` | MCP tool server (`80`, tools in `src/opsis/tools.py`), one episode per question through the OpenAI API, Codex CLI or Claude Code CLI (`81`, `81b` for the pan cancer test split in shards), rules condition (`81d`), reading conditions (`106`) |
| `baselines` | ABMIL and CLAM-SB (`57`, `57b_abmil.sbatch` over many tasks), PRISM2 (`58`, `83`, `85`, `87`), TITAN (`84`, `88`), CellViT++ (`59`), vision language models looking at slides and fields (`100`), TITAN and PRISM2 on CAMELYON16 (`101`), Virchow2 linear probes (`102`) |
| `transfer` | tile maps against BCSS necrosis (`71`), CAMELYON16 (`89`, `90`, `91`) |
| `collect` | `collect.py <table>` writes the result tables to `results/collected/` (`python scripts/collect/collect.py` lists them) |

Training and evaluation write checkpoints and predictions to `$OPSIS_DATA/runs/<run>/` and a copy of the metrics to
`results/runs/<run>/`, which the collectors read.

Run names used in the paper: `final_fov_s{0,1,2}` (Opsis-S fields), `final_fov_organ_s0` (PanNuke organ split),
`final_slide_pan_s{0,1,2}` (Opsis-S pan cancer slides), `jev_v2_op_s{0,1,2}` (Opsis-S breast slides), `final_tile_{noisyor,lse}_s{0,1,2}`
and `final_tile_attn_s0` (per tile heads), `final_vlm_s{0,1,2}` (Opsis-V), `abl_*` (ablations), `abmil_*`, `clam_*`, `titan_*`,
`prism2_*`, `virchow2_*_probe`, `zs_*`, `cellvit_nucls*`, `camelyon_*` (comparables and transfer).

## Citation

```bibtex
@article{ge2026opsis,
  title   = {One state, many questions: Opsis gives a frontier model typed eyes for pathology},
  author  = {Ge, Zhixuan and Qiu, Wei},
  journal = {arXiv preprint},
  year    = {2026}
}
```

The results shown here are in whole or part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga.
