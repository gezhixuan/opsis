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
| `OPSIS_WSI` | TCGA diagnostic slides, downloaded by `scripts/104` and `scripts/105` from the GDC |
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

The install also copies the organiser episodes of the paper into `results/organiser/`, so the organiser tables of `82` and `88f`
are recomputed from them without running a frontier model.

`release/build_release.py` is the script that assembled both Hugging Face releases from our data root.

## Pipeline

Scripts are numbered in the order they run. Long jobs have a SLURM wrapper in `scripts/slurm/` that takes its arguments
through `ARGS` (`sbatch --export=ALL,ARGS="..." scripts/slurm/20_train.sbatch`, submitted from the repository root).

| stage | scripts |
|---|---|
| data | `01` BCSS masks, `10` facts (nuclei, regions, tissue) from NuCLS and BCSS, `12` context canvases from the slides, `104`/`105` TCGA slides from the GDC |
| questions | `11` NuCLS questions and patient split, `67` mitotic and sTIL operators (`questions_final`), `80_pannuke_build` PanNuke, `50`/`52` breast slide questions, `54_extract_openai` + `54_run_pan_bulk.sh` + `55_validate_pan` report field extraction with verbatim quotes, `53` validation of the breast extraction, `66` pan cancer slide questions and split, `68` audit agreement |
| features | `13` Virchow2 tokens of the field and context canvases, `81_pannuke_tokens`, `70` SapBERT operator embeddings (one ablation) |
| Opsis-S | `20` field decoder, `51` slide decoder, `69` per tile heads (noisy-OR, log mean exp, gated attention), `21_train_tuned_state` the unfrozen encoder ablation |
| Opsis-V | `40` LoRA training and evaluation (also the untrained Qwen3-VL-8B and Patho-R1-7B with `--eval-only`), `41_vlm_shared_prefix` shared prefix timing, `41_train_vlm_state` Opsis-V reading the Virchow2 state |
| organiser | `80_mcp_server` the tool server (`src/opsis/tools.py`), `81_organiser_eval` one episode per question (OpenAI API, Codex CLI or Claude Code CLI), `81b` the pan cancer test split in shards, `81d` the rules condition, `106` the reading conditions |
| comparables | `57` ABMIL and CLAM-SB (`slurm/57b_abmil.sbatch` over many tasks), `58` and `83` PRISM2, `84` and `88` TITAN, `85` + `87` PRISM2 reports, `59` CellViT++, `100` vision language models looking at slides and fields, `101` TITAN and PRISM2 on CAMELYON16, `102` Virchow2 linear probes |
| transfer | `71` tile maps against BCSS necrosis, `89`, `90`, `91` CAMELYON16 |
| cost | `92` encoder timing, `111` slide question timing |
| collect | `60`, `65`, `72`, `74`, `75`, `82`, `86`, `88b` to `88f` write the result tables to `results/collected/` |

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
