---
license: other
license_name: mixed
pretty_name: Opsis typed pathology questions
task_categories:
- visual-question-answering
tags:
- pathology
- whole-slide-image
- TCGA
- NuCLS
- PanNuke
size_categories:
- 1M<n<10M
---

# Opsis typed pathology questions

The question sets of *One state, many questions: Opsis gives a frontier model typed eyes for pathology*. Code:
https://github.com/gezhixuan/opsis.

Every question is typed: an operator, its arguments, a box where the question is about a region, a fixed list of candidate
answers and the index of the correct one. Questions refer to public images by their identifiers. No image, slide, tile feature
or report text is included. The slides come from the GDC, NuCLS, BCSS and PanNuke from their own pages, and the report text
from the TCGA-Reports corpus.

## Contents

| folder | what | size | split | licence |
|---|---|---|---|---|
| `fields/nucls` | 19 operators on 1,744 NuCLS fields of 124 TCGA-BRCA patients (cells, regions, compositions), with the BCSS tissue context | 242,604 questions | by patient, 87 / 19 / 18 | CC BY 4.0 |
| `fields/pannuke` | 9 operators on 7,901 PanNuke fields of 19 organs, official folds and an organ split (breast, colon, lung, prostate held out) | 781,295 questions | folds and organs | CC BY-NC-SA 4.0 |
| `slides/pan_cancer` | 10,712 TCGA diagnostic slides of 8,861 patients in 32 projects: eight report fields, cancer type, molecular subtype, tumour purity and leukocyte fraction bins, five composite operators held out of training | 65,973 questions | per project by patient, 6,772 / 1,402 / 1,393 slides, plus KIRP, STAD, CESC and PAAD held out (1,145 slides) | CC BY 4.0 |
| `slides/breast` | 1,122 TCGA-BRCA slides with the synoptic report fields (Nottingham components, DCIS, LCIS, calcifications) and ER, PR, HER2, PAM50 | 18,215 questions | 780 / 172 / 170 slides | CC BY 4.0 |
| `reports` | report fields extracted from 8,138 TCGA-Reports by gpt-6-luna, each value with a verbatim quote checked against the report, and the independent audit extraction of 320 stratified reports by Claude Sonnet 5.5 | | | CC BY 4.0 |
| `organiser` | the organiser episodes of the paper: one record per question and condition with the tool calls and their results, the answer, its confidence, tokens and time, plus the summaries and the slide descriptions of the reading conditions | 100,222 episodes in 126 files | the evaluation sets of the paper | CC BY 4.0 |

Question columns: `qid`, `fov` (field or slide identifier), `slide`, `patient`, `family`, `op`, `args` (JSON), `atype`,
`candidates` (JSON list), `answer` (index into the candidates), `box` and `box2` (pixel boxes in the frame of the field image at 0.2 mpp, reaching beyond it for
questions about the context, or `[0, 0, 1, 1]` for a whole slide), `text` (one natural language rendering of the question), `derivation` (the source of the label) and `split`.
Slide sets add `project`. Report derived labels are agreement with the report: a report that does not mention a finding gives no
question.

## Organiser episodes

`organiser/<condition>.jsonl` holds one episode per line: the slide, the question id, the operator, the correct and the predicted
answer, the confidence, every tool call with its arguments and result, the token usage and the wall time. The file name gives the
model and the condition (`tools` asks Opsis-S, `vision` and `look` look at the slide, `describe6` and `describe48` read a description,
`both` has tools and vision, `structured` and `structured4` are the rules conditions without a language model, `eyes_*` are the
vision language models answering the typed questions directly, and `eyes_<model>_pan_ct_s<i>` rescore the 32 way cancer type
question of the two local models with the labels AA to AF scored by their own token, replacing those rows of `eyes_<model>_pan_s<i>`). `<condition>_summary.json` holds the accuracy and cost of a condition.
The episodes are kept as recorded: the tool server ran under an earlier project name, so tool calls appear as `mcp__pathjev__<tool>`.

## Use with the code

Unpack the folders into `$OPSIS_DATA/derived/` with the names the code expects (`questions_final`, `pannuke`,
`slide_questions_pan`, `slide_questions_v2`, `report_pan`) and the episodes into `results/organiser/` with
`release/install_questions.py` of the code repository. The tile features of a slide are
read from `$OPSIS_TCGA/gigapath_h5/<project>_trident_virchow2/h5_files/<slide>.h5`.

## Sources and licences

- NuCLS (Amgad et al., GigaScience 2022) and BCSS (Amgad et al., Bioinformatics 2019): CC0 1.0.
- PanNuke (Gamper et al., 2019 and 2020): CC BY-NC-SA 4.0. The PanNuke questions carry the same licence.
- TCGA-Reports (Kefeli et al., Mendeley Data 2024): CC BY 4.0. The report fields include short verbatim quotes from these reports.
- TCGA clinical and molecular tables, PanCanAtlas ABSOLUTE purity and the leukocyte fraction of Thorsson et al. (Immunity 2018).
  The results shown here are in whole or part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga.

No attempt to identify any individual may be made with these data.

## Citation

```bibtex
@article{ge2026opsis,
  title   = {One state, many questions: Opsis gives a frontier model typed eyes for pathology},
  author  = {Ge, Zhixuan and Qiu, Wei},
  journal = {arXiv preprint},
  year    = {2026}
}
```
