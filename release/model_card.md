---
license: apache-2.0
base_model: Qwen/Qwen3-VL-8B-Instruct
library_name: peft
pipeline_tag: image-text-to-text
tags:
- pathology
- lora
- visual-question-answering
---

# Opsis-V

Opsis-V is the typed question interface of *One state, many questions: Opsis gives a frontier model typed eyes for pathology*
inside Qwen3-VL-8B-Instruct: a LoRA adapter (rank 16, alpha 32, dropout 0.05) on the query, key, value, output, gate, up and down
projections of the language tower, with the vision tower frozen. It reads a field at 0.25 mpp and its 448 um context canvas as
pixels, with the canvases as a shared prefix, and answers each typed multiple choice question with the option letter. The answer
distribution is the softmax over the candidate letters. Code: https://github.com/gezhixuan/opsis.

## Training

NuCLS field questions (19 operators, `fields/nucls` of the question release), our patient split, 3 epochs, 16 questions per
field, learning rate 1e-4, bf16. The paper reports three seeds; this is the seed with the lowest validation NLL
(`seeds.json` lists all three). Prompt, canvases and scoring are in `scripts/opsis_v/40_train_vlm.py` of the code repository.

## Use

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "gezx1004/opsis-v")
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
```

Evaluation and shared prefix inference: `python scripts/opsis_v/40_train_vlm.py --eval-only` and `scripts/opsis_v/41_vlm_shared_prefix.py`.

## Limits

Trained on breast cancer fields of TCGA slides at one resolution. It answers the operators it was trained on, it is not a
diagnostic device and was not validated for clinical use.

## Licence

Apache 2.0, as the base model. The training labels come from NuCLS and BCSS (CC0) on TCGA slides.
