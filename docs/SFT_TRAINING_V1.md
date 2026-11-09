# Simple AI v3 — SFT pipeline

This pipeline fine-tunes the existing v2 decoder-only Transformer. It does not
retrain the tokenizer or modify the v2 checkpoint. Prompt tokens are masked
from the loss; only answer tokens and the answer EOS token are trained.

## 1. Export the Aksara records from the existing Colab session

The earlier Aksara download/audit cell keeps the records in a Python variable
named `records`. Export them without changing the source:

~~~python
import json

with open("/content/aksara_sft_raw.jsonl", "w", encoding="utf-8") as f:
    for row in records:
        f.write(json.dumps({
            "instruction": row["instruction"],
            "input": row.get("input", ""),
            "output": row["output"],
            "source": row.get("source", "aksara-sft-clean-v1"),
            "category": row.get("task_type", ""),
            "original_index": row.get("id", "")
        }, ensure_ascii=False) + "\n")

print("Exported:", len(records), "records")
~~~

Keep the original downloaded dataset and the GarudaCoder staging files unchanged.

## 2. Prepare deterministic splits

Install/update the repository in Colab, then run:

~~~bash
%cd /content/simple_ai
!git pull origin main
!python -m pip install -e ".[cloud]"
~~~

~~~bash
!python -m simple_ai.experiments.prepare_sft_v1 \
  --garudacoder \
    /content/simple_ai_v3_staging/programming_candidates.jsonl \
    /content/simple_ai_v3_staging/general_candidates.jsonl \
  --aksara /content/aksara_sft_raw.jsonl \
  --output-dir /content/drive/MyDrive/simple_ai_data/simple_ai_sft_v1
~~~

The normalizer removes exact duplicate instruction-answer pairs. Records
sharing the same normalized instruction are assigned to the same split to
reduce prompt leakage. Split assignment is deterministic (90/5/5). It writes
train.jsonl, validation.jsonl, test.jsonl, and manifest.json.

## 3. Fine-tune from the v2 best checkpoint

Use the checkpoint and tokenizer from the completed v2 run:

~~~bash
!python -m simple_ai.experiments.train_sft_v1 \
  --train /content/drive/MyDrive/simple_ai_data/simple_ai_sft_v1/train.jsonl \
  --validation /content/drive/MyDrive/simple_ai_data/simple_ai_sft_v1/validation.jsonl \
  --test /content/drive/MyDrive/simple_ai_data/simple_ai_sft_v1/test.jsonl \
  --pretrained-checkpoint /content/drive/MyDrive/simple_ai_runs/indonesian_v2_clean_v2/checkpoint_best.pt \
  --tokenizer /content/drive/MyDrive/simple_ai_runs/indonesian_v2_clean_v2/tokenizer.json \
  --output-dir /content/drive/MyDrive/simple_ai_runs/indonesian_v3_sft_v1 \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 0.00002 \
  --checkpoint-steps 100 \
  --max-steps 100
~~~

The first invocation is a short 100-step run to confirm that the pretrained
checkpoint, tokenizer, data, and GPU fit together. If it completes, continue
with the same arguments, remove `--max-steps 100`, and add:

~~~bash
--resume /content/drive/MyDrive/simple_ai_runs/indonesian_v3_sft_v1/checkpoint_latest.pt
~~~

Do not change the model, tokenizer, dataset files, batch size, learning rate,
epoch count, or seed when resuming. The trainer checks the checkpoint,
tokenizer, and dataset hashes.

## Outputs

- `checkpoint_latest.pt`: resumable SFT checkpoint.
- `checkpoint_best.pt`: lowest validation answer-token loss.
- `training_report.json`: run settings, split sizes, loss history, and test loss.

All outputs go to a new v3 directory. The v2 baseline checkpoint and tokenizer
are read-only inputs and are never overwritten.

## Important notes

- The v2 model has a 256-token context. Long examples are truncated to fit;
  the trainer preserves the end of the prompt and as much of the answer as
  possible.
- Loss is computed only on answer tokens, not the instruction prompt.
- This is full-model fine-tuning, not LoRA.
- Do not mix the pretraining corpus into the SFT prompt-answer files.
- Preserve upstream attribution and license obligations. The Aksara source is
  marked CC BY-SA 4.0; GarudaCoder declares Apache-2.0. Review upstream terms
  before redistributing combined data or model artifacts.
- A lower SFT loss is not proof of factual accuracy or robust IoT expertise.
