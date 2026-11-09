# Simple AI v2 — Cloud Training

This experiment is intentionally separate from the v1 train_torch.py pipeline and uses a new output directory. It does not overwrite v1 checkpoints.

## What changed in v2

- Byte-level BPE tokenizer implemented through the Rust-backed tokenizers library. The existing educational Python BPE remains available for learning and small examples.
- Default vocabulary: 8,000 tokens.
- Default context: 256 tokens.
- Default decoder-only Transformer: 6 layers, 8 attention heads, 256-dimensional embeddings (about 6.85 million parameters at an 8,000-token vocabulary).
- AdamW with linear learning-rate warmup and cosine decay.
- Atomic checkpoints and deterministic epoch sampler, so interrupted runs can resume.
- Separate generation command that encodes prompts with BOS and explicitly excludes EOS.

The model is still a small experimental language model. A larger vocabulary and more parameters do not guarantee coherent answers; corpus diversity and training quality remain important.

## Colab setup

Select a GPU runtime and mount Google Drive:

~~~python
from google.colab import drive
drive.mount("/content/drive")
~~~

If /content/simple_ai already exists from the v1 work, do not clone it again. Switch that checkout to the v2 branch:

~~~bash
%cd /content/simple_ai
!git fetch origin
!git checkout -B feature/indonesian-v2 origin/feature/indonesian-v2
!python -m pip install -e ".[cloud,dev]"
~~~

If the folder does not exist yet, clone the branch first:

~~~bash
!git clone -b feature/indonesian-v2 https://github.com/MazBayID/simple_ai.git
%cd /content/simple_ai
!python -m pip install -e ".[cloud,dev]"
~~~

Check the actual runtime GPU rather than assuming a specific model:

~~~python
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
~~~

Use the v1 prepared corpus in Drive. No need to download or prepare it again if these files already exist:

- /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/train.jsonl
- /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/validation.jsonl
- /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/test.jsonl

## First smoke test

Run only 20 optimizer steps to verify the tokenizer, GPU, model, loss, checkpoint, and resume path before starting a long run:

~~~bash
!python -m simple_ai.experiments.train_torch_v2 \
  --train /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/train.jsonl \
  --validation /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/validation.jsonl \
  --test /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/test.jsonl \
  --tokenizer byte-bpe \
  --vocab-size 8000 \
  --context-size 256 \
  --embedding-dim 256 \
  --num-heads 8 \
  --num-layers 6 \
  --batch-size 8 \
  --epochs 8 \
  --max-steps 20 \
  --checkpoint-steps 10 \
  --output-dir /content/drive/MyDrive/simple_ai_runs/indonesian_v2
~~~

Confirm that the report says device cuda, the vocabulary is close to the requested 8,000 (the trainer may stop earlier if no more merges are available), and the model has roughly 6.85 million parameters. Check that both checkpoint files and tokenizer.json appear in the v2 Drive folder.

## Full training

After the smoke test succeeds, run the same command without --max-steps. Keep the architecture, tokenizer, batch size, epochs, and schedule settings unchanged when resuming.

~~~bash
!python -m simple_ai.experiments.train_torch_v2 \
  --train /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/train.jsonl \
  --validation /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/validation.jsonl \
  --test /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/test.jsonl \
  --tokenizer byte-bpe \
  --vocab-size 8000 \
  --context-size 256 \
  --embedding-dim 256 \
  --num-heads 8 \
  --num-layers 6 \
  --batch-size 8 \
  --epochs 8 \
  --warmup-steps 500 \
  --checkpoint-steps 500 \
  --output-dir /content/drive/MyDrive/simple_ai_runs/indonesian_v2
~~~

If the GPU runs out of memory, lower --batch-size to 8 or 4. Do not change context size, model dimensions, tokenizer, or schedule when resuming an existing checkpoint.

## Resume after a Colab disconnect

Run the same full-training command with the same settings and append this option to that command:

~~~bash
--resume /content/drive/MyDrive/simple_ai_runs/indonesian_v2/checkpoint_latest.pt
~~~

The latest checkpoint stores model weights, optimizer state, epoch and batch position, global step, best validation loss, configuration, and random-number-generator states. The scheduler is derived from the saved global step and the immutable run configuration.

## Full final evaluation

For the final run, add --full-final-eval. This evaluates every validation and test window. Per-epoch validation is limited to 100 batches by default to avoid spending a long time evaluating the entire corpus after every epoch.

## Generate a continuation

After training, generate from the best checkpoint. The prompt is encoded with BOS but without EOS, because it is an unfinished document:

~~~bash
!python -m simple_ai.experiments.generate_torch_v2 \
  --checkpoint /content/drive/MyDrive/simple_ai_runs/indonesian_v2/checkpoint_best.pt \
  --tokenizer /content/drive/MyDrive/simple_ai_runs/indonesian_v2/tokenizer.json \
  --prompt "Internet of Things adalah" \
  --max-new-tokens 160 \
  --temperature 0.8 \
  --top-k 40
~~~

Try several prompts and inspect coherence, not just fluency. Compare v2 against v1 using the same held-out test split and corrected prompt encoding.

## Corpus licensing

The training corpus is CC BY-SA 4.0. Preserve its attribution and manifest and review attribution, share-alike, and model-weight implications before redistributing trained artifacts.
