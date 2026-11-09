# PyTorch Cloud Training

The PyTorch backend is optional. The original educational tensor/autograd backend
continues to work without PyTorch.

## Google Colab setup

1. Create a Colab notebook and select a GPU runtime if one is available.
2. Mount Google Drive so corpus files and checkpoints survive runtime resets.
3. Clone the repository and install the cloud extra:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone https://github.com/MazBayID/simple_ai.git
%cd /content/simple_ai
!python -m pip install -e ".[cloud,dev]"
```

Check the runtime before starting:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

A Colab GPU is not guaranteed. The training command automatically selects CUDA
when available and otherwise uses CPU. Do not assume a particular GPU model,
quota, or session duration.

## Prepare the corpus

Keep the data in Drive so it does not need to be downloaded and prepared again
after a Colab runtime reset:

```bash
!python -m simple_ai.experiments.prepare_indonesian_corpus \
  --source /content/drive/MyDrive/simple_ai_data/bahasa_wikipedia_ccbysa.jsonl \
  --output-dir /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia
```

The preparer writes the existing `.txt` splits and new `.jsonl` splits.
JSONL preserves one article per record, so training windows never cross article
boundaries. The training script prefers JSONL when it is present.

The source corpus is CC BY-SA 4.0. Keep `ATTRIBUTION.md` and `manifest.json`
with the data, and review attribution/share-alike and model-weight implications
before redistributing trained artifacts.

## First smoke test

Start with a short run before committing to a longer experiment:

```bash
!python -m simple_ai.experiments.train_torch \
  --train /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/train.jsonl \
  --validation /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/validation.jsonl \
  --test /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/test.jsonl \
  --tokenizer char \
  --context-size 128 \
  --embedding-dim 128 \
  --num-heads 4 \
  --num-layers 4 \
  --batch-size 16 \
  --epochs 3 \
  --max-steps 20 \
  --checkpoint-steps 10 \
  --output-dir /content/drive/MyDrive/simple_ai_runs/indonesian_v1
```

The character tokenizer is the initial baseline because the current educational
BPE trainer is CPU-intensive on a large corpus. We can optimize BPE separately
after the first end-to-end cloud run.

Inspect the reported device, parameter count, training loss, validation loss,
and checkpoint paths. A falling training loss alone does not establish useful
language ability.

## Resume training

After the smoke test, continue the same run by removing `--max-steps` and
passing the latest checkpoint. Keep the model dimensions, context size, and
tokenizer artifact unchanged:

```bash
!python -m simple_ai.experiments.train_torch \
  --train /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/train.jsonl \
  --validation /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/validation.jsonl \
  --test /content/drive/MyDrive/simple_ai_data/indonesian_wikipedia/test.jsonl \
  --context-size 128 \
  --embedding-dim 128 \
  --num-heads 4 \
  --num-layers 4 \
  --batch-size 16 \
  --epochs 3 \
  --checkpoint-steps 500 \
  --resume /content/drive/MyDrive/simple_ai_runs/indonesian_v1/checkpoint_latest.pt \
  --output-dir /content/drive/MyDrive/simple_ai_runs/indonesian_v1
```

The checkpoint contains model weights, AdamW state, epoch and batch position,
global step, best validation loss, configuration, and random-number-generator
states. Checkpoints are written atomically. Only load checkpoint files you
trust because PyTorch checkpoint loading uses Python deserialization.

Files written to the output directory:

- `tokenizer.json`: exact tokenizer vocabulary used by the run.
- `checkpoint_latest.pt`: most recent resumable state.
- `checkpoint_best.pt`: best validation-loss state observed so far.
- `training_report.json`: run configuration and evaluation summary.

The first run uses a modest 4-layer, 128-dimensional model. Benchmark GPU memory
and throughput before increasing batch size, context length, or model width.
