# Byte-pair tokenizer

Simple AI now includes a small deterministic byte-pair-style tokenizer written
from scratch with the Python standard library.

## Why it exists

The character tokenizer represents `makan` as five tokens. BPE starts from
Unicode characters and repeatedly merges the most frequent adjacent pair.
Common sequences can therefore become one token, reducing sequence length.

This implementation is intentionally educational, not a production tokenizer:
it operates on Unicode characters rather than UTF-8 bytes, uses a simple
greedy merge procedure, and has no pre-tokenization or normalization.

## Train an Indonesian model

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer bpe --vocab-size 128 --context-size 32 --epochs 20
```

The training report records vocabulary size and learned merge count. The
checkpoint and tokenizer are saved together and must always be used as a pair.

To compare with the character baseline:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer char --output-dir artifacts/indonesian_char
```

## Limitations

BPE reduces sequence length; it does not create knowledge or understanding.
The model is still the fixed-context MLP baseline. A separate, larger and
properly licensed corpus plus a stronger architecture are still required.
