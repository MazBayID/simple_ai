# Indonesian language and conversation experiment

## Purpose

This pipeline trains a small language model locally without a pretrained model
or third-party machine-learning framework. It supports an MLP baseline and an
optional one-block, single-head causal Transformer.

## Data formats

Plain UTF-8 text is accepted with `--input`. JSONL conversation data contains
one object per line:

```json
{"user":"Apa itu Python?","assistant":"Python adalah bahasa pemrograman."}
```

Role labels `Pengguna:` and `Asisten:` are plain text, not special tokens or
a role-aware instruction architecture. JSONL validation splits whole records
rather than splitting an exchange in the middle.

## Tokenizers

- `--tokenizer bpe` (default): deterministic Unicode-character BPE trained
  only on the training portion.
- `--tokenizer char`: original character-level baseline.

BPE reduces token sequence length, but does not add semantic understanding.
See `BPE_TOKENIZER.md` for its limitations.

## Training and validation

Default training reserves a contiguous tail of a plain-text source for
validation. For JSONL data, complete records are held out. For an independent
validation corpus, use `--validation-input`; the tokenizer is fitted only on
the training data. The report includes training/validation loss, validation
perplexity, vocabulary size, merge count, tokenization ratio, parameter count,
and optimizer configuration.

A lower loss or perplexity does not prove factuality, comprehension, or useful
chat ability. Keep a third, untouched test corpus for final comparisons.

## Architecture choices

Default MLP:
```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model
```

Causal Transformer (start with small settings on Android):
```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer bpe --architecture transformer --context-size 16 \
  --embedding-dim 8 --epochs 5 --batch-size 8
```

The Transformer is a single attention head and one block, implemented with
pure-Python matrix operations. It defaults to Adam; the MLP defaults to SGD.

## Chat

After training:
```bash
PYTHONPATH=python python -m simple_ai.chat
```

Chat loads the Indonesian model and matching tokenizer from
`artifacts/indonesian_model/`. It detects either supported architecture and
tokenizer format. It does not update weights while chatting and may generate
nonsense.

## Benchmark

```bash
PYTHONPATH=python python -m simple_ai.experiments.benchmark_model \
  --generation-tokens 32 --prompt "Halo, apa kabar?"
```

This measures checkpoint load time, generation speed, checkpoint size, and
Python-traced peak allocation. It does not measure total Android process RSS.

## Remaining work

1. Obtain a much larger, diverse Indonesian corpus with clear legal provenance.
2. Audit duplicates, encoding, language distribution, private/sensitive data,
   and train/validation/test separation.
3. Extend the architecture and optimize the numerical engine.
4. Evaluate held-out prompts for fluency, relevance, factuality, and consistency.
5. Profile time, peak RAM, model size, and generation speed on the target device.

The seed corpus is only an end-to-end smoke test, not a useful-scale dataset.
