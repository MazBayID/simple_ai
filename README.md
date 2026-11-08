# Simple AI

An open-source project exploring a small, fully offline AI built from scratch.

## Project goals

- Build a small language model and assistant from the ground up.
- Use Python for the educational framework and C++ for native/performance experiments.
- Run inference offline once local model files are available.
- Explore SQLite for local storage and memory.
- Measure training/runtime resources before setting device limits.
- Keep experiments reproducible and document limitations honestly.

## Current status

The repository contains a pure-Python tensor/autograd engine, neural-network
layers, a character tokenizer, a deterministic BPE tokenizer, next-token
datasets, an MLP language model, and an optional causal Transformer.

**Important:** Simple AI is a research prototype, not a capable assistant.
The included Indonesian seed corpus is only a smoke test. BPE and attention
improve the architecture, but do not create knowledge or guarantee understanding.

## Train an Indonesian baseline

The default is BPE + MLP:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model
```

Try the optional single-head causal Transformer with a small configuration:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer bpe --architecture transformer --vocab-size 128 \
  --context-size 16 --embedding-dim 8 --epochs 5 --batch-size 8
```

Compare with the character tokenizer:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer char --output-dir artifacts/indonesian_char
```

Use your own UTF-8 corpus with `--input`, or JSONL conversation pairs with
`--conversations`. For a separate validation corpus, add
`--validation-input /path/to/validation.txt`; the tokenizer is fitted only on
training text. Keep an additional untouched test corpus for final comparisons.
Only use data you have permission to train on; keep private corpora outside Git.

## Run the local chat harness

After training:

```bash
PYTHONPATH=python python -m simple_ai.chat
```

Chat loads the Indonesian checkpoint by default and detects either supported
model architecture and tokenizer format. It is a generation test harness, not
a reliable question-answering assistant.

## Architecture

- **MLP baseline:** token embedding → flatten → Linear → ReLU → vocabulary projection.
- **Transformer experiment:** learned token/position embeddings, one causal
  self-attention head, residual connection, feed-forward network, and output projection.

The Transformer is intentionally small and implemented using Python lists.
It defaults to Adam; the MLP baseline defaults to SGD. It has no layer
normalization, dropout, KV cache, or optimized matrix kernels.
Training may be slow on Android. Start with small contexts and batches.

## Benchmark inference

```bash
PYTHONPATH=python python -m simple_ai.experiments.benchmark_model \
  --generation-tokens 32 --prompt "Halo, apa kabar?"
```

The report includes load time, tokens per second, checkpoint size, and
Python-traced peak allocations (not total process RSS).

## Research milestones still required

- A substantially larger, diverse, legally obtained Indonesian corpus.
- Independent train/validation/test corpora and a repeatable quality benchmark.
- More Transformer blocks/heads and numerical/performance optimization.
- Profiling wall time, peak RAM, model size, and generation speed on Termux.
- Safety, factuality, and instruction-following evaluation before calling it a usable assistant.

See `docs/BPE_TOKENIZER.md`, `docs/TRANSFORMER.md`,
`docs/LANGUAGE_MODEL.md`, `docs/INDONESIAN_CHAT.md`, and
`docs/RELEASE_CHECKLIST.md`.

## License

License selection is pending. Do not assume the project is licensed until a license file is added.


## Prepare a provenance-tracked Indonesian corpus

A downloader and cleaner prepares a rights-cleared Indonesian Wikipedia subset (CC BY-SA 4.0). It verifies revision provenance, removes exact duplicate texts, and creates deterministic document-level train/validation/test splits. Raw and processed corpus files stay local and are ignored by Git.

    PYTHONPATH=python python -m simple_ai.experiments.prepare_indonesian_corpus

The pipeline requires at least 1,000,000 training characters and writes a manifest plus attribution record. See [docs/INDONESIAN_CORPUS.md](docs/INDONESIAN_CORPUS.md) for details.
