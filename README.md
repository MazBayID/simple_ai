# Simple AI

An open-source project exploring a small, fully offline AI built from scratch.

The project aims to be understandable, modular, and modifiable by contributors—not merely a chatbot wrapper.

## Project goals

- Build a small language model and AI assistant from the ground up.
- Python is the primary language for learning, experimentation, and contributor accessibility.
- C++ is the supporting language for native components and performance-critical operations.
- Explore compiling the native core as a C++ library and exposing it to Python.
- Fully offline runtime after model files are available locally.
- SQLite for local storage and memory.
- Keep runtime and training resource use measurable; exact RAM/CPU limits are deferred until profiling.
- Develop capabilities gradually and document experiments honestly.

## Current status

The project has a pure-Python autodiff and training engine, a character tokenizer,
a deterministic BPE tokenizer, next-token datasets, and an experimental fixed-context
MLP language model. Indonesian training and a local terminal chat harness are available.

**Important:** Simple AI is still a research prototype, not a capable assistant.
BPE reduces token sequence length but does not add semantic understanding. The
included seed corpus is only a smoke-test dataset.

## Development principles

1. Build from scratch; existing projects are references, not core dependencies.
2. Prefer small, testable milestones.
3. Keep runtime offline.
4. Measure RAM, CPU time, model size, and quality.
5. Separate training and validation data.
6. Document limitations and failed experiments.
7. Never describe falling training loss as proof of understanding.

## Train an Indonesian baseline

BPE is now the default tokenizer for the Indonesian experiment:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model
```

Choose the tokenizer, architecture, vocabulary, and context sizes explicitly:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer bpe --architecture transformer --vocab-size 128 \\\n  --context-size 16 --embedding-dim 8 --epochs 5 --batch-size 8
```

For a character-level comparison:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer char --output-dir artifacts/indonesian_char
```

Use a local UTF-8 corpus or JSONL dialogue file with `--input` or `--conversations`.
Do not commit private or unlicensed training data.

## Run the chat harness

After training:

```bash
PYTHONPATH=python python -m simple_ai.chat
```

The chat loads the Indonesian checkpoint by default. It can also load the older
character checkpoint because tokenizer format is detected automatically. This is
a text-generation experiment, not a reliable question-answering assistant.

## Architecture and next research milestones

The current model is embedding → flatten → Linear → ReLU → vocabulary projection.
It has no attention, recurrence, or Transformer blocks. The next major engineering
milestone is a small causal-attention model, followed by independent evaluation,
larger licensed data, and device profiling.

See `docs/BPE_TOKENIZER.md`, `docs/TRANSFORMER.md`, `docs/LANGUAGE_MODEL.md`, and
`docs/INDONESIAN_CHAT.md`.

## License

License selection is pending. Do not assume the project is licensed until a license file is added.
