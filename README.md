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

## Resource policy

Exact RAM and CPU ceilings are intentionally deferred. Training and inference must be measured separately before setting numerical targets. The current model is small, but no performance guarantee is made for every Android device.

## Current status

The repository now has a pure-Python neural training engine, character tokenizer, text pipeline, and a first fixed-context character-level MLP language model. An Indonesian text seed corpus, JSONL conversation format, validation-loss reporting, and a local terminal chat loop are included as research tools.

**Important:** this is still an experimental character model, not a capable assistant. The included Indonesian corpus is deliberately small and only verifies the end-to-end training path. It does not provide broad Indonesian knowledge or reliable answers.

## Planned structure

- `docs/` — research notes, design decisions, and benchmarks
- `data/` — small public/example training data; private corpora should remain local
- `python/` — Python framework and experiments
- `cpp/` — native C++ components
- `tests/` — correctness tests
- `.github/workflows/` — continuous integration and training automation

## Development principles

1. Build from scratch; existing projects are references, not core dependencies.
2. Prefer small, testable milestones.
3. Keep runtime offline.
4. Measure RAM, CPU time, model size, and quality.
5. Separate training and validation data.
6. Document limitations and failed experiments.
7. Never describe falling training loss as proof of understanding.

## Current executable capabilities

- Train and sample the original toy character-transition model.
- Save and reload JSON checkpoints.
- Store local experiment metadata and conversation records in SQLite.
- Run a pure-Python 2-D tensor engine with reverse-mode automatic differentiation.
- Train small neural networks with Linear, ReLU, MSE, SGD, and cross-entropy.
- Build a small C++ prototype with CMake.
- Tokenize Unicode text at character level and prepare fixed-context next-token examples.
- Train an experimental fixed-context MLP language model from scratch.
- Read Indonesian JSONL conversation pairs and render them as plain-text dialogue.
- Split a corpus into contiguous training and validation regions and report both losses.
- Run an interactive local terminal chat demo using a saved model and tokenizer.

## Indonesian experiment

Default training uses the small included seed corpus:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model
```

Train from conversation pairs instead:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model --conversations data/conversations.example.jsonl --context-size 32 --epochs 20
```

For a larger local corpus:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model --input /path/to/indonesian_corpus.txt --context-size 48 --epochs 30
```

The model, tokenizer, and training report are saved in `artifacts/indonesian_model/`. Start the experimental chat:

```bash
PYTHONPATH=python python -m simple_ai.chat
```

The chat program is an inference interface only. It does not add knowledge to the model, and out-of-vocabulary characters become `<unk>`. Better dialogue requires much more clean Indonesian text and many diverse, high-quality conversation examples.

## Current architecture and limitations

The language model is an embedding → flatten → Linear → ReLU → vocabulary projection MLP. It predicts one character at a time from a fixed-size context. It has no attention, recurrence, transformer blocks, subword tokenizer, retrieval, or semantic representation. The simple chat loop is therefore a test harness, not evidence of useful conversation.

Validation loss is measured on a held-out contiguous portion of the same source file. This is a basic leakage-aware check, not a comprehensive benchmark. For meaningful evaluation, use a separate corpus and questions never seen during training.

See `docs/LANGUAGE_MODEL.md`, `docs/TOKENIZER_DATASET.md`, and `docs/INDONESIAN_CHAT.md`.

## License

License selection is pending. Do not assume the project is licensed until a license file is added.
