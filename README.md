# Simple AI

An open-source project exploring a small, fully offline AI built from scratch.

The project aims to be understandable, modular, and modifiable by contributors—not merely a chatbot wrapper.

## Project goals

- Build a small language model and AI assistant from the ground up.
- Python is the primary language for learning, experimentation, and contributor accessibility.
- C++ is the supporting language for native components and performance-critical operations.
- Explore compiling the native core as a C++ library and exposing it to Python.
- Fully offline runtime.
- SQLite for local storage and memory.
- Keep runtime and training resource use measurable and suitable for constrained devices; exact RAM/CPU limits are deferred until profiling.
- Develop capabilities gradually and document experiments honestly.

## Resource policy

Exact RAM and CPU ceilings are intentionally deferred. We will measure training and inference separately before setting numerical targets.

The initial stages focus on tiny educational models, efficient data pipelines, and reproducible experiments. We will not claim resource efficiency until it has been benchmarked.

## Current status

This repository is in its **early neural-engine and language-data research stage**. It has a small neural training engine, a deterministic character tokenizer, and a next-token dataset preparation pipeline. No trained language model or conversational inference engine is available yet.

## Planned structure

- `docs/` — research notes, design decisions, and benchmarks
- `python/` — Python framework and experiments
- `cpp/` — native C++ components
- `tests/` — correctness and resource tests
- `.github/workflows/` — continuous integration and future build/release automation

## Releases and GitHub Actions

GitHub Actions validates the repository and can run a genuine, small neural regression experiment. Its checkpoint is an experiment artifact, not a language model or release binary.

Future release packages may include source archives, a Python package, a native C++ library, documented model checkpoints, benchmark reports, and checksums after they are properly tested.

## Development principles

1. Build from scratch; existing projects are references, not core dependencies.
2. Prefer small, testable milestones.
3. Keep the project offline at runtime.
4. Measure RAM, CPU time, model size, and quality.
5. Document limitations and failed experiments.
6. Do not publish a model or binary as usable until it has been tested.

## First runnable prototype

The repository contains a tiny character-transition model, a Python CLI, a SQLite local storage layer, and a C++ native prototype. These are foundation experiments, not a semantic language model or full assistant.

For Android/Termux setup and test commands, see [README_TERMUX.md](README_TERMUX.md).

## Current executable capabilities

- Train the original toy character-transition model and generate short sequences.
- Save and reload its JSON checkpoint.
- Store local experiment metadata and conversation records in SQLite.
- Run a pure-Python 2-D tensor engine with reverse-mode automatic differentiation.
- Train a tiny neural network using Linear, ReLU, MSE, SGD, and a full-batch training loop.
- Build and run a small C++ prototype with CMake.
- Prepare local text as character token IDs and fixed-context next-token examples.
- Map token IDs to trainable, position-aware embedding features with gradient accumulation.
- Calculate numerically stable next-token cross-entropy directly from logits and backpropagate classification gradients.

## Current milestones

**Neural training engine / experimental.** The regression experiment validates parameter learning on the simple target y = 2x; it is not language-model training.

**Tokenizer and dataset pipeline / experimental.** The character tokenizer and next-token dataset establish the text data path. A complete language-model architecture, vocabulary projection, inference loop, and quality evaluation are still future milestones. Embeddings and cross-entropy are isolated building blocks, not a complete model.

See [docs/TOKENIZER_DATASET.md](docs/TOKENIZER_DATASET.md), [docs/TENSOR_ENGINE.md](docs/TENSOR_ENGINE.md), and [docs/TRAINING_ENGINE.md](docs/TRAINING_ENGINE.md).

## License

License selection is pending. Do not assume the project is licensed until a license file is added.
