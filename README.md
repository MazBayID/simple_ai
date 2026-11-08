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

This repository is in its **early neural-engine research stage**. A small neural training engine and a reproducible regression experiment now exist; no trained language model or conversational inference engine is available yet.

## Planned structure

The structure will evolve as the project is implemented. Early planned areas:

- `docs/` — research notes, design decisions, and benchmarks
- `python/` — Python framework and experiments
- `cpp/` — native C++ components
- `tests/` — correctness and resource tests
- `.github/workflows/` — continuous integration and future build/release automation

## Releases and GitHub Actions

GitHub Actions validates the repository and can run a genuine, small neural regression experiment. Its checkpoint is an experiment artifact, not a language model or release binary.

Future release packages may include:

- Source code archive (GitHub provides source archives for tagged releases).
- Python package, once packaging and APIs are stable.
- Native C++ library for supported platforms, after it can be built and tested.
- Tiny model checkpoints only after training, provenance, license, and resource usage are documented.
- Benchmark reports and checksums as supporting release files.

Build outputs from ordinary workflow runs will be CI artifacts; versioned distributions will be attached to GitHub Releases.

## Development principles

1. Build from scratch; existing projects are references, not core dependencies.
2. Prefer small, testable milestones.
3. Keep the project offline at runtime.
4. Measure RAM, CPU time, model size, and quality.
5. Document limitations and failed experiments.
6. Do not publish a model or binary as usable until it has been tested.

## License

License selection is pending. Do not assume the project is licensed until a license file is added.


## First runnable prototype

The repository now contains a tiny character-transition model, a Python CLI, a SQLite local storage layer, and a C++ native prototype. These are foundation experiments, not a semantic language model or full assistant.

For Android/Termux setup and test commands, see [README_TERMUX.md](README_TERMUX.md).

## Current executable capabilities

- Train the Python toy model from text.
- Generate short character sequences from learned transition counts.
- Save and reload a JSON checkpoint.
- Store local experiment metadata and conversation records in SQLite.
- Build and run a small C++ prototype with CMake.
- Run tests and native compilation through GitHub Actions.

## Current milestone

**Milestone: neural training engine / experimental.**

This does not mean the planned AI is complete. The next milestones are tokenizer design, a language-model dataset pipeline, a small language model, inference, and measured resource profiling.


## Tensor engine milestone

A first pure-Python 2-D tensor engine is now available at
`python/simple_ai/core/tensor.py`. It supports matrix multiplication,
element-wise arithmetic, transpose, ReLU, reductions, and reverse-mode
automatic differentiation. It uses only the Python standard library.

This is a correctness-first educational engine, not a performance library.
See [Tensor Engine documentation](docs/TENSOR_ENGINE.md) for its API,
limitations, and examples. Neural-network layers, an SGD optimizer, a loss function, and a training loop are now implemented. See [Training Engine documentation](docs/TRAINING_ENGINE.md).


## Neural training engine milestone

A minimal trainable neural-network stack now includes `Linear`, `ReLU`, mean-squared error, SGD, and a full-batch training loop. Run a deterministic regression experiment with `PYTHONPATH=python python -m simple_ai.experiments.train_regression`. GitHub Actions can run the same experiment and upload its checkpoint. This is a training-engine validation task, **not** a language model or conversational AI.
