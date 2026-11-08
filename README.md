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
- Target resource ceiling: **1.2 GB RAM and 1 GHz CPU, including training and inference**.
- Develop capabilities gradually and document experiments honestly.

## Important resource constraint

The RAM and CPU limits apply to **training as well as inference**. This is a hard project constraint, not a future optimization target.

Training capable language models from scratch under this limit is a major research challenge. The initial stages will focus on tiny educational models, efficient data pipelines, and measurable experiments. We will not claim a model meets the target until it has been benchmarked.

## Current status

This repository is in its **foundation / research stage**. No trained model, inference engine, or distributable binary is available yet.

## Planned structure

The structure will evolve as the project is implemented. Early planned areas:

- `docs/` — research notes, design decisions, and benchmarks
- `python/` — Python framework and experiments
- `cpp/` — native C++ components
- `tests/` — correctness and resource tests
- `.github/workflows/` — continuous integration and future build/release automation

## Releases and GitHub Actions

At this stage, Actions should validate the repository and run tests when code exists. We will not upload placeholder model weights or fake binaries.

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

**Milestone: first prototype / ready for initial Termux smoke testing.**

This does not mean the planned AI is complete. The next milestones are tokenizer design, data pipeline, neural-network implementation, training experiments, and measured resource profiling.
