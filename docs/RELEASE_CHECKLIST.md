# Release readiness checklist

Simple AI must not be called a usable offline assistant until all sections
below are backed by repeatable evidence.

## Engineering correctness

- [ ] Python test suite passes on CI and Termux.
- [ ] C++ native build and demo pass.
- [ ] MLP and Transformer checkpoints save, reload, and generate.
- [ ] Tokenizer/model compatibility is checked at load time.
- [ ] Training and inference commands work from a clean checkout.

## Data quality and rights

- [ ] Training corpus has documented provenance and usage rights.
- [ ] Duplicate, malformed, low-quality, private, and sensitive records are audited.
- [ ] Train, validation, and final test sets are independent.
- [ ] The final test set is not used to tune hyperparameters.
- [ ] Dataset size and language coverage are reported.

## Model quality

- [ ] Held-out Indonesian prompts test fluency and relevance.
- [ ] Factual questions are scored against trusted references.
- [ ] Instruction following and multi-turn consistency are measured.
- [ ] Hallucination and unsafe-output behavior are reviewed.
- [ ] Baselines are compared with fixed seeds and identical data splits.

## Target-device readiness

- [ ] Training wall time is measured on the target Android device.
- [ ] Peak process RSS is measured (Python tracemalloc is not enough).
- [ ] Generation tokens/second and latency are recorded.
- [ ] Model and tokenizer sizes are recorded.
- [ ] Resource limits are set from measurements, not guesses.

## Distribution

- [ ] License is selected and added.
- [ ] Versioned release artifacts are reproducible.
- [ ] Model card documents architecture, data, limitations, and intended use.
- [ ] Offline installation and inference are tested without network access.

The current repository is an experimental prototype. Passing unit tests proves
software behavior, not that the model has learned useful Indonesian language
ability. The remaining data, quality, and Android checks require real corpora
and device measurements.
