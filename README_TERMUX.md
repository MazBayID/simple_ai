# Test Simple AI on Termux (Android)

Simple AI is an early offline-AI research project. The neural training engine and text data pipeline are educational prototypes; this repository does not yet provide a conversational assistant or capable SLM.

## 1. Install tools

```bash
pkg update
pkg upgrade
pkg install python git clang cmake
python -m pip install pytest
```

## 2. Clone or update the repository

```bash
git clone https://github.com/MazBayID/simple_ai.git
cd simple_ai
```

For an existing clone:

```bash
cd ~/simple_ai
git pull origin main
```

## 3. Run the test suite

```bash
PYTHONPATH=python python -m pytest -q
```

## 4. Run the original Python demo

```bash
PYTHONPATH=python python -m simple_ai.cli demo
```

The generated text may be repetitive or nonsensical. The original model learns character transition counts, not meaning.

## 5. Run the neural training experiment

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_regression
```

The experiment learns the simple relationship y = 2x and writes a checkpoint to `artifacts/regression_checkpoint.json`. This validates the training engine, not a language model.

## 6. Prepare your own text corpus

Create or place a UTF-8 text file in the repository root. For example:

```bash
nano sample.txt
```

Then prepare it:

```bash
PYTHONPATH=python python -m simple_ai.pipeline --input sample.txt --context-size 16
```

The pipeline writes these files to `artifacts/text_pipeline/`:

- `tokenizer.json`
- `token_ids.json`
- `dataset.json`

The source `sample.txt` is read-only from the pipeline's perspective; it is not modified. The first tokenizer is character-level, not BPE or a word tokenizer. If the corpus is too short for the chosen context size, rerun with a smaller value such as 4 or 8.

Local `sample.txt` and `artifacts/` are ignored by Git by default. Do not commit private or unlicensed training text accidentally.

See [docs/TOKENIZER_DATASET.md](docs/TOKENIZER_DATASET.md) for design details and limitations.

## 7. Build the C++ demo

```bash
cd cpp
cmake -S . -B build
cmake --build build
./build/simple_ai_native_demo
```

## Important limitations

The character tokenizer and next-token dataset only prepare data. They do not train a language model yet. The current Tensor engine now has a basic trainable embedding lookup, but still lacks softmax/cross-entropy, efficient batching, and a vocabulary projection layer. These will be developed and tested as separate milestones.

Exact RAM/CPU targets are deferred until profiling. Do not interpret a successful regression experiment as proof that a future SLM will meet any resource target.
