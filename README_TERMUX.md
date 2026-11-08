# Test Simple AI on Termux (Android)

This repository now includes a small neural training experiment. It is still not a conversational assistant or a capable SLM.

## 1. Install tools

In Termux:

```bash
pkg update
pkg upgrade
pkg install python git clang cmake
```

## 2. Clone the repository

```bash
git clone https://github.com/MazBayID/simple_ai.git
cd simple_ai
```

If you already cloned it:

```bash
cd ~/simple_ai
git pull origin main
```

## 3. Run Python demo (no third-party runtime dependencies)

```bash
PYTHONPATH=python python -m simple_ai.cli demo
```

Expected: prints training statistics and generated text. The generated text may be repetitive or nonsensical; this is a tiny character transition model, not a chatbot.

## 4. Train using your own text

Create a small text file:

```bash
nano sample.txt
```

Write some text, save, then run:

```bash
PYTHONPATH=python python -m simple_ai.cli train --data sample.txt --output tiny_model.json
PYTHONPATH=python python -m simple_ai.cli generate --model tiny_model.json --prompt "Halo" --tokens 100
```

## 5. Install the CLI package (optional)

```bash
python -m pip install -e .
simple-ai demo
```

## 6. Run tests

```bash
python -m pip install pytest
PYTHONPATH=python python -m pytest -q
```

## 7. Try the tensor engine

The first tensor engine uses only Python's standard library:

```bash
PYTHONPATH=python python -c "from simple_ai.core import Tensor; x=Tensor([[1, 2], [3, 4]]); print((x @ x).data)"
```

Expected output:

```text
[[7.0, 10.0], [15.0, 22.0]]
```

Read [docs/TENSOR_ENGINE.md](docs/TENSOR_ENGINE.md) for the API and gradient
example.

## 8. Build the C++ demo

```bash
cd cpp
cmake -S . -B build
cmake --build build
./build/simple_ai_native_demo
```

## 9. Run the neural training experiment

From the repository root:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_regression
```

Expected: the final loss is lower than the initial loss, a learned weight is printed, and `artifacts/regression_checkpoint.json` is created. This is a simple regression experiment, not language-model training.

## Important limitation

The first Python and C++ programs are **tiny character-transition prototypes**. They do not understand meaning, follow instructions, or perform reasoning. They are intended to verify the repository, training loop concept, serialization, CLI, SQLite, and native build before we design a real tokenizer and neural network.

Exact RAM/CPU targets are deferred until profiling. Do not interpret the regression experiment's success as proof that a future SLM will meet any resource target.
