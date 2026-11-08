# Test Simple AI on Termux (Android)

Simple AI is an offline AI research prototype built from scratch. Its current
language model is experimental and does not yet provide reliable answers.

## 1. Install tools

```bash
pkg update
pkg upgrade
pkg install python git clang cmake
python -m pip install pytest
```

## 2. Update and test

```bash
cd ~/simple_ai
git pull origin main
PYTHONPATH=python python -m pytest -q
```

## 3. Train the Indonesian BPE baseline

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model
```

Defaults: BPE tokenizer, vocabulary target 128, context 32, 20 epochs.
The actual vocabulary may be smaller if the corpus has too few repeated pairs.
The run saves model, tokenizer, and report under `artifacts/indonesian_model/`.

Try the chat:

```bash
PYTHONPATH=python python -m simple_ai.chat
```

The chat loads the Indonesian checkpoint by default. Keep the model and tokenizer
from the same training run together.

## 4. Compare character tokenization

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer char --output-dir artifacts/indonesian_char
```

## 5. Train from your own data

Plain text:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --input /path/to/indonesian_corpus.txt --tokenizer bpe --vocab-size 512
```

JSONL conversations (one JSON object per line with non-empty `user` and
`assistant` strings):

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --conversations data/conversations.example.jsonl
```

Use only data you have permission to train on. Keep private corpora outside Git.

## 6. Build the C++ prototype

```bash
cd cpp
cmake -S . -B build
cmake --build build
./build/simple_ai_native_demo
```

## Important limitations

- BPE is a small educational Unicode-character merge algorithm, not a production tokenizer.
- The model is still a fixed-context MLP, not a Transformer.
- The chat is a text-generation test harness, not proof of comprehension.
- Validation is held out from tokenizer fitting, but one document is not an independent benchmark.
- RAM/CPU limits remain deferred until we benchmark on the target Android device.
