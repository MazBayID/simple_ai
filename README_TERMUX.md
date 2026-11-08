# Test Simple AI on Termux (Android)

Simple AI is an offline AI research prototype built from scratch. It is not
yet a reliable assistant.

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

## 3. Train the Indonesian baseline

Default: BPE tokenizer, MLP architecture, vocabulary target 128, context 32,
20 epochs. The actual BPE vocabulary may be smaller if there are not enough
repeated pairs in the corpus.

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model
```

The model, tokenizer, and report are saved under
`artifacts/indonesian_model/`.

## 4. Try the causal Transformer

Start small because this pure-Python implementation is slower than the MLP:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer bpe --architecture transformer --vocab-size 128 \
  --context-size 16 --embedding-dim 8 --epochs 5 --batch-size 8
```

This overwrites the default Indonesian checkpoint. Keep its model and tokenizer
together, then run:

```bash
PYTHONPATH=python python -m simple_ai.chat
```

## 5. Compare character tokenization

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --tokenizer char --output-dir artifacts/indonesian_char
```

## 6. Train from your own data

Plain UTF-8 text:

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --input /path/to/indonesian_corpus.txt --tokenizer bpe --vocab-size 512
```

JSONL dialogue records (one object per line with non-empty `user` and
`assistant` strings):

```bash
PYTHONPATH=python python -m simple_ai.experiments.train_conversation_model \
  --conversations data/conversations.example.jsonl
```

For a separate validation corpus (recommended for comparisons), add
`--validation-input /path/to/validation.txt`. The tokenizer is fitted only on
the training file; unseen validation characters map to the unknown token.

Only use data you have permission to train on. Keep private corpora outside Git.

## 7. Build the C++ prototype

```bash
cd cpp
cmake -S . -B build
cmake --build build
./build/simple_ai_native_demo
```

## Important limitations

- BPE is an educational Unicode-character merge tokenizer, not a production tokenizer.
- The Transformer is one block and one attention head; it is a learning prototype.
- The chat is a text-generation harness, not proof of comprehension.
- Validation comes from one held-out region of the source, not an independent benchmark.
- RAM/CPU limits remain deferred until we benchmark on the target Android device.
